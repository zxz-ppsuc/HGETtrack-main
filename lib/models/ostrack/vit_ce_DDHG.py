import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from lib.models.channelsplit.ssse import ssse
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt
from ..layers.DyT import DynamicTanh
_logger = logging.getLogger(__name__)

class LKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3,
                        groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x
    
class GFU(nn.Module):
    def __init__(self, channel_dim=768):
        super().__init__()

        self.feature_enhence = LKA(channel_dim)
        self.gate = nn.Sequential(
            nn.Sigmoid()
        )
        
    def forward(self, rgb, dte):
        rgb_enhence = self.feature_enhence(rgb)
        dte_enhence = self.feature_enhence(dte)
        gate_weights = self.gate(rgb_enhence + dte_enhence)
        
        fused = gate_weights * rgb + (1 - gate_weights) * dte
        return fused
    
class VisionTransformerCE(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
                 
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.ssse = ssse()
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_dte = embed_layer( 
            img_size=img_size, patch_size=patch_size, in_chans=16, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.gate = GFU()
        self.init_weights(weight_init)
        self.dynamic_tanh = DynamicTanh(
        normalized_shape = embed_dim,
        channels_last=False,
        alpha_init_value=0.5
        )
        self.gatenorm = norm_layer(embed_dim)
    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        x, z = x_rgb, z_rgb
        def weighted_fusion(fi, wi):
            sum_per_batch = wi.sum(dim=0, keepdim=True)
            normalized_wi = wi / sum_per_batch.clamp(min=1e-8)
            weights = [normalized_wi[i].view(-1, 1, 1) for i in range(5)]
            return sum(emb * weight for emb, weight in zip(fi, weights))
        bs_x  = self.ssse(x_dte) 
        x_embedded = [self.patch_embed(tensor) for tensor in bs_x[0]]
        x_dte = weighted_fusion(x_embedded, bs_x[1])
        
        bs_z  = self.ssse(z_dte) 
        z_embedded = [self.patch_embed(tensor) for tensor in bs_z[0]]
        z_dte = weighted_fusion(z_embedded, bs_z[1])

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        
        z_feat = token2feature(self.gatenorm(z))
        x_feat = token2feature(self.gatenorm(x))
        z_dte_feat = token2feature(self.gatenorm(z_dte))
        x_dte_feat = token2feature(self.gatenorm(x_dte))

        z_fused = self.gate(z_feat, z_dte_feat)
        x_fused = self.gate(x_feat, x_dte_feat)
        z = feature2token(z_fused)
        x = feature2token(x_fused)

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        z += self.temporal_pos_embed_z
        x += self.temporal_pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):       
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            checkpoint_model = checkpoint['model']
            key_names = list(checkpoint_model.keys())
            for key_name in key_names:
                if 'decoder' in key_name:
                    del checkpoint_model[key_name]

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            print(missing_keys)
            print(unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce_DDHG(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_DDHG(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model 