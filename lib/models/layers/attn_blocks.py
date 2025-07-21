import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
from .DyT import DynamicTanh

def candidate_elimination_prompt(tokens: torch.Tensor, lens_t: int, global_index: torch.Tensor):
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=global_index.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)

    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index

class FNA(nn.Module):
    def __init__(
            self,
            in_features=768,
            hidden_dim=4,
            groups=2,
            scale=1.0
    ):
        super().__init__()
        r = hidden_dim
        self.conv_A = nn.Conv1d(in_features, r, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(r, in_features, 1, groups=groups, bias=True)
        self.conv_M = nn.Conv1d(r, r, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.zeros_(self.conv_M.weight)
        nn.init.zeros_(self.conv_M.bias)
        self.groups = groups
        self.r = r
        self.scale = scale
        self.route_weight = nn.Parameter(torch.ones(1, in_features, 1))
        nn.init.ones_(self.route_weight)

        self.dynamic_tanh = DynamicTanh(
            normalized_shape=in_features,
            channels_last=False,
            alpha_init_value=0.5
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x

        routed_x = x * self.route_weight

        gated_x = self.dynamic_tanh(routed_x) 

        x_a = self.act(self.conv_A(gated_x))
        x_m = self.conv_M(self.dropout(x_a))
        x_combined = self.act(x_a + x_m)

        x_up = self.conv_B(x_combined)
        result = x_up * self.scale + residual

        return result.transpose(1, 2).contiguous()
    

class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, use_adapter=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.attn_rep = FNA(in_features=dim) if use_adapter else nn.Identity()
        self.mlp_rep = FNA(in_features=dim) if use_adapter else nn.Identity()
    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x_attn = self.attn_rep(x_attn)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp_rep(self.mlp(self.norm2(x))))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
