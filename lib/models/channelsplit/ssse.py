import torch.nn as nn
import torch


def sc(fc, od):
    obi, odbd = od[0], od[1]  # odbd shape: (B, 16) torch.Size([16, 16])
    B = fc.size(0)
    
    # 将索引转换为每5组，每组3通道（处理前15通道）
    group_indices = odbd[:, :15].view(B, 5, 3)  # (B,5,3)
    h, w = fc.shape[2], fc.shape[3]
    
    # 向量化提取所有组
    fc_expanded = fc.unsqueeze(1)  # (B,1,16,H,W) torch.Size([16, 1, 16, 384, 384])
    fc_expanded = fc_expanded.expand(-1,5,-1,-1,-1)  # 显式扩展组维度
    indices_exp = group_indices.view(B,5,3,1,1).expand(-1,-1,-1,h,w)

    selected = torch.gather(fc_expanded, 2, indices_exp)  # (B,5,3,H,W)
    fi = [selected[:, i] for i in range(5)]  # 5*(B,3,H,W)
    
    # 计算权重（保持张量）
    obi_groups = obi[:, :15].view(B, 5, 3)  # (B,5,3)
    wi = obi_groups.mean(dim=2).permute(1, 0)  # (5,B)
    
    return fi, wi



class ssse(nn.Module):
    def __init__(self):
        super(ssse, self).__init__()
        channel = 16

        self.spectral = nn.Sequential(nn.Conv2d(channel, channel * 2,1, 1, bias=False),
                                      nn.BatchNorm2d(channel * 2),
                                      nn.ReLU(),
                                      nn.Conv2d(channel * 2, channel * 4,1, 1,bias=False),
                                      nn.BatchNorm2d(channel * 4),
                                      nn.ReLU(),
                                      )

        self.spatial = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 2,3, 1, 1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(),
            nn.Conv2d(channel * 2, channel * 2,3, 1, 1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(),
        )

        self.mlpd = nn.Sequential(nn.Linear(channel * 2, channel * 2, bias=True),nn.Tanh(),
                                         nn.Linear(channel * 2, channel, bias=True),
                                         nn.Tanh(), )

    def forward(self, x):
        # select

        # x = x.mul(1 / 255.0).clamp(0.0, 1.0)  ()
        it = x #B,16,256,256
        b1, c1, w1, h1 = x.size()
        x2 = self.spatial(self.spectral(x))  #torch.Size([32, 32, 256, 256])
        b, c, w, h = x2.size()
        x3 = x2.view(b, c, -1) #b,c,h*w  torch.Size([32, 32, 65536])?
        w0 = self.mlpd(x3.permute(0, 2, 1)) #b,h*w,c

        res = w0.permute(0, 2, 1) #b,c,h*w torch.Size([32, 16, 65536])
        y0 = res.mean(dim=2) # (b,c=16) torch.Size([32, 16])
        y1 = y0.view(b, 16, 1) #b,16,1 torch.Size([32, 16, 1])
        ty = y1.permute(0, 2, 1) #b,1,16

        C = torch.bmm(y1, ty) #y1 * y1的转置  => b,16,16 torch.Size([32, 16, 16])
        #for i in range(16):
         #   w0[:, i, i] = 0.0 #将对角线设为0
        # 关键修改：替换对角线置零操作
        # 生成与w0通道对齐的掩码
        diag_mask = torch.eye(C.size(1), device=C.device)          # (16,16)
        diag_mask = diag_mask.view(1, C.size(1), C.size(1), 1)      # (1,16,16,1)
        diag_mask = diag_mask.expand(C.size(0), -1, -1, -1)        # (B,16,16,1)
        diag_mask = diag_mask.reshape(C.size(0), C.size(1), -1)    # (B,16,16)
        C = C.masked_fill(diag_mask.bool(), 0.0)  # 使用bool类型掩码

        w = torch.norm(C, p=2, dim=2) #b,16 计算出权重矩阵 torch.Size([32, 16])

        oy = torch.sort(w, dim=-1, descending=True, out=None) #将w按照最后一个维度进行排序 b,16

        fi, wi = sc(x, oy) #均为2维列表，其中每个子列表fi[i] wi[i] 的torch.Size为([32, 3, 256, 256])

        a = it.view(b1, c1, -1) #b,16,h*w torch.Size([32, 16, 65536])
        cgh = torch.bmm(C, a) #b,16,16 torch.Size([32, 16, 65536])

        return fi, wi, oy, cgh