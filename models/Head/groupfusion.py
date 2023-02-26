import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class SingleGroupFusionHead(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SingleGroupFusionHead, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w).contiguous()
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1)).contiguous()
        return out
    

class MultiGroupFusionHead(nn.Module):
    def __init__(self, out_scale=0.001, pool = "avg", muti_level_nums=4):
        super(MultiGroupFusionHead, self).__init__()
        self.out_scale = out_scale
        if pool == "avg":
            self.pool = ChannelPool(mode=pool)
        elif pool == "linear":
            self.pool = nn.Linear(muti_level_nums,1)
        else:
            self.pool = nn.Identity()

    def forward(self, z, x):
        # assert len(z)==len(x), "输入尺度不对应！！！"
        res = []
        for z_part in z:
            res.append(self._fast_xcorr(z_part, x) * self.out_scale)
        res = torch.concat(res,dim=1)
        res = self.pool(res)
        return res

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.contiguous().view(-1, nz * c, h, w).contiguous()
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1)).contiguous()
        return out
    

class ChannelPool(nn.Module):
    def __init__(self,mode = "avg"):
        super(ChannelPool,self).__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "avg":
            res = torch.mean(x,dim=1,keepdim=True)
        return res
