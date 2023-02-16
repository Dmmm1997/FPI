from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
from .FSRA.backbones.vit_pytorch import Block
import numpy as np


__all__ = ['SiamFC_HEAD','FeatureFusion']


class SwinTrack(nn.Module):
    def __init__(self, opt):
        super(SwinTrack, self).__init__()
        original_dim = 384
        opt.dim = 48
        self.linear1 = nn.Linear(original_dim, opt.dim)
        self.linear2 = nn.Linear(original_dim, opt.dim)
        self.z_patches = opt.UAVhw[0] // 16 * opt.UAVhw[1] // 16
        self.x_patches = opt.Satellitehw[0] // 16 * opt.Satellitehw[1] // 16
        num_patches = self.z_patches+self.x_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, opt.dim))
        self.attnBlock = nn.Sequential(Block(dim=opt.dim,num_heads=12))
        self.out_norm = nn.LayerNorm(opt.dim)
        # cls and loc
        self.cls_linear = nn.Linear(opt.dim, 1)
        self.loc_linear = nn.Linear(opt.dim, 2)

    def forward(self, z, x):
        z_feat = self.linear1(z)
        x_feat = self.linear2(x)
        concat_feature = torch.concat((z_feat, x_feat), dim=1)
        concat_feature += self.pos_embed
        out_feature = self.attnBlock(concat_feature)[:,self.z_patches:,:]
        out_feature = self.out_norm(out_feature)
        cls_feat = self.cls_linear(out_feature)
        # loc_feat = self.loc_linear(decoder_feat)

        return cls_feat  # B*1*25*25 B*2*25*25


class SiamFC_HEAD(nn.Module):
    def __init__(self, out_scale=0.001, muti_head=False):
        super(SiamFC_HEAD, self).__init__()
        self.out_scale = out_scale
        self.muti_head = muti_head

    def forward(self, z, x , cnn=False):
        if self.muti_head:
            assert len(z)==len(x), "输入尺度不对应！！！"
            res = []
            for i,j in zip(z,x):
                i = self.vector2array(i)
                j = self.vector2array(j)
                res.append(self._fast_xcorr(i, j) * self.out_scale)
            return sum(res)/len(res)
        else:
            if not cnn:
                z = self.vector2array(z)
                x = self.vector2array(x)
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

    def vector2array(self, vector):
        n, p, c = vector.shape
        h = w = np.sqrt(p)
        if int(h) * int(w) != int(p):
            raise ValueError("p can not be sqrt")
        else:
            h = int(h)
            w = int(w)
        array = vector.permute(0, 2, 1).contiguous().view(n, c, h, w)
        return array


class SiamRPN_HEAD(nn.Module):
    def __init__(self, out_scale = 0.001,in_channels=384):
        super(SiamRPN_HEAD, self).__init__()
        self.out_scale = out_scale
        loc_output = 2

        self.template_cls_conv = nn.Conv2d(in_channels,
                                           in_channels * loc_output, kernel_size=1)


        self.search_cls_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=1)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)
        self.norm = nn.BatchNorm2d(loc_output)


    def forward(self, z, x):
        template = self.template_cls_conv(z)
        search = self.search_cls_conv(x)
        cls = self._rpn_fast_xcorr(z, x)*self.out_scale
        loc = self._rpn_fast_xcorr(template,search)
        loc = self.loc_adjust(loc)*self.out_scale
        # loc = self.norm(loc)
        return cls,loc


    def _rpn_fast_xcorr(self,kernel,x):
        """group conv2d to calculate cross correlation, fast version
            """
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])  # [b,384,7,7]
        px = x.view(1, -1, x.size()[2], x.size()[3])  # [1,2*384,25,25]
        po = F.conv2d(px, pk, groups=batch,padding=kernel.size()[2]//2)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po



class FeatureFusion(nn.Module):
    def __init__(self,opt):
        super(FeatureFusion, self).__init__()
        if opt.backbone == "Deit-S":
            ndim = 384
        elif opt.backbone == "Vit-S":
            ndim = 768
        else:
            raise NameError("!!!!!!!")
        self.cross_attn1 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.cross_attn2 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.cross_attn3 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.norm1 = nn.LayerNorm(ndim)
        self.norm2 = nn.LayerNorm(ndim)
        self.norm3 = nn.LayerNorm(ndim)

        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(ndim, 256)

        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)

        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, 64)

        self.proj = nn.Linear(64, 1)


    def forward(self,z,x):
        k = v = z.transpose(0,1).contiguous()
        q = x.transpose(0,1).contiguous()
        k1 = self.cross_attn1(query=k, key=q, value=q)[0]+v  # 25*25，B，384
        k1 = v1 = self.norm1(k1)

        src1 = self.cross_attn2(query=q,key=k1,value=v1)[0]+q # 25*25，B，384
        src1 = self.norm2(src1)
        src2 = self.cross_attn3(query=src1,key=src1,value=src1)[0] + src1
        src2 = self.norm3(src2)
        src2 = src2.transpose(0,1).contiguous()

        res1 = self.dropout1(self.activation1(self.linear1(src2)))
        res2 = self.dropout2(self.activation2(self.linear2(res1)))+res1
        res = self.dropout3(self.activation3(self.linear3(res2)))
        res = self.proj(res)
        return res

