import torch.nn as nn
import numpy as np
import torch
from tool.networktools import init_weights
from .Backbone.backbone import make_backbone
from .Neck.neck import make_neck
from .Head.head import make_head
import time


class FPI(nn.Module):
    def __init__(self, opt):
        super(FPI, self).__init__()
        self.opt = opt
        # backbone init
        self.backbone_uav = make_backbone(opt, opt.UAVhw)
        self.backbone_satellite = make_backbone(opt, opt.Satellitehw)
        opt.backbone_output_channel = self.backbone_uav.backbone_out_channel
        
        # neck init
        opt.neck_output_channel = self.backbone_uav.backbone_out_channel[0]
        self.neck = make_neck(opt)

        # head init
        opt.nums_head_input = len(self.backbone_uav.backbone_out_channel)
        self.head = make_head(opt)


    def forward(self, z, x):
        # start_time = time.time()
        # backbone forward
        z = self.backbone_uav(z)
        x = self.backbone_satellite(x)
        # print("backbone_time:{}".format(time.time()-start_time))
        # start_time = time.time()
        # neck forward
        neck_z = self.neck(z)
        neck_x = self.neck(x)
        # print("neck_time:{}".format(time.time()-start_time))
        # start_time = time.time()
        # head forward
        # z = self.vector2array(z)
        # x = self.vector2array(x)
        cls_out = self.head(neck_z, neck_x[0])
        # print("head_time:{}".format(time.time()-start_time))
        
        return cls_out, None

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

    def get_part(self, x, padding_time):
        h, w = x.shape[-2:]
        cx, cy = h // 2, w // 2
        ch, cw = h // (padding_time + 1) // 2, w // (padding_time + 1) // 2
        x1, y1, x2, y2 = cx - ch, cy - cw, cx + ch + 1, cy + cw + 1
        part = x[:, :, int(x1):int(x2), int(y1):int(y2)]
        return part
    

def make_model(opt):
    model = FPI(opt)
    return model
