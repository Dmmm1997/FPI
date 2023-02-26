import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import timm
from .fpn import FPN
from .pafpn import PAFPN
from .channel_convert import CCN



def make_neck(opt):
    backbone_model = Neck(opt)
    return backbone_model


class Neck(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.neck = self.init_neck(opt)

        
    def init_neck(self, opt):
        if opt.neck=="FPN":
            neck_model = FPN(opt.backbone_output_channel,opt.neck_output_channel,len(opt.backbone_output_channel))
        elif opt.neck=="PAFPN":
            neck_model = PAFPN(opt.backbone_output_channel,opt.neck_output_channel,len(opt.backbone_output_channel))
        elif opt.neck=="CCN":
            neck_model = CCN(opt.backbone_output_channel,opt.neck_output_channel)
        else:
            raise NameError("{} not in the neck list!!!".format(opt.neck))
        return neck_model

    def forward(self, features):
        features = self.neck(features)
        return features



    
