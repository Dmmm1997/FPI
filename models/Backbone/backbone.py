import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from models.Backbone.convnext import convnext_small, convnext_tiny
import timm
from .pcpvt import pcpvt_small

from models.FSRA.backbones.pvt import pvt_small, pvt_tiny


def make_backbone(opt,img_size):
    backbone_model = Backbone(opt,img_size)
    return backbone_model



class Backbone(nn.Module):
    def __init__(self, opt, img_size):
        super().__init__()
        self.opt = opt
        self.backbone, self.backbone_out_channel = self.init_backbone(opt.backbone,img_size)

    def init_backbone(self, backbone,img_size):
        if backbone=="Resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True,features_only=True,img_size = img_size)
            backbone_out_channel = [384]
        elif backbone=="Vit-S":
            backbone_model = timm.create_model("vit_small_patch16_224",pretrained=True, img_size=img_size)
            backbone_out_channel = [384]
        elif backbone=="Deit-S":
            backbone_model = timm.create_model("deit_small_distilled_patch16_224",pretrained=True, img_size=img_size)
        elif backbone == "Pvt-T":
            backbone_model = pvt_tiny(pretrained=True)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone == "pvt_small":
            backbone_model = pvt_small(pretrained=True)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone=="convnext_small":
            backbone_model = convnext_small(pretrained=True)
            backbone_out_channel = [96, 192, 384, 768]
        elif backbone=="convnext_tiny":
            backbone_model = convnext_tiny(pretrained=True)
            backbone_out_channel = [96, 192, 384, 768]
        elif backbone=="pcpvt_small":
            backbone_model = pcpvt_small(pretrained=True)
            backbone_out_channel = [64, 128, 320, 512]
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,backbone_out_channel

    def forward(self, image):
        backbone = self.opt.backbone
        if backbone=="Vit-S":
            features = self.backbone.forward_features(image)[:,1:]
        elif backbone in ["convnext_small","convnext_tiny"]:
            features = self.backbone.forward_features(image)
        elif backbone in ["Pvt-T","pvt_small","pcpvt_small"]:
            features = self.backbone.forward_features(image)
        return features