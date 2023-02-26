import torch.nn as nn
from .FSRA.make_model import make_transformer_model, make_cnn_model
from .head import SiamFC_HEAD, FeatureFusion, SiamRPN_HEAD, SwinTrack
import numpy as np
# from .SwinTrack.network import SwinTrack
import torch
from tool.networktools import init_weights


Transformer_model_list = ["Deit-S", "Vit-S",
                          "Swin-Transformer-S", "Pvt-T", "Pvt-S"]
CNN_model_list = ["Resnet50", "Resnest50", "VAN-S"]


class SiamUAV_Transformer_Model(nn.Module):
    def __init__(self, opt):
        super(SiamUAV_Transformer_Model, self).__init__()
        backbone = opt.backbone
        self.model_uav = make_transformer_model(
            opt, opt.UAVhw, transformer_name=backbone)
        if not opt.share:
            self.model_satellite = make_transformer_model(
                opt, opt.Satellitehw, transformer_name=backbone)
        else:
            self.model_satellite = self.model_uav

        # self.head = FeatureFusion(opt)
        # self.head = SiamRPN_HEAD()
        self.head = SiamFC_HEAD(muti_head=False)

        # self.head = SwinTrack(opt)
        # init_weights(self.head)
        self.opt = opt

    def forward(self, z, x):
        z = self.model_uav(z)
        x = self.model_satellite(x)
        # siamrpn
        # z = self.vector2array(z)
        # x = self.vector2array(x)
        # # if self.opt.padding:
        # #     a_z = self.get_part(a_z, self.opt.padding)
        # # # cls
        cls = self.head(z, x,cnn=True)
        return cls, None

        # swintrack head
        # cls = self.head(z, x)
        # cls = self.vector2array(cls)
        # return cls,None

        # featurefusion
        # map = self.head(z,x)
        # map = self.vector2array(map)

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

    def load_params(self, opt):
        # load pretrain param
        if opt.backbone == "Vit-S":
            pretrain = "/media/dmmm/4T-3/demo/SiamUAV/pretrain_model/vit_small_p16_224-15ec54c9.pth"
        if opt.backbone == "Deit-S":
            pretrain = "/media/dmmm/4T-3/demo/SiamUAV/pretrain_model/deit_small_distilled_patch16_224-649709d9.pth"
        if opt.backbone == "Swin-Transformer-S":
            pretrain = "/media/dmmm/4T-3/demo/SiamUAV/pretrain_model/swin_small_patch4_window7_224.pth"
        if opt.backbone == "Pvt-T":
            pretrain = "/media/dmmm/4T-3/demo/SiamUAV/pretrain_model/pvt_tiny.pth"
        if opt.backbone == "Pvt-S":
            pretrain = "/media/dmmm/4T-3/demo/SiamUAV/pretrain_model/pvt_small.pth"
        self.model_uav.transformer.load_param(pretrain)
        if not opt.share:
            self.model_satellite.transformer.load_param(pretrain)
        if opt.checkpoint:
            pretran_model = torch.load(opt.checkpoint)
            model2_dict = self.state_dict()
            state_dict = {k: v for k, v in pretran_model.items()
                          if k in model2_dict.keys()}
            model2_dict.update(state_dict)
            self.load_state_dict(model2_dict)


class SiamUAV_CNN_Model(nn.Module):
    def __init__(self, opt):
        super(SiamUAV_CNN_Model, self).__init__()
        backbone = opt.backbone
        self.model_uav = make_cnn_model(backbone)
        if not opt.share:
            self.model_satellite = make_cnn_model(backbone)
        self.head = SiamFC_HEAD()
        self.opt = opt

    def forward(self, z, x):
        z = self.model_uav(z)
        if self.opt.share:
            x = self.model_uav(x)
        else:
            x = self.model_satellite(x)
        map = self.head(z, x, cnn=True)
        return map, None


def make_model(opt, pretrain=True):
    if opt.backbone in Transformer_model_list:
        model = SiamUAV_Transformer_Model(opt)
        if pretrain:
            model.load_params(opt)
    if opt.backbone in CNN_model_list:
        model = SiamUAV_CNN_Model(opt)
    return model
