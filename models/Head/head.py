import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .attentionfusion import AttentionFusion
from .groupfusion import MultiGroupFusionHead, SingleGroupFusionHead


def make_head(opt):
    head_model = Head(opt)
    return head_model


class Head(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.head = self.init_head(opt.head)

    def init_head(self, head):
        if head == "SingleGroupFusionHead":
            head_model = SingleGroupFusionHead()
        elif head == "MultiGroupFusionHead":
            head_model = MultiGroupFusionHead(pool = self.opt.head_pool, muti_level_nums = self.opt.nums_head_input)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, z, x):
        return self.head(z, x)
