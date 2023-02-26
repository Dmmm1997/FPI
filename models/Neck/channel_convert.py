# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

class CCN(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(CCN,self).__init__()
        self.Conv2dList = nn.ModuleList()
        for in_channel in input_channels:
            self.Conv2dList.append(nn.Conv2d(in_channel,output_channels,kernel_size=(3,3),padding=1))

    def forward(self, x):
        output_features = []
        for ind in range(len(x)):
            output_features.append(self.Conv2dList[ind](x[ind]))
        return output_features
