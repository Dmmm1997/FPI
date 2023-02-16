import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .modules.encoder.concatenated_fusion.concatenated_fusion import ConcatenatedFusion
from .modules.decoder.concatenated_fusion import ConcatenationBasedDecoder



class SwinTrack(nn.Module):
    def __init__(self, opt):
        super(SwinTrack, self).__init__()
        original_dim = 384
        opt.dim=48
        self.linear1 = nn.Linear(original_dim,opt.dim)
        self.linear2 = nn.Linear(original_dim,opt.dim)
        self.encoder = ConcatenatedFusion(opt)
        self.decoder = ConcatenationBasedDecoder(opt)
        self.out_norm = nn.LayerNorm(opt.dim)
        # cls and loc
        self.cls_linear = nn.Linear(opt.dim,1)
        self.loc_linear = nn.Linear(opt.dim,2)
        # self.head = head
        #
        # self.z_backbone_out_stage = z_backbone_out_stage
        # self.x_backbone_out_stage = x_backbone_out_stage
        # self.z_input_projection = z_input_projection
        # self.x_input_projection = x_input_projection
        #
        # self.z_pos_enc = z_pos_enc
        # self.x_pos_enc = x_pos_enc

        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        # self.encoder.apply(_init_weights)
        # self.decoder.apply(_init_weights)

    def forward(self, z, x):
        z_pos = None
        x_pos = None
        z_feat = self.linear1(z)
        x_feat = self.linear2(x)
        z_feat, x_feat = self.encoder(z_feat, x_feat, z_pos, x_pos)
        decoder_feat = self.decoder(z_feat, x_feat, z_pos, x_pos)
        decoder_feat = self.out_norm(decoder_feat)
        cls_feat = self.cls_linear(decoder_feat)
        # loc_feat = self.loc_linear(decoder_feat)

        return cls_feat  # B*1*25*25 B*2*25*25
