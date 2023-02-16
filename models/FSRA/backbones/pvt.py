import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], F4=False, num_stages=4):
        super().__init__()
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)
        # self.init_weights(pretrained)


    def load_param(self,checkpoint):
        pretran_model = torch.load(checkpoint)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs[2]

    def forward(self, x):
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict




class pvt_tiny(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_tiny, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class pvt_small(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_small, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
#
#
# @BACKBONES.register_module()
# class pvt_small_f4(PyramidVisionTransformer):
#     def __init__(self, **kwargs):
#         super(pvt_small_f4, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
#             sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, F4=True, pretrained=kwargs['pretrained'])
#
#
# @BACKBONES.register_module()
# class pvt_medium(PyramidVisionTransformer):
#     def __init__(self, **kwargs):
#         super(pvt_medium, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
#             sr_ratios=[8, 4, 2, 1], pretrained=kwargs['pretrained'])
#
#
# @BACKBONES.register_module()
# class pvt_large(PyramidVisionTransformer):
#     def __init__(self, **kwargs):
#         super(pvt_large, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
#             sr_ratios=[8, 4, 2, 1], pretrained=kwargs['pretrained'])


# class PyramidFeatures(nn.Module):
#     def __init__(self, C3_size, C4_size, C5_size, feature_size=384): #384 1
#         super(PyramidFeatures, self).__init__()
#
#         # upsample C5 to get P5 from the FPN paper
#         self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
#         self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
#
#         # add P5 elementwise to C4
#         self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
#         self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
#
#         # add P4 elementwise to C3
#         self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
#         self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
#
#         # "P6 is obtained via a 3x3 stride-2 conv on C5"
#         self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
#
#         # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
#         self.P7_1 = nn.ReLU()
#         self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
#
#     def forward(self, inputs):
#         C3, C4, C5= inputs
#
#         P5_x = self.P5_1(C5)
#         P5_upsampled_x = self.P5_upsampled(P5_x)
#         P5_x = self.P5_2(P5_x)
#
#         P4_x = self.P4_1(C4)
#         P4_x = P5_upsampled_x + P4_x
#         P4_upsampled_x = self.P4_upsampled(P4_x)
#         P4_x = self.P4_2(P4_x)
#
#         P3_x = self.P3_1(C3)
#         P3_x = P3_x + P4_upsampled_x
#         P3_x = self.P3_2(P3_x)
#
#         P6_x = self.P6(C5)
#         #panet
#         P7_x = self.P7_1(P6_x)
#         P7_x = self.P7_2(P7_x)
#         return [P3_x, P4_x, P5_x]
#         # fpn
#         # return P3_x
#
# class Panet(nn.Module):
#     def __init__(self, class_number=512):
#         super(Panet, self).__init__()
#         self.fpn = PyramidFeatures(1,1,1)
#         self.convN = nn.Conv2d(1, 1, 3, 2, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.gelu = nn.GELU()
#         # self.convN3 = nn.Conv2d(555, 384, 3, 2, 1)
#         self.upsampled = nn.Upsample(scale_factor=2, mode='nearest')
#
#     def forward(self, x):
#         # P2, P3, P4 = self.fpn(x)
#         # N2 = P2
#         # N2_ = self.convN(N2)
#         # N2_ = self.relu(N2_)
#         #
#         # N3_ = torch.cat((N2_, P3), dim=1)
#         # N3 = self.convN2(N3_)
# ##################################################
#         #P2, P3, P4 = self.fpn(x)
#         #N2 = P2
#         #N2_ = self.convN(N2)
#        # N2_ = self.gelu(N2_)
#
#        # N3_ = torch.cat((N2_, P3), dim=1)
#        # N3 = self.convN2(N3_)
#         # N3= self.relu(N3)
#         # N3 =  self.upsampled(N3)
#         # N3 = torch.cat((N2, N3), dim=1)
#         # N3 = self.convN2(N3)
#
#
# ################################################
#         P2, P3, P4 = self.fpn(x)
#         N2 = P2
#         N2_ = self.convN(N2)
#         N2_ = self.relu(N2_)
#         #
#         N3 = N2_ + P3
#         #
#         N3_ = self.convN(N3)
#         N3_ = self.relu(N3_)
#
#         # N3 = self.upsampled(N3)
#         # N2 = N3 +N2
#         #
#         N4 = N3_ + P4
#         #N4 = torch.cat((N3_, P4), dim=1)
#         #
#         # # N4_ = self.convN(N4)
#         # # N4_ = self.relu(N4_)
#         # # # N5 = N4_ + P5
# #####################################################
#         return N2, N3, N4
#
#
#
#
# class FPN(nn.Module):
#     def __init__(self, block, layers):
#         super(FPN, self).__init__()
#         self.in_planes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         # Bottom-up layers
#         self.layer1 = self._make_layer(block,  64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         # Top layer
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_planes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#         layers = []
#         layers.append(block(self.in_planes, planes, stride, downsample))
#         self.in_planes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#         return nn.Sequential(*layers)
#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
#     def forward(self, x):
#         # Bottom-up
#         c1 = F.relu(self.bn1(self.conv1(x)))
#         c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
#         #print(f'c1:{c1.shape}')
#         c2 = self.layer1(c1)
#         #print(f'c2:{c2.shape}')
#         c3 = self.layer2(c2)
#         #print(f'c3:{c3.shape}')
#         c4 = self.layer3(c3)
#         #print(f'c4:{c4.shape}')
#         c5 = self.layer4(c4)
#         #print(f'c5:{c5.shape}')
#         # Top-down
#         p5 = self.toplayer(c5)
#         #print(f'p5:{p5.shape}')
#         p4 = self._upsample_add(p5, self.latlayer1(c4))
#         #print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
#         p3 = self._upsample_add(p4, self.latlayer2(c3))
#         #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
#         p2 = self._upsample_add(p3, self.latlayer3(c2))
#         #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
#         # Smooth
#         p4 = self.smooth1(p4)
#         p3 = self.smooth2(p3)
#         p2 = self.smooth3(p2)
#         return p2, p3, p4, p5