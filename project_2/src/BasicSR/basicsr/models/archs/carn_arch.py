import torch
from torch import nn as nn

from basicsr.models.archs.arch_util import (ResidualBlockNoBN, Upsample, default_init_weights)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, pytorch_init=False):
        super(BasicBlock, self).__init__()

        conv1 = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)
        relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([conv1], 0.1)

        self.body = nn.Sequential(
            conv1,
            relu
        )

    def forward(self, x):
        out = self.body(x)
        return out


class Block(nn.Module):
    """
    https://github.com/nmhkahn/CARN-pytorch/blob/master/carn/model/carn.py
    """
    def __init__(self, num_feat, res_scale):
        super(Block, self).__init__()

        self.b1 = ResidualBlockNoBN(num_feat=num_feat, res_scale=res_scale)
        self.b2 = ResidualBlockNoBN(num_feat=num_feat, res_scale=res_scale)
        self.b3 = ResidualBlockNoBN(num_feat=num_feat, res_scale=res_scale)
        self.c1 = BasicBlock(num_feat * 2, num_feat, 1, 1, 0)
        self.c2 = BasicBlock(num_feat * 3, num_feat, 1, 1, 0)
        self.c3 = BasicBlock(num_feat * 4, num_feat, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3

class CARN(nn.Module):
    """CARN network structure.

    Paper: Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    Ref git repo: https://github.com/nmhkahn/CARN-pytorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 #num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(CARN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.entry = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.b1 = Block(num_feat, res_scale)
        self.b2 = Block(num_feat, res_scale)
        self.b3 = Block(num_feat, res_scale)
        self.c1 = BasicBlock(num_feat * 2, num_feat, 1, 1, 0)
        self.c2 = BasicBlock(num_feat * 3, num_feat, 1, 1, 0)
        self.c3 = BasicBlock(num_feat * 4, num_feat, 1, 1, 0)

        self.upsample = Upsample(upscale, num_feat)
        self.exit = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.entry(x)

        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        x = self.exit(self.upsample(o3))
        x = x / self.img_range + self.mean

        return x
