import torch
from torch import nn as nn

from basicsr.models.archs.arch_util import (ResidualBlockNoBN, Upsample, default_init_weights)

"""
https://github.com/clovaai/cutblur/blob/master/model/carn.py
"""

class Group(nn.Module):
    def __init__(self, num_channels, num_block, res_scale=1.0):
        super().__init__()

        for nb in range(num_block):
            setattr(self,
                "b{}".format(nb+1),
                ResidualBlockNoBN(num_channels, res_scale)
            )
            setattr(self,
                "c{}".format(nb+1),
                nn.Conv2d(num_channels*(nb+2), num_channels, 1, 1, 0)
            )
        self.num_block = num_block

    def forward(self, x):
        c = out = x
        for nb in range(self.num_block):
            unit_b = getattr(self, "b{}".format(nb+1))
            unit_c = getattr(self, "c{}".format(nb+1))

            b = unit_b(out)
            c = torch.cat([c, b], dim=1)
            out = unit_c(c)

        return out

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
                 num_block=3,
                 num_group=3,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(CARN, self).__init__()
        self.num_group = num_group
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.entry = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        for ng in range(num_group):
            setattr(self,
                    "c{}".format(ng + 1),
                    nn.Conv2d(num_feat * (ng + 2), num_feat, 1, 1, 0)
                    )
            setattr(self,
                    "b{}".format(ng + 1),
                    Group(num_feat, num_block, res_scale)
                    )

        # self.b1 = Block(num_feat, res_scale)
        # self.b2 = Block(num_feat, res_scale)
        # self.b3 = Block(num_feat, res_scale)
        # self.c1 = BasicBlock(num_feat * 2, num_feat, 1, 1, 0)
        # self.c2 = BasicBlock(num_feat * 3, num_feat, 1, 1, 0)
        # self.c3 = BasicBlock(num_feat * 4, num_feat, 1, 1, 0)

        self.upsample = Upsample(upscale, num_feat)
        self.exit = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.entry(x)

        c = out = x

        for ng in range(self.num_group):
            group = getattr(self, "b{}".format(ng + 1))
            conv = getattr(self, "c{}".format(ng + 1))

            g = group(out)
            c = torch.cat([c, g], dim=1)
            out = conv(c)
        res = out
        res += x

        x = self.exit(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
