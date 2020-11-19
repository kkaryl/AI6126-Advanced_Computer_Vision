import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import (ResidualBlockNoBN, Upsample, default_init_weights)

"""
https://github.com/njulj/RFDN/blob/master/block.py
"""

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = 0
    modules = [
        nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups),
        nn.LeakyReLU(0.05, inplace=True)
    ]
    return nn.Sequential(*modules) #sequential(p, c, n, a)

class RFDN(nn.Module):
    """RFDN network structure.

    Paper: Residual Feature Distillation Network for Lightweight Image Super-Resolution
    Ref git repo: https://arxiv.org/pdf/2009.11551.pdf

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
                 num_block=5,
                 upscale=4):
        super(RFDN, self).__init__()
        self.fea_conv = conv_layer(num_in_ch, num_feat, kernel_size=3)

        self.B1 = RFDB(in_channels=num_feat)
        self.B2 = RFDB(in_channels=num_feat)
        self.B3 = RFDB(in_channels=num_feat)
        self.B4 = RFDB(in_channels=num_feat)
        self.B5 = RFDB(in_channels=num_feat)
        #self.B6 = RFDB(in_channels=num_feat)
        self.c = conv_block(num_feat * num_block, num_feat, kernel_size=1)

        self.LR_conv = conv_layer(num_feat, num_feat, kernel_size=3)

        self.upsample = Upsample(upscale, num_feat)
        self.exit = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        #self.scale_idx = 0

    def forward(self, x):
        out_fea = self.fea_conv(x)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        #out_B6 = self.B6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.exit(self.upsample(out_lr))

        return output
