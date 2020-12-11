"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""

import numpy as np
import torch.nn.functional as F
import torch

"""
    Rethinking Data Augmentation for Image Super-resolution (CVPR 2020)
    https://arxiv.org/pdf/2004.00448.pdf
    https://github.com/clovaai/cutblur/
"""


def match_resolution(im1, im2):
    if im1.size() != im2.size():
        scale = im1.size(2) // im2.size(2)
        im2 = F.interpolate(im2, scale_factor=scale, mode="bilinear", align_corners=False)
    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
    # print(cut_ratio, ch, cw)
    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        # print("cut inside")
        im2[..., cy:cy + ch, cx:cx + cw] = im1[..., cy:cy + ch, cx:cx + cw]
    else:
        # print("cut outside")
        im2_aug = im1.clone()
        im2_aug[..., cy:cy + ch, cx:cx + cw] = im2[..., cy:cy + ch, cx:cx + cw]
        im2 = im2_aug

    ## resize back
    # im2 = F.interpolate(im2, scale_factor=1/scale, mode="nearest")

    return im1, im2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 1)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1 - v) * rim1
    im2 = v * im2 + (1 - v) * rim2

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1 - v) * im1[r_index, :]
    im2 = v * im2 + (1 - v) * im2[r_index, :]
    return im1, im2
