import numpy as np
import torch.nn.functional as F
import random

"""
    Rethinking Data Augmentation for Image Super-resolution (CVPR 2020)
    https://arxiv.org/pdf/2004.00448.pdf
    https://github.com/clovaai/cutblur/
"""
def match_resolution(im1, im2):
    if im1.size() != im2.size():
        scale = im1.size(2) // im2.size(2)
        im2 = F.interpolate(im2, scale_factor=scale, mode="bilinear")
    return im1, im2

def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    # match the resolution of (LR, HR) due to CutBlur
    if im1.size() != im2.size():
        scale = im1.size(2) // im2.size(2)
        im2 = F.interpolate(im2, scale_factor=scale, mode="nearest")

    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    # resize back
    im2 = F.interpolate(im2, scale_factor=1/scale, mode="nearest")

    return im1, im2

def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2

# def _cutmix(im2, prob=1.0, alpha=1.0):
#
#     if alpha <= 0 or np.random.rand(1) >= prob:
#         return None
#
#     cut_ratio = np.random.randn() * 0.01 + alpha
#
#     h, w = im2.size(2), im2.size(3)
#     ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
#
#     fcy = np.random.randint(0, h-ch+1)
#     fcx = np.random.randint(0, w-cw+1)
#     tcy, tcx = fcy, fcx
#     rindex = torch.randperm(im2.size(0)).to(im2.device)
#
#     return {
#         "rindex": rindex, "ch": ch, "cw": cw,
#         "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
#     }
#
# def cutmix(im1, im2, prob=1.0, alpha=1.0):
#     """
#     https://github.com/clovaai/cutblur/
#     """
#     c = _cutmix(im2, prob, alpha)
#     if c is None:
#         return im1, im2
#
#     scale = im1.size(2) // im2.size(2)
#     rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
#     tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]
#
#     hch, hcw = ch*scale, cw*scale
#     hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale
#
#     im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
#     im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]
#
#     return im1, im2