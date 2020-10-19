from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

__all__ = ['add_weight_decay', 'mixup_data']

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    Prevents bias or batch norm decay.
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    
    Usage:
        parameters = add_weight_decay(model, weight_decay)
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


@torch.no_grad()
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    https://github.com/PistonY/torch-toolbox/blob/c1227fce136de0e0e271769efff8755eeb1c11a5/torchtoolbox/tools/mixup.py#L11
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    y_pairs = torch.stack([y_a, y_b], dim=0)
    return mixed_x, y_pairs, lam
