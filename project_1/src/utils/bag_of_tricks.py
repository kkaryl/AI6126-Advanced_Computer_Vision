from __future__ import print_function
import torch
import torch.nn as nn

__all__ = ['add_weight_decay']

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