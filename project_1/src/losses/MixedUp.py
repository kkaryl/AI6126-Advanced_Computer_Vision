import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import *

__all__ = ['MixedUp']

# References:
# fastai course

class MixedUp(nn.Module):
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
        self.lam = 1
        self.name = "MU_" + self.base_criterion.name
        
    def set_lambda(self, lam):
        self.lam = lam

    def forward(self, preds, targets, mixed=True):
        if mixed:
            y_a, y_b = targets

            with NoneReduce(self.base_criterion) as loss_func:
                loss1 = loss_func(preds, y_a)
                loss2 = loss_func(preds, y_b)

            #lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
            loss = lin_comb(loss1, loss2, self.lam) 
            return reduce_loss(loss, getattr(self.base_criterion, 'reduction', 'mean'))
        else:
            return self.base_criterion(preds, targets)

            
