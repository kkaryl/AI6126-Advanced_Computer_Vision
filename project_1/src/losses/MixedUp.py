import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import *

__all__ = ['MixedUp']

# References:
# fastai course

class MixedUp(nn.Module):
    def __init__(self, old_criterion):
        self.old_criterion = old_criterion
        self.lam = 1
        self.name = "MU_" + self.old_criterion.name
        
    def set_lambda(self, lam):
        self.lam = lam

    def forward(self, preds, targets):
        y_a, y_b = targets
        with NoneReduce(self.old_criterion) as loss_func:
            loss1 = loss_func(preds, y_a)
            loss2 = loss_func(preds, y_b)
            
        #lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
        loss = lin_comb(loss1, loss2, self.lam) 
        return reduce_loss(loss, getattr(self.old_criterion, 'reduction', 'mean'))
