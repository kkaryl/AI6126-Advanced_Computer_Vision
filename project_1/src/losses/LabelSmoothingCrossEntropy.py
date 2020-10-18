import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import *

__all__ = ['LabelSmoothingCrossEntropy']

# References:
# fastai course

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy
    """
    def __init__(self, ls:float=0.1, reduction='mean'):
        super().__init__()
        self.ls = ls
        self.reduction = reduction
        self.name = "CELS"
    
    def forward(self, preds, target):
        n_classes = preds.size()[-1] #2
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/n_classes, nll, self.ls)
    

    