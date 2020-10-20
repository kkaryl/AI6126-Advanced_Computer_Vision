import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

#References:
#https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
#https://github.com/zhezh/focalloss/blob/master/focalloss.py
#https://www.kaggle.com/jimitshah777/bilinear-efficientnet-focal-loss-label-smoothing

__all__ = ['FocalLossLS']

def focal_loss_ls(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        ls: float = 0.1,
        classes: int = 2,
        eps: float = 1e-10) -> torch.Tensor:
    """Function that computes Focal loss with Label Smoothing.
    """
    if not torch.is_tensor(pred):
        raise TypeError("Pred type is not a torch.Tensor. Got {}"
                        .format(type(pred)))

    if not len(pred.shape) >= 2:
        raise ValueError("Invalid pred shape, we expect BxCx*. Got: {}"
                         .format(pred.shape))

    if pred.size(0) != target.size(0):
        raise ValueError('Expected pred batch_size ({}) to match target batch_size ({}).'
                         .format(pred.size(0), target.size(0)))

    n = pred.size(0)
    out_size = (n,) + pred.size()[2:]
    if target.size()[1:] != pred.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not pred.device == target.device:
        raise ValueError(
            "pred and target must be in the same device. Got: {} and {}" .format(
                pred.device, target.device))
    
    # compute softmax over the classes axis
    pred_soft: torch.Tensor = F.softmax(pred, dim=1) + eps
        
    # new: label smoothing
    if ls > 0:
        pred_ls = (1 - ls) * pred_soft + ls / classes
        pred_soft = torch.clamp(pred_ls, eps, 1.0-eps)    

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=pred.shape[1],
        device=pred.device, dtype=pred.dtype)

    # compute the actual focal loss
    weight = torch.pow(-pred_soft + 1., gamma)

    focal = -alpha * weight * torch.log(pred_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss

class FocalLossLS(nn.Module):
    """Criterion that computes Focal loss.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Pred: (N, C, *) where C = number of classes.
        - Target: (N, *)
    References:
        [1] https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none', ls:float = 0.1, classes: int = 2) -> None:
        super(FocalLossLS, self).__init__()
        if ls > 0:
            self.name = 'FLLS'
        else:
            self.name = 'FL'
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.ls: float = ls
        self.classes: int = classes
        self.eps: float = 1e-10
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss_ls(pred, target, self.alpha, self.gamma, self.reduction, self.ls, self.classes, self.eps)
    
def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    """Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape (N, C, *),
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps