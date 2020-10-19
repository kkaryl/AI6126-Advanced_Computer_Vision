from __future__ import print_function

import os
import sys
import torch

__all__ = ['AverageMeter', 'adjust_learning_rate', 'accuracy', 'reset_gpu_cache', 'print_attribute_acc', 'create_dir_ifne']

        
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class AttributeAverageMeter(AverageMeter):
    
    def __init__(self, idx, name):
        super(AttributeAverageMeter, self).__init__()
        self.idx = idx
        self.name


def adjust_learning_rate(optimizer, decay_type, epoch,
                         gamma=1., step=1, total_epochs=1,
                         turning_point=100, schedule=None):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if decay_type == 'step':
        lr = lr * (gamma ** (epoch // step))
    elif decay_type == 'cos':
        from math import cos, pi
        lr = lr * (1 + cos(pi * epoch / total_epochs)) / 2
    elif decay_type == 'linear':
        lr = lr * (1 - epoch / total_epochs)
    elif decay_type == 'linear2exp':
        if epoch < turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = lr * (1 - epoch / int(turning_point * 1.0526))
        else:
            lr *= gamma
    elif decay_type == 'schedule':
        if epoch in schedule:
            lr *= gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(decay_type))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def accuracy(output, target, topk=(1,), mixedup=0):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if mixedup > 0:
        target = target.gt(0.5).int()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reset_gpu_cache(model, optimizer, criterio, device):
    
    def _wipe_memory(cuda_obj, is_optimizer=False): 
        if is_optimizer:
            _gpu_to(cuda_obj, torch.device('cpu'))        
            del cuda_obj
            import gc
            gc.collect()
        else:
            cpu_obj = cuda_obj.to(torch.device('cpu'))
            cuda_obj = cuda_obj.to(device)
            
        torch.cuda.empty_cache()

    def _gpu_to(cuda_obj, device):
        for param in cuda_obj.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
                            
    if model:
        _wipe_memory(model)
    if optimizer:                        
        _wipe_memory(optimizer, is_optimizer=True)
    if criterion:
        _wipe_memory(criterion)

    
                            
def print_attribute_acc(top1, attribute_names):
    assert len(top1) == len(attribute_names)
    assert type(top1[0]) == AverageMeter
    for t, a in zip(top1, attribute_names):
        print(f"{a}: {t.avg}")
    return {a:t.avg for t, a in zip(top1, attribute_names)}
        
def view_bar(num, total):
    """

    :param num:
    :param total:
    :return:
    """
    rate = float(num + 1) / total
    rate_num = int(rate * 100)
    if num != total:
        r = '\r[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    else:
        r = '\r[%s%s]%d%%' % ("=" * 100, " " * 0, 100,)
    sys.stdout.write(r)
    sys.stdout.flush()

def create_dir_ifne(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)