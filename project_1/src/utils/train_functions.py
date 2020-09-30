from __future__ import print_function

import sys

__all__ = ['AverageMeter', 'adjust_learning_rate', 'accuracy']

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
