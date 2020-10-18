from functools import partial

# References:
# fastai course

def lin_comb(value1, value2, beta): 
    return beta*value1 + (1-beta)* value2

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class NoneReduce():
    def __init__(self, loss_func):
        self.loss_func,self.old_red = loss_func, None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)