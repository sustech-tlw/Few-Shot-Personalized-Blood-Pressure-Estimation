import numpy as np
from sklearn.metrics import r2_score
import os
import sys
from io import StringIO
import torch
import shutil


def norm(x):
    # norm a tensor on the last dimension
    mean = x.mean(dim=-1, keepdim=True)  # [b, h, 1]
    x = x - mean
    std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
    x = x / std
    return x

def save_checkpoint(state, is_best, fold_path='', file_name='checkpoint.pth.tar'):
    path = os.path.join(fold_path, file_name)
    path_best = os.path.join(fold_path, 'model_best.pth.tar')
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, path_best)

def regression_metric(output, target):
    R2 = r2_score(target, output)
    ME = np.mean(target - output)
    SDE = np.std(abs(target - output))
    SD = np.std(target - output)
    MAE = np.mean(np.abs(output - target))
    MSE = np.mean((output - target)**2)
    return (MAE, SD, R2, ME, MSE, SDE)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def log_and_print(metric_epoch, epoch, save_path, save_name):
    save_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    print(
        " * Train Epoch {Epoch} MSE {MSE:.3f} MAE {MAE:.3f} SDE {SDE:.3f} R2 {R2:.3f} ME {ME:.3f} SD {SD:.3f}".format(
            Epoch=epoch, MAE=metric_epoch[0], SD=metric_epoch[1], R2=metric_epoch[2], ME=metric_epoch[3],
            MSE=metric_epoch[4], SDE=metric_epoch[5])

    )
    sys.stdout = save_stdout
    with open(os.path.join(save_path, save_name + '.txt'), 'a') as f:
        f.write(result.getvalue())

    print(
        " * Train Epoch {Epoch} MSE {MSE:.3f} MAE {MAE:.3f} SDE {SDE:.3f} R2 {R2:.3f} ME {ME:.3f} SD {SD:.3f}".format(
            Epoch=epoch, MAE=metric_epoch[0], SD=metric_epoch[1], R2=metric_epoch[2], ME=metric_epoch[3],
            MSE=metric_epoch[4], SDE=metric_epoch[5])

    )


def log_and_print_gridsim(cos_sim, epoch, save_path, save_name):
    save_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    print(
        " * Train Epoch {Epoch:.4f} grad_sim {cos_sim}".format(
            Epoch=epoch, cos_sim=cos_sim)

    )
    sys.stdout = save_stdout
    with open(os.path.join(save_path, save_name + '.txt'), 'a') as f:
        f.write(result.getvalue())

    print(
        " * Train Epoch {Epoch:.4f} grad_sim {cos_sim}".format(
            Epoch=epoch, cos_sim=cos_sim)

    )