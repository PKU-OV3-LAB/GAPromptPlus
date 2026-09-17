from math import log10
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import logging

class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]

class ConfusionMatrix:
    """Accumulate a confusion matrix for a classification task.
    ignore_index only supports index <0, or > num_classes
    """

    def __init__(self, num_classes, ignore_index=None):
        self.value = 0
        self.num_classes = num_classes
        self.virtual_num_classes = num_classes + 1 if ignore_index is not None else num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        true = true.flatten()
        pred = pred.flatten()
        if self.ignore_index is not None:
            if (true == self.ignore_index).sum() > 0:
                pred[true == self.ignore_index] = self.virtual_num_classes -1
                true[true == self.ignore_index] = self.virtual_num_classes -1
        unique_mapping = true.flatten() * self.virtual_num_classes + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.virtual_num_classes**2)
        self.value += bins.view(self.virtual_num_classes, self.virtual_num_classes)[:self.num_classes, :self.num_classes]

    def reset(self):
        """Reset all accumulated values."""
        self.value = 0

    @property
    def tp(self):
        """Get the true positive samples per-class."""
        return self.value.diag()

    @property
    def actual(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=1)

    @property
    def predicted(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=0)

    @property
    def fn(self):
        """Get the false negative samples per-class."""
        return self.actual - self.tp

    @property
    def fp(self):
        """Get the false positive samples per-class."""
        return self.predicted - self.tp

    @property
    def tn(self):
        """Get the true negative samples per-class."""
        actual = self.actual
        predicted = self.predicted
        return actual.sum() + self.tp - (actual + predicted)

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        # return self.tp + self.fn
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denomenator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        return self.value.sum()

    @property
    def overall_accuray(self):
        return self.tp.sum() / self.total

    @property
    def union(self):
        return self.value.sum(dim=0) + self.value.sum(dim=1) - self.value.diag()

    def all_acc(self):
        return self.cal_acc(self.tp, self.count)

    @staticmethod
    def cal_acc(tp, count):
        acc_per_cls = tp / count.clamp(min=1) * 100
        over_all_acc = tp.sum() / count.sum() * 100
        macc = torch.mean(acc_per_cls)  # class accuracy
        return macc.item(), over_all_acc.item(), acc_per_cls.cpu().numpy()

    @staticmethod
    def print_acc(accs):
        out = '\n    Class  ' + '   Acc  '
        for i, values in enumerate(accs):
            out += '\n' + str(i).rjust(8) + f'{values.item():.2f}'.rjust(8)
        out += '\n' + '-' * 20
        out += '\n' + '   Mean  ' + f'{torch.mean(accs).item():.2f}'.rjust(8)
        logging.info(out)

    def all_metrics(self):
        tp, fp, fn = self.tp, self.fp, self.fn,

        iou_per_cls = tp / (tp + fp + fn).clamp(min=1) * 100
        acc_per_cls = tp / self.count.clamp(min=1) * 100
        over_all_acc = tp.sum() / self.total * 100

        miou = torch.mean(iou_per_cls)
        macc = torch.mean(acc_per_cls)  # class accuracy
        return miou.item(), macc.item(), over_all_acc.item(), iou_per_cls.cpu().numpy(), acc_per_cls.cpu().numpy()


def get_mious(tp, union, count):
    iou_per_cls = (tp + 1e-10) / (union + 1e-10) * 100
    acc_per_cls = (tp + 1e-10) / (count + 1e-10) * 100
    over_all_acc = tp.sum() / count.sum() * 100

    miou = torch.mean(iou_per_cls)
    macc = torch.mean(acc_per_cls)  # class accuracy
    return miou.item(), macc.item(), over_all_acc.item(), iou_per_cls.cpu().numpy(), acc_per_cls.cpu().numpy()
