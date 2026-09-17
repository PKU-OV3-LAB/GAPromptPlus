import sys
import time
import builtins
import numpy as np

from typing import Dict
from sklearn.metrics import confusion_matrix


class AverageTracker:

  def __init__(self):
    self.value = dict()
    self.num = dict()
    self.start_time = time.time()

  def update(self, value: Dict):
    r'''Update the tracker with the given value, which is called at the end of each iteration.
    '''

    if not value:
      return    # empty input, return

    for key, val in value.items():
      self.value[key] = self.value.get(key, 0) + np.asarray(val)
      self.num[key] = self.num.get(key, 0) + 1

  def average(self):
    return {key: val.item() / self.num[key] for key, val in self.value.items()}


class Metrics(dict):
    '''
    storing calculated metrics with easy printing
    '''

    def __init__(self, *args, scale=1, order=['mIoU', 'mIoU_cld', 'OA', 'mACC'], task=None, names=None, **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        self.scale = scale
        self.order = [order] if isinstance(order, str) else list(order)  # the importance rank of metrics - main key = order[0]
        self._scalar_to_list = {'mIoU': 'IoUs', 'mACC': 'ACCs'}

        main_list = [self._scalar_to_list[i] for i in self.order if i in self._scalar_to_list]
        self.main_list = main_list[0] if main_list else None
        self.names = names

        if task in ['seg', 'segmentation']:
            self.seg()
        elif task in ['cls', 'classification']:
            self.cls()
        elif task:
            raise ValueError(f'Metrics not support predefined task={task}')
        return

    # def __missing__(self, key):
    #     return None

    def cls(self):
        self.order = ['mACC', 'OA']
        self.main_list = 'ACCs'
        return self

    def seg(self):
        self.order = ['mIoU', 'mIoU_cld', 'OA', 'mACC']
        self.main_list = 'IoUs'
        return self

    # Comparison
    # ------------------------------------------------------------------------------------------------------------------

    def _is_valid(self, other, raise_invalid=True):
        if self.order[0] not in other:
            if raise_invalid:
                raise ValueError(f'missing main key - {self.order[0]}, in order {self.order}')
            return False
        return True

    def __eq__(self, other):  # care only the main key
        self._is_valid(self)
        self._is_valid(other)
        return self[self.order[0]] == other[self.order[0]]

    def __gt__(self, other):
        self._is_valid(self)
        self._is_valid(other)
        for k in self.order:
            if k not in self:  # skip if not available
                continue
            if k not in other or self[k] > other[k]:  # True if more completed
                return True
            elif self[k] < other[k]:
                return False

        # all equal (at least for main key)
        return False

    # Pretty print
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def scalar_str(self):
        scalar_m = [k for k in self.order if k in self and self[k]]
        s = ''.join([f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m])
        return s
    @property
    def list_str(self):
        if self.main_list is None:
            return ''
        list_m = [k for k in [self.main_list] if k in self and self[k] is not None]
        s = []
        for k in list_m:
            m = self.list_to_line(k)
            s += [m]
        s = ' | '.join(s)
        return s

    def print(self, conf=True, logger=None):
        s = self.full()
        print = builtins.print
        if logger is not None:
            print = logger.info
        if conf and 'conf' in self:
            conf = self['conf']
            # assert np.issubdtype(conf.dtype, np.integer)
            with np.printoptions(linewidth=sys.maxsize, threshold=sys.maxsize, precision=3):
                print(self['conf'])
        print(s)

    def full(self, get_list=False, keys=None):
        # separate line print each group of metrics
        scalar_m = [k for k in ['OA', 'mACC', 'mIoU_cld', 'mIoU'] if k in self and self[k] and k in self.order]
        str_d = {k: f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m}  # scalar_m -> str

        lines = []
        for k_scalar in str_d:
            if k_scalar not in self._scalar_to_list:
                lines += [f'{str_d[k_scalar]}']
                continue
            k_list = self._scalar_to_list[k_scalar]
            lines += [f'{str_d[k_scalar]} | {self.list_to_line(k_list, names=self.names)}']

        max_len = max(len(v) for v in lines)
        s = ['-' * max_len, *lines, '-' * max_len]
        if self.names:
            width = max([len(i) for i in self.names])
            str_names = ' '.join(['{i:<{width}}'.format(i=i, width=width) for i in self.names])
            str_names = ' ' * (len(lines[-1].split('|')[0]) + 2) + str_names
            s = [str_names] + s

        s = s if get_list else '\n'.join(s)
        return s

    def __repr__(self):
        return ' | '.join([k for k in [self.scalar_str, self.list_str] if k])

    def list_to_line(self, k, names=None):
        if isinstance(k, list):
            score_list = k
        elif k in self:
            score_list = self[k]
        else:
            return ''
        width = max([len(i) for i in names]) if names else 5
        score_list = [i/self.scale*100 for i in score_list]
        score_list = ['{i:<{width}.2f}'.format(i=i, width=width) for i in score_list]
        m = ' '.join(score_list)
        return m

def metrics_from_confusions(confusions, proportions=None, remap=False):
    '''
    Computes IoU from confusion matrices.
    Args:
        confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes; gt (row) x pred (col).
    '''

    confusions = confusions.astype(np.float32)
    if proportions is not None:
        # Balance with real proportions
        confusions *= np.expand_dims(proportions.astype(np.float32) / (confusions.sum(axis=-1) + 1e-6), axis=-1)

    if remap:
        # Hungarian to maximize TP
        from scipy.optimize import linear_sum_assignment
        _, col_inds = linear_sum_assignment(confusions, maximize=True)
        confusions = confusions[np.argsort(col_inds), :]  # assignment: gt i -> pred col_inds[i]

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)
    ACC = TP / (TP_plus_FN + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)
    mACC = np.sum(ACC, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later, or simply denotes absence with nan
    IoU += mask * mIoU
    # IoU[mask] = float('nan')
    ACC[mask] = float('nan')

    # Compute Accuracy
    OA = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)
    m = {
        'mIoU': mIoU.mean(),
        'mACC': mACC.mean(),
        'OA': OA,
        'IoUs': IoU,
        'ACCs': ACC,
        '_valid_mask': np.logical_not(mask),  # valid mask
        'conf': confusions.astype(int),
    }
    m = Metrics(m)
    return m
