
# from addict import Dict
from functools import partial

import re
import math
import numpy as np
from typing import Union
from itertools import islice
from collections import OrderedDict, Counter

import torch
import torch.backends
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

# from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

class OpsCounter(Counter):
    _elem0 = ''
    def __getitem__(self, key):
        cnt : int = super().__getitem__(key)
        self[key] = cnt + 1
        return cnt if cnt else self._elem0


class FastBatchNorm1d(nn.Module):
    """ torch points 3d """
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    @property
    def nb_params(self):
        """ This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

    def get_weight_decay(self):  # 0-decay
        return [[n, p, 0] for n, p in self.named_parameters()]

    def _forward_dense(self, x):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x):
        """ Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x):
        if x.dim() == 2:  # [BN, d]
            return self._forward_sparse(x)
        elif x.dim() == 3:  # [B, N, d]
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class MLPs(PointModule):

    @property
    def default_kwargs(self):
        return {
            'norm_layer': 'ln',
            'ln_eps': 1e-5,
            'bn_eps': 1e-3,
            'bn_momentum': 0.01,  # 1-0.99
            # 'bias': True,
            'act_layer': nn.GELU,

            'drop': None,
            'shortcut': None,
            'drop_path': None,

            'pre_norm': False,
            'post_norm': False,
        }

    @property
    def valid_norm(self):
        return ['bn', 'ln', 'fastbn']

    @property
    def valid_act(self):
        return ['relu', 'gelu']

    def __init__(
        self,
        ops,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        ratio=1,
        **kwargs,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or int(in_channels * ratio)

        assert set(kwargs.keys()).issubset(set(self.default_kwargs.keys())), f'unexpected keys in kwargs={kwargs}'
        kwargs = dict(self.default_kwargs, **kwargs)

        # dropout
        self.drop = kwargs.pop('drop')
        if self.drop:
            self.drop = nn.Dropout(self.drop)
        # shortcut
        self.shortcut = kwargs.pop('shortcut')
        if self.shortcut:
            self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        # drop path
        self.drop_path = kwargs.pop('drop_path')
        if self.drop_path:
            self.drop_path = DropPath(self.drop_path)
            assert self.shortcut is not None, f'using drop-path without shortcut'

        # collect ops
        ops_list = []
        if 'lin' in ops.lower() and 'linear' not in ops.lower():
            ops = ops.replace('lin', 'linear').replace('liN', 'linearN')
        assert 'mlp' in ops or ops in ['linear', 'linearN'] + [f'linear{n}' for n in self.valid_norm], f'invalid ops = {ops}'

        num_mlp = re.search('\d+', ops)
        num_mlp = int(num_mlp.group()) if num_mlp else 1
        linear = 'linear' in ops or not ops.endswith('mlp')  # linear / linearbn / mlp2 to ends with linear

        # - pre/post/out -norm
        pre_norm = kwargs.pop('pre_norm')
        post_norm = kwargs.pop('post_norm')
        out_norm = ops in ['linearN'] + [f'linear{n}' for n in self.valid_norm]
        self.pre_norm = self.post_norm = None
        if pre_norm:
            assert not out_norm and not post_norm
            kwargs_norm = dict(**kwargs, act_layer=None, norm_layer=pre_norm if isinstance(pre_norm, str) else kwargs['norm_layer'])
            self.pre_norm = self.get_norm(**kwargs_norm)
            kwargs['norm_layer'] = None  # disable mid-norm
        if post_norm:
            assert not out_norm and not pre_norm
            kwargs_norm = dict(**kwargs, act_layer=None, norm_layer=post_norm if isinstance(post_norm, str) else kwargs['norm_layer'])
            self.post_norm = self.get_norm(**kwargs_norm)
            kwargs['norm_layer'] = None  # disable mid-norm
        if out_norm:
            assert not pre_norm and not post_norm
            kwargs_norm = dict(**kwargs, act_layer=None, norm_layer=ops.replace('linear', '').replace('N', kwargs['norm_layer']))
            self.post_norm = self.get_norm(**kwargs_norm)  # as post_norm

        # - mlps - mid
        fdims = in_channels
        fdims_out = hidden_channels
        for ops_i in range(1, num_mlp):
            ops_list += self.get_mlp(din=fdims, dout=fdims_out, ops_i=ops_i, **kwargs)
            fdims = fdims_out
        ops_i = num_mlp

        # - linear/mlp - out
        fdims_out = out_channels
        kwargs_out = dict(act_layer=None, norm_layer=None) if linear else dict()
        kwargs_out = dict(**kwargs, **kwargs_out)
        ops_list += self.get_mlp(din=fdims, dout=fdims_out, ops_i=ops_i, **kwargs_out)

        # register
        self.ops_list = ops_list
        for ops_n, ops_module in ops_list:
            if ops_n is None:
                continue
            self.add_module(name=str(ops_n), module=ops_module)

        return

    def get_mlp(self, din, dout, ops_i='', **kwargs):
        ops_list = [(f'linear{ops_i}', nn.Linear(din, dout))]

        norm_layer = self.get_norm(**kwargs)
        if norm_layer:
            ops_list += [(f'norm{ops_i}', norm_layer())]

        act_layer = self.get_act(**kwargs)
        if act_layer:
            ops_list += [(f'act{ops_i}', act_layer())]

        if self.drop is not None:
            ops_list += [(None, self.drop)]
        return ops_list

    def get_norm(self, norm_layer, **kwargs):
        if norm_layer == 'ln':
            norm_layer = partial(nn.LayerNorm, eps=kwargs['ln_eps'])
        elif norm_layer == 'bn':
            norm_layer = partial(nn.BatchNorm1d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])
        elif norm_layer == 'fastbn':
            norm_layer = partial(FastBatchNorm1d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])
        elif norm_layer:
            assert not isinstance(norm_layer, str), f'not support norm_layer ({type(norm_layer)}) = {norm_layer}, not in {self.valid_norm}'
        return norm_layer

    def get_act(self, act_layer, **kwargs):
        if act_layer == 'gelu':
            act_layer = nn.GELU
        elif act_layer == 'relu':
            act_layer = nn.ReLU
        elif act_layer:
            assert not isinstance(act_layer, str), f'not support act_layer ({type(act_layer)}) = {act_layer}, not in {self.valid_act}'
        return act_layer

    def forward(self, point: Point):
        features = point.feat

        if self.shortcut is not None:  # sc --
            shortcut = self.shortcut(features)

        x = features
        if self.pre_norm:
            x = self.pre_norm(x)

        for ops_n, ops_fn in self.ops_list:
            # lin - norm - act - drop
            x = ops_fn(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.shortcut is not None:  # -- sc
            x = shortcut + x

        if self.post_norm is not None:
            x = self.post_norm(x)

        point.feat = x
        if 'sparse_conv_feat' in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class Pooling(nn.Module):
    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce
        assert reduce in ['sum', 'mean', 'min', 'max']

    def extra_repr(self):
        return self.reduce

    def forward(self, x, idx_ptr, indices=None):
        if indices is not None:
            x = x[indices]
        x = torch_scatter.segment_csr(x, idx_ptr, reduce=self.reduce)
        return x


class Projection(PointModule):

    @property
    def default_kwargs(self):
        return {
            'norm_layer': 'ln',
            'ln_eps': 1e-5,
            'bn_eps': 1e-3,
            'bn_momentum': 0.01,  # 1-0.99
            # 'bias': True,
            'act_layer': nn.GELU,

            'drop': None,
            'shortcut': None,
            'drop_path': None,

            'pre_norm': False,
            'post_norm': False,

            'indice_key': None,
        }

    @property
    def valid_norm(self):
        return ['bn', 'ln', 'fastbn']

    @property
    def valid_act(self):
        return ['relu', 'gelu']

    @property
    def valid_pool(self):
        return ['pool_sum', 'pool_mean', 'pool_min', 'pool_max']

    def __init__(
        self,
        ops,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        ratio=1,
        **kwargs,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or int(out_channels * ratio)

        assert set(kwargs.keys()).issubset(set(self.default_kwargs.keys())), f'unexpected keys in kwargs={kwargs}'
        kwargs = dict(self.default_kwargs, **kwargs)

        # dropout
        self.drop = kwargs.pop('drop')
        if self.drop:
            self.drop = nn.Dropout(self.drop)
        # shortcut
        self.shortcut = kwargs.pop('shortcut') or None
        if self.shortcut:
            self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        # drop path
        self.drop_path = kwargs.pop('drop_path')
        if self.drop_path:
            self.drop_path = DropPath(self.drop_path)
            assert self.shortcut is not None, f'using drop-path without shortcut'

        # solve ops_str
        str_list = []
        ops_str = ops.split('-') if isinstance(ops, str) else ops
        for ops in ops_str:
            if 'lin' in ops.lower() and 'linear' not in ops.lower():
                ops = ops.replace('lin', 'linear').replace('liN', 'linearN')
            if ops in ['linear', 'linearN'] + [f'linear{n}' for n in self.valid_norm]:
                n = ops[len('linear'):]
                str_list += ['linear', kwargs['norm_layer'] if n == 'N' else n]
            else:
                str_list.append(ops)
        str_list = [ops for ops in str_list if ops]

        # - pre/post -norm
        pre_norm = kwargs.pop('pre_norm')
        post_norm = kwargs.pop('post_norm')
        self.pre_norm = self.post_norm = None
        if pre_norm:
            assert not post_norm
            kwargs_norm = dict(**kwargs, act_layer=None, norm_layer=pre_norm if isinstance(pre_norm, str) else kwargs['norm_layer'])
            self.pre_norm = self.get_norm(**kwargs_norm)
            kwargs['norm_layer'] = None  # disable mid-norm
        if post_norm:
            assert not pre_norm
            kwargs_norm = dict(**kwargs, act_layer=None, norm_layer=post_norm if isinstance(post_norm, str) else kwargs['norm_layer'])
            self.post_norm = self.get_norm(**kwargs_norm)
            kwargs['norm_layer'] = None  # disable mid-norm

        # collect ops
        # - dims
        fdims = in_channels
        fdims_list = [hidden_channels] * len(str_list)
        for ops_i, ops in reversed(list(enumerate(str_list))):
            if ops not in (*self.valid_norm, 'norm', *self.valid_act, 'act', *self.valid_pool):
                break
        fdims_list[ops_i] = out_channels

        # - build
        ops_list = []
        ops_counter = OpsCounter()
        for ops_i, ops in enumerate(str_list):
            fdims_out = fdims_list[ops_i]

            if 'mlp' in ops:
                num_mlp = re.search('\d+', ops)
                num_mlp = int(num_mlp.group()) if num_mlp else 1
                linear = not ops.endswith('mlp')  # mlp2 to ends with linear

                # - mlps - mid
                for _ in range(1, num_mlp):
                    ops_list += self.get_mlp(din=fdims, dout=hidden_channels, counter=ops_counter, **kwargs)
                    fdims = hidden_channels

                # - linear/mlp - out
                kwargs_out = dict(act_layer=None, norm_layer=None) if linear else dict()
                kwargs_out = dict(kwargs, **kwargs_out)
                ops_list += self.get_mlp(din=fdims, dout=fdims_out, counter=ops_counter, **kwargs_out)
                fdims = fdims_out

            elif 'linear' in ops:
                kwargs_lin = dict(kwargs, act_layer=None, norm_layer=None)
                ops_list += self.get_mlp(din=fdims, dout=fdims_out, counter=ops_counter, **kwargs_lin)
                fdims = fdims_out

            elif any(ops.startswith(n) for n in ['spconv', 'subm']):
                kwargs_conv = dict(kwargs)
                kernel_size = re.search(r'\d+', ops)
                if kernel_size:
                    kwargs_conv.update(kernel_size=int(kernel_size.group()))
                conv_type = re.match(r'[a-zA-Z]+', ops).group()
                if conv_type == 'subm':
                    conv_list = self.get_submconv(din=fdims, dout=fdims_out, counter=ops_counter, **kwargs_conv)
                else:
                    raise ValueError(f"not support conv_type={conv_type}")
                ops_list += conv_list
                fdims = fdims_out

            elif ops in (*self.valid_norm, 'norm'):
                norm_kwargs = dict(kwargs)
                if ops in self.valid_norm:
                    norm_kwargs.update(norm_layer=ops)
                norm_layer = self.get_norm(**norm_kwargs)
                ops_n = 'norm'
                ops_i = ops_counter[ops_n]
                ops_list += [(f'{ops_n}{ops_i}', norm_layer(fdims))]

            elif ops in (*self.valid_act, 'act'):
                act_kwargs = dict(kwargs)
                if ops in self.valid_act:
                    act_kwargs.update(act_layer=ops)
                act_layer = self.get_act(**act_kwargs)
                ops_n = 'act'
                ops_i = ops_counter[ops_n]
                ops_list += [(f'{ops_n}{ops_i}', act_layer())]

            elif ops.startswith('pool_'):
                ops_n = 'pool'
                ops_i = ops_counter[ops_n]
                ops_list += [(f'{ops_n}{ops_i}', Pooling(reduce=ops[len('pool_'):]))]

            elif ops.startswith('dp'):
                ops_n = 'dropout'
                ops_i = ops_counter[ops_n]
                dropout_p = float(ops[2:])
                ops_list += [(f'{ops_n}{ops_i}', nn.Dropout(dropout_p))]

            else:
                raise ValueError(f'not support ops={ops}')

        # register
        self.ops_list = ops_list
        for ops_n, ops_module in ops_list:
            if ops_n is None:
                continue
            self.add_module(name=str(ops_n), module=ops_module)

        return

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: Union[slice, int, str]):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        size = len(self)
        if not (-size <= idx < size):
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        # it = iter(self._modules.values())
        # for i in range(idx):
        #     next(it)
        return next(islice(self._modules.values(), idx, None))

    def get_mlp(self, din, dout, counter : OpsCounter = None, **kwargs):
        counter = counter if counter is not None else self.counter

        ops_n = 'linear'
        ops_i = counter[ops_n]  # not showing 0 - OpsCounter
        ops_list = [(f'{ops_n}{ops_i}', nn.Linear(din, dout))]

        norm_layer = self.get_norm(**kwargs)
        if norm_layer:
            ops_n = 'norm'
            ops_i = counter[ops_n]
            ops_list += [(f'{ops_n}{ops_i}', norm_layer(dout))]

        act_layer = self.get_act(**kwargs)
        if act_layer:
            ops_n = 'act'
            ops_i = counter[ops_n]
            ops_list += [(f'{ops_n}{ops_i}', act_layer())]

        if self.drop is not None:
            ops_list += [(None, self.drop)]
        return ops_list

    @classmethod
    def get_norm(self, norm_layer, **kwargs):
        if norm_layer == 'ln':
            norm_layer = partial(nn.LayerNorm, eps=kwargs['ln_eps'])
        elif norm_layer == 'bn':
            norm_layer = partial(nn.BatchNorm1d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])
        elif norm_layer == 'fastbn':
            norm_layer = partial(FastBatchNorm1d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])
        elif norm_layer:
            assert not isinstance(norm_layer, str), f'not support norm_layer ({type(norm_layer)}) = {norm_layer}, not in {self.valid_norm}'
        return norm_layer

    @classmethod
    def get_act(self, act_layer, **kwargs):
        if act_layer == 'gelu':
            act_layer = nn.GELU
        elif act_layer == 'relu':
            act_layer = nn.ReLU
        elif act_layer in ['tan', 'tanh']:
            act_layer = nn.Tanh
        elif act_layer:
            assert not isinstance(act_layer, str), f'not support act_layer ({type(act_layer)}) = {act_layer}, not in {self.valid_act}'
        return act_layer

    def get_submconv(self, din, dout, counter=None, **kwargs):
        counter = counter if counter is not None else self.counter

        ops_n = 'conv'
        ops_i = counter[ops_n]
        kernel_size = kwargs.get('kernel_size', 3)
        padding = kwargs.get('padding', 1)
        indice_key = kwargs.get('indice_key', None)
        conv = spconv.SubMConv3d(din, dout, kernel_size=kernel_size, padding=padding, indice_key=indice_key)
        ops_list = [(f'{ops_n}{ops_i}', conv)]
        return ops_list

    def forward(self, point: Union[Point, torch.tensor], idx_ptr=None, indices=None):
        features = point.feat if isinstance(point, Point) else point

        if self.shortcut is not None:  # sc --
            shortcut = self.shortcut(features)

        x = features
        if self.pre_norm:
            x = self.pre_norm(x)

        for ops_n, ops_fn in self.ops_list:
            # lin/conv - norm - act - drop
            if spconv.modules.is_spconv_module(ops_fn):
                # assume existing 'sparse_conv_feat' if using spconv module
                x = ops_fn(point.sparse_conv_feat.replace_feature(x)).features
            elif isinstance(ops_fn, Pooling):
                # pooling ops
                x = x[indices] if indices is not None else x
                x = ops_fn(x, idx_ptr, indices=indices)
            else:
                x = ops_fn(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.shortcut is not None:  # -- sc
            x = shortcut + x

        if self.post_norm is not None:
            x = self.post_norm(x)

        if isinstance(point, Point):
            point.feat = x
            if 'sparse_conv_feat' in point.keys():
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
            return point
        return x
