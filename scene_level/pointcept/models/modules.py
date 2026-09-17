import sys
import torch.nn as nn
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None

from collections import OrderedDict
from pointcept.models.utils.structure import Point
from pointcept.engines.hooks import HookBase


def is_ocnn_module(module):
    if ocnn is not None:
        ocnn_modules = (
            ocnn.nn.OctreeConv,
            ocnn.nn.OctreeDeconv,
            ocnn.nn.OctreeGroupConv,
            ocnn.nn.OctreeDWConv,
        )
        return isinstance(module, ocnn_modules)
    else:
        return False


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # following spconv.SparseSequential
        # - nn.Sequential
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # - allow dict-based (ordered)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        # - support list indexing & default iter (eg. for-loop)
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    # def __iadd__(self, rval):
    #     # enable `+=`
    #     if isinstance(rval, list):
    #         for v in rval:
    #             self.add(v)
    #     elif isinstance(rval, dict, PointSequential):
    #         if isinstance(rval, PointSequential):
    #             rval = rval._modules
    #         for k, v in rval.items():
    #             if k.isdigit():
    #                 k = str(len(self._modules))
    #             while k in self._modules:
    #                 cnt : str = k.split('_')[-1] if '_' in k else ''
    #                 if cnt.isdigit():
    #                     k = '_'.join(k.split('_')[:-1]) + f'_{int(cnt) + 1}'
    #                 else:
    #                     k += f'_1'
    #             self.add_module(k, v)
    #     else:
    #         raise NotImplementedError(f'not support += with type {type(rval)}:\n{rval}')
    #     return self

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            elif is_ocnn_module(module):
                if isinstance(input, Point):
                    input.octree.features[-1] = module(
                        input.feat[input.octree_order], input.octree, input.octree.depth
                    )
                    input.feat = input.octree.features[-1][input.octree_inverse]
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                    else:
                        raise IndexError(f"input.indices reaches 0 - shape={input.indices.shape}")
                else:
                    input = module(input)
        return input


class PointModel(PointModule, HookBase):
    r"""PointModel
    placeholder, PointModel can be customized as a Pointcept hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
