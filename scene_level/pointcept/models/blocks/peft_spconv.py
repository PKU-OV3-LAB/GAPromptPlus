import re
import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._buffer_dict import BufferDict
from .peft import LoraLayer

import spconv.pytorch as spconv

class _SubMConvNd(spconv.SparseModule, LoraLayer):
    # Lora implemented in a sub-manifold conv(2,3)d layer
    def __init__(
        self,
        base_layer: spconv.SparseModule,
        adapter_name: str,
        r: int = 0,
        ratio: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
        # - eg:
        # fan_in_fan_out=bool(lora_config.fan_in_fan_out),    # bool - default False
        # ephemeral_gpu_offload=bool(lora_config.ephemeral_gpu_offload),  # bool - default False
    ):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LoraLayer.__init__(self, base_layer)

        if ratio:
            assert not r, f"setting r={r} and ratio={ratio} simultaneously"
            r = int(base_layer.in_channels / ratio)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        return

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora = False, lora_bias = False):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_alpha = self.init_lora_alpha(lora_alpha)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)

        # Actual trainable parameters
        base_layer : spconv.conv.SparseConvolution = self.get_base_layer()
        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        indice_key = base_layer.indice_key

        conv_layer = type(base_layer)
        _kernel_dim = base_layer.weight.dim()
        out_kernel = out_stride = (1,) * (_kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(in_channels, r, kernel_size, stride, padding, bias=False, indice_key=indice_key)
        self.lora_B[adapter_name] = conv_layer(r, out_channels, out_kernel, out_stride, bias=lora_bias, indice_key=indice_key)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
        return

    def _cast_input_dtype(self, x: spconv.SparseConvTensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.features.dtype == dtype):
            return x
        return x.replace_feature(x.features.to(dtype=dtype))

    def forward(self, x: spconv.SparseConvTensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        result : spconv.SparseConvTensor
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.features.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_B.weight.dtype)

                if not self.use_dora[active_adapter]:
                    x = x.replace_feature(dropout(x.features))
                    x = lora_B(lora_A(x))
                    x = x.replace_feature(x.features * scaling)
                    result = result + x
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = x.replace_feature(dropout(x.features))
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.replace_feature(result.features.to(torch_result_dtype))
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class SubMConv1d(_SubMConvNd):
    pass
class SubMConv2d(_SubMConvNd):
    pass
class SubMConv3d(_SubMConvNd):
    pass
class SubMConv4d(_SubMConvNd):
    pass

class _SubMConvNd_Bottleneck(_SubMConvNd):
    # conv1x1-conv-conv1x1
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora = False, lora_bias = False):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_alpha = self.init_lora_alpha(lora_alpha)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)

        # Actual trainable parameters
        base_layer : spconv.conv.SparseConvolution = self.get_base_layer()
        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        indice_key = base_layer.indice_key

        conv_layer = type(base_layer)
        _kernel_dim = base_layer.weight.dim()
        out_kernel = out_stride = (1,) * (_kernel_dim - 2)

        self.lora_A[adapter_name] = spconv.SparseSequential(
            conv_layer(in_channels, r, out_kernel, out_stride, bias=False, indice_key=indice_key),  # conv1x1
            conv_layer(r, r, kernel_size, stride, padding, bias=False, indice_key=indice_key)
        )
        self.lora_B[adapter_name] = conv_layer(r, out_channels, out_kernel, out_stride, bias=lora_bias, indice_key=indice_key)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
        return


class SubMConv1d_Bottleneck(_SubMConvNd_Bottleneck):
    pass
class SubMConv2d_Bottleneck(_SubMConvNd_Bottleneck):
    pass
class SubMConv3d_Bottleneck(_SubMConvNd_Bottleneck):
    pass
class SubMConv4d_Bottleneck(_SubMConvNd_Bottleneck):
    pass

# lora_kwargs = dict(
#     r=lora_config.r,
#     lora_alpha=lora_config.alpha,
#     lora_dropout=lora_config.lora_dropout,
#     fan_in_fan_out=bool(lora_config.fan_in_fan_out),    # bool - default False
#     init_lora_weights=lora_config.init_lora_weights,    # str/bool
#     use_rslora=bool(lora_config.use_rslora),            # bool - default False
#     use_dora=bool(lora_config.use_dora),                # bool - default False
#     ephemeral_gpu_offload=bool(lora_config.ephemeral_gpu_offload),  # bool - default False
#     lora_bias=lora_config.lora_bias,
# )

def dispatch_spconv(target: torch.nn.Module, adapter_name: str, **kwargs) -> Optional[torch.nn.Module]:
    if kwargs.pop("use_bottleneck", False):
        return dispatch_spconv_bottleneck(target, adapter_name, **kwargs)

    new_module = None
    if isinstance(target, spconv.SubMConv1d):
        new_module = SubMConv1d(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv2d):
        new_module = SubMConv2d(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv3d):
        new_module = SubMConv3d(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv4d):
        new_module = SubMConv4d(target, adapter_name, **kwargs)

    return new_module

def dispatch_spconv_bottleneck(target: torch.nn.Module, adapter_name: str, **kwargs) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, spconv.SubMConv1d):
        new_module = SubMConv1d_Bottleneck(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv2d):
        new_module = SubMConv2d_Bottleneck(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv3d):
        new_module = SubMConv3d_Bottleneck(target, adapter_name, **kwargs)
    elif isinstance(target, spconv.SubMConv4d):
        new_module = SubMConv4d_Bottleneck(target, adapter_name, **kwargs)

    return new_module
