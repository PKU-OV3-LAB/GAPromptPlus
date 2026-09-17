"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import re
import copy
import torch
import torch.nn as nn
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")

from torch.nn.modules import batchnorm
from torch.nn.modules import instancenorm
from torch.nn.modules import normalization

def get_decay_params(
        model: nn.Module,
        weight_decay,
        skip_names=("bias",),
        skip_modules=(
            instancenorm._InstanceNorm,
            batchnorm._BatchNorm,
            normalization.LayerNorm,
            normalization.GroupNorm,
        ),
        skip_decay=0.0,
    ):
    """ Collect desired params for weight-decay
        - name check to avoid bias
        - type check to avoid normalization layer - default init: weight (gamma) = 1, bias (beta) = 0
    """
    # following `nn.Module.named_parameters (recurse=True)`
    # - invoke `nn.Module._named_members (recurse=True)`
    def get_members_fn(module):
        wd = skip_decay if isinstance(module, skip_modules) else weight_decay
        for k, v in module._parameters.items():
            if v is None:  # conv/linear with bias=False - NOTE: originally checked in `_named_members`
                continue
            yield k, (v, wd)
    gen = model._named_members(get_members_fn)

    basename = lambda n: n.split(".")[-1]
    decay_params = [(n, p, wd if basename(n) not in skip_names else skip_decay) for n, (p, wd) in gen if p.requires_grad]
    return decay_params

def get_decay_params_by_modules(
        model: nn.Module,
        weight_decay,
        skip_names=("bias"),
        skip_modules=(instancenorm._InstanceNorm, batchnorm._BatchNorm),
        skip_decay=0.0,
    ):
    memo = set()
    decay_params = []
    # using named_modules & avoid potential duplicate
    for name, module in model.named_modules():
        need_skip = isinstance(module, skip_modules)
        wd = skip_decay if need_skip else weight_decay
        for param_name, param in module.named_parameters(recurse=need_skip):
            full_name = f"{name}.{param_name}" if name else param_name
            if full_name in memo:
                continue
            decay_params.append((full_name, param, skip_decay if param_name in skip_names else wd))
            memo.add(full_name)
    return decay_params


def build_optimizer(cfg, model : nn.Module, param_dicts=None, verbose=1):
    cfg = copy.deepcopy(cfg)  # use as kwargs
    finetune = cfg.pop("finetune", None)

    cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
    # [0] as default param_group
    param_dicts = param_dicts if param_dicts is not None else []
    for i in range(len(param_dicts)):
        param_group = dict(names=[], params=[])
        if "lr" in param_dicts[i].keys():
            param_group["lr"] = param_dicts[i].lr
        if "momentum" in param_dicts[i].keys():
            param_group["momentum"] = param_dicts[i].momentum
        if "weight_decay" in param_dicts[i].keys():
            param_group["weight_decay"] = param_dicts[i].weight_decay
        cfg.params.append(param_group)

    for n, p in model.named_parameters():
        flag = False
        for i in range(len(param_dicts)):
            if param_dicts[i].keyword in n:
                cfg.params[i + 1]["names"].append(n)
                cfg.params[i + 1]["params"].append(p)
                flag = True
                break
        if not flag:
            if p.requires_grad:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

    if finetune is not None:
        assert finetune, f"specified finetune but empty: {finetune}"
        if not isinstance(finetune, str):
            finetune = "|".join([f"({s})" for s in finetune])
        finetune = re.compile(finetune)

        for p in model.parameters():
            p.requires_grad = False
        for param_group in cfg.params:
            train_inds = []
            for i, (n, p) in enumerate(zip(param_group["names"], param_group["params"])):
                if finetune.search(n):
                    train_inds.append(i)
                    p.requires_grad = True
            param_group["names"] = [param_group["names"][i] for i in train_inds]
            param_group["params"] = [param_group["params"][i] for i in train_inds]
        cfg.params = [param_group for param_group in cfg.params if len(param_group["names"]) > 0]

    if verbose:
        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = "\n\t".join(cfg.params[i].pop("names"))
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params:\n\t{param_names}")
        n_all = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen = n_all - n_train
        percent_all = n_frozen / n_all * 100
        logger.info(f"Num params: {n_train}, frozen params: {n_frozen}, all params: {n_all} (frozen percent: {percent_all:.2f}%)")

    return OPTIMIZERS.build(cfg=cfg)
