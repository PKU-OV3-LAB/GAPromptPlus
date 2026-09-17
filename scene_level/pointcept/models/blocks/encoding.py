
from addict import Dict
from functools import partial
from collections import abc

import re
import copy
import math
import numpy as np

import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import spconv.pytorch as spconv
from timm.models.layers import DropPath

from .projection import MLPs, Projection
from libs import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.modules import PointModule, PointSequential
from pointcept.utils.registry import Registry

ENCODING = Registry("encoding")

def _check_at_most_one(*args):
    if any(args):
        chk = iter(args)
        assert any(chk) and not any(chk), f"multiple True in {args}"

_stage_record = dict()  # stage -> latent_dims
def check_new_latents(channels, stage, granularity, key_prefix=''):
    _errmsg = f"unexpected stage={stage}"
    assert isinstance(stage, str), _errmsg
    assert re.fullmatch(r"(enc|dec)\d+", stage), _errmsg
    stage_n = re.match(r"enc|dec", stage).group()
    stage_i = int(re.search(r"\d+", stage).group())

    if not granularity:  # not sharing at all
        return True, channels
    elif granularity == "stage":  # sharing in same stage - each (enc|dec){i} stage
        pass
    elif granularity == "scale":  # sharing at same scale - same i across enc/dec
        stage = stage_i
    elif granularity == "comp":  # sharing in same process - within enc/dec
        stage = stage_n
    elif granularity in [True, "glb"]:  # sharing across the whole
        stage = None
    else:
        raise ValueError(f"not support granularity={granularity}")

    if key_prefix:
        stage = f"{key_prefix}/{stage}"
    if stage in _stage_record:
        is_new = False
        channels = _stage_record[stage]
    else:
        is_new = True
        _stage_record[stage] = channels
    return is_new, channels

class PrefixEncoding(PointModule):
    def __init__(
        self,
        channels,
        num_latents,
        # - qkv proj
        qkv_proj=False,
        qkv_bias=True,
        qkv_share=False,
        hidden_channels=None,
        # - out proj
        proj=False,
        proj_drop=0.0,
        proj_bias=True,
        # - norm & act
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=False,
        # - latent
        latent_dims=None,               # fdims of the latents
        latent_in_proj=None,            # proj before latent qkv
        latent_share=False,             # sharing latent to later blks
        latent_update=False,            # if update latent
        latent_update_proj=False,       # proj before latent update
        latent_update_shortcut=True,    # shortcut for latent update
        stage=None,
    ):
        super().__init__()
        self.latent_update = latent_update
        self.latent_update_shortcut = latent_update_shortcut

        self.stage = stage
        self.channels = channels
        self.num_latents = num_latents
        out_channels = (hidden_channels or channels) * (3 if latent_update else 2)
        latent_dims = (latent_dims or channels) if qkv_proj or qkv_share else out_channels
        is_new, latent_dims = check_new_latents(latent_dims, stage, granularity=latent_share, key_prefix=self.__class__.__name__)

        self.latents = None
        self.latent_dims = latent_dims
        if is_new and num_latents > 0:
            # create new (separate) latents
            self.latents = self.get_latents(num_latents, latent_dims=latent_dims)

        proj_kwargs = dict(
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.latent_in_proj = None
        if latent_in_proj:
            latent_in_proj_dout = channels if qkv_share else latent_dims
            self.latent_in_proj = Projection(ops=latent_in_proj, in_channels=latent_dims, hidden_channels=hidden_channels, out_channels=latent_in_proj_dout, **proj_kwargs)

        self.norm = None
        self.qkv_proj = None
        self.qkv_share = qkv_share
        if qkv_proj:
            # qkv - projection to encode the prefix
            if isinstance(qkv_proj, str):
                self.qkv_proj = Projection(ops=qkv_proj, in_channels=latent_dims, hidden_channels=hidden_channels, out_channels=out_channels, **proj_kwargs)
            else:
                self.qkv_proj = nn.Linear(latent_dims, out_channels, bias=qkv_bias)
        if pre_norm and norm_layer is not None:
            # pre-norm on latents
            self.norm = norm_layer(latent_dims)

        self.proj = None
        if proj:
            # out-proj - separate output projection for latent update
            proj_dout = (hidden_channels or channels) if latent_update_proj else latent_dims
            if isinstance(proj, str) and proj == "i":
                self.proj = nn.Identity()
            elif isinstance(proj, str):
                self.proj = Projection(ops=proj, in_channels=channels, hidden_channels=hidden_channels, out_channels=proj_dout, **proj_kwargs)
            else:
                self.proj = nn.Sequential(
                    nn.Linear(channels, proj_dout, bias=proj_bias),
                    nn.Dropout(proj_drop),
                )

        self.latent_update_proj = None
        if latent_update_proj:
            # further projection before latent update
            latent_update_proj_din = proj_dout if proj and proj != "i" else channels
            self.latent_update_proj = Projection(ops=latent_update_proj, in_channels=latent_update_proj_din, hidden_channels=hidden_channels, out_channels=latent_dims, **proj_kwargs)
        return

    def get_latents(self, num_latents, latent_dims, init_scale=0.02):
        latents = nn.Parameter(torch.empty(num_latents, latent_dims), requires_grad=True)
        with torch.no_grad():
            latents.normal_(0.0, init_scale)
        return latents

    def extra_repr(self):
        if self.latents is not None:
            return f"(latents): {tuple(self.latents.shape)}"

    def forward(self, point):
        if self.latents is not None:
            latents = self.latents
            if self.latent_update:
                B = len(point.offset)
                latents = latents.repeat(B, 1)
            point.latents = latents  # update point.latents
        else:
            latents = point.latents

        if self.latent_in_proj is not None:
            latents = self.latent_in_proj(latents)

        if self.norm is not None:
            latents = self.norm(latents)
        return latents

    # def update_latent(self, point, latents):
    #     if self.latent_proj:
    #         latents = self.latent_proj(latents)
    #     if self.latent_shortcut:
    #         latents = latents + point.latents
    #     return latents

class Adapter(PointModule):
    def __init__(
        self,
        channels,
        hidden_channels=None,    # rank
        ratio=None,              # ratio - if not specified hidden_channels
        projection=None,         # special down-ops
        projection_up=None,      # special up-ops
        # - skip-conn
        shortcut=True,
        dropout=0.0,
        dropout_in=0.0,
        drop_path=0.0,
        # - norm & act
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=False,          # pre-norm
        post_norm=False,         # post-norm
        # scaling=None,            # scaling
        init=None,
        # - fusing
        alpha=None,
        beta=None,
        gate=None,
        gate_mean=False,
        gate_blend=False,
        gate_channel=False,
        stage=None,
        **kwargs,
    ):
        super().__init__()
        self.stage = stage
        self.channels = channels
        self.hidden_channels = hidden_channels
        if ratio is not None:
            _check_at_most_one(ratio, hidden_channels)
            self.hidden_channels = hidden_channels = int(channels / ratio)

        act_layer = Projection.get_act(act_layer)
        norm_layer = Projection.get_act(norm_layer)
        kwargs = dict(
            act_layer=act_layer,
            norm_layer=norm_layer,
            **kwargs
        )
        if projection is not None:
            down = Projection(ops=projection, in_channels=channels, hidden_channels=hidden_channels, out_channels=hidden_channels, **kwargs)
        else:
            down =  [
                norm_layer() if pre_norm else None,
                nn.Linear(channels, hidden_channels),
                act_layer(),
            ]
            down = PointSequential(*[i for i in down if i is not None])
        self.down = down

        if projection_up == "identity":
            up = nn.Identity()
        elif projection_up is not None:
            up = Projection(ops=projection_up, in_channels=hidden_channels, out_channels=channels, **kwargs)
        else:
            up = nn.Linear(hidden_channels, channels)
        self.up = up

        self.dropout = nn.Dropout(dropout)
        self.dropout_in = nn.Dropout(dropout_in) if dropout_in > 0.0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

        self.norm = None
        if pre_norm or post_norm:
            self.norm = norm_layer()
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        _check_at_most_one(pre_norm, post_norm)

        # fusing
        _check_at_most_one(alpha, beta)
        self.alpha = self.beta = None
        if alpha is not None:
            self.alpha = self._get_weight(alpha)
        if beta is not None:
            self.beta = self._get_weight(beta)

        self.gate = None
        if gate is not None:
            ops = gate if isinstance(gate, str) else "linear"
            out_channels = channels if gate_channel else 1
            self.gate = Projection(ops=ops, in_channels=channels, out_channels=out_channels, **kwargs)
        self.gate_mean = gate_mean
        self.gate_blend = gate_blend

        self.shortcut = shortcut
        if not shortcut:
            assert not (self.beta or self.gate_blend), f"shortcut not performed in adapter"

        self.reset_parameters(init=init)
        return

    def reset_parameters(self, init=None):
        if init is None:
            self.reset_parameters_default()
            return
        elif isinstance(init, (list, tuple)):
            for i in init:
                self.reset_parameters(init=i)
            return

        if init == "lora":
            self.reset_parameters_lora()
        elif init == "normal":
            self.reset_parameters_normal(std=0.02)
        elif init == "trunc":
            self.reset_parameters_normal(std=0.01, trunc=0.02)
        else:
            raise ValueError(f"{self.__class__} not support init={init}")
        return

    def reset_parameters_default(self):  # bias & ln only
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_parameters_normal(self, std=0.02, trunc=None):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if trunc is not None:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-trunc, b=trunc)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_parameters_lora(self):
        self.reset_parameters_default()
        if self.up is not None:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)
        return

    def _get_weight(self, weight_str):
        if weight_str.startswith('ws'):
            w = torch.zeros(size=[self.channels], dtype=torch.float32)
            if weight_str[2:] == 'N':
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif weight_str[2:]:
                w.data.fill_(float(weight_str[2:]))
            w = nn.Parameter(w)
        elif weight_str.startswith('w'):
            w = torch.tensor(float(weight_str[1:]) if weight_str[1:] else 0.0)
            w = nn.Parameter(w)
        else:
            w = float(weight_str)
        return w

    def fuse_features(self, point: Point, shortcut: torch.Tensor):
        # fusion
        # - point    : adapter output
        # - shortcut : adapter input feat
        if self.gate is not None:
            shortcut_point = Point(point, feat=shortcut)
            gate = F.sigmoid(self.gate(shortcut_point).feat)  # [N, 1/C]
            if self.gate_mean:
                # mean within each cloud
                _offset = torch.cat([point.offset.new_zeros(1), point.offset])
                _counts = _offset[1:] - _offset[:-1]
                gate = torch_scatter.segment_csr(
                    src=gate,
                    indptr=_offset,
                    reduce="mean",
                ).repeat_interleave(_counts, dim=0)

            point.feat = point.feat * gate
            if self.gate_blend:
                shortcut = shortcut * (1 - gate)

        else:
            if self.alpha is not None:
                point.feat = point.feat * self.alpha
            elif self.beta is not None:
                point.feat = point.feat * self.beta
                shortcut = shortcut * (1 - self.beta)

        if self.shortcut:
            point.feat = point.feat + shortcut
        return point

    def forward(self, point: Point, shortcut=None):
        point = Point(point)
        feat = point.feat
        if shortcut is None:
            shortcut = feat
        elif isinstance(shortcut, Point):
            shortcut = shortcut.feat

        if self.pre_norm:
            point.feat = self.norm(point.feat)
        if self.dropout_in is not None:
            point.feat = self.dropout_in(point.feat)

        point = self.down(point)
        point.feat = self.dropout(point.feat)
        point.feat = self.up(point.feat)
        if self.drop_path is not None:
            point.feat = self.drop_path(point.feat)

        point = self.fuse_features(point, shortcut)
        if self.post_norm:
            point.feat = self.norm(point.feat)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class LatAttnAdapter(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        num_latents,
        # - in proj
        in_proj=None,
        in_shortcut=False,
        # - qkv proj
        qkv_proj=True,
        qkv_bias=True,
        head_ratio=None,
        head_channels=None,
        hidden_channels=None,
        # - attn
        qk_scale=None,
        attn_drop=0.0,
        # - out proj & ffn
        proj=True,
        proj_drop=0.0,
        proj_bias=True,
        ffn=False,
        ffn_ratio=4.0,
        # - skip-conn
        q_shortcut=True,
        shortcut=True,
        drop_path=0.0,
        # - norm & act
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        # - fusing
        alpha=None,
        gate=None,
        gate_mean=False,
        weights=None,
        init=None,
        # - latent
        latent_dims=None,               # fdims of the latents
        latent_ffn=False,               # ffn of latent attns
        latent_proj=False,              # proj of latent attns
        latent_attn=0,                  # num of latent self-attns
        latent_shortcut=True,           # shortcut for latents
        latent_update_proj=False,       # proj before latents update
        latent_update_shortcut=False,   # shortcut for latents update
        latent_share=False,             # sharing latent to later blks
        # - impl
        impl_attn="torch",
        upcast_attention=False,
        upcast_softmax=False,
        stage=None,
        indice_key=None,
    ):
        super().__init__()
        from .attentions import LatentAttentions

        if head_ratio is not None:
            num_heads = max(int(num_heads / head_ratio), 1)
        if head_channels is not None:
            _check_at_most_one(head_channels, hidden_channels)
            hidden_channels = int(num_heads * head_channels)
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.stage = stage

        latent_dims = latent_dims or hidden_channels or channels
        is_new, latent_dims = check_new_latents(latent_dims, stage, granularity=latent_share)
        num_latents = num_latents # if is_new else 0

        indice_key = indice_key or "stage" + re.search(r"\d+", stage).group()
        proj_kwargs = dict(
            norm_layer=norm_layer,
            act_layer=act_layer,
            indice_key=indice_key,
        )

        in_proj = in_proj or "linear-act"
        in_proj_kwargs = dict(proj_kwargs, hidden_channels=hidden_channels, shortcut=in_shortcut)
        self.down = Projection(ops=in_proj, in_channels=channels, out_channels=hidden_channels, **in_proj_kwargs)

        self.latent_attn = LatentAttentions(
            channels=hidden_channels,
            num_heads=num_heads,
            num_latents=num_latents,
            qkv_proj=qkv_proj,
            qkv_bias=qkv_bias,
            hidden_channels=hidden_channels,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj=proj,
            proj_drop=proj_drop,
            proj_bias=proj_bias,
            ffn=ffn,
            ffn_ratio=ffn_ratio,
            shortcut=q_shortcut, # shortcut for q (feat cross-out)
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            # weights=weights,
            # init=init,
            latent_dims=latent_dims,
            latent_ffn=latent_ffn,
            latent_proj=latent_proj,
            latent_attn=latent_attn,
            latent_shortcut=latent_shortcut,
            latent_update_proj=latent_update_proj,
            latent_update_shortcut=latent_update_shortcut,
            # latent_share=latent_share,
            impl_attn=impl_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.up = Projection("lin", in_channels=hidden_channels, out_channels=channels)
        self.shortcut = shortcut

        # fusing
        self.alpha = float(alpha) if alpha is not None else None
        self.weights = self._get_weight(weights) if weights is not None else None
        self.gate = None
        if gate is not None:
            ops = gate if isinstance(gate, str) else "linear"
            self.gate = Projection(ops=ops, in_channels=channels, out_channels=1, **proj_kwargs)
            self.gate_mean = gate_mean
        _check_at_most_one(alpha, gate, weights)

        self.reset_parameters(init=init)
        return

    def _get_norm(self, norm_str):
        if norm_str in ["ln", "layer_norm"]:
            norm = nn.LayerNorm
        elif norm_str in ["bn", "batch_norm"]:
            norm = nn.BatchNorm1d
        elif norm_str == "":
            norm = nn.Identity
        else:
            raise ValueError(f"not support norm={norm_str}")
        return norm

    def _get_weight(self, weight_str):
        if weight_str.startswith('ws'):
            w = torch.zeros(size=[self.channels], dtype=torch.float32)
            if weight_str[2:] == 'N':
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif weight_str[2:]:
                w.data.fill_(float(weight_str[2:]))
            w = nn.Parameter(w)
        elif weight_str.startswith('w'):
            w = torch.tensor(float(weight_str[1:]) if weight_str[1:] else 0.0)
            w = nn.Parameter(w)
        else:
            w = float(weight_str)
        return w

    def reset_parameters(self, init=None):
        self.reset_parameters_default()
        if init == "lora":
            self.reset_parameters_lora()
        else:
            assert init is None, f"{self.__class__} not support init={init}"
        return

    def reset_parameters_default(self):  # bias & ln only
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_parameters_lora(self):
        if self.up is not None:
            proj = self.up[-1]
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        return

    def forward(self, point: Point, shortcut=None):
        point = Point(point)
        if shortcut is None:
            shortcut = point.feat

        point = self.down(point)
        point = self.latent_attn(point)
        point = self.up(point)

        # fusing
        feat = point.feat
        if self.alpha is not None:
            feat = feat * self.alpha
        if self.weights is not None:
            feat = feat * self.weights
        if self.gate is not None:
            gate = F.sigmoid(self.gate(shortcut))
            if self.gate_mean:  # mean within each cloud
                _offset = F.pad(point.offset, (1, 0))
                bincount = offset2bincount(point.offset)
                gate = torch_scatter.segment_csr(src=gate, indptr=_offset, reduce="mean").repeat_interleave(bincount, dim=0)
            feat = feat * gate

        if self.shortcut:
            feat = feat + shortcut

        point.feat = feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

def LatAttnEncoding(
    channels,
    num_heads,
    num_latents,
    # - qkv proj
    qkv_proj=True,
    qkv_bias=True,
    head_ratio=None,
    head_channels=None,
    hidden_channels=None,
    # - attn
    qk_scale=None,
    attn_drop=0.0,
    # - out proj & ffn
    proj=True,
    proj_drop=0.0,
    proj_bias=True,
    ffn=False,
    ffn_ratio=4.0,
    # - skip-conn
    shortcut=True,
    drop_path=0.0,
    # - norm & act
    norm_layer=nn.LayerNorm,
    act_layer=nn.GELU,
    pre_norm=True,
    weights=None,
    init=None,
    # - latent
    latent_dims=None,               # fdims of the latents
    latent_ffn=False,               # ffn of latent attns
    latent_proj=False,              # proj of latent attns
    latent_attn=0,                  # num of latent self-attns
    latent_shortcut=True,           # shortcut for latents
    latent_update_proj=False,       # proj before latents update
    latent_update_shortcut=False,   # shortcut for latents update
    latent_share=False,             # sharing latent to later blks
    # - impl
    impl_attn="torch",
    upcast_attention=False,
    upcast_softmax=False,
    stage=None,
):
    from .attentions import LatentAttentions
    latent_dims = latent_dims or channels

    is_new, latent_dims = check_new_latents(latent_dims, stage, granularity=latent_share)
    num_latents = num_latents # if is_new else 0

    if head_ratio is not None:
        num_heads = max(int(num_heads / head_ratio), 1)
    if head_channels is not None:
        _check_at_most_one(head_channels, hidden_channels)
        hidden_channels = int(num_heads * head_channels)

    _attn = LatentAttentions(
        channels=channels,
        num_heads=num_heads,
        num_latents=num_latents,
        qkv_proj=qkv_proj,
        qkv_bias=qkv_bias,
        hidden_channels=hidden_channels,
        qk_scale=qk_scale,
        attn_drop=attn_drop,
        proj=proj,
        proj_drop=proj_drop,
        proj_bias=proj_bias,
        ffn=ffn,
        ffn_ratio=ffn_ratio,
        shortcut=shortcut,
        drop_path=drop_path,
        norm_layer=norm_layer,
        act_layer=act_layer,
        pre_norm=pre_norm,
        weights=weights,
        init=init,
        latent_dims=latent_dims,
        latent_ffn=latent_ffn,
        latent_proj=latent_proj,
        latent_attn=latent_attn,
        latent_shortcut=latent_shortcut,
        latent_update_proj=latent_update_proj,
        latent_update_shortcut=latent_update_shortcut,
        # latent_share=latent_share,
        impl_attn=impl_attn,
        upcast_attention=upcast_attention,
        upcast_softmax=upcast_softmax,
    )
    return _attn


def build_encoding(cfg, _default=None):
    if isinstance(cfg, abc.Sequence):
        encoding_list = [build_encoding(c) for c in cfg]
        return nn.Sequential(*encoding_list)
    if cfg is _default:
        return _default

    cfg = copy.deepcopy(cfg)
    if cfg.type in globals():
        _cls = globals()[cfg.pop('type')]
        encoding = _cls(**cfg)
    else:
        encoding = ENCODING.build(**cfg)
        # raise ValueError(f'not support Encoding - cfg={cfg}')
    return encoding


def build_encoding_ptv3(embed_cfg, channels, norm_layer, act_layer, stage, _default=None, **kwargs):
    if embed_cfg is _default:
        return _default
    if isinstance(embed_cfg, abc.Sequence):
        embed_list = [build_encoding_ptv3(c, channels=channels, norm_layer=norm_layer, act_layer=act_layer, stage=stage, _default=_default, **kwargs) for c in embed_cfg]
        return nn.Sequential(*embed_list)
    if embed_cfg.stage:
        target_stage = [embed_cfg.stage] if isinstance(embed_cfg.stage, str) else embed_cfg.stage
        if not any(s in stage for s in target_stage):  # eg target_stage = ['enc', 'dec0']
            return _default

    # from pointcept.models.blocks.encoding import build_encoding
    embed_kwargs = copy.deepcopy(embed_cfg)
    embed_kwargs.channels = channels
    embed_kwargs.indice_key = "stage" + re.search(r"\d+", stage).group()
    if embed_kwargs.pop("pdnorm", False):
        embed_kwargs.norm_layer = norm_layer
    if embed_kwargs.act_layer == "act":
        embed_kwargs.act_layer = act_layer
    if embed_kwargs.drop_path == True:
        embed_kwargs.drop_path = kwargs.pop("drop_path", 0.0)

    if embed_cfg.type in ["LatAttnEncoding", "LatAttnAdapter"]:
        assert stage is not None
        embed_kwargs.stage = stage
        embed_kwargs.num_heads = embed_kwargs.num_heads or kwargs["num_heads"]
    if embed_cfg.type in ["LatAttnEncoding", ]:
        embed_kwargs.pop("indice_key", None)
    if embed_cfg.type in ["LatAttnEncoding", "LatAttnAdapter"]:
        embed_kwargs.impl_attn = kwargs["impl_attn"]

    return build_encoding(embed_kwargs)

class SparsePointHook(nn.Module):
    def __init__(self, keys=None):
        super().__init__()
        self._keys = tuple(keys or [])
        self._dict = {}

    def forward_pre_hook(self, module, inputs):
        (x,) = inputs  # expect args list
        # x - spconv.SparseConvTensor
        batch, grid_coord = x.indices.tensor_split([1], dim=1)
        point = Point(
            feat=x.features,
            batch=batch.reshape(-1),
            grid_coord=grid_coord,
            sparse_shape=x.spatial_shape,
            sparse_conv_feat=x,
        )
        for k, v in self._dict.items():
            point[k] = v
        return (point,)  # replacing args list

    def forward_hook(self, module, input, output):
        point : Point = output
        x = point.sparse_conv_feat
        for k in self._keys:
            if k in point:
                self._dict[k] = point[k]
        return x  # replacing output

    def __repr__(self):
        return f'SparsePointHook(id={id(self)})'

    def forward(self):
        raise RuntimeError

def build_encoding_spunet(embed_cfg, channels, norm_layer, act_layer, stage, indice_key=None, hook=None, _default=None, **kwargs):
    if embed_cfg is _default:
        return _default
    if isinstance(embed_cfg, abc.Sequence):
        embed_list = [build_encoding_spunet(c, channels=channels, norm_layer=norm_layer, act_layer=act_layer, stage=stage, indice_key=indice_key, hook=hook, _default=_default, **kwargs) for c in embed_cfg]
        return nn.Sequential(*embed_list)
    if embed_cfg.stage:
        target_stage = [embed_cfg.stage] if isinstance(embed_cfg.stage, str) else embed_cfg.stage
        if not any(s in stage for s in target_stage):  # eg target_stage = ['enc', 'dec0']
            return _default

    from pointcept.models.blocks.encoding import build_encoding
    embed_kwargs = copy.deepcopy(embed_cfg)
    embed_kwargs.channels = channels
    embed_kwargs.indice_key = indice_key or "stage" + re.search(r"\d+", stage).group()
    if embed_kwargs.pop("pdnorm", False):
        embed_kwargs.norm_layer = norm_layer
    if embed_kwargs.act_layer == "act":
        embed_kwargs.act_layer = act_layer
    if embed_kwargs.drop_path == True:
        embed_kwargs.drop_path = kwargs.pop("drop_path", 0.0)

    if embed_cfg.type in ["LatAttnEncoding", "LatAttnAdapter"]:
        assert stage is not None
        embed_kwargs.stage = stage
        # embed_kwargs.num_heads = embed_kwargs.num_heads or kwargs["num_heads"]
    if embed_cfg.type in ["LatAttnEncoding"]:
        embed_kwargs.pop("indice_key", None)
    if embed_cfg.type in ["LatAttnEncoding", "LatAttnAdapter"]:
        embed_kwargs.impl_attn = embed_kwargs.impl_attn or kwargs.pop("impl_attn", "flash")
    else:
        embed_kwargs.pop("impl_attn", None)

    blk = build_encoding(embed_kwargs)
    if hook is not None:
        blk.register_forward_pre_hook(hook.forward_pre_hook, with_kwargs=False)
        blk.register_forward_hook(hook.forward_hook, with_kwargs=False)
        blk._hook = hook
    return blk
