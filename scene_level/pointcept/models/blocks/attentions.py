
# from addict import Dict
from functools import partial

import re
import math
import numpy as np

import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None
try:
    import xformers.ops as xops
except ImportError:
    xops = None


from .projection import MLPs, Projection
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        channels: int,
        channels_v: int = None,
        # - attn
        qk_scale: float = None,
        attn_drop: float = 0.0,
        # - impl
        impl_attn: str = "flash",
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.channels_v = channels_v or channels
        assert self.channels % num_heads == 0, f"incompatible channels={channels}, num_heads={num_heads}"
        assert self.channels_v % num_heads == 0, f"incompatible channels={self.channels_v}, num_heads={num_heads}"

        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.impl_attn = impl_attn
        assert impl_attn in ["flash", "xops", "sdpa", "torch"]
        if impl_attn == "flash":
            assert upcast_attention is False, "Set upcast_attention to False when enable Flash Attention"
            assert upcast_softmax is False, "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.attn_drop = attn_drop
            # self.scale = qk_scale  # flash-attn default (None) to (channels // num_heads) ** -0.5
        elif impl_attn == "xops":
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            assert upcast_attention is False, "Set upcast_attention to False when using xops (slow in fp32)"
            assert upcast_softmax is False, "Set upcast_softmax to False when using xops (slow in fp32)"
            assert xops is not None, "Make sure xformer is installed."
            # self.patch_size_max = patch_size
            # self.patch_size = 0
            self.attn_drop = attn_drop
            self.scale = qk_scale  # xops default (None) to (channels // num_heads) ** -0.5
        elif impl_attn == "sdpa":
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            assert upcast_attention is False, "Set upcast_attention to False when using sdpa"
            assert upcast_softmax is False, "Set upcast_softmax to False when using sdpa"
            assert qk_scale is None, "Set qk_scale to None when using sdpa"
            self.attn_drop = attn_drop
            # torch.backends.cuda.enable_flash_sdp(False)
            # torch.backends.cuda.enable_mem_efficient_sdp(False)
            # torch.backends.cuda.enable_math_sdp(True)
        elif impl_attn == "torch":
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.attn_drop = torch.nn.Dropout(attn_drop)
        else:
            raise ValueError(f"not support impl_attn={impl_attn}")

        self.softmax = nn.Softmax(dim=-1)
        return

    def forward(self, q, k=None, v=None, cu_seqlens=None, cu_seqlens_k=None, attn_mask=None):
        """ multi-head scaled-dot-product attn
        input:
            q   : [N, 3C]           - var-len packed qkv
                : [N, C_q/C_k/C_v]  - var-len separate q/k/v
                : [B, N_cld, 3C]            - packed qkv
                : [B, N_cld, C_q/C_k/C_v]   - separate q/k/v

            cu_seqlens  : [B + 1]   - cumulative seq-lens (batch size)
                                    - start inds of each batch & ending with idx of last-pts of last-batch
        """

        C = self.channels
        H = self.num_heads

        if self.impl_attn == "flash":
            if cu_seqlens is not None:
                # [N, 3C / C_qkv]
                cu_seqlens = cu_seqlens.int()
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                if k is None and v is None:
                    feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                        # [N, 3, H, C//H]
                        qkv=q.half().reshape(-1, 3, H, C // H),
                        # [#patches + 1] - start-end of each attn patch
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).reshape(-1, C)
                elif k is not None and v is None:
                    if cu_seqlens_k is None:
                        cu_seqlens_k = cu_seqlens
                        max_seqlen_k = max_seqlen
                    else:
                        cu_seqlens_k = cu_seqlens_k.int()
                        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
                    feat = flash_attn.flash_attn_varlen_kvpacked_func(
                        # [N, H, C//H]
                        q=q.half().reshape(-1, H, C // H),
                        # [N, 2, H, C//H]
                        kv=k.half().reshape(-1, 2, H, C // H),
                        # [#patches + 1] - start-end of each attn patch
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen_k,
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).reshape(-1, C)
                else:
                    if cu_seqlens_k is None:
                        cu_seqlens_k = cu_seqlens
                        max_seqlen_k = max_seqlen
                    else:
                        cu_seqlens_k = cu_seqlens_k.int()
                        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
                    feat = flash_attn.flash_attn_varlen_func(
                        q=q.half().reshape(-1, H, C // H),
                        k=k.half().reshape(-1, H, C // H),
                        v=v.half().reshape(-1, H, self.channels_v // H),
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen_k,
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).reshape(-1, C)
            else:
                # [B, N_cld, 3C / C_qkv]
                if k is None and v is None:
                    feat = flash_attn.flash_attn_qkvpacked_func(
                        # [B, N_cld, 3C] => [B, N_cld, 3, H, C//H]
                        qkv=q.half().unflatten(-1, [3, H, C // H]),
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).flatten(-2)
                elif k is not None and v is None:
                    feat = flash_attn.flash_attn_kvpacked_func(
                        # [B, N_cld, H, C//H]
                        q=q.half().unflatten(-1, [H, C // H]),
                        # [B, N_cld, 2, H, C//H]
                        kv=k.half().unflatten(-1, [2, H, C // H]),
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).flatten(-2)
                else:
                    feat = flash_attn.flash_attn_func(
                        q=q.half().unflatten(-1, [H, C // H]),
                        k=k.half().unflatten(-1, [H, C // H]),
                        v=v.half().unflatten(-1, [H, self.channels_v // H]),
                        dropout_p=self.attn_drop if self.training else 0,
                        softmax_scale=self.scale,
                    ).flatten(-2)
            feat = feat.to(q.dtype)

        elif self.impl_attn == "xops":
            if cu_seqlens is not None:
                if k is None and v is None:
                    # [N, 3C] => [1, N, 3, H, C//H] => 3 x [1, N, H, C//H]
                    q, k, v = q.reshape(-1, 3, H, C // H).unsqueeze(0).unbind(dim=-3)
                elif v is None:
                    q = q.reshape(-1, H, C // H).unsqueeze(0)
                    k, v = k.reshape(-1, 2, H, C // H).unsqueeze(0).unbind(dim=-3)
                else:
                    q = q.reshape(-1, H, C // H).unsqueeze(0)
                    k = k.reshape(-1, H, C // H).unsqueeze(0)
                    v = v.reshape(-1, H, self.channels_v // H).unsqueeze(0)

                # np_seqlens = cu_seqlens.numpy()
                # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(np_seqlens[1:] - np_seqlens[:-1], device=cu_seqlens.device)
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # .to(torch.int32)
                seqinfo = xops.fmha.attn_bias._SeqLenInfo(seqstart=cu_seqlens, max_seqlen=seqlens.max(), min_seqlen=seqlens.min(), seqstart_py=None)
                if cu_seqlens_k is None:
                    seqinfo_k = seqinfo
                else:
                    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                    k_seqinfo = xops.fmha.attn_bias._SeqLenInfo(seqstart=cu_seqlens_k, max_seqlen=seqlens_k.max(), min_seqlen=seqlens_k.min(), seqstart_py=None)
                attn_bias = xops.fmha.BlockDiagonalMask(q_seqinfo=seqinfo, k_seqinfo=k_seqinfo)
            else:
                if k is None and v is None:
                    # [B, N_cld, 3C] => [B, N_cld, 3, H, C//H] => 3 x [B, N_cld, H, C//H]
                    q, k, v = q.unflatten(-1, [3, H, C // H]).unbind(dim=-3)
                else:
                    q = q.unflatten(-1, [H, C // H])
                    k = k.unflatten(-1, [H, C // H])
                    v = v.unflatten(-1, [H, self.channels_v // H])
                attn_bias = None

            if self.upcast_attention:
                q = q.float()
                k = k.float()
                v = v.float()
            feat = xops.memory_efficient_attention(
                query=q, key=k, value=v,
                attn_bias=attn_bias,
                scale=self.scale,
                p=self.attn_drop if self.training else 0,
            )
            feat = feat.reshape(-1, C)
            feat = feat.to(q.dtype)

        elif self.impl_attn == "sdpa":
            if k is None and v is None:
                # q: [...N, 3C] => [...N, 3, H, C//H] => 3 x [...H, N, C//H]
                q, k, v = q.unflatten(-1, [3, H, C // H]).transpose(-4, -2).unbind(dim=-3)
            elif v is None:
                q = q.unflatten(-1, [H, C // H]).transpose(-3, -2)
                k, v = k.unflatten(-1, [2, H, C // H]).transpose(-4, -2).unbind(dim=-3)
            else:
                # [...N, C] => [...N, H, C//H] => [...H, N, C//H]
                q = q.unflatten(-1, [H, C // H]).transpose(-3, -2)
                k = k.unflatten(-1, [H, C // H]).transpose(-3, -2)
                v = v.unflatten(-1, [H, C // H]).transpose(-3, -2)

            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.tolist()
                cu_seqlens_k = cu_seqlens_k.tolist() if cu_seqlens_k is not None else cu_seqlens
                # B x [H, N_cld, C//H]
                q = torch.tensor_split(q, cu_seqlens[1:-1], dim=1)
                k = torch.tensor_split(k, cu_seqlens_k[1:-1], dim=1)
                v = torch.tensor_split(v, cu_seqlens_k[1:-1], dim=1)
                # - nested [B, H, N_cld, C//H]
                q = torch.nested.as_nested_tensor(list(q))
                k = torch.nested.as_nested_tensor(list(k))
                v = torch.nested.as_nested_tensor(list(v))

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            feat = nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0,
                scale=self.scale,
            )

            if cu_seqlens is not None:
                # - unbind B x [H, N_cld, C//H] => [H, N, C//H]
                feat = torch.concat(feat.unbind(), dim=1)
            # [...N, H, C//H] => [...N, C]
            feat = feat.transpose(-3, -2).flatten(-2)
            feat = feat.to(q.dtype)

        elif self.impl_attn == "torch":
            # TODO: check torch1.13 compatibility

            if k is None and v is None:
                # q: [...N, 3C] => [...N, 3, H, C//H] => 3 x [...H, N, C//H]
                q, k, v = q.unflatten(-1, [3, H, C // H]).transpose(-4, -2).unbind(dim=-4)
            elif v is None:
                q = q.unflatten(-1, [H, C // H]).transpose(-3, -2)
                k, v = k.unflatten(-1, [2, H, C // H]).transpose(-4, -2).unbind(dim=-3)
            else:
                # [...N, C] => [...N, H, C//H] => [...H, N, C//H]
                q = q.unflatten(-1, [H, C // H]).transpose(-3, -2)
                k = k.unflatten(-1, [H, C // H]).transpose(-3, -2)
                v = v.unflatten(-1, [H, C // H]).transpose(-3, -2)

            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.tolist()
                cu_seqlens_k = cu_seqlens_k.tolist() if cu_seqlens_k is not None else cu_seqlens
                # B x [H, N_cld, C//H]
                q = torch.tensor_split(q, cu_seqlens[1:-1], dim=1)
                k = torch.tensor_split(k, cu_seqlens_k[1:-1], dim=1)
                v = torch.tensor_split(v, cu_seqlens_k[1:-1], dim=1)
                # - nested [B, H, N_cld, C//H]
                q = torch.nested.as_nested_tensor(list(q))  # torch.jagged supports only 1st-dim ragged B x [*, fdims...]
                k = torch.nested.as_nested_tensor(list(k))
                v = torch.nested.as_nested_tensor(list(v))

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, N_cld, N_cld]
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(q.dtype)
            feat = (attn @ v)

            if cu_seqlens is not None:
                # - unbind B x [H, N_cld, C//H] => [H, N, C//H]
                feat = torch.concat(feat.unbind(), dim=1)
            # [...N, H, C//H] => [...N, C]
            feat = feat.transpose(-3, -2).flatten(-2)
            feat = feat.to(q.dtype)

        else:
            raise NotImplementedError
        return feat


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        # bias=True,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # linear - gelu - drop - linear - drop
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        # - qkv proj
        qkv_proj=True,
        qkv_bias=True,
        hidden_channels=None,
        # - attn
        qk_norm=False,
        qk_scale=None,
        attn_drop=0.0,
        # - out proj & ffn
        proj=True,
        proj_drop=0.0,
        proj_bias=True,
        ffn=True,
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
        # - impl
        impl_attn="flash",
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.pre_norm = pre_norm
        self.qk_norm = qk_norm
        self.hidden_channels = hidden_channels = hidden_channels or channels
        assert hidden_channels % num_heads == 0, f"incompatible hidden_channels={hidden_channels}, num_heads={num_heads}"
        assert hidden_channels == channels or proj, f"incompatible channels={channels}, hidden_channels={hidden_channels}, with proj={proj}"

        self.qkv = None
        if qkv_proj:
            self.qkv = nn.Linear(channels, hidden_channels * 3, bias=qkv_bias)

        self.q_norm = self.k_norm = None
        if qk_norm:
            self.q_norm = norm_layer(hidden_channels // num_heads)
            self.k_norm = norm_layer(hidden_channels // num_heads)

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            channels=hidden_channels,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            impl_attn=impl_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.proj = None
        if proj:
            self.proj = nn.Linear(hidden_channels, channels, bias=proj_bias)
            self.proj_drop = nn.Dropout(proj_drop)
        self.norm = norm_layer(channels)

        self.ffn = None
        if ffn:
            self.ffn = FFN(
                in_channels=channels,
                hidden_channels=int(channels * ffn_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
            self.ffn_norm = norm_layer(channels)
            self.ffn_weights = nn.Parameter(weights * torch.ones([channels]), requires_grad=True) if weights is not None else None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.weights = nn.Parameter(weights * torch.ones([channels]), requires_grad=True) if weights is not None else None

        self.reset_parameters(init)
        return

    def reset_parameters(self, init=None):
        if init == "lora":
            self.reset_parameters_lora()
        else:
            assert init is None, f"{self.__class__} not support init={init}"
        return

    def reset_parameters_lora(self):
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        if self.ffn is not None:
            nn.init.zeros_(self.ffn.fc2.weight)
            nn.init.zeros_(self.ffn.fc2.bias)
        return

    def forward(self, feat, cu_seqlens=None, attn_mask=None):
        shortcut = feat

        if self.pre_norm:  # pre-norm
            feat = self.norm(feat)
        # qkv-mha-proj
        if self.qkv is not None:
            feat = self.qkv(feat)
        if self.qk_norm:  # qk-norm
            q, k, v = feat.reshape(-1, 3, self.num_heads, self.hidden_channels // self.num_heads).unbind(dim=-3)
            feat = torch.concat([self.q_norm(q), self.k_norm(k), v], dim=-1).reshape(-1, 3 * self.hidden_channels)
        feat = self.attn(feat, cu_seqlens=cu_seqlens, attn_mask=attn_mask)
        if self.proj is not None:
            feat = self.proj(feat)
            feat = self.proj_drop(feat)
        # drop-shortcut
        if self.weights is not None:
            feat = self.weights * feat
        feat = self.drop_path(feat)
        if self.shortcut:
            feat = shortcut + feat
        if not self.pre_norm:  # post-norm
            feat = self.norm(feat)

        # ffn
        if self.ffn is not None:
            shortcut = feat
            if self.pre_norm:  # pre-norm
                feat = self.ffn_norm(feat)
            feat = self.ffn(feat)
            if self.ffn_weights is not None:
                feat = self.ffn_weights * feat
            feat = self.drop_path(feat)
            if self.shortcut:
                feat = shortcut + feat
            if not self.pre_norm:  # post-norm
                feat = self.ffn_norm(feat)

        return feat


class CrossAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        # - qkv proj
        qkv_proj=True,
        qkv_bias=True,
        kv_channels=None,
        hidden_channels=None,
        # - attn
        qk_norm=False,
        qk_scale=None,
        attn_drop=0.0,
        # - out proj & ffn
        proj=True,
        proj_drop=0.0,
        proj_bias=True,
        ffn=True,
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
        # - impl
        impl_attn="flash",
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.qk_norm = qk_norm
        self.pre_norm = pre_norm
        self.kv_channels = kv_channels = kv_channels or channels
        self.hidden_channels = hidden_channels = hidden_channels or channels
        assert hidden_channels % num_heads == 0, f"incompatible hidden_channels={hidden_channels}, num_heads={num_heads}"
        assert hidden_channels == channels or proj or not shortcut, f"incompatible channels={channels}, hidden_channels={hidden_channels}, with proj={proj}, shortcut={shortcut}"

        self.q = self.kv = None
        if qkv_proj == "q":
            self.q = nn.Linear(channels, hidden_channels, bias=qkv_bias)
        elif qkv_proj == "kv":
            self.kv = nn.Linear(kv_channels, hidden_channels * 2, bias=qkv_bias)
        elif qkv_proj:
            self.q = nn.Linear(channels, hidden_channels, bias=qkv_bias)
            self.kv = nn.Linear(kv_channels, hidden_channels * 2, bias=qkv_bias)

        self.q_norm = self.k_norm = False
        if qk_norm:
            self.q_norm = norm_layer(hidden_channels // self.num_heads)
            self.k_norm = norm_layer(hidden_channels // self.num_heads)

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            channels=hidden_channels,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            impl_attn=impl_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.proj = None
        if proj:
            self.proj = nn.Linear(hidden_channels, channels, bias=proj_bias)
            self.proj_drop = nn.Dropout(proj_drop)

        self.norm = norm_layer(channels) if pre_norm in [True, False, "q"] else nn.Identity()
        self.norm_kv = norm_layer(kv_channels) if pre_norm in [True, "kv"] else nn.Identity()
        assert pre_norm in (True, False, "q", "kv", None), f"not support pre_norm={pre_norm}"

        self.ffn = None
        if ffn:
            self.ffn = FFN(
                in_channels=channels,
                hidden_channels=int(channels * ffn_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
            self.ffn_norm = norm_layer(channels)
            self.ffn_weights = nn.Parameter(weights * torch.ones([channels]), requires_grad=True) if weights is not None else None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.weights = nn.Parameter(weights * torch.ones([channels]), requires_grad=True) if weights is not None else None

        self.reset_parameters(init)
        return

    def reset_parameters(self, init=None):
        if init == "lora":
            self.reset_parameters_lora()
        else:
            assert init is None, f"{self.__class__} not support init={init}"
        return

    def reset_parameters_lora(self):
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        if self.ffn is not None:
            nn.init.zeros_(self.ffn.fc2.weight)
            nn.init.zeros_(self.ffn.fc2.bias)
        return

    def forward(self, feat, feat_kv, cu_seqlens=None, cu_seqlens_k=None, shortcut=None):
        if shortcut is None:
            shortcut = feat
        if self.pre_norm:  # pre-norm
            feat = self.norm(feat)
            feat_kv = self.norm_kv(feat_kv)
        # qkv-mha-proj
        if self.q is not None:
            feat = self.q(feat)
        if self.kv is not None:
            feat_kv = self.kv(feat_kv)
        if self.qk_norm:  # qk-norm
            feat = self.q_norm(feat.reshape(-1, self.num_heads, self.hidden_channels // self.num_heads)).reshape(-1, self.hidden_channels)
            k, v = feat_kv.reshape(-1, 2, self.num_heads, self.hidden_channels // self.num_heads).unbind(dim=-3)
            feat_kv = torch.concat([self.k_norm(k), v], dim=-1).reshape(-1, 2 * self.hidden_channels)
        feat = self.attn(feat, k=feat_kv, cu_seqlens=cu_seqlens, cu_seqlens_k=cu_seqlens_k)
        if self.proj is not None:
            feat = self.proj(feat)
            feat = self.proj_drop(feat)
        # drop-shortcut
        if self.weights is not None:
            feat = self.weights * feat
        feat = self.drop_path(feat)
        if self.shortcut:
            feat = shortcut + feat
        if not self.pre_norm:  # post-norm
            feat = self.norm(feat)

        # ffn
        if self.ffn is not None:
            shortcut = feat
            if self.pre_norm:  # pre-norm
                feat = self.ffn_norm(feat)
            feat = self.ffn(feat)
            if self.ffn_weights is not None:
                feat = self.ffn_weights * feat
            feat = self.drop_path(feat)
            if self.shortcut:
                feat = shortcut + feat
            if not self.pre_norm:  # post-norm
                feat = self.ffn_norm(feat)

        return feat


class LatentAttentions(PointModule):
    """
    attn complexity (M = #latents, s = #self-attn, N = #points, P = patch_size):
        latent attn : + 2NM + s*M^2
        attn adapter: N/P*(P+M)^2 => +2NM + N/P*M^2
    """
    def __init__(
        self,
        channels,
        num_heads,
        num_latents,
        # - qkv proj
        qkv_proj=True,
        qkv_bias=True,
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
        shortcut=True,                  # shortcut for feat q (cross-out)
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
        # latent_share=False,             # sharing latent to later blks
        # - impl
        impl_attn="flash",
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels or channels
        self.latent_dims = latent_dims = latent_dims or channels
        self.latent_update_shortcut = latent_update_shortcut
        # self.latent_share = latent_share

        kwargs = dict(
            num_heads=num_heads,
            qkv_proj=qkv_proj,
            qkv_bias=qkv_bias,
            hidden_channels=hidden_channels,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            # proj - latent
            proj=latent_proj,
            proj_drop=proj_drop,
            proj_bias=proj_bias,
            # ffn - latent
            ffn=latent_ffn,
            ffn_ratio=ffn_ratio,
            shortcut=latent_shortcut,
            drop_path=drop_path,
            # - norm & act
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            # - impl
            impl_attn=impl_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.latents = self.get_latents(num_latents) if num_latents > 0 else None  # [M, C]
        self.cross_in = CrossAttention(channels=latent_dims, kv_channels=channels, **kwargs)
        if not latent_proj:
            latent_dims = hidden_channels

        self.latent_attn = []
        for i in range(latent_attn):
            self.latent_attn += [SelfAttention(channels=latent_dims, **kwargs)]
        self.latent_attn = nn.Sequential(*self.latent_attn) if self.latent_attn else None

        out_kwargs = dict(kwargs, proj=proj, ffn=ffn, shortcut=shortcut, weights=weights, init=init)
        self.cross_out = CrossAttention(channels=channels, kv_channels=latent_dims, **out_kwargs)

        self.latent_update_proj = None
        if latent_update_proj:
            proj_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)
            self.latent_update_proj = Projection(ops=latent_update_proj, in_channels=latent_dims, out_channels=self.latent_dims, **proj_kwargs)
        return

    def get_latents(self, num_latents, init_scale=0.02):
        latents = nn.Parameter(torch.empty(num_latents, self.latent_dims))
        with torch.no_grad():
            latents.normal_(0.0, init_scale)
        return latents

    def extra_repr(self):
        if self.latents is not None:
            return f"(latents): {tuple(self.latents.shape)}"

    def forward(self, point: Point):
        feat = point.feat  # [N, C]
        cu_seqlens = F.pad(point.offset, (1, 0)).int()  # 0-start offset (max_pts-end)

        B = len(point.offset)
        if self.latents is not None:
            latents = point.latents = self.latents.repeat(B, 1)
            num_latents = self.latents.shape[0]
        else:
            latents = point.latents  # [BM, C]
            num_latents = latents.shape[0] // B
        cu_seqlens_lat = torch.arange(B + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device) * num_latents

        latents = self.cross_in(feat=latents, feat_kv=feat, cu_seqlens=cu_seqlens_lat, cu_seqlens_k=cu_seqlens)
        if self.latent_attn is not None:
            latents = latents.reshape(B, num_latents, self.latent_dims)  # [B, M, C]
            latents = self.latent_attn(latents)
            latents = latents.reshape(-1, self.latent_dims)  # [BM, C]
        feat = self.cross_out(feat=feat, feat_kv=latents, cu_seqlens=cu_seqlens, cu_seqlens_k=cu_seqlens_lat)

        if self.latent_update_proj is not None:
            latents = self.latent_update_proj(latents)
        if self.latent_update_shortcut:
            latents = point.latents + latents

        point.feat = feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.latents = latents
        return point


class BiAttentions(PointModule):
    """
    attn complexity (l = #latents, N = #points, P = patch_size):
        latent attn : + 2Nl
        attn adapter: N/P*(P+l)^2 => +2Nl + N/P*l^2
    """
    def __init__(
        self,
        channels,
        num_heads,
        num_latents,
        # - qkv proj
        qkv_proj=True,
        qkv_bias=True,
        hidden_channels=None,
        # - attn
        qk_scale=None,
        attn_drop=0.0,
        # - out proj & ffn?
        proj=True,
        proj_drop=0.0,
        proj_bias=True,
        # - skip-conn
        # shortcut=True,
        drop_path=0.0,
        # - norm & act
        norm_layer=nn.LayerNorm,
        # act_layer=nn.GELU,
        pre_norm=True,
        weights=1e-4,
        # init=None,
        # - latent
        latent_dims=None,       # fdims of the latents
        latent_proj=True,       # proj of latent attns
        # latent_share="stage",   # sharing latent to later blks
        # - impl
        impl_attn="torch",
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels = hidden_channels or channels
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.impl_attn = impl_attn
        assert impl_attn in ["torch", "loop"], f"requireing explicit calculation of attn-mask, not support impl={impl_attn}"
        assert hidden_channels % num_heads == 0
        assert hidden_channels == channels or (proj and latent_proj)

        self.latent_dims = latent_dims = latent_dims or channels
        self.latents = nn.Parameter(torch.empty(num_latents, latent_dims)) if num_latents > 0 else None  # [M, C]
        # self.latent_share = latent_share

        self.norm = norm_layer(channels)
        self.norm_lat = norm_layer(latent_dims)
        self.pre_norm = pre_norm

        # attn
        self.qkv = self.qkv_lat = None
        if qkv_proj:
            self.qkv = nn.Linear(channels, 2 * hidden_channels, bias=qkv_bias)
            self.qkv_lat = nn.Linear(latent_dims, 2 * hidden_channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = self.proj_lat = None
        if proj:
            self.proj = nn.Linear(hidden_channels, channels, bias=proj_bias)
        if latent_proj:
            self.proj_lat = nn.Linear(hidden_channels, latent_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.weights = nn.Parameter(weights * torch.ones([channels]), requires_grad=True)
        self.weights_lat = nn.Parameter(weights * torch.ones([latent_dims]), requires_grad=True)

        # impl
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self._reset_parameters()
        return

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if any(name.startswith(n) for n in ["norm", "weights"]):
                continue
            if name.endswith(".weight"):
                nn.init.xavier_uniform_(param)
            elif name.endswith(".bias"):
                param.data.fill_(0)
            elif name == "latents":
                nn.init.normal_(param, std=0.02)
            else:
                raise ValueError(f"shall not init param with name={name}")

    def forward(self, point: Point):
        offset = point.offset
        B = len(offset)
        C = self.channels
        H = self.num_heads

        shortcut = feat = point.feat  # [N, C]
        shortcut_lat = latents = self.latents if self.latents is not None else point.latents  # [M/BM, C]

        # pre-norm
        if self.pre_norm:
            feat = self.norm(feat)
            latents = self.norm_lat(latents)

        # - qkv
        if self.qkv is not None:
            feat_qkv = self.qkv(feat)
            latents_qkv = self.qkv_lat(latents)
        else:
            feat_qkv = feat.repeat(1, 2)
            latents_qkv = latents.repeat(1, 2)

        if self.impl_attn == "torch":
            # - transpose & split
            feat_qk, feat_v = feat_qkv.unflatten(-1, [2, H, C // H]).transpose(-4, -2).unbind(-3)  # 2 x [H, N, C//H]
            feat_qk = torch.tensor_split(feat_qk, offset[:-1].tolist(), dim=-2)  # B x [H, N_cld, C//H]
            feat_v = torch.tensor_split(feat_v, offset[:-1].tolist(), dim=-2)

            if self.latents is not None:
                latents_qk, latents_v = latents_qkv.unflatten(-1, [2, H, C // H]).transpose(-4, -2).unbind(-3)  # 2 x [H, M, C//H]
                latents_qk = latents_qk.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, M, C//H]
                latents_v = latents_v.unsqueeze(0).repeat(B, 1, 1, 1)
                shortcut_lat = shortcut_lat.repeat(B, 1)  # [BM, C]
            else:
                latents_qk, latents_v = latents_qkv.reshape([B, -1, 2, H, C//H]).transpose(-4, -2).unbind(-3)  # 2 x [B, H, M, C//H]

            # - nested
            feat_qk = torch.nested.as_nested_tensor(list(feat_qk))  # [B, H, N_cld, C//H]
            feat_v = torch.nested.as_nested_tensor(list(feat_v))
            latents_qk = torch.nested.as_nested_tensor(list(latents_qk.unbind(0)))  # [B, H, M, C//H]
            latents_v = torch.nested.as_nested_tensor(list(latents_v.unbind(0)))

            # - attn
            if self.upcast_attention:
                feat_qk = feat_qk.float()
                latents_qk = latents_qk.float()
            attn = feat_qk @ latents_qk.transpose(-2, -1)  # [B, H, N_cld, M]

            # NOTE - nested_tensor not support: clamp / max / ops with broadcasting
            # attn = attn - attn.max()
            # if not self.upcast_attention:
            #     # Do not de/increase +/-50000, data type half has quite limited range
            #     attn = torch.clip(attn, min=-50000, max=50000)

            attn_T = attn.transpose(-2, -1).contiguous()  # [B, H, M, N_cld]
            # attn_T = attn_T - torch.max(attn_T, dim=-1, keepdim=True).values

            attn = self.softmax(attn.float()).to(feat.dtype)  # need to upcast for nested_tensor ???
            attn_T = self.softmax(attn_T.float()).to(latents.dtype)

            attn = self.attn_drop(attn)
            attn_T = self.attn_drop(attn_T)

            # - attn out
            feat_out = attn @ latents_v
            latents_out = attn_T @ feat_v

            feat_out = torch.concat(feat_out.unbind(), dim=-2).permute(1, 0, 2)  # B x [H, N_cld, C//H] => [N, H, C//H]
            feat_out = feat_out.flatten(-2)  # [N, C]
            latents_out = torch.concat(latents_out.unbind(), dim=-2).permute(1, 0, 2)  # B x [H, M, C//H] => [BM, H, C//H]
            latents_out = latents_out.flatten(-2)  # [BM, C]

            if self.proj is not None:
                feat_out = self.proj_drop(self.proj(feat_out))
                latents_out = self.proj_drop(self.proj_lat(latents_out))

            # scale-droppath-shortcut
            feat = shortcut + self.drop_path(self.weights * feat_out)
            latents = shortcut_lat + self.drop_path(self.weights_lat * latents_out)

            # post-norm
            if not self.pre_norm:
                feat = self.norm(feat)
                latents = self.norm_lat(latents)

        elif self.impl_attn == "loop":
            # - transpose
            feat_qkv = feat_qkv.unflatten(-1, [2, H, C // H]).transpose(-4, -2)  # [H, 2, N, C//H]
            if self.latents is not None:
                latents_qkv = latents_qkv.unflatten(-1, [2, H, C // H]).transpose(-4, -2)  # [H, 2, M, C//H]
                latents_qkv = latents_qkv.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, 2, M, C//H]
                shortcut_lat = shortcut_lat.repeat(B, 1)  # [BM, C]
            else:
                latents_qkv = latents_qkv.reshape([B, -1, 2, H, C//H]).transpose(-4, -2)  # [B, H, 2, M, C//H]

            # - attn
            if self.upcast_attention:
                feat_qk = feat_qk.float()
                latents_qk = latents_qk.float()
            # - loop-over split
            feat_out = []
            latents_out = []
            _i_start = 0
            for B_i in range(len(point.offset)):
                _i_end = point.offset[B_i]
                _feat_qk, _feat_v = feat_qk[:, :, _i_start:_i_end, :].unbind(-3)  # [H, N_cld, C//H]
                _latents_qk, _latents_v = latents_qkv[B_i].unbind(-3)  # [H, M, C//H]

                attn = _feat_qk @ _latents_qk.transpose(-2, -1)  # [H, N_cld, M]

                attn = attn - attn.max()
                if not self.upcast_attention:
                    # Do not de/increase +/-50000, data type half has quite limited range
                    attn = torch.clip(attn, min=-50000, max=50000)

                attn_T = attn.transpose(-2, -1)  # [H, M, N_cld]
                attn_T = attn_T - torch.max(attn_T, dim=-1, keepdim=True).values

                attn = self.softmax(attn)
                attn_T = self.softmax(attn_T)

                attn = self.attn_drop(attn)
                attn_T = self.attn_drop(attn_T)

                # - attn out
                feat_out.append(attn @ _latents_v)
                latents_out.append(attn_T @ _feat_v)
                _i_start = _i_end

            feat_out = torch.concat(feat_out, dim=-2).permute(1, 0, 2)  # B x [H, N_cld, C//H] => [N, H, C//H]
            feat_out = feat_out.flatten(-2)  # [N, C]
            latents_out = torch.concat(dim=-2).permute(1, 0, 2)  # B x [H, M, C//H] => [BM, H, C//H]
            latents_out = latents_out.flatten(-2)  # [BM, C]

            if self.proj is not None:
                feat_out = self.proj_drop(self.proj(feat_out))
                latents_out = self.proj_drop(self.proj_lat(latents_out))

            # scale-droppath-shortcut
            feat = shortcut + self.drop_path(self.weights * feat_out)
            latents = shortcut_lat + self.drop_path(self.weights_lat * latents_out)

            # post-norm
            if not self.pre_norm:
                feat = self.norm(feat)
                latents = self.norm_lat(latents)

        else:
            raise NotImplementedError(f"impl_attn={self.impl_attn}")

        point.feat = feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.latents = latents_out
        return point
