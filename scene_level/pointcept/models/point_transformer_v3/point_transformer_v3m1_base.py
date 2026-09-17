"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from collections import abc
from functools import partial
from addict import Dict
import re
import copy
import math
import weakref

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        # coord - [#patch, patch_size, patch_size, 3] - relative grid_coord (3d index)
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        # channels=enc_channels[s],  # (32, 64, 128, 256, 512),
        num_heads,
        # num_heads=enc_num_head[s],  # (2, 4, 8, 16, 32), => head_dims = 16
        patch_size,
        # patch_size=enc_patch_size[s],  # (1024, 1024, 1024, 1024, 1024),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        # order_index=i % len(self.order),
        enable_rpe=False,
        impl_attn="flash",
        upcast_attention=True,
        # upcast_attention=False,
        upcast_softmax=True,
        # upcast_softmax=False,
        stage=None,
        embed_adap=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.impl_attn = impl_attn
        assert impl_attn in ["flash", "sdpa", "torch"]
        if impl_attn == "flash":
            assert enable_rpe is False, "Set enable_rpe to False when enable Flash Attention"
            assert upcast_attention is False, "Set upcast_attention to False when enable Flash Attention"
            assert upcast_softmax is False, "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
            # self.scale = qk_scale  # flash-attn default (None) to (channels // num_heads) ** -0.5
        elif impl_attn == "sdpa":
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            assert upcast_attention is False, "Set upcast_attention to False when using sdpa"
            assert upcast_softmax is False, "Set upcast_softmax to False when using sdpa"
            assert qk_scale is None, "Set qk_scale to None when using sdpa"
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = attn_drop
            # torch.backends.cuda.enable_flash_sdp(False)
            # torch.backends.cuda.enable_mem_efficient_sdp(False)
            # torch.backends.cuda.enable_math_sdp(True)
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

        self.stage = stage
        self.embed_adap = self.build_embed(embed_adap, stage=stage)
        return

    def build_embed(self, config, stage):
        if config is None:
            return None
        # setup
        embed_kwargs = copy.deepcopy(config)
        embed_kwargs.stage = stage
        embed_kwargs.channels = self.channels

        embed_type = embed_kwargs.type
        if embed_type in ["PrefixEncoding"]:
            self.forward = self.forward_prefix

        elif embed_type in ["Adapter"]:
            self.forward = self.forward_parallel

        else:
            raise ValueError(f"not support embed_type={embed_type}")

        from pointcept.models.blocks.encoding import build_encoding
        embed_module = build_encoding(embed_kwargs)
        if hasattr(embed_module, "hooks"):
            with torch.no_grad():
                embed_module.hooks(weakref.proxy(self))
        return embed_module

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        # relative posenc - using grid coord & used as attn bias
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            # [#patch, patch_size, patch_size, 3]
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        # calc mapping to/from pts - after per-cloud padding
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            # [B] - #pts of each cloud (batches_len)
            bincount = offset2bincount(offset)
            bincount_pad = (
                # round up to patch (multiple of patch_size)
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            # - either original (no pad, less than patch_size), or pad (multiple of patch_size)
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            # 0-start offset
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            # create mapping
            # - pad:    [N_pad] - inds into original pts    - padd_pts = pts[pad]
            # - unpad:  [N]     - inds into padded pts      - pts = padd_pts[unpad]
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            # - cu_seqlens: [#patch] - inds into padded pts - patches_start = padded_pts[cu_seqlens]
            cu_seqlens = []
            for i in range(len(offset)):
                # i-th cloud - shift inds for each cloud [start-end] by aligning start

                # - init inds into original pts - shifting inds into padded pts
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    # padding by repeating the cloud
                    pad[
                        # i-th cloud [padded part] = [ end - #padded : end ] = [ end - (patch_size - bincount[i] % patch_size) : end]
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        # i-th cloud [padded part & shift left 1 patch_size]
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]

                # - init inds into padded pts - shifting inds back into original pts
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]

                # - inds into padded pts - the start of each patch
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                # - ending with idx of last pts of last patch
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        # attn - out proj - drop

        if self.impl_attn in ["torch", "sdpa"]:
            self.patch_size = min(
                # batches_len.min().item()
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # inds mapping:
        # - pad:    [N_pad] - inds in padded -> inds into original
        # - unpad:  [N]     - inds in original -> inds into padded
        # - cu_seqlens: [#patches]  - inds in patches -> inds into padded pts
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # merge the mapping:
        # - padded_serialized_pts = pts[order][pad]
        order = point.serialized_order[self.order_index][pad]
        # - pts = padded_serialized_pts[unpad][inverse]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        # [N', K, 3, H, C']
        # - [N_pad, 3C], where N_pad = N'(#patch) * K (patch_size), C = H (#head) * C' (head_dim)
        qkv = self.qkv(point.feat)[order]

        if self.impl_attn == "torch":
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        elif self.impl_attn == "sdpa":
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn_mask = None
            if self.enable_rpe:
                attn_mask = self.rpe(self.get_rel_pos(point, order))

            feat = nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0,
                # scale=self.scale,
            )
            feat = feat.reshape(-1, C)
            feat = feat.to(qkv.dtype)

        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                # [N_pad, 3, #head, head_dim]
                # qkv.half().reshape(-1, 3, H, C // H),
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                # [#patches + 1] - start-end of each attn patch
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # out-proj & drop
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        point.feat = feat
        return point

    @torch.no_grad()
    # @torch.inference_mode()  # NOTE: inference tensors cannot be saved for backward - index need to be stored in compute graph
    def get_padding_and_inverse_prefix(self, point):
        # calc mapping to/from pts, with latents as per-patch prefix - after per-cloud padding
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        latent_pad_key = "latent_pad_key"
        latent_unpad_key = "latent_unpad_key"
        latent_seqlens_key = "latent_seqlens_key"
        latent_count_key = "latent_count_key"

        full_unpad_key = "full_pad_key"

        num_latents = self.embed_adap.num_latents

        patch_size = self.patch_size
        # patch_size = self.patch_size - num_latents
        patch_size_full = patch_size + num_latents

        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            # [B] - #pts of each cloud (batches_len)
            bincount = offset2bincount(offset)
            # round up to patch (multiple of patch_size)
            patch_count = torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc")
            bincount_pad = patch_count * patch_size
            # - only pad point when num of points larger than patch_size
            mask_pad = bincount > patch_size
            # - either original (no pad, less than patch_size), or pad (multiple of patch_size)
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            # - allocate prefix into each patch
            latent_count = patch_count * num_latents
            bincount_full = bincount_pad + latent_count
            # 0-start offset
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            _offset_full = nn.functional.pad(torch.cumsum(bincount_full, dim=0), (1, 0))
            _offset_latent = nn.functional.pad(torch.cumsum(latent_count, dim=0), (1, 0))

            # create mapping
            # - pad:    [N_pad] - inds into original pts    - padded_pts = pts[pad]
            pad = torch.empty(_offset_pad[-1], device=offset.device, dtype=torch.long)
            # - unpad:  [N]     - inds into padded pts      - pts = padded_pts[unpad]
            unpad = torch.arange(_offset[-1], device=offset.device)
            # - cu_seqlens: [#patch] - inds into padded pts - patches_start = padded_pts[cu_seqlens]
            cu_seqlens = []

            # - lat_pad:    [N_full] - inds into original pts (with latents) - padded_pts = pts[lat_pad]
            lat_pad = torch.empty(_offset_full[-1], device=offset.device, dtype=torch.long)
            # - lat_unpad:  [#patch x #prefix] - inds into padded pts (with latents) - latent = padded_pts[lat_unpad]
            lat_unpad = torch.empty(_offset_latent[-1], device=offset.device, dtype=torch.long) if self.embed_adap.latent_update else None
            # - lat_seqlens: [#patch] - inds into padded pts (with latents) - patches_start = padded_pts[cu_seqlens]
            lat_seqlens = []

            # - full_unpad: [N] - inds into padded pts (with latents)  - pts = padded_pts[full_unpad]
            full_unpad = torch.arange(_offset[-1], device=offset.device)

            _lat_inds = _offset[-1] + torch.arange(num_latents, device=offset.device)
            B = len(offset)
            for i in range(B):
                # i-th cloud - shift inds for each cloud [start-end] by aligning start

                # - init inds into original pts - shifting inds into padded pts
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                full_unpad[_offset[i] : _offset[i + 1]] += _offset_full[i] - _offset[i]

                # - init size of padded pts - filling inds back into original pts
                pad_inds = _offset[i] + torch.arange(bincount_pad[i], device=offset.device) % bincount[i]  # fill last-patch if necessary
                pad[_offset_pad[i] : _offset_pad[i + 1]] = pad_inds

                # - inds into padded pts - the start of each patch
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

                lat_inds = _lat_inds
                if self.embed_adap.latents is None and self.embed_adap.latent_update:
                    lat_inds = lat_inds + i * num_latents  # updated queries for each cloud in batch

                lat_pad_inds = torch.cat(
                    [pad_inds.reshape(patch_count[i], -1), lat_inds.unsqueeze(0).repeat(patch_count[i], 1)],
                    dim=-1,
                ).reshape(-1)
                lat_pad[_offset_full[i] : _offset_full[i + 1]] = lat_pad_inds

                if lat_unpad is not None:

                    lat_unpad_inds = torch.arange(num_latents, device=offset.device).unsqueeze(0).repeat(patch_count[i], 1)
                    lat_unpad_inds += torch.arange(patch_count[i], device=offset.device).unsqueeze(1) * patch_size_full + min(patch_size, bincount_pad[i]) + _offset_full[i]
                    lat_unpad[_offset_latent[i] : _offset_latent[i + 1]] = lat_unpad_inds.reshape(-1)

                lat_seqlens.append(
                    torch.arange(
                        _offset_full[i],
                        _offset_full[i + 1],
                        step=patch_size_full,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                # - ending with idx of last pts of last patch
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
            point[latent_pad_key] = lat_pad
            point[latent_unpad_key] = lat_unpad
            point[latent_seqlens_key] = nn.functional.pad(
                torch.concat(lat_seqlens), (0, 1), value=_offset_full[-1]
            )
            point[latent_count_key] = latent_count
            point[full_unpad_key] = full_unpad
        return (
            point[pad_key],
            point[unpad_key],
            point[cu_seqlens_key],
            point[latent_pad_key],
            point[latent_unpad_key],
            point[latent_seqlens_key],
            point[latent_count_key],
            point[full_unpad_key]
        )

    def forward_prefix(self, point, _collect_attn=False):
        # attn - out proj - drop
        # - with latents as prefix in patches

        if self.impl_attn in ["torch", "sdpa"]:
            self.patch_size = min(
                # batches_len.min().item()
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        B = point.offset.shape[0]

        # inds mapping:
        # - pad:    [N_pad] - inds in padded -> inds into original
        # - unpad:  [N]     - inds in original -> inds into padded
        # - cu_seqlens: [#patches]  - inds in patches -> inds into padded pts
        pad, unpad, cu_seqlens, lat_pad, lat_unpad, lat_seqlens, lat_count, full_unpad = self.get_padding_and_inverse_prefix(point)

        # merge the mapping:
        # - padded_serialized_pts = pts[order][pad]
        order = point.serialized_order[self.order_index]
        # - pts = padded_serialized_pts[unpad][inverse]
        inverse = point.serialized_inverse[self.order_index]

        feat = point.feat
        latents = self.embed_adap(point)  # get latents

        # qkv projection
        qkv = self.qkv(feat)
        if self.embed_adap.qkv_proj:
            # separete q/kv
            latents = self.embed_adap.qkv_proj(latents)
        elif self.embed_adap.qkv_share:
            # sharing q/kv
            latents = self.qkv(latents)

        if _collect_attn:
            # collect feat-q latent-k
            point._feat_q = qkv[:, :C]
            point._latents_k =  latents[:, -2*C:-C]

        # padding the serialized points with latents
        # [N_pad, 3C], where N_pad = #patch * (patch_size + #latents)
        L = self.embed_adap.num_latents
        if self.embed_adap.latent_update:
            qkv = torch.cat([qkv[order], latents], dim=0)[lat_pad]
        else:
            # latents as kv only
            q, kv = torch.tensor_split(qkv, [C], dim=1)
            q = q[order[pad]]
            if self.embed_adap.qkv_share:
                latents = latents[:, C:]
            kv = torch.cat([kv[order], latents], dim=0)[lat_pad]

        if self.impl_attn == "torch":
            if self.embed_adap.latent_update:
                # reshape qkv: [N', K+L, 3, H, C'] => [3, N', H, K+L, C']
                # - N_pad = N'(#patch) * ( K (patch_size) + L (#latents) ), C = H (#head) * C' (head_dim)
                q, k, v = qkv.reshape(-1, K+L, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            else:
                q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                k, v = kv.reshape(-1, K+L, 2, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        elif self.impl_attn == "sdpa":
            if self.embed_adap.latent_update:
                # reshape qkv: [N', K+L, 3, H, C'] => [3, N', H, K+L, C']
                # - N_pad = N'(#patch) * ( K (patch_size) + L (#latents) ), C = H (#head) * C' (head_dim)
                q, k, v = qkv.reshape(-1, K+L, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            else:
                q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                k, v = kv.reshape(-1, K+L, 2, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn_mask = None
            if self.enable_rpe:
                attn_mask = self.rpe(self.get_rel_pos(point, order))

            feat = nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0,
                # scale=self.scale,
            )
            feat = feat.reshape(-1, C)
            feat = feat.to(qkv.dtype)

        else:
            if self.embed_adap.latent_update:
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    # [N_pad, 3, #head, head_dim]
                    qkv.half().reshape(-1, 3, H, C // H),
                    # [#patches + 1] - start-end of each attn patch (with latents)
                    lat_seqlens,
                    max_seqlen=K + L,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
            else:
                feat = flash_attn.flash_attn_varlen_kvpacked_func(
                    # [N_pad, 3, #head, head_dim]
                    q=q.half().reshape(-1, H, C // H),
                    kv=kv.half().reshape(-1, 2, H, C // H),
                    # [#patches + 1] - start-end of each attn patch
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=lat_seqlens,
                    max_seqlen_q=K,
                    max_seqlen_k=K + L,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)

            feat = feat.to(qkv.dtype)

        if self.embed_adap.latent_update:
            latents = feat[lat_unpad]
            feat = feat[full_unpad][inverse]
            # update latents - avg over patches
            latents_batch_inds = torch.repeat_interleave(torch.arange(B, device=feat.device), lat_count)
            latents_accum = torch.zeros((B, L, C), device=feat.device, dtype=feat.dtype)

            latents_accum = latents_accum.scatter_add_(
                dim=0,
                index=latents_batch_inds.reshape(-1, 1).expand(-1, C).reshape(-1, L, C),
                src=latents.reshape(-1, L, C),  # [#patch (total), L, C]
            )
            latents = latents_accum / lat_count.view(B, 1, 1)
            # out-proj & drop
            if self.embed_adap.proj:
                feat = self.proj(feat)
                feat = self.proj_drop(feat)
                latents = self.embed_adap.proj(latents.reshape(-1, C))
            else:
                # sharing out-proj
                feat = torch.cat([feat, latents.reshape(-1, C)], dim=0)
                feat = self.proj(feat)
                feat = self.proj_drop(feat)
                feat, latents = torch.tensor_split(feat, [-B * L], dim=0)

            if self.embed_adap.latent_update_proj:
                latents = self.embed_adap.latent_update_proj(latents)
            if self.embed_adap.latent_update_shortcut:
                latents += point.latents
            point.latents = latents

            # # out-proj & drop
            # if self.latent_type in ["s", "so"]:
            #     feat = torch.cat([feat, latents], dim=0)
            #     feat = self.proj(feat)
            #     feat = self.proj_drop(feat)
            #     latent_num = len(lat_unpad)
            #     latents = feat[-latent_num:]
            #     feat = feat[:-latent_num]
            # else:
            #     feat = self.proj(feat)
            #     feat = self.proj_drop(feat)
            # # update latents - avg over patches
            # latents_batch_inds = torch.repeat_interleave(torch.arange(B, device=feat.device), lat_count)
            # latents_accum = torch.zeros((B, self.latent_num, C), device=feat.device, dtype=feat.dtype)
            # latents_accum = latents_accum.scatter_add_(
            #     dim=0,
            #     index=latents_batch_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, self.latent_num, C),
            #     src=latents.reshape(-1, self.latent_num, C),
            # )
            # latents = latents_accum / lat_count.view(B, 1, 1)
            # point.latents = latents

        else:
            feat = feat[unpad][inverse]
            # out-proj & drop
            feat = self.proj(feat)
            feat = self.proj_drop(feat)

        point.feat = feat
        return point

    def forward_parallel(self, point):
        # - parallel adapter
        feat_adap = self.embed_adap(point).feat

        # attn - out proj - drop
        if self.impl_attn in ["torch", "sdpa"]:
            self.patch_size = min(
                # batches_len.min().item()
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # inds mapping:
        # - pad:    [N_pad] - inds in padded -> inds into original
        # - unpad:  [N]     - inds in original -> inds into padded
        # - cu_seqlens: [#patches]  - inds in patches -> inds into padded pts
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # merge the mapping:
        # - padded_serialized_pts = pts[order][pad]
        order = point.serialized_order[self.order_index][pad]
        # - pts = padded_serialized_pts[unpad][inverse]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        # [N', K, 3, H, C']
        # - [N_pad, 3C], where N_pad = N'(#patch) * K (patch_size), C = H (#head) * C' (head_dim)
        qkv = self.qkv(point.feat)[order]

        if self.impl_attn == "torch":
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        elif self.impl_attn == "sdpa":
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn_mask = None
            if self.enable_rpe:
                attn_mask = self.rpe(self.get_rel_pos(point, order))

            feat = nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0,
                # scale=self.scale,
            )
            feat = feat.reshape(-1, C)
            feat = feat.to(qkv.dtype)

        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                # [N_pad, 3, #head, head_dim]
                qkv.half().reshape(-1, 3, H, C // H),
                # [#patches + 1] - start-end of each attn patch
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # out-proj & drop
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        # - joint adapter
        feat = feat + feat_adap

        point.feat = feat
        return point

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
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


class Block(PointModule):
    def __init__(
        self,
        channels,
        # channels=enc_channels[s],  # (32, 64, 128, 256, 512),
        num_heads,
        # num_heads=enc_num_head[s],  # (2, 4, 8, 16, 32),
        patch_size=48,
        # patch_size=enc_patch_size[s],  # (1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        # drop_path=enc_drop_path_[i],  # linspace(0, drop_path=0.3, sum(enc_depths))
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        # order_index=i % len(self.order),
        cpe_indice_key=None,
        # cpe_indice_key=f"stage{s}",
        enable_rpe=False,
        impl_attn="flash",
        impl_spconv=None,
        upcast_attention=True,
        # upcast_attention=False
        upcast_softmax=True,
        # upcast_softmax=False,
        stage=None,  # (enc|dec){i}
        embed_blk=None,
        embed_pos=None,
        embed_ppos=None,
        embed_patt=None,
        embed_pattsc=None,
        embed_pstatt=None,
        embed_attn=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.impl_attn = impl_attn
        self.stage = stage

        from pointcept.models.blocks.encoding import build_encoding_ptv3
        _kwargs = dict(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            order_index=order_index,
            norm_layer=norm_layer,
            act_layer=act_layer,
            drop_path=drop_path,
            impl_attn=impl_attn,
            stage=stage,
            _default=None,
        )
        _build_embed = partial(build_encoding_ptv3, **_kwargs)
        self.embed_blk = _build_embed(embed_blk)        # after blk calling

        self.cpe = PointSequential(
            # SubMConv 3x3 - linear (conv 1x1) - norm (ln)
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                # indice_key - to save indice generation time (indice_data - in-kernel-out mapping)
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.embed_pos = _build_embed(embed_pos)        # after enc-pos
        self.embed_ppos = _build_embed(embed_ppos)      # parallel enc-pos

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            # order/pad - attn - unpad/inverse - proj - drop (0)
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            impl_attn=impl_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            stage=stage,
            embed_adap=embed_patt,                      # parallel attn
        )
        self.embed_attn = _build_embed(embed_attn)      # after attn
        self.embed_pattsc = _build_embed(embed_pattsc)  # parallel attn-sc
        self.embed_pstatt = _build_embed(embed_pstatt)  # post attn (before sc)

        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            # linear - gelu - drop (0) - linear - drop (0)
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.impl_spconv = impl_spconv
        assert impl_spconv in ["sync", None], f"{self.__class__} not support impl_spconv={impl_spconv}"
        return

    def forward(self, point: Point):
        # sc | posenc
        # sc | pre-norm - attn - drop path
        # sc | pre-norm - mlp(ffn) - drop path

        if self.embed_blk is not None:
            point = self.embed_blk(point)

        shortcut = point.feat
        if self.embed_ppos is not None:  # parallel to posenc
            feat = self.cpe(point).feat
            point = self.embed_ppos(Point(point, feat=shortcut))
            point.feat = point.feat + feat
        else:
            point = self.cpe(point)
            point.feat = shortcut + point.feat
        if self.embed_pos is not None:
            point = self.embed_pos(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        if self.embed_pattsc is not None:  # parallel to attn-sc
            feat = self.drop_path(self.attn(point)).feat
            point = self.embed_pattsc(Point(point, feat=shortcut))
            point.feat = point.feat + feat
        elif self.embed_pstatt is not None:  # post attn (before sc)
            point = self.drop_path(self.embed_pstatt(self.attn(point)))
            point.feat = shortcut + point.feat
        else:
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)
        if self.embed_attn is not None:
            point = self.embed_attn(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)

        if self.impl_spconv == "sync":
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        # norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
        act_layer=None,
        # act_layer=nn.GELU,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # check stride = 2 ** n, exists an int n>0 - stride is 2's powers
        assert stride > 1
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

        return


    def forward(self, point: Point):
        # calc pooling cluster & pooled pts
        # - proj - pool

        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            # too-large pooling: avoid pooling into single cluster
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        # the bit loc for pooling - slicing to pool
        # - points (keys) with same higher bits & diff lower bits pooled into 1
        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "latents" in point.keys():
            point_dict["latents"] = point.latents

        if self.traceable:
            # for unpooling
            # - per-point cluster id (pooled pts inds)
            # - current point & feat
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)

        point.sparsify()
        return point

class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        # norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
        act_layer=None,
        # act_layer=nn.GELU,
        traceable=False,  # record parent and cluster
        unpool_keys=None,
        impl_spconv=None,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable
        self.unpool_keys = unpool_keys
        self.impl_spconv = impl_spconv
        assert impl_spconv in ["sync", None], f"{self.__class__} not support impl_spconv={impl_spconv}"

    def forward(self, point):
        # proj - unpool | skip-proj
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")    # `Point` for pooling (feat from last block of encoder)
        inverse = point.pop("pooling_inverse")  # per-point cluster inds (into pooled point)
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        # NOTE: would not affect enc-stage point, due to the tensor version control (feat) & exlicit replacement (spase_conv_feat)

        if self.traceable:
            parent["unpooling_parent"] = point

        if self.unpool_keys is not None:
            for k in self.unpool_keys:  # update pooling_parent
                parent[k] = point[k]

        if self.impl_spconv == "sync":
            parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        # norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
        act_layer=None,
        # act_layer=nn.GELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            # TODO: padding no effect?
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        # [rgb, normal-xyz]
        in_channels=6,
        order=("z", "z-trans"),
        # order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        # enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        # dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        unpool_keys=None,
        impl_attn="flash",
        impl_spconv=None,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        init=None,
        # for additional encoding
        embed_blk=None,
        embed_pos=None,
        embed_ppos=None,
        embed_patt=None,
        embed_pattsc=None,
        embed_pstatt=None,
        embed_attn=None,
        # for Point Prompt Trainint (PPT)
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            # PDNorm - Prompt-driven Norm - multi-dataset Point Prompt Trainint (PPT)
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = partial(nn.LayerNorm, eps=1e-5)
        # activation layers
        act_layer = nn.GELU

        # stem: SubMConv 5x5 - norm (bn) - act (gelu)
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            # stage: [pooling (s>0), block * depth]

            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    # pooling: linear - pool - norm (bn) - act (gelu)
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                # stage-s, block-i
                enc.add(
                    # attn-block: posenc | attn | ffn
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        impl_attn=impl_attn,
                        impl_spconv=impl_spconv,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        stage=f"enc{s}",
                        embed_blk=embed_blk,
                        embed_pos=embed_pos,
                        embed_ppos=embed_ppos,
                        embed_patt=embed_patt,
                        embed_pattsc=embed_pattsc,
                        embed_pstatt=embed_pstatt,
                        embed_attn=embed_attn,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                # stage: [unpooling, block * depth]

                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    # unpooling: proj (linear-bn-gelu) - unpool | skip-proj (linear-bn-gelu)
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        unpool_keys=unpool_keys,
                        impl_spconv=impl_spconv,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    # stage-s, block-i
                    dec.add(
                        # attn-block: posenc | attn | ffn
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            impl_attn=impl_attn,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            stage=f"dec{s}",
                            embed_blk=embed_blk,
                            embed_pos=embed_pos,
                            embed_ppos=embed_ppos,
                            embed_patt=embed_patt,
                            embed_pattsc=embed_pattsc,
                            embed_pstatt=embed_pstatt,
                            embed_attn=embed_attn,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if init is True:
            self.reset_parameters()
        elif init:
            raise ValueError(f"not support init={init}")
        return None

    def reset_parameters(self):
        module_list = list(self.children())
        for module in module_list:
            # - actual init
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, spconv.SubMConv3d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif hasattr(module, "reset_parameters"):
                # - inited
                continue
            else:
                module_list += list(module.children())
        return

    def forward(self, data_dict):
        point = Point(data_dict)
        # - serialized_code, serialized_order, serialized_inverse
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        # - sparse_shape, sparse_conv_feat
        point.sparsify()

        # stem - conv
        point = self.embedding(point)

        # enc - [pooling, block * depth], ...
        point = self.enc(point)

        if not self.cls_mode:
            # dec - [unpooling, block * depth], ...
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
