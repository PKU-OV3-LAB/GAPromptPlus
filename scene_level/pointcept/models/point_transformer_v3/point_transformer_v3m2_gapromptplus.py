"""
Point Transformer - V3 Mode2

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
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
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
            self.flash_dtype = None
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

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
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
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
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
        else:
            if self.flash_dtype is None:
                self.flash_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(self.flash_dtype).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
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
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.ls1 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.ls2 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = PointSequential(
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

    def forward(self, point: Point, non_linear = None):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if non_linear:
            point.feat = point.feat + non_linear(point.feat)
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            point_dict["idx_ptr"] = idx_ptr
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
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

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
        return point


@MODELS.register_module("PT-v3m2-gapromptplus")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        prompt_stages=[2,3,4]
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.enc_mode = enc_mode
        self.freeze_encoder = freeze_encoder
        self.enc_depths = enc_depths

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
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
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        self.enc_nonlinar_mappings = nn.ModuleDict()
        for s in range(self.num_stages):
            for i in range(self.enc_depths[s]):
                self.enc_nonlinar_mappings[f'enc{s}-nonlinar{i}'] = Adapter(enc_channels[s], max(min(64,8*2**s),16))

        self.point_shift_prompter = ShiftNet(3, 3, enc_channels[0], perturbation=0.0005)
        self.prompt_stages = prompt_stages
        self.keypoint_prompters = nn.ModuleList([InstancePointPrompter(point_number=64//2**s, hidden_dimension=enc_channels[s]) for s in self.prompt_stages])

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
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
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)

        point = self.point_shift_prompter(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

        if 0 in self.prompt_stages:
            point, prompt_indexs = self.keypoint_prompters[self.prompt_stages.index(0)](point, require_prompt_positions=True)

        point.sparsify()

        # point = self.enc(point)
        for s in range(self.num_stages):
            stage = getattr(self.enc, f"enc{s}")
            if s > 0:
                down = getattr(stage, "down")
                point = down(point)
                if s in self.prompt_stages:
                    point, prompt_indexs = self.keypoint_prompters[self.prompt_stages.index(s)](point, require_prompt_positions=True)
                    point.sparsify()

            for i in range(self.enc_depths[s]):
                block = getattr(stage, f"block{i}")
                point = block(point, non_linear=self.enc_nonlinar_mappings[f'enc{s}-nonlinar{i}'])

                if s in self.prompt_stages:
                    point = prompt_propagation(point, prompt_indexs=prompt_indexs, k=8, fusion_ratio=0.05)

        point = self.keypoint_prompters[0].remove_prompt(point)

        if not self.enc_mode:
            point = self.dec(point)
        return point


class ShiftNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dimension=384, perturbation=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dimension = hidden_dimension
        self.mlp_position = nn.Sequential(
            nn.Linear(hidden_dimension, 16),
            nn.ReLU(),
            nn.Linear(16, self.out_channels),
            nn.Sigmoid()
        )
        self.perturbation = perturbation
        for layer in self.mlp_position:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)
                nn.init.constant_(layer.bias, val=0.0)

    def forward(self, point: Point):
        batch = len(point.offset)
        shift_prompt = self.mlp_position(point.feat) * self.perturbation * point.coord
        point.coord = point.coord + shift_prompt
        for key in ["grid_coord", "serialized_code", "serialized_order", "serialized_inverse"]:
            if key in point:
                point.pop(key)
        point = Point(point)
        point.grid_size = point.grid_size.mean()

        return point


class Adapter(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0.1
                ):
        super(Adapter, self).__init__()
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.layer_norm = nn.LayerNorm(self.embed_dims)
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = nn.GELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init_weights)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
        self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]
        self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
        self.sampled_bias_1 =  self.ln2.bias

    def forward(self, x):
        x = self.layer_norm(x)
        scale = 0.7
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        return out*scale


def sequence_uniform_sampling(point: Point, sample_number: int, order_type=0):
    '''Serialization-based uniform sampling for prompt keypoint selection.'''
    batch = len(point.offset)
    sample_coords = []
    sample_features = []
    sample_orders = []
    for i in range(batch):
        start = 0 if i == 0 else point.offset[i - 1]
        end = point.offset[i]
        point_number = end - start
        sample_step = max(point_number // sample_number, 1)
        order_index = torch.arange(0, point_number, sample_step, device=point.coord.device)[:sample_number]
        serialized_order = point.serialized_order[order_type][start:end]
        sample_order = serialized_order[order_index]
        sample_coords.append(point.coord[sample_order])
        sample_features.append(point.feat[sample_order])
        sample_orders.append(sample_order)
    sample_coords = torch.stack(sample_coords, dim=0)
    sample_features = torch.stack(sample_features, dim=0)
    sample_orders = torch.stack(sample_orders, dim=0)

    return sample_coords, sample_features, sample_orders


def sequence_knn(point, center_indexs, k, order_type=0):
    """
    Args:
        center_indexs: [B, M] (in serialized space)
    Returns:
        neighbor_orders: [B, M, K] (point indices)
    """
    batch = len(point.offset)
    half_k = k // 2
    center_positions = point.serialized_inverse[order_type][center_indexs.reshape(-1)].reshape(batch, -1)
    offsets = torch.cat([torch.arange(-half_k, 0, device=center_positions.device), torch.arange(1, half_k + 1, device=center_positions.device)])  # [K]
    neighbor_indexs_all = []
    serial_order = point.serialized_order[order_type]  # [N]

    for b in range(batch):
        start = 0 if b == 0 else point.offset[b - 1]
        end = point.offset[b]
        center_pos = center_positions[b] - start  # [M]
        N = end - start
        neighbor_pos = center_pos[:, None] + offsets[None, :]  # [M, K]
        neighbor_pos = neighbor_pos.clamp(0, N - 1)
        # pos -> point index
        neighbor_idx = serial_order[start:end][neighbor_pos]  # [M, K]
        neighbor_indexs_all.append(neighbor_idx)
    neighbor_indexs = torch.stack(neighbor_indexs_all, dim=0)

    return neighbor_indexs


class InstancePointPrompter(nn.Module):
    def __init__(self, point_number=20, hidden_dimension=384, scale=0.1, factor=5, ):
        super().__init__()
        self.scale = scale
        self.factor = factor
        self.point_number = point_number
        self.prompt_mark = 'prompt_point_number'
        self.hidden_dimension = hidden_dimension
        self.prompter = nn.Sequential(nn.Linear(hidden_dimension, 4))
        self.prompt_tokens = nn.Parameter(torch.zeros([self.point_number, hidden_dimension]))
        nn.init.kaiming_normal_(self.prompter[0].weight)  # Initialize
        nn.init.zeros_(self.prompter[0].bias)  # Initialize bias to zero
        nn.init.kaiming_uniform_(self.prompt_tokens)

    def forward(self, point: Point, order_type=0, require_prompt_positions=False):
        batch = len(point.offset)
        candidate_points, candidate_features, candidate_indexs = sequence_uniform_sampling(point, self.point_number*4, order_type=order_type)
        candidate_scores = self.prompter(candidate_features)
        coord_prompts = candidate_scores[..., :3]
        scores = candidate_scores[..., 3]
        _, idx = torch.topk(scores, self.point_number, dim=1)
        prompt_points = torch.gather(candidate_points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))+self.scale*torch.gather(coord_prompts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        prompt_features = torch.gather(candidate_features, 1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dimension))+self.prompt_tokens[None]
        for i in range(batch):
            if i==0:
                prompted_coord = torch.concat([point.coord[0:point.offset[i]], prompt_points[i]], dim=0)
                prompted_feature = torch.concat([point.feat[0:point.offset[i]], prompt_features[i]], dim=0)
            else:
                prompted_coord = torch.concat([prompted_coord, point.coord[point.offset[i-1]:point.offset[i]], prompt_points[i]], dim=0)
                prompted_feature = torch.concat([prompted_feature, point.feat[point.offset[i-1]:point.offset[i]], prompt_features[i]], dim=0)
        point.coord = prompted_coord
        point.feat = prompted_feature
        point.offset = point.offset + torch.arange(1, batch+1).cuda()*self.point_number
        point[self.prompt_mark] = self.point_number

        for key in ['grid_coord', 'batch', 'serialized_order_index', 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 'sparse_shape', 'sparse_conv_feat']:
            if key in point:
                point.pop(key)

        point = Point(point)
        point.grid_size = point.grid_size.mean()
        shuffle_orders = ('serialized_order_index' in point.keys())
        point.serialization(order=point.order, shuffle_orders=shuffle_orders)
        if require_prompt_positions:
            prompt_indexs = point.offset[None].transpose(0,1) - torch.arange(1, self.point_number+1, device=point.offset.device)[None] # [B, prompt_number]
            return point, prompt_indexs

        return point


    def remove_prompt(self, point: Point):
        """Remove prompt points from Point hierarchy."""

        batch = len(point.offset)

        hierarchies = 1
        count_cur = point
        while "pooling_parent" in count_cur.keys():
            hierarchies += 1
            count_cur = count_cur.pooling_parent

        cur = point
        prev = None
        for i in range(hierarchies):
            prompt_number = cur[self.prompt_mark] if self.prompt_mark in cur.keys() else 0
            if prompt_number==0:
                prev = cur
                cur = cur.pooling_parent
                continue

            original_offset = cur.offset - torch.arange(1, batch+1).cuda()*prompt_number
            keep_indices = []
            for i in range(batch):
                cur_start = 0 if i == 0 else cur.offset[i - 1]
                cur_end = cur.offset[i]
                ori_start = 0 if i == 0 else original_offset[i - 1]
                ori_end = original_offset[i]
                num_ori = ori_end - ori_start
                keep_indices.append(torch.arange(cur_start, cur_start+num_ori, device=cur.offset.device))
            keep_indices = torch.cat(keep_indices)

            for key in ["coord", "grid_coord", "feat", "batch"]:
                if key in cur.keys() and torch.is_tensor(cur[key]):
                    if cur[key].shape[0] == cur.offset[-1]:
                        cur[key] = cur[key][keep_indices]
            for key in ["pooling_inverse"]:
                if prev and key in prev.keys():
                    if prev[key].shape[0] == cur.offset[-1]:
                        prev[key] = prev[key][keep_indices]
            cur.offset = original_offset
            prev = cur
            cur = cur.pooling_parent

        return point


def propagate(xyz1, xyz2, feat2, fusion_ratio = 0.3):
    """
    Input:
        xyz1: input points position data, [B, M, K, 3]
        xyz2: sampled input points position data, [B, M, 3]
        feat2: input points data, [B, M, D]
    Return:
        delta_feat: upsampled points data, [B, M, K, D]
    """
    B, M, K, _ = xyz1.shape
    dists = torch.sum((xyz1 - xyz2[:,:,None])**2, dim=-1)
    dist_recip = 1.0 / (dists + 1e-5)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    weight = weight.view(B, M, K)
    interpolated_points = feat2[:, :, None] * weight[..., None]
    delta_feat = fusion_ratio*interpolated_points

    return delta_feat


def prompt_propagation(point: Point, prompt_indexs: torch.Tensor, k: int, order_type: int = 0, fusion_ratio=0.3):
    prompt_coords = point.coord[prompt_indexs]
    prompt_feat = point.feat[prompt_indexs]
    neighbor_indexs = sequence_knn(point, prompt_indexs, k, order_type)
    neighbor_coords = point.coord[neighbor_indexs.reshape(-1)].reshape(*neighbor_indexs.shape,3)
    delta_feat = propagate(neighbor_coords, prompt_coords, prompt_feat, fusion_ratio=fusion_ratio)
    point.feat.index_add_(0, neighbor_indexs.reshape(-1), delta_feat.reshape(-1, delta_feat.shape[-1]))
    if "sparse_conv_feat" in point.keys():
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

    return point
