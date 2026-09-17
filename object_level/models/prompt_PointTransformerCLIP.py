import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from collections import OrderedDict
from .Point_MAE import Encoder, Group
from .Point_PN import Point_PN
from .PointPrompt import ShiftNet, PointPrompt, InstancePointPrompter, PatchPrompter, Group, propagate, pooling, FNetBlock


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=10, idx=0, config=None):
        super().__init__()
        if config is not None:
            self.config = config
        self.dim = dim
        self.ln_1 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.ln_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, mlp_hidden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_hidden_dim, dim))
        ]))

        # GAPrompt++
        self.prompt_dropout = nn.Dropout(0.1)
        self.num_tokens = num_tokens
        if idx in self.config.prompt_layers:
            self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens, dim))
        # self.FT = FNetBlock()
        self.adapter = Adapter(embed_dims=dim, reduction_dims=config.adapter_config.adapter_dim)
        # self.adapter1 = Adapter(embed_dims=dim, reduction_dims=12)
        self.out_transform = nn.Sequential(nn.BatchNorm1d(dim), nn.GELU())

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x, global_feature=None, multi_scale_prompt=None, layer_id=None, level1_center=None, level1_index=None, level2_center=None, level2_index=None):
        if global_feature is not None and multi_scale_prompt is not None and layer_id in self.config.prompt_layers:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.expand(x.shape[0], -1, -1))
            token_prompt = token_prompt+0.5*torch.concat([multi_scale_prompt, global_feature], dim=1)
            x = torch.cat((x[:,0:1], token_prompt, x[:,1:]), 1)
        elif global_feature is not None and layer_id in self.config.prompt_layers:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.expand(x.shape[0], -1, -1))
            token_prompt = token_prompt + 0.5*global_feature.expand([-1, self.num_tokens, -1])
            x = torch.cat((x[:,0:1], token_prompt, x[:,1:]), 1)
        elif global_feature is not None and layer_id in self.config.prompt_layers:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.expand(x.shape[0], -1, -1))
            token_prompt = token_prompt + global_feature.expand(x.shape[0], -1, -1)
            x = torch.cat((x[:,0:1], token_prompt, x[:,1:]), 1)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if self.config.propagation_type == 'permutation_after_attention':
            B,G,_ = x.shape
            cls_x = x[:,0:1]
            x = x[:,1:]
            G = G-1
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
        else:
            cls_x = x[:,0:1]
            x = x[:,1:]
        if self.adapter is not None:
            if global_feature is not None and layer_id in self.config.prompt_layers:
                x = x + self.adapter(x+global_feature*0.5)
            else:
                x = x + self.adapter(x)
        if layer_id in self.config.prompt_layers:
            x = torch.concat([cls_x, x[:,self.num_tokens:,:]], dim=1)
        else:
            x = torch.concat([cls_x, x], dim=1)
        return x

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
            self.scale = nn.Linear(self.embed_dims, 1)
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
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

    def calc_sampled_param_num(self):
        return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops


# finetune model
@MODELS.register_module()
class PointTransformerCLIP_GAPromptPlus(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.level2_group_divider = Group(num_group=self.num_group//2, group_size=self.group_size//2)

        # self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.encoder = Point_PN(k_neighbors=64)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.resblocks = nn.ModuleList([
            Block(dim=self.trans_dim, num_heads=self.num_heads, mlp_ratio=4.,
                  qkv_bias=True, qk_scale=None,
                  drop=0., attn_drop=0.,
                  drop_path = dpr[i] if isinstance(dpr, list) else dpr,
                  num_tokens=sum(config.prompt_granularity),
                  idx=i, config=config)
                for i in range(self.depth)])

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 3, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        if config.point_prompt == True:
            # self.point_prompt = PointPrompt(point_number=config.point_number, init_type='uniform', scale=config.scale, factor=config.factor)
            self.instance_point_prompter = InstancePointPrompter(point_number=config.point_number, hidden_dimension=self.trans_dim, scale=config.scale, factor=config.factor)
        if config.shift_net == True:
            self.point_shift_prompter = ShiftNet(3, 3, hidden_dimension=config.encoder_dims, perturbation=config.perturbation, num_group=config.num_group, group_size=config.group_size, prompt_granularity=config.prompt_granularity)

        self.build_loss_func()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, finetuned=False):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, weights_only=False)
            if hasattr(ckpt, "state_dict"):
                ckpt = ckpt.state_dict()

            for key in ckpt.keys():
                if key in ['model', 'net', 'network', 'state_dict', 'base_model']:
                    ckpt = ckpt[key]
            ckpt_state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}

            base_ckpt = ckpt_state_dict if finetuned else {}
            #clip
            for k, v in ckpt_state_dict.items():
                if not finetuned and 'visual' not in k:
                    if 'positional_embedding' in k:
                        base_ckpt[k] = v
                    elif 'transformer' in k:
                        k = k.replace("transformer.", "")
                        base_ckpt[k] = v
                    elif 'ln_final' in k:
                        k = k.replace("ln_final", "norm")
                        base_ckpt[k] = v

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, **kwargs):
        x = pts.clone().transpose(1, 2).contiguous()
        center, group_input_tokens = self.encoder(x, pts)  # B G N
        group_input_tokens = group_input_tokens.transpose(1, 2)

        if self.config.shift_net:
            pts, shape_feature, multi_scale_prompt = self.point_shift_prompter(pts, require_global_feature=True)

        if self.config.point_prompt:
            pts = self.instance_point_prompter(pts, shape_feature)

        neighborhood, level1_center = self.group_divider(pts)
        level2_neighborhood, level2_center, level1_idx, level2_idx = self.level2_group_divider(center, require_index=True)

        # neighborhood, center = self.group_divider(pts)
        # group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        # x = x + pos
        for idx, block in enumerate(self.resblocks):
            if idx in self.config.prompt_layers:
                x = block(x + pos, global_feature=shape_feature, multi_scale_prompt=multi_scale_prompt, layer_id=idx, level1_center=level1_center, level1_index=level1_idx, level2_center=level2_center, level2_index=level2_idx)
            else:
                x = block(x + pos, layer_id=idx, level1_center=level1_center, level1_index=level1_idx, level2_center=level2_center, level2_index=level2_idx)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0], x[:, 1:].mean(1)], dim=-1)
        ret = self.cls_head_finetune(concat_f)

        return ret
