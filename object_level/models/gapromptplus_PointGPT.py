import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from models.PointGPT import get_z_values
from models.gapromptplus_MAE import Adapter
from .PointPrompt import ShiftNet, PointPrompt, InstancePointPrompter, PatchPrompter, Group, propagate, pooling, FNetBlock
# from .GAPrompt import ShiftNet, PointPrompt, Group2, propagate, pooling

class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder_small(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group_GPT(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.knn_2 = KNN(k=1, transpose_mode=True)

    def simplied_morton_sorting(self, xyz, center):
        '''
        Simplifying the Morton code sorting to iterate and set the nearest patch to the last patch as the next patch, we found this to be more efficient.
        '''
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        all_indices = []
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            idxs = np.zeros((self.num_group), dtype=np.int32)
            temp = np.arange(self.num_group)
            z_ind = np.argsort(z[temp])
            idxs = temp[z_ind]
            all_indices.append(idxs)
        all_indices = torch.tensor(all_indices, device=xyz.device)

        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        sorted_indices = all_indices + idx_base
        sorted_indices = sorted_indices.view(-1)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)[0]  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        # can utilize morton_sorting by choosing morton_sorting function
        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.view(
            batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, 3).contiguous()

        return neighborhood, center


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_tokens=10, idx=0, config=None):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.config = config
        self.drop_path = DropPath(0.1)
        self.adapter = Adapter(embed_dims=embed_dim, reduction_dims=config.adapter_config.adapter_dim, drop_rate_adapter=config.adapter_config.adapter_drop_path_rate)
        self.prompts = None
        self.num_tokens = num_tokens
        if idx in self.config.prompt_layers:
            self.prompts = nn.Parameter(torch.zeros(num_tokens, embed_dim))
            nn.init.xavier_uniform_(self.prompts)
        self.bnorm = nn.BatchNorm1d(embed_dim)

    def forward(self, x, attn_mask, idx=None, propagation_dict=None):
        prompt_tokens = None
        global_shape_prompt = propagation_dict.get('global_shape_prompt')
        multi_scale_prompt = propagation_dict.get('multi_scale_prompt')
        if global_shape_prompt is not None and multi_scale_prompt is not None and idx in self.config.prompt_layers:
            prompt_tokens = self.prompts.expand(x.shape[0], -1, -1)
            prompt_tokens = prompt_tokens+0.5*torch.concat([multi_scale_prompt, global_shape_prompt], dim=1)
            x = torch.cat((x[:,0:2], prompt_tokens, x[:,2:]), 1) # before position
            # x = torch.cat((x, prompt_tokens), 1) # after position

        elif idx in self.config.prompt_layers:
            prompt_tokens = self.prompts.expand(x.shape[0], -1, -1)
            # x = torch.cat((x[:,0:1], prompt_tokens, x[:,1:]), 1)
            x = torch.cat((x[:,0:2], prompt_tokens, x[:,2:]), 1) # before position
            # x = torch.cat((x, prompt_tokens), 1) # after position

        # if attn_mask is not None:
        #     attn_mask = attn_mask[:x.shape[1], :x.shape[1]]


        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        if prompt_tokens is not None and propagation_dict.get('prompt_propagation_after'):
            B,G,_ = x.shape
            sos_cls_x = x[:,0:2]
            x = x[:,2:]
            G = G-2

            level1_center = propagation_dict['center1']
            level1_index = propagation_dict['center1_idx']
            level2_center = propagation_dict['center2']
            level2_index = propagation_dict['center2_idx']
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.embed_dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.embed_dim)

            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            x_centers = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.embed_dim), transform=self.bnorm)+0.3*x_centers
            prompt_tokens = x[:, :-propagate_range] # before position
            x = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=x_centers)
            x = torch.concat((sos_cls_x, prompt_tokens, x), dim=1)# before position
        else:
            level1_center = propagation_dict['center1']
            propagate_range = level1_center.shape[1]

        if prompt_tokens is not None:
            x = torch.concat((x[:, 0:2], x[:, -propagate_range:]), 1)  # before position
            # x = x[:, :2+propagate_range] # after position

        if self.adapter is not None:
            x = x + self.adapter(x)

        return x

class GPT_extractor(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size, pretrained=False, config=None
    ):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.config = config
        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads, num_tokens=sum(config.prompt_granularity), idx=i, config=config))

        self.ln_f = nn.LayerNorm(embed_dim)
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

        if pretrained == False:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * self.config.head_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

            self.cls_norm = nn.LayerNorm(self.trans_dim)

    def forward(self, h, pos, attn_mask, classify=False, propagation_dict=None):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        # prepend sos token
        sos = torch.ones(batch, 1, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:, :-1, :]], axis=1)
        else:
            h = torch.cat([sos, h], axis=1)

        # transformer
        for idx, layer in enumerate(self.layers):
            h = layer(h + pos, attn_mask, idx=idx, propagation_dict=propagation_dict)

        h = self.ln_f(h)

        # encoded_points = h.transpose(0, 1)
        # if not classify:
        #     return encoded_points

        h = self.cls_norm(h)
        if self.config.head_dim == 3:
            concat_f = torch.cat([h[:, 0], h[:, 1:].max(1)[0], h[:, 2:].mean(1)], dim=-1)
        elif self.config.head_dim == 2:
            concat_f = torch.cat([h[:, 0], h[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret, 0


class GPT_generator(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, trans_dim, group_size
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads, idx=i))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

    def forward(self, h, pos, attn_mask):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape
        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)
        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask)
        h = self.ln_f(h)
        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(1, 2).transpose(0, 1).reshape(batch * length, -1, 3)
        return rebuild_points


@MODELS.register_module()
class GPTPointTransformer_GAPromptPlus(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group_GPT(num_group=self.num_group, group_size=self.group_size)
        self.level2_group_divider = Group(num_group=self.num_group//2, group_size=self.group_size//2)
        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            config=config
        )

        if config.point_prompt == True:
            # self.point_prompt = PointPrompt(point_number=config.point_number, init_type='uniform', scale=config.scale, factor=config.factor)
            self.instance_point_prompter = InstancePointPrompter(point_number=config.point_number, hidden_dimension=self.trans_dim, scale=config.scale, factor=config.factor)
        if config.shift_net == True:
            self.point_shift_prompter = ShiftNet(3, 3, hidden_dimension=config.encoder_dims, perturbation=config.perturbation, num_group=config.num_group, group_size=config.group_size, prompt_granularity=config.prompt_granularity)

        # self.generator_blocks = GPT_generator(
        #     embed_dim=self.encoder_dims,
        #     num_heads=self.num_heads,
        #     num_layers=self.decoder_depth,
        #     trans_dim=self.trans_dim,
        #     group_size=self.group_size
        # )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.norm = nn.LayerNorm(self.trans_dim)

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self, loss_type='cdl12'):
        self.loss_ce = nn.CrossEntropyLoss()
        if loss_type == "cdl1":
            self.loss_func_p = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, finetuned=False):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if 'cls_head_finetune' in k and not finetuned:
                    del base_ckpt[k]

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
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(
                f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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

    def forward(self, pts,  batch_idx=None):
        global_shape_prompt = None
        if self.config.shift_net:
            pts, global_shape_prompt, multi_scale_prompt = self.point_shift_prompter(pts, require_global_feature=True)
        if self.config.point_prompt:
            # pts = self.point_prompt(pts)
            pts = self.instance_point_prompter(pts, global_shape_prompt)

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        _, center2, center1_idx, center2_idx = self.level2_group_divider(center, require_index=True)
        propagation_dict = {
            'center1': center,
            'center1_idx': center1_idx,
            'center2': center2,
            'center2_idx': center2_idx,
            'gather_idx': False,
            'prompt_propagation_after': self.config.prompt_propagation_after,
            'global_shape_prompt': global_shape_prompt,
            'multi_scale_prompt': multi_scale_prompt,
        }
        B, L, _ = group_input_tokens.shape

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat([center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        device, dtype = x.device, x.dtype
        # attn_mask = torch.full((L+2+10, L+2+10), -float("Inf"), device=device, dtype=dtype).to(torch.bool)
        # attn_mask = torch.triu(attn_mask, diagonal=1)

        # transformer
        ret, encoded_features = self.blocks(x, pos, attn_mask=None, classify=True, propagation_dict=propagation_dict)

        # encoded_features = torch.cat(
        #     [encoded_features[:, 0, :].unsqueeze(1), encoded_features[:, 2:-1, :]], dim=1)

        # attn_mask = torch.full(
        #     (L, L), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        # ).to(torch.bool)

        # attn_mask = torch.triu(attn_mask, diagonal=1)

        # generated_points = self.generator_blocks(
        #     encoded_features, pos_relative, attn_mask)

        # neighborhood = neighborhood + center.unsqueeze(2)

        # gt_points = neighborhood.reshape(
        #     B*(self.num_group), self.group_size, 3)

        # loss1 = self.loss_func_p1(generated_points, gt_points)
        # loss2 = self.loss_func_p2(generated_points, gt_points)

        # return ret, loss1 + loss2
        return ret
