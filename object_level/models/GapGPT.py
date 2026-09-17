import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .GAPrompt import propagate, pooling
from timm.models.layers import DropPath

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

    def calc_sampled_param_num(self):

        return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops



class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, idx=None, config=None):
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
        self.adapter = Adapter(embed_dims=embed_dim, reduction_dims=16, drop_rate_adapter=0.1)
        self.prompts = None
        if idx is not None and idx < self.config.prompt_depth:
            self.prompts = nn.Parameter(torch.zeros(getattr(self.config, "legacy_prompt_tokens", 21), embed_dim))
            nn.init.xavier_uniform_(self.prompts)
        self.bnorm = nn.BatchNorm1d(embed_dim)

    def forward(self, x, attn_mask, idx=None, propagation_dict=None):
        prompt_tokens = None
        if idx is not None and idx < self.config.prompt_depth and self.prompts is not None:
            prompt_tokens = self.prompts.repeat(x.shape[0], 1, 1)
        if prompt_tokens is not None:
            # x = torch.cat((x[:,0:1], prompt_tokens, x[:,1:]), 1)
            # x = torch.cat((x[:,0:2], prompt_tokens, x[:,2:]), 1)
            x = torch.cat((x, prompt_tokens), 1)
        else:
            if attn_mask is not None:
                attn_mask = attn_mask[:x.shape[1], :x.shape[1]]

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
            prompt_tokens = x[:,:-propagate_range]
            x = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=x_centers, de_neighbors=8, dist_e=1e-3)
            x = torch.concat((sos_cls_x, prompt_tokens, x), dim=1)
        if prompt_tokens is not None:
            # x = torch.cat((x[:,0:2], x[:,10+2:]), 1)
            x = x[:, :-prompt_tokens.shape[1]]

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
            self.layers.append(Block(embed_dim, num_heads, idx=i, config=config))

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

    def forward(self, h, pos, attn_mask, classify=False, propagation_dict=None, shape_feature=None):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        # h = h.transpose(0, 1)
        # pos = pos.transpose(0, 1)

        # # prepend sos token
        sos = torch.ones(batch, 1, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:, :-1, :]], axis=1)
        else:
            h = torch.cat([sos, h], axis=1)

        # h = h.transpose(0, 1)
        # pos = pos.transpose(0, 1)

        # transformer
        for idx, layer in enumerate(self.layers):
            h = layer(h + pos, attn_mask, idx=idx, propagation_dict=propagation_dict)

        h = self.ln_f(h)

        # encoded_points = h.transpose(0, 1)
        # if not classify:
        #     return encoded_points

        # h = h.transpose(0, 1)
        h = self.cls_norm(h)
        if self.config.head_dim == 3:
            concat_f = torch.cat([h[:, 0], h[:, 1:].max(1)[0], shape_feature[:,0]], dim=-1)
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
