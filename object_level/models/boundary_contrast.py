import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# import pointops
# from utils import misc
from knn_cuda import KNN

_inf = 1e9
_eps = 1e-12

def get_subscene_label(stage_i, points_list, target, nstride, num_classes, **kwargs):
    # o - num of points in each batch - b batch together - using BxN, used for finding the points in each example
    # calc the reduced size of points for the batch - n = [BxN], with each b clouds, i-th cloud containing N = b[i] points
    x = F.one_hot(target, num_classes)
    x = get_subscene_features(stage_i, points_list, x, nstride, **kwargs)
    return x

def get_subscene_features(stage_i, points_list, x, nstride, kr=None, return_neighbor=False):
    if stage_i == 3:
        return x.float()

    if kr is None:  # infer from sub-sampling (nstride) as default
        kr = nstride
    p_from = points_list[3]
    p_to = points_list[stage_i]
    knn_function = KNN(k=kr, transpose_mode=True)
    _, neighbor_idx = knn_function(p_from, p_to)  # (m, kr) - may have invalid neighbor
    x = torch.gather(x, 1, neighbor_idx.reshape(x.shape[0], -1, 1).expand(-1,-1,x.shape[-1]))
    x = x.reshape(x.shape[0], -1, kr, x.shape[-1])
    x = x.float().mean(-2)
    if return_neighbor:
        return x, neighbor_idx, kr
    return x

def get_boundary_mask(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, get_plain=False, get_cnt=False):
    """ assume all label valid indicated by valid_mask """
    labels_shape = labels.shape
    if neighbor_label is None:
        shape = [*neighbor_idx.shape, *labels.shape[1:]]  # [BxN, kr, ncls]
        neighbor_label = labels[neighbor_idx.view(-1).long(), ...].view(shape)
        print(shape, neighbor_idx.shape, labels.shape, neighbor_label.shape, flush=True)

    valid_neighbor = neighbor_label >= 0
    labels = labels.unsqueeze(-1)

    neq = labels != neighbor_label
    neq = torch.logical_and(neq, valid_neighbor)
    if get_cnt:
        bound = torch.sum(neq, dim=-1)
        bound = bound * valid_mask if valid_mask is not None else bound
    else:
        bound = torch.any(neq, dim=-1)
        bound = torch.logical_and(bound, valid_mask) if valid_mask is not None else bound  # mask out row of invalid center
    # assert len(bound.shape) == len(labels_shape), f'invalid shape - bound {bound.shape}, label {labels_shape}, neighbor label {neighbor_label.shape}, with valid_mask = {valid_mask}'

    if get_plain:
        # assert not get_cnt, 'no need to get plain if having cnt of boundary (together with valid neighbor)'
        eq = labels == neighbor_label  # same - T, diff - F, invalid center - F, invalid neighbor - F
        eq = torch.logical_or(eq, torch.logical_not(valid_neighbor))  # valid -> all eq => plain if neighbor all invalid
        plain = torch.all(eq, dim=-1)
        plain = torch.logical_and(plain, valid_mask) if valid_mask is not None else plain
        return bound, plain
    return bound  # [BxN]

class ContrastHead(nn.Module):
    """ currently used as criterion - need to be wrapped with DataParallel if params used (eg. project)
    """
    def __init__(self, config, nsample=[8, 8, 8, 16], nstride=8):
        super().__init__()
        self.nsample = torch.tensor(nsample) # [24, 24, 24, 36] [12, 12, 12, 18]
        self.nstride = torch.tensor(nstride) # 4 8
        self.num_classes = torch.tensor(config.cls_dim)

        self.dist_func = self.dist_l2
        self.posmask_func = self.posmask_cnt
        self.contrast_func = self.contrast_softnn
        self.temperature = 1.0

    def forward(self, output, target, points_list, features_list):
        loss_list = []
        for i in range(len(points_list)):
            loss = self.point_contrast(i, points_list, features_list, target)
            loss_list += [loss]
        return sum(loss_list)

    def sample_label(self, i, points_list, features_list, target):
        points = points_list[i]
        features = features_list[i]
        nsample = self.nsample[i]
        labels = get_subscene_label(i, points_list, target, self.nstride, self.num_classes)  # (m, ncls) - distribution / onehot
        knn_function = KNN(k=nsample, transpose_mode=True)
        _, neighbor_idx = knn_function(points, points)
        # exclude self-loop
        nsample -= 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()

        neighbor_label = torch.gather(labels, 1, neighbor_idx.reshape(labels.shape[0],-1,1).expand(-1,-1,labels.shape[2])).reshape(-1, nsample, labels.shape[-1])
        neighbor_feature = torch.gather(features, 1, neighbor_idx.reshape(features.shape[0],-1,1).expand(-1,-1,features.shape[2])).reshape(-1, nsample, features.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
        features = features.reshape(-1, features.shape[-1])

        return nsample, labels, neighbor_label, features, neighbor_feature


    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps) # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        # kl dist from featuers (gt) to neighbors (pred)
        if normalized in [False, 'softmax']:  # if still not a prob distribution - prefered
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')

        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        dist = dist.sum(-1)  # [m, nsample]
        return dist


    def posmask_cnt(self, labels, neighbor_label):
        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
        neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
        mask = (labels == neighbor_label)  # [m, nsample]
        return mask

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        # each Log term an example; per-pos vs. all negs
        neg = torch.sum(exp * (1 - posmask), axis=-1)  # (m)
        under = exp + neg
        loss = (exp / (exp + neg))[posmask]  # each Log term an example
        loss = -torch.log(loss)
        return loss

    def point_contrast(self, i, points_list, features_list, target):
        points = points_list[i]
        features = features_list[i]
        nsample = self.nsample[i]
        labels = get_subscene_label(i, points_list, target, self.nstride, self.num_classes)  # (m, ncls) - distribution / onehot
        knn_function = KNN(k=nsample, transpose_mode=True)
        _, neighbor_idx = knn_function(points, points)

        # exclude self-loop
        nsample = self.nsample[i] - 1  # nsample -= 1 can only be used if nsample is py-number - results in decreasing number if is tensor, e.g. 4,3,2,1,...
        neighbor_idx = neighbor_idx[..., 1:].contiguous()

        neighbor_label = torch.gather(labels, 1, neighbor_idx.reshape(labels.shape[0],-1,1).expand(-1,-1,labels.shape[2])).reshape(-1, nsample, labels.shape[-1])
        neighbor_feature = torch.gather(features, 1, neighbor_idx.reshape(features.shape[0],-1,1).expand(-1,-1,features.shape[2])).reshape(-1, nsample, features.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
        features = features.reshape(-1, features.shape[-1])
        posmask = self.posmask_cnt(labels, neighbor_label)  # (m, nsample) - bool
        # select only pos-neg co-exists
        point_mask = torch.sum(posmask.int(), -1)  # (m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)

        if not torch.any(point_mask):
            if i == 0:
                o = torch.cat([torch.tensor([0]).to(o.device), o])
                for bi, (start, end) in enumerate(zip(o[:-1], o[1:])):
                    print('bi / labelcnt - ', bi , ' / ', torch.unique(labels[start:end].argmax(dim=1)))
                print(point_mask.sum(0), len(point_mask))
                print(labels[0], neighbor_label[0])
                print(labels[100], neighbor_label[100])
                print(labels[900], neighbor_label[900])
                print(flush=True)
                raise
            return torch.tensor(.0)

        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]

        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)  # (m)

        loss = torch.mean(loss)
        return loss
