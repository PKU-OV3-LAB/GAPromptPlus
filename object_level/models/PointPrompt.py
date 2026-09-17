import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from utils import misc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def pooling(knn_x_w, transform=None):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(dim=2)[0]
        if transform is not None:
            lc_x = transform(lc_x.permute(0, 2, 1)).permute(0,2,1)
        return lc_x

def propagate(xyz1, xyz2, points1, points2, de_neighbors=6, remove_self_loop=False):
    """
    Input:
        xyz1: input points position data, [B, N, 3]
        xyz2: sampled input points position data, [B, S, 3]
        points1: input points data, [B, N, D']
        points2: input points data, [B, S, D'']
    Return:
        new_points: upsampled points data, [B, N, D''']
    """

    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape

    dists = square_distance(xyz1, xyz2)
    dists, idx = dists.sort(dim=-1)
    if remove_self_loop:
        dists, idx = dists[:, :, 1:de_neighbors+1], idx[:, :, 1:de_neighbors+1]  # [B, N, S]
    else:
        dists, idx = dists[:, :, :de_neighbors], idx[:, :, :de_neighbors]  # [B, N, S]
    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    weight = weight.view(B, N, -1, 1)
    interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)#B, N, 6, C->B,N,C

    new_points = points1+0.3*interpolated_points # B,N,C

    return new_points


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, require_index=False):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center,center_idx = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        center_idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * num_points
        center_idx = center_idx + center_idx_base
        center_idx = center_idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        if require_index:
            return neighborhood, center, idx, center_idx
        else:
            return neighborhood, center


class PositionalEmbedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)#[2^0,2^1,...,2^(n-1)]
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)

class PointNet(nn.Module):   ## Embedding module
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
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_group, group_size, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.group_divider = Group(num_group, group_size)
        self.num_group = num_group
        self.group_size = group_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        neighborhood, center, idx, center_idx = self.group_divider(xyz.float(), require_index=True) #[B, G, n, 3]
        new_xyz = center.reshape((B, self.num_group, -1))
        new_points = points.reshape((B*N, -1))[idx].reshape((B, self.num_group, self.group_size, -1))

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, group_size, npoint/num_group]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0,2,1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, interpolate_neighbors=32):
        super(PointNetFeaturePropagation, self).__init__()
        self.interpolate_neighbors = interpolate_neighbors
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C] 64
            xyz2: sampled input points position data, [B, S, C] 1024
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.interpolate_neighbors], idx[:, :, :self.interpolate_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-4)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.interpolate_neighbors, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0,2,1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0,2,1)
        return new_points

class PatchPrompter(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dimension=384):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dimension = hidden_dimension
        self.propagation = PointNetFeaturePropagation(in_channel=in_channels, mlp=[64,self.out_channels], interpolate_neighbors=6)

    def forward(self, center, x):
        x = self.propagation(center, center, None, x)
        return x


class ShiftNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dimension=384, perturbation=0.1, num_group = 128, group_size = 32, prompt_granularity = [16, 4, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dimension = hidden_dimension
        self.num_group = num_group
        self.group_size = group_size
        self.group_divider = Group(num_group, group_size)

        self.abstraction_level1 = PointNetSetAbstraction(self.num_group, self.group_size, in_channels, mlp=[64, 64])
        self.abstraction_level2 = PointNetSetAbstraction(self.num_group//2, self.group_size//2, 64, mlp=[128])
        # multi-scale prompt
        self.abstraction_level3 = PointNetSetAbstraction(prompt_granularity[0], self.group_size//4, 128, mlp=[128])
        self.abstraction_level4 = PointNetSetAbstraction(prompt_granularity[1], self.group_size//8, 128, mlp=[128])


        self.propagation1 = PointNetFeaturePropagation(in_channel=32+in_channels, mlp=[32, 32])
        self.propagation2 = PointNetFeaturePropagation(in_channel=128+64, mlp=[64,32])
        # multi-scale prompt

        self.shape_feature_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dimension)
        )
        for layer in self.shape_feature_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)

        self.mlp_position = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.out_channels),
        )

        self.perturbation = perturbation
        for layer in self.mlp_position:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)
                nn.init.constant_(layer.bias, val=0.0)

    def forward(self, points, require_global_feature=False):
        B, N, _ = points.shape  #[32, 2048, 3]
        feature = points
        center1, center1_feature = self.abstraction_level1(points, feature)
        center2, center2_feature = self.abstraction_level2(center1, center1_feature)
        # multi-scale prompt
        center3, center3_feature = self.abstraction_level3(center2, center2_feature)
        center3_feature = propagate(center3,center2,center3_feature,center2_feature)
        center4, center4_feature = self.abstraction_level4(center3, center3_feature)
        center4_feature = propagate(center4,center3,center4_feature,center3_feature)
        center3_feature = self.shape_feature_mlp(center3_feature)
        center4_feature = self.shape_feature_mlp(center4_feature)
        multi_scale_prompt = torch.concat([center4_feature,center3_feature],dim=1)

        center1_feature = self.propagation2(center1, center2, center1_feature, center2_feature)
        feature = self.propagation1(points, center1, feature, center1_feature)
        global_feature = torch.mean(center4_feature, dim=1).unsqueeze(1)  # B 1 hidden_dimension
        y = self.mlp_position(feature) * self.perturbation * points
        y = y + points
        if require_global_feature:
            return y, global_feature, multi_scale_prompt
        else:
            return y

class PointPrompt(nn.Module):
    def __init__(self, point_number=20, init_type='uniform', scale=0.01, factor=5):   # peak learning rate 0.0005
        super().__init__()
        self.points = nn.Parameter(torch.zeros([1, point_number, 3], dtype=torch.float32), requires_grad=True)
        self.factor = factor
        if init_type == 'uniform':
            nn.init.uniform_(self.points, -scale, scale)
        elif init_type == 'cluster':
            means = [
                [-scale, 0, 0],
                [scale, 0, 0],
                [0, -scale, 0],
                [0, scale, 0]
            ]
            cov = np.eye(3) * 0.04
            num_points_per_cluster = point_number//len(means)
            points = []
            for mean in means:
                cluster_points = np.random.multivariate_normal(mean, cov, num_points_per_cluster)
                points.append(cluster_points)
            with torch.no_grad():
                self.points.copy_(torch.tensor(np.vstack(points).reshape([1, point_number, 3])))

    def forward(self, x):
        learnable_points = self.points.expand(x.shape[0], 1, 1) * self.factor
        x = torch.cat([x, learnable_points], dim=1)
        return x

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_topk(logits, k, tau=1.0, dim=1):
    gumbels = sample_gumbel(logits.shape).to(logits.device)
    y = (logits + gumbels) / tau
    # softmax + topk approximation
    topk = torch.topk(y, k, dim=dim).indices
    return topk

class InstancePointPrompter(nn.Module):
    def __init__(self, point_number=20, hidden_dimension=384, scale=0.01, factor=5):
        super().__init__()
        self.scale = scale
        self.factor = factor
        self.point_number = point_number
        # self.points = nn.Parameter(torch.zeros([1, point_number, 3], dtype=torch.float32), requires_grad=True)
        # nn.init.uniform_(self.points, -self.scale, self.scale)
        # self.prompter = nn.Sequential(nn.Linear(hidden_dimension, 3*point_number), nn.Tanh())
        # self.context_prompter = Block(dim=hidden_dimension, num_heads=6, mlp_ratio=0.5, qkv_bias=None, qk_scale=None, drop=0.1, attn_drop=0, drop_path = 0.1, num_tokens=self.point_number)
        self.prompter = nn.Sequential(nn.Linear(3+hidden_dimension, 4))
        nn.init.kaiming_normal_(self.prompter[0].weight)  # Initialize
        nn.init.zeros_(self.prompter[0].bias)  # Initialize bias to zero

    def forward(self, x, global_shape_feature):
        B = x.shape[0]
        prompt_points, prompt_points_idx = misc.fps(x, self.point_number*4)
        prompts_scores = self.prompter(torch.concat([prompt_points, global_shape_feature.expand(-1,prompt_points.shape[1],-1)], dim=-1))
        prompts = prompts_scores[..., :3]
        scores = prompts_scores[..., 3]
        _, idx = torch.topk(scores, self.point_number, dim=1)
        idx = idx.unsqueeze(-1).expand(-1, -1, 3)
        prompt_points = torch.gather(prompt_points, 1, idx)+self.scale*torch.gather(prompts, 1, idx)
        x = torch.cat([x, prompt_points], dim=1)
        return x

    # def forward(self, x, global_shape_feature):
    #     B = x.shape[0]
    #     prompt_points, prompt_points_idx = misc.fps(x, self.point_number*4)
    #     prompts_scores = self.prompter(torch.concat([prompt_points, global_shape_feature.expand(1,prompt_points.shape[1],1)], dim=-1))
    #     prompts = prompts_scores[..., :3]
    #     scores = prompts_scores[..., 3]
    #     # _, idx = torch.topk(scores, self.point_number, dim=1)
    #     idx = gumbel_topk(scores, self.point_number, tau=1.0, dim=1)
    #     idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    #     prompt_points = torch.gather(prompt_points, 1, idx)+self.scale*torch.gather(prompts, 1, idx)

    #     x = torch.cat([x, prompt_points], dim=1)
    #     return x

# Fast Fourier Transform in hidden and sequence dimension
class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x
