import numpy as np
import torch
import random
import collections
from scipy.linalg import expm, norm
from pointnet2_ops import pointnet2_utils
try:
    from collections.abc import Iterable  # above Python 3.10
except ImportError:
    from collections import Iterable      # below Python 3.9

class PointcloudRotate(object):
    def __init__(self, angle=[0.0, 1.0, 0.0]):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()

        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()

        return pc

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

class RandomHorizontalFlip(object):
  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords

class SinPoint:
    def __init__(self, rand_center_num=0, w=2.5, A=0.5, sample='RPS'):
        self.rand_center_num = rand_center_num
        self.w = w
        self.A = A
        self.sample = sample

    def Local(self, data):
        """
        Args:
            data (B,N,3)
        """
        device = data.device
        B, N, C = data.shape
        if self.sample == "RPS":
            idxs = self.generate_random_permutations_batch(B,N,self.rand_center_num)
        if self.sample == "FPS":
            idxs = self.farthest_point_sample(data, self.rand_center_num)
        dist = torch.zeros_like(data).to(device)
        for i in range(self.rand_center_num):
            center = self.index_points(data, idxs[:,i]).unsqueeze(1)
            dist = dist + data - center
        dist = dist / self.rand_center_num
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * dist)
        newdata = data + move
        return newdata

    def Global(self, data):
        """
        Args:
            data (B,N,3)
        """
        device = data.device
        B, N, C = data.shape
        newdata = torch.zeros_like(data)
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * data)
        newdata = data + move
        return newdata

    def __call__(self, pc):
        """
        Args:
            data (B,N,3)
            label (B)
        """
        B = pc.shape[0]
        newdata, shift, scale = self.normalize_point_clouds(pc)
        if self.rand_center_num == 0:
            newdata = self.Global(newdata)
        else:
            newdata = self.Local(newdata)
        newdata = newdata * scale + shift
        return newdata

    def index_points(self, points, idx):
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

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            fps_idx: sampled pointcloud index, [B, npoint]
        """
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
        return fps_idx

    def generate_random_permutations_batch(self, B, N, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        all_permutations = torch.stack([torch.randperm(N) for _ in range(B)])
        centroids = all_permutations[:, :npoint]
        return centroids

    def square_distance(self,src, dst):
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

    def normalize_point_clouds(self, pcs):
        B, N, C = pcs.shape
        shift = torch.mean(pcs, dim=1).unsqueeze(1)
        scale = torch.std(pcs.view(B, N * C), dim=1).unsqueeze(1).unsqueeze(1)
        newpcs = (pcs - shift) / scale
        return newpcs, shift, scale
