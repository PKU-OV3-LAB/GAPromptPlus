"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)

@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        # limited reconstruction (semi-sup)
        lr_file=None,
        # limited annotation (sparse labels)
        la_file=None,
        la_sample="full",
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        self.la_sample = la_sample
        assert la_sample in ["random", "full"], f"not support limited-anno with sample={la_sample}"
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            # ...[train,val,test]/scene_name
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data_aligned(self, idx):
        data = self.get_data(idx)

        data_path = self.data_list[idx % len(self.data_list)]
        info_file = os.path.join(data_path, 'meta.txt')
        # Rotating the mesh to axis aligned
        info_dict = {}
        with open(info_file) as f:
            for line in f:
                (key, val) = line.split(" = ")
                info_dict[key] = np.fromstring(val, sep=' ')

        if 'axisAlignment' not in info_dict:
            rot_matrix = np.identity(4)
        else:
            rot_matrix = info_dict['axisAlignment'].reshape(4, 4)
        r_coords = data["coord"].transpose()
        r_coords = np.append(r_coords, np.ones((1, r_coords.shape[1])), axis=0)
        r_coords = np.dot(rot_matrix, r_coords)
        data["coord"] = r_coords.transpose()[:, :3]
        return data

    def get_data(self, idx):
        # called in `__getitem__`, followed by `transform` - for both prepare_train/test_data
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        for k in list(data_dict.keys()):
            if k.startswith("eigval"):  # map eigval / eigval_k* / eigvalraw[_k*] -> eigval
                data_dict["eigval"] = data_dict.pop(k).astype(np.float32)

        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            if self.la_sample == "full":
                data_dict["sampled_index"] = sampled_index
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)

    def get_cats_mask(self):
        masks = dict()
        labals_arr = np.array(CLASS_LABELS_200)
        for name, cats in {
            "head": HEAD_CATS_SCANNET_200,
            "common": COMMON_CATS_SCANNET_200,
            "tail": TAIL_CATS_SCANNET_200,
        }.items():
            cats_arr = np.array(cats)
            cats_mask = np.isin(labals_arr, cats_arr)  # [200] per-elem isin [#CATS]
            masks[name] = cats_mask
        return masks
