"""
@author: Zixiang Ai
@email: zxAi25@stu.pku.edu.cn
@file: GSModelDataset.py
@time: 2025/11/11 19:16
"""

import zlib
import torch
import torch.utils.data as data
import numpy as np
import json
from pathlib import Path

from .io import IO
from .build import DATASETS
from utils.logger import *
import math

def read_gaussian_attribute(vertex, attribute=['xyz']):
    assert "xyz" in attribute, "At least need xyz attribute"
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data = np.stack((x, y, z), axis=-1)  # [n, 3]

    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        data = np.concatenate((data, opacity), axis=-1)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        scales = np.exp(scales)  # scale normalization
        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data = np.concatenate((data, scales, rots), axis=-1)

    if "sh" in attribute:
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)

    return data


@DATASETS.register_module()
class GSModel60Dataset(data.Dataset):
    def __init__(self, config):
        self.data_root = Path(config.DATA_PATH).expanduser().resolve()
        configured_gs_path = Path(getattr(config, "GS_PATH", self.data_root)).expanduser()
        self.gs_path = configured_gs_path.resolve()
        self.num_category = config.NUM_CATEGORY
        self.attribute = config.ATTRIBUTE
        self.subset = config.subset
        self.norm_attribute = config.norm_attribute
        self.deterministic_eval = getattr(config, "DETERMINISTIC_EVAL", False)
        self.eval_seed = int(getattr(config, "EVAL_SEED", 0))
        print_log(f"config: {config}", logger="Gaussian")

        if self.subset not in {"train", "test"}:
            raise ValueError(f"Unsupported GSModel60 split: {self.subset}")

        self.catfile = self.data_root / "category.txt"
        if self.subset == "train":
            self.data_list_file = self.data_root / "train.json"
        else:
            self.data_list_file = self.data_root / "test.json"

        for required in (self.catfile, self.data_list_file):
            if not required.is_file():
                raise FileNotFoundError(f"Missing GSModel60 metadata file: {required}")

        self.cat = [line.rstrip().split(',', 1)[1] for line in self.catfile.open()]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.sample_points_num = config.N_POINTS

        print_log(f"[DATASET] Using Guassian Attribute {self.attribute}",logger="Gaussian")
        print_log(f"[DATASET] Sample out {self.sample_points_num} points",logger="Gaussian")
        print_log(f"[DATASET] Open file {self.data_list_file}", logger="Gaussian")
        with self.data_list_file.open("r") as f:
            lines = json.load(f)

        self.file_list = []
        for line in lines:
            line = line.strip()
            relative_path = Path(line)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Unsafe relative path in {self.data_list_file}: {line}"
                )
            if len(relative_path.parts) < 2:
                raise ValueError(
                    f"Malformed GSModel60 record in {self.data_list_file}: {line}"
                )
            cls = int(relative_path.parts[0])
            taxonomy_id = relative_path.stem
            model_id = relative_path.with_suffix("").as_posix()
            file_path = self.gs_path / relative_path
            self.file_list.append(
                {
                    "taxonomy_id": taxonomy_id,
                    "model_id": model_id,
                    "file_path": str(file_path),
                    "cls": cls,
                }
            )
        print_log(f"[DATASET] {len(self.file_list)} instances were loaded", logger="Gaussian")


    def pc_norm_gs(self, pc, attribute=["xyz"]):
        """pc: NxC, return NxC"""
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
        pc_xyz = pc_xyz / m
        pc[..., :3] = pc_xyz
        pc[..., 4:7] = pc[..., 4:7] / m

        if "opacity" in attribute:
            min_opacity = 0
            max_opacity = 1
            pc[..., 3] = (pc[..., 3] - min_opacity) / (max_opacity - min_opacity) * 2 - 1
        if "scale" in attribute:
            s_center = np.mean(pc[..., 4:7], axis=0)
            pc[..., 4:7] = pc[..., 4:7] - s_center
            s_m = np.max(np.sqrt(np.sum(pc[..., 4:7] ** 2, axis=1)))
            pc[..., 4:7] = pc[..., 4:7] / s_m
        else:
            s_center = np.zeros(3)
            s_m = 1
        if "sh" in attribute:
            sh = pc[..., 11:14]
            sh = sh * 0.28209479177387814
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)
            pc[..., 11:14] = sh

        return pc, s_center, s_m

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        label = sample["cls"]
        try:
            gs = IO.get(sample["file_path"])
        except Exception as error:
            raise RuntimeError(
                f"Failed to load GSModel60 sample: {sample['file_path']}"
            ) from error

        vertex = gs["vertex"]
        data = read_gaussian_attribute(vertex, self.attribute)
        data, _, _ = self.pc_norm_gs(data, self.norm_attribute)
        if self.deterministic_eval and self.subset != "train":
            sample_seed = self.eval_seed + zlib.crc32(sample["model_id"].encode("utf-8"))
            rng = np.random.default_rng(sample_seed)
            choice_gs = rng.choice(len(data), self.sample_points_num, replace=True)
        else:
            choice_gs = np.random.choice(len(data), self.sample_points_num, replace=True)
        data = data[choice_gs, :]

        data = torch.from_numpy(data).float()
        return "GSModel", sample["file_path"], (data, label)

    def __len__(self):
        return len(self.file_list)
