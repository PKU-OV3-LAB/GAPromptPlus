from .io import IO
from .build import DATASETS
from utils.logger import *
import torch.utils.data as data
import json
import torch
import numpy as np
import zlib
from pathlib import Path


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
class uCO3D(data.Dataset):
    def __init__(self, config, prepare=False):
        self.data_root = Path(config.DATA_PATH).expanduser().resolve()
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.deterministic_eval = getattr(config, "DETERMINISTIC_EVAL", False)
        self.eval_seed = int(getattr(config, "EVAL_SEED", 0))
        if prepare:
            self.prepare_data()
        if self.subset not in {"train", "test"}:
            raise ValueError(f"Unsupported uCO3D80 split: {self.subset}")

        split_path = self.data_root / f"{self.subset}_split.json"
        categories_path = self.data_root / "uCO3D80_categories.json"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing fixed split file: {split_path}. "
                "Download the published uCO3D80 dataset directory."
            )
        if not categories_path.is_file():
            raise FileNotFoundError(
                f"Missing category metadata: {categories_path}"
            )

        with split_path.open("r") as f:
            self.file_list = json.load(f)
        for index, sample in enumerate(self.file_list):
            missing = {"taxonomy_id", "model_id", "file_path"} - set(sample)
            if missing:
                raise ValueError(
                    f"Malformed record {index} in {split_path}: missing {sorted(missing)}"
                )
            relative_path = Path(sample["file_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Unsafe relative path in {split_path}: {sample['file_path']}"
                )
        print(f'[DATASET] {len(self.file_list)} instances were loaded.')
        self.permutation = np.arange(self.npoints)

    def prepare_data(self):
        raise RuntimeError(
            "Dynamic uCO3D80 split generation is disabled in the public release. "
            "Use the published train_split.json and test_split.json files unchanged."
        )

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num, sample_key):
        if self.deterministic_eval and self.subset != "train":
            sample_seed = self.eval_seed + zlib.crc32(sample_key.encode("utf-8"))
            permutation = np.random.default_rng(sample_seed).permutation(self.permutation)
        else:
            np.random.shuffle(self.permutation)
            permutation = self.permutation
        return pc[permutation[:num]]

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        ply = IO.get(str(self.data_root / sample['file_path']))
        data = read_gaussian_attribute(ply['vertex'], attribute=['xyz'])
        label = sample['taxonomy_id']
        data = self.random_sample(data, self.npoints, sample["model_id"])
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], (data, label)

    def __len__(self):
        return len(self.file_list)
