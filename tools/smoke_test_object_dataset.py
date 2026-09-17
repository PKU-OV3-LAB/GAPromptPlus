#!/usr/bin/env python3
"""Load real samples through the public GAPrompt++ dataset classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = ROOT / "classification"
sys.path.insert(0, str(CLASSIFICATION))


def test_uco(root: Path, split: str) -> None:
    from datasets.uCO3D80Dataset import uCO3D

    config = SimpleNamespace(
        DATA_PATH=str(root),
        subset=split,
        N_POINTS=8192,
        DETERMINISTIC_EVAL=True,
        EVAL_SEED=0,
    )
    dataset = uCO3D(config)
    expected = {"train": 13468, "test": 6996}[split]
    if len(dataset) != expected:
        raise ValueError(f"Expected {expected} samples, found {len(dataset)}")
    for index in {0, len(dataset) - 1}:
        taxonomy_id, model_id, (points, label) = dataset[index]
        if tuple(points.shape) != (8192, 3):
            raise ValueError(f"Unexpected tensor shape: {tuple(points.shape)}")
        if int(taxonomy_id) != int(label):
            raise ValueError(f"Label mismatch for {model_id}")
        print(split, index, model_id, tuple(points.shape), int(label))


def test_gs(root: Path, split: str) -> None:
    from datasets.GSModel60Dataset import GSModel60Dataset

    config = SimpleNamespace(
        DATA_PATH=str(root),
        GS_PATH=str(root),
        NUM_CATEGORY=60,
        ATTRIBUTE=["xyz"],
        subset=split,
        norm_attribute=["xyz"],
        N_POINTS=8192,
        DETERMINISTIC_EVAL=True,
        EVAL_SEED=0,
    )
    dataset = GSModel60Dataset(config)
    expected = {"train": 7757, "test": 6090}[split]
    if len(dataset) != expected:
        raise ValueError(f"Expected {expected} samples, found {len(dataset)}")
    for index in {0, len(dataset) - 1}:
        _, path, (points, label) = dataset[index]
        if tuple(points.shape) != (8192, 3):
            raise ValueError(f"Unexpected tensor shape: {tuple(points.shape)}")
        print(split, index, path, tuple(points.shape), int(label))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", choices=("uco3d80", "gsmodel60"), required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        if args.dataset == "uco3d80":
            test_uco(root, split)
        else:
            test_gs(root, split)


if __name__ == "__main__":
    main()
