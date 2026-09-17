#!/usr/bin/env python3
"""Validate the published uCO3D80 or GSModel60 directory layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_relative_path(value: str, source: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in {source}: {value}")
    return path


def verify_uco(root: Path, check_files: bool) -> dict:
    categories = load_json(root / "uCO3D80_categories.json")
    train = load_json(root / "train_split.json")
    test = load_json(root / "test_split.json")

    if len(categories) != 80:
        raise ValueError(f"Expected 80 categories, found {len(categories)}")
    if len(train) != 13468 or len(test) != 6996:
        raise ValueError(
            f"Unexpected split sizes: train={len(train)}, test={len(test)}"
        )

    seen: set[str] = set()
    missing: list[str] = []
    for split_name, records in (("train", train), ("test", test)):
        for index, record in enumerate(records):
            required = {"taxonomy_id", "model_id", "file_path"}
            absent = required - set(record)
            if absent:
                raise ValueError(
                    f"{split_name}[{index}] is missing fields: {sorted(absent)}"
                )
            relative = validate_relative_path(
                str(record["file_path"]), root / f"{split_name}_split.json"
            )
            key = relative.as_posix()
            if key in seen:
                raise ValueError(f"Duplicate sample path across splits: {key}")
            seen.add(key)
            if check_files and not (root / relative).is_file():
                missing.append(key)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} uCO3D80 files are missing; first: {missing[0]}"
        )

    return {
        "dataset": "uCO3D80",
        "categories": len(categories),
        "train": len(train),
        "test": len(test),
        "total": len(seen),
        "files_checked": check_files,
    }


def verify_gs(root: Path, check_files: bool) -> dict:
    categories = [
        line for line in (root / "category.txt").read_text().splitlines() if line
    ]
    train = load_json(root / "train.json")
    test = load_json(root / "test.json")

    if len(categories) != 60:
        raise ValueError(f"Expected 60 categories, found {len(categories)}")
    if len(train) + len(test) != 13847:
        raise ValueError(
            f"Unexpected GSModel60 total: {len(train) + len(test)}"
        )

    seen: set[str] = set()
    missing: list[str] = []
    for split_name, paths in (("train", train), ("test", test)):
        for value in paths:
            relative = validate_relative_path(str(value), root / f"{split_name}.json")
            key = relative.as_posix()
            if key in seen:
                raise ValueError(f"Duplicate sample path across splits: {key}")
            seen.add(key)
            if check_files and not (root / relative).is_file():
                missing.append(key)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} GSModel60 files are missing; first: {missing[0]}"
        )

    return {
        "dataset": "GSModel60",
        "categories": len(categories),
        "train": len(train),
        "test": len(test),
        "total": len(seen),
        "files_checked": check_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--dataset", choices=("uco3d80", "gsmodel60"), required=True
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every PLY referenced by the fixed split to exist.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = (
        verify_uco(root, args.check_files)
        if args.dataset == "uco3d80"
        else verify_gs(root, args.check_files)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
