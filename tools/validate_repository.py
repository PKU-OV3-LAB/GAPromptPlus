#!/usr/bin/env python3
"""Validate the public source tree without requiring datasets or checkpoints."""

from __future__ import annotations

import compileall
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def remove_caches() -> None:
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        path.unlink()


def main() -> None:
    remove_caches()
    forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".pyc"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in forbidden:
            raise ValueError(f"Forbidden generated or weight file: {path}")
        if path.name == ".DS_Store" or path.name.startswith("._"):
            raise ValueError(f"Forbidden platform file: {path}")

    expected = {
        "object_classification_configs": (
            ROOT / "object_level/cfgs/classification",
            "*.yaml",
            18,
        ),
        "object_part_segmentation_configs": (
            ROOT / "object_level/cfgs/part_segmentation",
            "*.yaml",
            3,
        ),
        "scene_configs": (
            ROOT / "scene_level/configs",
            "*-gapromptplus/*.py",
            6,
        ),
    }
    counts = {}
    for name, (folder, pattern, required) in expected.items():
        count = len(list(folder.glob(pattern)))
        if count != required:
            raise ValueError(f"{name}: expected {required}, found {count}")
        counts[name] = count

    remove_caches()
    try:
        if not compileall.compile_dir(ROOT, quiet=1):
            raise RuntimeError("Python compilation failed")
    finally:
        remove_caches()

    print(json.dumps({"python_compile": "pass", **counts}, indent=2))


if __name__ == "__main__":
    main()
