# Datasets

## Object-level datasets

Set dataset paths in `object_level/cfgs/dataset_configs/`. Default paths are under `object_level/data/`; use symbolic links there or edit the dataset config to point to an external directory.

- ModelNet40, ScanObjectNN, and ShapeNetPart follow their official layouts.
- [GSModel60](https://huggingface.co/datasets/zxAi/GSModel60) uses
  `category.txt`, `train.json`, and `test.json`; its default path is `object_level/data/GSModel60`.
- [uCO3D80](https://huggingface.co/datasets/zxAi/uCO3D80) uses the published
  `train_split.json` and `test_split.json`; its default path is `object_level/data/uCO3D80`. Do not regenerate this split.

Verify the two released datasets after download:

```bash
python tools/verify_dataset_bundle.py /path/to/GSModel60 --dataset gsmodel60 --check-files
python tools/verify_dataset_bundle.py /path/to/uCO3D80 --dataset uco3d80 --check-files
```

## Scene-level datasets

Preprocessing scripts for S3DIS, ScanNet, and NuScenes are under
`scene_level/pointcept/datasets/preprocessing/`. Keep raw and processed data
outside the repository. Link the processed directories to
`scene_level/data/{S3DIS,scannet,nuScenes}`, or set `S3DIS_ROOT`,
`SCANNET_ROOT`, and `NUSCENES_ROOT` before launching an experiment.
