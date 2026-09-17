# Datasets

## Object-level datasets

Set dataset paths in `object_level/cfgs/dataset_configs/`.

- ModelNet40, ScanObjectNN, and ShapeNetPart follow their official layouts.
- [GSModel60](https://huggingface.co/datasets/zxAi/GSModel60) uses
  `category.txt`, `train.json`, and `test.json`.
- [uCO3D80](https://huggingface.co/datasets/zxAi/uCO3D80) uses the published
  `train_split.json` and `test_split.json`. Do not regenerate this split.

Verify the two released datasets after download:

```bash
python tools/verify_dataset_bundle.py /path/to/GSModel60 --dataset gsmodel60 --check-files
python tools/verify_dataset_bundle.py /path/to/uCO3D80 --dataset uco3d80 --check-files
```

## Scene-level datasets

Preprocessing scripts for S3DIS, ScanNet, and NuScenes are under
`scene_level/pointcept/datasets/preprocessing/`. Keep raw and processed data
outside the repository.
