# Object-level experiments

This directory contains GAPrompt++ object classification and ShapeNetPart part
segmentation.

## Configs

- `cfgs/classification/`: the 18 classification settings reported in the paper.
- `cfgs/part_segmentation/`: the three ShapeNetPart settings.
- `cfgs/dataset_configs/`: dataset roots and loader options.

Run commands from this directory so `_base_` dataset paths resolve correctly.

```bash
python main.py \
  --config cfgs/classification/gapromptpp-point-pqae-gsmodel60.yaml \
  --ckpts /path/to/pretrained_backbone.pth
```

For evaluation, add `--test` and provide the fine-tuned checkpoint through the
corresponding command-line option used by the selected runner.


## Pretrained weights

See [`../docs/PRETRAINED_MODELS.md`](../docs/PRETRAINED_MODELS.md) for official
upstream downloads and exact target filenames. The PointGPT-L ScanObjectNN
OBJ_BG config is evaluation-only and requires `--test --ckpts` with its
fine-tuned GAPrompt++ checkpoint.
