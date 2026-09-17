# Scene-level experiments

This directory contains GAPrompt++ semantic segmentation on S3DIS, ScanNet, and
NuScenes.

The six paper configs are under `configs/concerto-gapromptplus/` and
`configs/utonia-gapromptplus/`.

```bash
python main.py \
  --cfg_path configs/utonia-gapromptplus/semseg-utonia-v1m1-4a-nuscenes-lin.py \
  --mode test \
  --weight /path/to/model_best.pth \
  --save_path results/nuscenes \
  --gpus 1
```

Paper evaluation uses one GPU, batch size 1, and full-fragment voting. Reuse
the same `save_path` to resume cached scene predictions.


## Data and pretrained weights

Default paths are relative to this directory. You may create links under
`data/` and `weights/`, or use the environment variables documented in
[`../docs/PRETRAINED_MODELS.md`](../docs/PRETRAINED_MODELS.md). The configured
Concerto weights are public upstream artifacts; the Utonia weight is pending
public release.
