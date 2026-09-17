# Pretrained models

The repository contains all model definitions used by the released configs.
Model weights are external artifacts and are not committed to Git. Run commands
from `object_level/` or `scene_level/`; paths below are relative to those
directories. A command-line checkpoint path overrides the configured default.

## Object-level backbones

| Backbone | Target path | Official source | Availability |
|---|---|---|---|
| PointGPT-L post-pretraining | `object_level/pretrained_bases/pointgpt-post-pretrained-L.pth` | [PointGPT-L post-pretrained model](https://drive.google.com/file/d/1Kh6f6gFR12Y86FAeBtMU9NbNpB5vZnpu/view?usp=sharing) ([repository](https://github.com/CGuangyan-BIT/PointGPT)) | Public upstream |
| Point-FEMAE | `object_level/pretrained_bases/femae-epoch-300.pth` | [Point-FEMAE pretrained models](https://drive.google.com/drive/folders/1q0A-yXC1fmKKg38fbaqIxM79lvXpj4AO?usp=drive_link) ([repository](https://github.com/zyh16143998882/AAAI24-PointFEMAE)) | Public upstream |
| ReCon Base | `object_level/pretrained_bases/recon_base.pth` | [ReCon pretrained Base model](https://drive.google.com/file/d/1L-TlZUi7umBCDpZW-1F0Gf4X-9Wvf_Zo/view?usp=share_link) ([repository](https://github.com/qizekun/ReCon)) | Public upstream |
| Point-PQAE | `object_level/pretrained_bases/point-pqae-ckpt-epoch-300.pth` | No stable public upstream URL is available yet | Pending |

Downloaded files may have different upstream names. Rename or symlink them to
the target paths above, or pass an explicit path with `--ckpts`.

The PointGPT-L ScanObjectNN OBJ_BG config is a legacy evaluation-only
compatibility config. It requires a GAPrompt++ fine-tuned checkpoint through
`--test --ckpts`; it must not be used to train from an upstream backbone.

## Scene-level backbones

| Backbone | Target path | Official source | Availability |
|---|---|---|---|
| Concerto Base | `scene_level/weights/Concerto/concerto_base.pth` | [Pointcept/Concerto](https://huggingface.co/Pointcept/Concerto/blob/main/concerto_base.pth) | Public upstream |
| Concerto Large Outdoor | `scene_level/weights/Concerto/concerto_large_outdoor.pth` | [Pointcept/Concerto](https://huggingface.co/Pointcept/Concerto/blob/main/concerto_large_outdoor.pth) | Public upstream |
| Utonia | `scene_level/weights/Utonia/utonia.pth` | No stable public upstream URL is available yet | Pending |

The six scene configs also accept environment-variable overrides:

```bash
export CONCERTO_BASE_CKPT=/path/to/concerto_base.pth
export CONCERTO_LARGE_OUTDOOR_CKPT=/path/to/concerto_large_outdoor.pth
export UTONIA_CKPT=/path/to/utonia.pth
export SCANNET_ROOT=/path/to/processed/scannet
export S3DIS_ROOT=/path/to/processed/S3DIS
export NUSCENES_ROOT=/path/to/processed/nuScenes
```

## GAPrompt++ checkpoints

Fine-tuned GAPrompt++ checkpoints are not bundled with the source repository.
Their public links will be added after the model-weight license is finalized.
Do not confuse these downstream checkpoints with the upstream pretrained
backbones listed above.
