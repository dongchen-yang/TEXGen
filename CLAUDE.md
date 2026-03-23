# CLAUDE.md

## Project Overview

**LightGen** is an adaptation of TEXGen (SIGGRAPH Asia 2024) for **emission map generation** — predicting RGB emission maps in UV space given geometry and material properties. It uses Flow Matching diffusion with classifier-free guidance (CLIP conditioning). The codebase lives inside the `TEXGen/` git submodule.

## Baseline Variants

There are exactly **3 baseline variants**. All other configs (pretrained, unfiltered, bs2, simple U-Net, base) are deprecated and should be ignored.

| Variant | Full Config | Overfit Config | Key Difference |
|---------|------------|----------------|----------------|
| **Vanilla** | `lightgen_pointuv_256_batch32_emission_filtered.yaml` | `lightgen_pointuv_overfit.yaml` | Standard 12-ch input, MSE+L1 loss |
| **GT Mask Cond** | `lightgen_pointuv_256_batch32_emission_filtered_gt_mask_cond.yaml` | `lightgen_pointuv_overfit_gt_mask_cond.yaml` | 13-ch input (stacks thresholded GT emission mask as extra channel) |
| **Mask Cls Loss** | `lightgen_pointuv_256_batch32_emission_filtered_mask_cls.yaml` | `lightgen_pointuv_overfit_mask_cls.yaml` | 12-ch input, adds BCE mask classification loss (`lambda_pred_mask_cls: 1.0`, threshold 0.01) |

## Commands

All commands run from `TEXGen/` directory.

### Training
```bash
# Overfitting tests (single-sample sanity checks, small architecture)
python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --train
python launch.py --config configs/lightgen_pointuv_overfit_gt_mask_cond.yaml --gpu 0 --train
python launch.py --config configs/lightgen_pointuv_overfit_mask_cls.yaml --gpu 0 --train

# Full dataset training (256x256, batch 32, emission-filtered)
python launch.py --config configs/lightgen_pointuv_256_batch32_emission_filtered.yaml --gpu 0 --train
python launch.py --config configs/lightgen_pointuv_256_batch32_emission_filtered_gt_mask_cond.yaml --gpu 0 --train
python launch.py --config configs/lightgen_pointuv_256_batch32_emission_filtered_mask_cls.yaml --gpu 0 --train

# With W&B logging
python launch.py --config <config.yaml> --gpu 0 --train --wandb

# SLURM cluster
sbatch slurm_train.sh
```

### Testing / Inference
```bash
python launch.py --config <config.yaml> --gpu 0 --test system.resume=/path/to/checkpoint.ckpt
python launch.py --config <config.yaml> --gpu 0 --validate system.resume=/path/to/checkpoint.ckpt
python inference_specific_samples.py
```

### Config Overrides (CLI)
OmegaConf-style overrides appended directly:
```bash
python launch.py --config <config.yaml> --gpu 0 --train system.optimizer.args.lr=1e-4 data.batch_size=2
```

### Debugging
- `--verbose` — DEBUG-level logging
- `--typecheck` — dynamic type checking via jaxtyping
- `--benchmark` — record running times
- `python test_dataloader.py` / `python test_full_data.py` — data verification

## Architecture

### Repository Structure
```
lightgen/                            # Top-level project
├── TEXGen/                          # Git submodule — main implementation
│   ├── launch.py                    # Entry point (train/test/validate/export)
│   ├── configs/                     # YAML configs (OmegaConf)
│   ├── slurm_train.sh              # SLURM job script
│   ├── spuv/                        # Core package
│   │   ├── systems/
│   │   │   ├── base.py              # BaseSystem — abstract LightningModule
│   │   │   ├── texgen_base.py       # TEXGenDiffusion — Flow Matching, EMA, schedulers
│   │   │   ├── texgen_test.py       # TEXGenDiffusion — validation/test (Euler ODE inference)
│   │   │   └── lightgen_system.py   # LightGenSystem — emission losses, visualization
│   │   ├── models/
│   │   │   ├── sparse_networks/
│   │   │   │   ├── lightgen_pointuvnet.py  # LightGenPointUVNet (primary backbone)
│   │   │   │   ├── texgen_network.py       # Original PointUVNet (base architecture)
│   │   │   │   └── serialization/          # Voxel ordering (z-order, Hilbert)
│   │   │   ├── simple_uv_unet.py           # SimpleUVUNet (lightweight, for quick tests)
│   │   │   ├── tokenizers/clip.py          # CLIP conditioning
│   │   │   ├── lpips.py                    # Perceptual loss
│   │   │   └── renderers/                  # NVDiffRast rendering
│   │   ├── data/
│   │   │   ├── lightgen_uv.py       # LightGenDataModule — loads NPZ, parquet indexing
│   │   │   └── mesh_uv.py           # Original TEXGen loader (unused)
│   │   └── utils/
│   │       ├── config.py            # OmegaConf config loading
│   │       ├── image_metrics.py     # SSIM, PSNR
│   │       ├── memory_tracker.py    # GPU OOM debugging
│   │       ├── lit_ema.py           # EMA weight averaging
│   │       └── snr_utils.py         # SNR-based loss weighting
│   └── inference_outputs/           # Saved inference results
├── data/
│   └── baked_uv_local_subset/       # Dataset (NPZ + parquet + split JSONs)
├── evaluation/                      # Standalone evaluation scripts
│   ├── PSNR.py                      # Emission PSNR
│   ├── IoU.py / 3DIoU.py           # 2D/3D IoU on emission masks
│   ├── VLM.py / VLM_multiview.py   # VLM-based evaluation
│   ├── patch_psnr_emission.py       # PSNR fix (handles NaN/inf)
│   └── generate_result_html.py      # HTML result visualization
├── external/xgutils/                # Git submodule — utility library
└── visualization/                   # Visualization tools
```

### Inheritance Chain
```
BaseSystem (base.py)
  └─ TEXGenDiffusion (texgen_base.py) — Flow Matching, noise schedule, EMA
       └─ TEXGenDiffusion (texgen_test.py) — Euler ODE inference, test pipeline
            └─ LightGenSystem (lightgen_system.py) — emission losses, data prep
```

### Dynamic Class Loading
Classes are referenced by string paths in YAML configs. `spuv.find()` resolves them at runtime:
```yaml
system_cls: spuv.systems.lightgen_system.LightGenSystem
data_cls: spuv.data.lightgen_uv.LightGenDataModule
backbone_cls: spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet
image_tokenizer_cls: spuv.models.tokenizers.clip.ClipTokenizer
```

### Backbone: LightGenPointUVNet
Modified PointUVNet operating on **pre-baked UV data** (no online rasterization). 5-stage encoder-decoder:
- Block types: UV, Point-UV (hybrid), UV-DIT (attention)
- TorchSparse for 3D-aware voxel grouping
- Skip connections (adaptive)
- Classifier-free guidance via CLIP embedding dropout

Full-scale architecture:
```yaml
block_out_channels: [32, 256, 1024, 1024, 2048]
block_type: ["uv", "point_uv", "uv_dit", "uv_dit", "uv_dit"]
window_size: [0, 128, 128, 256, 256]  # for 256x256
```

Overfit architecture (smaller, 3-stage):
```yaml
block_out_channels: [32, 128, 256]
block_type: ["uv", "point_uv", "uv_dit"]
window_size: [0, 128, 128]
```

### Flow Matching Diffusion
- Velocity prediction: `v = x0 - noise`
- Training: sample `t ~ U[0,1]` with power=2 bias toward smaller t
- Inference: Euler ODE integration over `test_num_steps` (default 50) with `test_cfg_scale` (default 2.0)
- `rescale_betas_zero_snr: true` for terminal SNR fix

### Loss Functions (`lightgen_system.py`)
All variants share:
- **MSE loss** (`lambda_mse: 1.0`) on velocity, masked to valid UV pixels
- **L1 loss** (`lambda_l1: 0.5`) on velocity, masked to valid UV pixels

Variant-specific:
- **GT Mask Cond**: No extra loss — the GT emission mask is concatenated as an input channel (`in_channels: 13`)
- **Mask Cls Loss**: BCE on predicted vs GT emission mask (`lambda_pred_mask_cls: 1.0`, `mask_cls_threshold: 0.01`)
- **Dark region loss** (`lambda_dark_region`): available but set to 0.0 in all current configs

### EMA (Exponential Moving Average)
- Enabled for full training (`ema_decay: 0.9999`), disabled for overfitting
- Validation uses EMA weights (`val_with_ema: true`)
- `LitEma` class handles weight shadowing and restoration

## Data

### Format
NPZ files (`somage.npz`) at 512x512 UV resolution, downsampled to config resolution (256x256):
```
occupancy      [H, W, 1]  bool    — valid UV pixel mask
position       [H, W, 3]  uint16  — 3D positions (decoded to [-2, 2])
objnormal      [H, W, 3]  uint16  — normals (decoded to [-1, 1])
color          [H, W, 3]  uint8   — albedo (decoded to [0, 1])
metal          [H, W, 1]  uint8   — metallic (decoded to [0, 1])
rough          [H, W, 1]  uint8   — roughness (decoded to [0, 1])
emission_color [H, W, 3]  uint8   — GT emission (decoded to [0, 1], normalized to [-1, 1])
```

### Input Channels
Backbone input tensor (concatenated in channel dim):
```
noisy_emission(3) + position(3) + albedo(3) + metallic(1) + roughness(1) + baked_weights/mask(1) = 12
```
GT Mask Cond variant adds: `+ gt_emission_mask(1) = 13`

### Dataset
Emission-filtered Objaverse subset: **878 train / 112 val / 109 test** (1099 total, zero-emission samples removed).

- Parquet indexing: `df_SomgProc_emission_filtered.parquet`
- Split file: `data_splits_emission_filtered.json`
- Overfit split: `overfit_split_single.json` (single sample)
- Data root: `../data/baked_uv_local_subset/` (relative to `TEXGen/`)

## Config System

YAML via OmegaConf. Key sections:
```yaml
name / tag                    # experiment naming → output dir
auto_resume: true             # resume from latest checkpoint
custom_output_dir             # override output location (used in overfit configs)

data_cls / system_cls         # dynamic class paths
data.batch_size               # 32 (full) or 1 (overfit)
data.uv_height / uv_width    # 256 for all current configs
data.train_indices            # JSON file path or tuple range

system.backbone.in_channels   # 12 (vanilla/mask_cls) or 13 (gt_mask_cond)
system.loss.diffusion_loss_dict.lambda_pred_mask_cls  # mask cls variant only
system.loss.diffusion_loss_dict.mask_cls_threshold    # mask cls variant only
system.condition_drop_rate    # 0.1 (full training) or 0.0 (overfit)
system.use_ema                # true (full training) or false (overfit)
system.test_num_steps         # 50 (inference denoising steps)
system.test_cfg_scale         # 2.0 (classifier-free guidance scale)

trainer.max_epochs            # 5000 (full) or 10000 (overfit)
trainer.precision             # bf16-mixed

checkpoint.dirpath            # external checkpoint storage (full training on /scratch/)
checkpoint.monitor            # val/psnr
```

### Overfit vs Full Training Config Differences
| Setting | Overfit | Full |
|---------|---------|------|
| Architecture | 3-stage (small) | 5-stage (full) |
| batch_size | 1 | 32 |
| condition_drop_rate | 0.0 | 0.1 |
| use_ema | false | true |
| weight_decay | 0.0 | 0.01 |
| max_epochs | 10000 | 5000 |
| check_val_every_n_epoch | 5 | 100 |
| data split | overfit_split_single.json | data_splits_emission_filtered.json |
| checkpoint.dirpath | local (outputs/) | cluster (/scratch/) |

## Output Structure
```
outputs/{name}/{tag}@{timestamp}/
├── ckpts/      # checkpoints (last.ckpt + top-k by val/psnr)
├── tb_logs/    # TensorBoard
├── save/       # visualization images
├── configs/    # config snapshot
└── cmd.txt     # command used
```

Full training checkpoints go to `/scratch/dya78/lightgen/TEXGen/output_emission_filtered*/`.

## Key Implementation Details

- **PyTorch 2.6 compat**: `launch.py` patches `torch.load()` to force `weights_only=False`
- **Auto-resume**: finds latest `last*.ckpt` in checkpoint dir when `auto_resume: true`
- **WandB resume**: run ID saved in checkpoint, auto-restored on resume. Stale cache cleaned.
- **Memory management**: `memory_tracker.py` logs GPU usage; `cleanup_after_validation_step: true` runs `gc.collect()` + `torch.cuda.empty_cache()` after each val step
- **Signal handling**: graceful shutdown on SIGTERM/SIGUSR1/KeyboardInterrupt
- **Conda env**: `texgen`
- **Critical env var**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (prevents OOM from fragmentation)

## Hardware Requirements
- Training (full): >=40GB VRAM (A100), bf16-mixed precision
- Inference: >=24GB VRAM
- SLURM: 1 GPU, 32G RAM, 8 CPUs, 48h wall time
