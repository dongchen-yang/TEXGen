# CLAUDE.md

## Project Overview

**LightGen** is an adaptation of TEXGen (SIGGRAPH Asia 2024) for **emission map generation** — predicting RGB emission maps in UV space given geometry and material properties. It uses Flow Matching diffusion with classifier-free guidance (CLIP conditioning). The codebase lives inside the `TEXGen/` git submodule.

## Baseline Variants

There are exactly **3 baseline variants**. Active workflow map: [`LIGHTGEN_WORKFLOW.md`](LIGHTGEN_WORKFLOW.md).
Superseded overfit10 / sweep configs and stale docs live under `deprecated/` (not `deprecated/configs/` of older pretrained variants — those were never in this tree).

| Variant | 1k Full Config | 74k Scaled Config | Overfit Config | Key Difference |
|---------|----------------|-------------------|----------------|----------------|
| **Vanilla** | `lightgen_pointuv_256_batch32_emission_filtered.yaml` | `lightgen_pointuv_256_batch32_emissive_74k.yaml` | `lightgen_pointuv_overfit.yaml` | Standard 12-ch input, MSE+L1 loss |
| **GT Mask Cond** | `lightgen_pointuv_256_batch32_emission_filtered_gt_mask_cond.yaml` | _(not yet)_ | `lightgen_pointuv_overfit_gt_mask_cond.yaml` | 13-ch input (stacks thresholded GT emission mask as extra channel) |
| **Mask Cls Loss** | `lightgen_pointuv_256_batch32_emission_filtered_mask_cls.yaml` | _(not yet)_ | `lightgen_pointuv_overfit_mask_cls.yaml` | 12-ch input, adds BCE mask classification loss (`lambda_pred_mask_cls: 1.0`, threshold 0.01) |

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

# Scaled 74k training on fir (4× H100 DDP, per-GPU bs=32, global bs=128)
python launch.py --config configs/lightgen_pointuv_256_batch32_emissive_74k.yaml --gpu 0,1,2,3 --train --wandb

# With W&B logging
python launch.py --config <config.yaml> --gpu 0 --train --wandb

# SLURM cluster (1k baseline on fir; uses ~/scratch baked_uv_local_subset.tar)
sbatch slurm_train.sh

# SLURM cluster (74k scaled on fir; auto extracts NPZ tars to $SLURM_TMPDIR)
sbatch slurm_train_74k.sh
# Variant wrappers: scripts/fir/texgen_256_batchsize32_filter_emission*.sh

# Per-GPU max-batch probe on fir (H100; runs 3 forward+backward steps per bs)
DATA_ROOT=$SLURM_TMPDIR/baked_uv bash batch_sweep.sh
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

## Architecture

### Repository Structure
```
lightgen/                            # Top-level project
├── TEXGen/                          # Git submodule — main implementation
│   ├── launch.py                    # Entry point (train/test/validate/export)
│   ├── configs/                     # Active YAML configs (OmegaConf)
│   ├── scripts/fir/                 # Canonical fir launch wrappers
│   ├── slurm_train_74k.sh           # 74k SLURM job script
│   ├── LIGHTGEN_WORKFLOW.md         # Active LightGen workflow map
│   ├── deprecated/                  # Archived overfit10/sweep configs & helpers
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

### Datasets

Two scales coexist. Pick based on the config you're running.

**1k emission-filtered** (legacy single-GPU baseline): **878 train / 112 val / 109 test** (1099 total, zero-emission samples removed).

- Parquet indexing: `df_SomgProc_emission_filtered.parquet`
- Split file: `data_splits_emission_filtered.json`
- Overfit split: `overfit_split_single.json` (single sample)
- Data root: `../data/baked_uv_local_subset/` (relative to `TEXGen/`); on workstation a symlink to `/cs/3dlg-falas/.../somages/v1201_homages_512charts/somages/`. Local copy is ~8.5 GB.

**74k emissive-complete** (scaled 4-GPU, 2026-04 onwards): **73,251 train / 112 val / 109 test** (74,353-sample emissive-complete pool; val/test pinned to the 1k baseline for direct PSNR comparability).

- Parquet indexing: `df_SomgProc_final.parquet` (854,287 rows; dataloader filters to `success==True` → 824,858 rows that the indices reference positionally)
- Split file: `data_processing/annotation/data_splits_emissive_74k_pinned.json` (built by `data_processing/create_splits_74k_pinned.py` — preserves the 1k val/test, excludes 1k-train leakage)
- Storage layout (post-migration, see "Data Migration" below):
    - **NPZ tars (training)**: 8 chunks of ~57 GB each, total ~490 GB. `npz_chunk_00.tar … npz_chunk_07.tar`. Each contains `<shard>/<ditem_id>/somage.npz` + `_dproc_*.json`. Per-sample = 2 entries; total = 7×27,885 + 27,864 = 223,059 entries.
    - **Thumbnail tar (CLIP conditioning)**: `thumbnails_emissive.tar`, ~12 GB, 80,735 PNGs (covers all 74,353 ditem_ids in the pool plus a few extras). Each at `emissive_thumbnails/<ditem_id>.png` inside the tar. Required for correct CLIP image conditioning — without it the dataloader falls back to the albedo UV map (substantively wrong input). At job start, extracted into `$SLURM_TMPDIR/baked_uv/emissive_thumbnails/`, then symlinked as `$SLURM_TMPDIR/baked_uv/thumbnails -> emissive_thumbnails` to match the path the dataloader expects.
    - **GLB tars (eval / rendering)**: 8 chunks of ~104 GB each, total ~840 GB. `glb_chunk_00.tar … glb_chunk_07.tar`. Each contains `<shard>/<ditem_id>_1024.glb`. Total 74,353 entries.
- **Locations**:
    - Jupiter NFS (archive, source of truth): `/cs/3dlg-jupiter-project/lightgen/dataset/{npz,glb}_chunk_*.tar` + `thumbnails_emissive.tar`
    - Fir Lustre `/scratch` (ready for training): `/home/dya78/scratch/lightgen/data/tars/{npz,glb}_chunk_*.tar` + `thumbnails_emissive.tar`
- **Job-time staging**: `slurm_train_74k.sh` extracts the 8 NPZ tars (~2 min via `detar_progress.py`, parallel-of-8 with single tqdm bar) **plus** the thumbnail tar (~30 s) into `$SLURM_TMPDIR/baked_uv/` on job start. Total extracted ~468 GB into node-local SSD. GLB tars stay on `/scratch` — only needed for downstream evaluation, not training.

### Data Migration (2026-04-23, with thumbnail follow-up 2026-04-28)

The 74k subset was migrated from `/cs/3dlg-falas/` symlinks to standalone tar archives so it's portable to clusters without 3dlg-falas access (i.e. fir).

- Manifests + drivers under `_staging_migrate/` on the workstation:
    - `manifests/{npz,glb}_chunk_*.paths` (74,353 ditem_ids resolved to shard/id; 8 chunks of ~9,295 each)
    - `tar_driver.sh` — builds 8+8 tars on jupiter NFS reading from /cs/3dlg-falas (4h 31m wall time)
    - `rsync_driver.sh` — pushes tars jupiter→fir over WAN (8h 32m wall time at ~42 MB/s)
    - `verify_jupiter.sh` — `tar -tf | wc -l` per tar to confirm entry counts
- 2026-04-28 follow-up: built `thumbnails_emissive.tar` (12 GB) from `processed_data/emissive_thumbnails/` directly to jupiter NFS (~3 min) and rsync'd to fir (~4 min). Discovered after observing 4,637 "Thumbnail not found, using albedo UV map as fallback" warnings in the first successful 74k training run — model had been getting wrong CLIP conditioning (albedo UV map instead of rendered preview) for the entire 8-epoch debug run.
- Total wall time end-to-end: ~13h 10m for NPZ+GLB; +~7 min for thumbnails. 1.3 TB total on each side.

### Distributed sampler caveat under DDP

PyTorch Lightning auto-injects `DistributedSampler` for every dataloader under DDP, including val/test. With 4 ranks, each rank only sees 1/4 of the val/test set, and only rank 0 logs to wandb — so the qualitative panel shows only 1/4 of the visualizations vs the 1k baseline (28 of 112 val samples observed in first 74k run vs 109 in `htiandl5`).

**Fix in `lightgen_uv.py`** (commit 056a5ad, 2026-04-28): `val_dataloader()`, `test_dataloader()`, and `predict_dataloader()` pass an explicit `SequentialSampler`. Lightning respects an explicit sampler and skips its auto-shard. Every rank now evaluates the full val/test set in parallel (same wall-time as 1-rank since they're independent on each GPU; 4× redundant compute on val, but val is < 1% of total job time). Train still uses Lightning's auto `DistributedSampler` (we want sharded training).

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
| batch_size | 1 or 10 | 32 |
| condition_drop_rate | 0.1 | 0.1 |
| use_ema | true (decay=0.9999) | true (decay=0.9999) |
| val_with_ema | true | true |
| weight_decay | 0.01 | 0.01 |
| dropout | [0.0, 0.0, 0.1] | [0.0, 0.0, 0.0, 0.0, 0.1] |
| max_epochs | 10000 | 5000 |
| check_val_every_n_epoch | 100 | 100 |
| data split | overfit_split_single.json or overfit_split_10.json | data_splits_emission_filtered.json |
| checkpoint.dirpath | local (outputs/) | cluster (/scratch/) |

**Important:** All new experiments (including overfit) must use the full regularization suite (allreg): EMA, dropout, weight decay, and condition dropout. The Mar26 allreg experiment showed +4–5 dB improvement and eliminated training degradation. Legacy no-reg overfit configs are deprecated.

## Output Structure
```
outputs/{name}/{tag}@{timestamp}/
├── ckpts/      # checkpoints (last.ckpt + top-k by val/psnr)
├── tb_logs/    # TensorBoard
├── save/       # visualization images
├── configs/    # config snapshot
└── cmd.txt     # command used
```

Full training checkpoints go to `/scratch/dya78/lightgen/TEXGen/output_emission_filtered*/` (1k baseline) or `/home/dya78/scratch/lightgen/TEXGen/output_emissive_74k_vanilla/` (74k scaled).

## Key Implementation Details

- **PyTorch 2.6 compat**: `launch.py` patches `torch.load()` to force `weights_only=False`
- **Auto-resume**: finds latest `last*.ckpt` in checkpoint dir when `auto_resume: true`
- **WandB resume**: run ID saved in checkpoint, auto-restored on resume. Stale cache cleaned.
- **Memory management**: `memory_tracker.py` logs GPU usage; `cleanup_after_validation_step: true` runs `gc.collect()` + `torch.cuda.empty_cache()` after each val step
- **Signal handling**: graceful shutdown on SIGTERM/SIGUSR1/KeyboardInterrupt
- **Conda env**: `texgen`
- **Critical env var**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (prevents OOM from fragmentation)

## Experiment Findings

### Overfit Test — 3 Baseline Variants (2026-03-23)

Single-sample overfit comparison of Vanilla, GT Mask Cond, and Mask Cls Loss (λ=1.0).

**Results (Best / Final PSNR):**
| Variant | Best PSNR | Final PSNR (10k) |
|---------|-----------|-------------------|
| GT Mask Cond | **44.53 dB** | **43.24 dB** |
| Vanilla | 41.32 dB | 39.09 dB |
| Mask Cls (λ=1.0) | 37.30 dB | 31.18 dB |

**Key Takeaways:**
1. **GT Mask Cond wins clearly** — +3.2 dB over vanilla. The GT emission mask as an extra input channel is a strong oracle signal.
2. **Mask Cls Loss (λ=1.0) hurts overfitting** — peaks at 37.3 dB (worse than vanilla 41.3 dB) and diverges badly in the second half of training (drops to 31.2 dB). The BCE loss destabilizes training at this weight.
3. **Vanilla is the most stable** — steady convergence, mild degradation after peak (41.3→39.1 dB).
4. **Mask Cls needs lambda tuning** — λ=1.0 is too high. This motivated the lambda sweep below.

### Mask Cls Lambda Sweep (2026-03-23)

Sweep of `lambda_pred_mask_cls` ∈ {0.0 (vanilla), 0.01, 0.1, 1.0, 10.0}. All other settings identical.

**Results (Best / Final PSNR):**
| Lambda | Best PSNR | @ Step | Final PSNR (10k) |
|--------|-----------|--------|-------------------|
| **0.01** | **41.80 dB** | 8,414 | **40.01 dB** |
| 0.0 (vanilla) | 41.32 dB | 7,884 | 39.09 dB |
| 0.1 | 41.53 dB | 5,809 | 34.96 dB |
| 1.0 | 37.30 dB | 5,854 | 31.18 dB |
| 10.0 | 33.20 dB | 4,859 | 27.25 dB |

**Key Takeaways:**
1. **λ=0.01 is the sweet spot** — slightly outperforms vanilla (41.80 vs 41.32 dB peak, 40.01 vs 39.09 dB final) while achieving perfect mask IoU=1.0. The mask prediction head is essentially free.
2. **Higher λ hurts reconstruction monotonically** — PSNR degrades as λ increases: ~40 dB (0.01) → ~35 dB (0.1) → ~31 dB (1.0) → ~27 dB (10.0).
3. **Mask IoU saturates at 1.0 for all lambdas** by step ~8k, so even λ=0.01 is sufficient to learn the mask perfectly.
4. **λ=0.01 and vanilla are the most stable** at end of training (~−1.8 dB drop from peak). Higher lambdas degrade much more (λ=1.0 drops −6.1 dB).
5. **Recommendation: Use λ=0.01** as the default for mask cls loss — vanilla-level PSNR + a usable emission mask predictor.

### Mask Cls Lambda Sweep Configs
| Lambda | Config |
|--------|--------|
| 0.01 | `configs/lightgen_pointuv_overfit_mask_cls_lambda001.yaml` (kept active) |
| 0.1 | `deprecated/configs/lightgen_pointuv_overfit_mask_cls_lambda01.yaml` |
| 1.0 | `configs/lightgen_pointuv_overfit_mask_cls.yaml` (baseline overfit) |
| 10.0 | `deprecated/configs/lightgen_pointuv_overfit_mask_cls_lambda10.yaml` |

Sweep script (archived): `deprecated/scripts/root/run_mask_cls_lambda_sweep.sh`

## Hardware Requirements

| Workload | GPU | RAM | CPUs | Wall time |
|---|---|---|---|---|
| 1k full training | 1× A100 / H100, ≥40 GB | 32 GB | 8 | 48 h |
| 74k scaled training | 4× H100 80 GB (DDP) | 1000 GB (full fir node) | 48 | 3-5 days |
| Inference / eval | 1× ≥24 GB VRAM | — | — | — |

bf16-mixed precision throughout. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required to avoid fragmentation.

### 74k batch-size sweep on H100 (2026-04-27)

Per-GPU memory probe on a single H100 80 GB (`bash batch_sweep.sh`, 3 train batches per bs, no val/test):

| bs | Peak VRAM | Verdict |
|---|---|---|
| 32 | 58.8 GB | ✅ comfortable (~22 GB headroom) |
| 48 | 80.7 GB | ⚠️ at the cliff (~0.8 GB headroom — outlier samples can OOM) |
| 64 | OOM during forward | ❌ |

Per-sample marginal ~1.37 GB; static overhead ~14 GB (model + optimizer + CLIP). Default for production: **per-GPU bs=32 → global 128 on 4× H100**. Memory peaks at backward (model accumulates activations + grads at the same instant, then drops back to ~10 GB after `optimizer.step()`).
