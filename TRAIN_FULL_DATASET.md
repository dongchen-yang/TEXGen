# Training LightGen on Full Annotated Dataset

This guide explains how to train LightGen on the full annotated emissive dataset with proper train/val/test splits.

## Dataset Overview

- **Total annotated samples**: 1,161 emissive objects
- **Successfully matched**: 1,154 samples
- **Train set**: 923 samples (80%)
- **Validation set**: 115 samples (10%)
- **Test set**: 116 samples (10%)

The splits are stored in: `/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json`

## Data Split Generation

The train/val/test splits were generated using:

```bash
cd /localhome/dya78/code/lightgen
conda activate texgen
python data_processing/create_splits.py
```

This script:
1. Loads emissive annotations from `emissive_annotations.json`
2. Matches them with the parquet dataframe
3. Creates 8:1:1 splits (train:val:test)
4. Saves indices to `data_splits.json`

## Configuration

The full training configuration is in `configs/lightgen_full.yaml`:

### Key Settings:

**Data**:
- Batch size: 4 (increased for efficiency)
- Num workers: 4 (parallel data loading)
- Image size: 512x512 (UV space)

**Model**:
- Backbone: SimpleUVUNet (lightweight U-Net for UV space)
- Base channels: 64
- Input: 11 channels (position + normal + albedo + metal + rough)
- Output: 3 channels (emission RGB)

**Training**:
- Max epochs: 100
- Validation: **Every 1 epoch** (check_val_every_n_epoch: 1)
- Precision: bf16-mixed (for efficiency)
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR
- EMA: Enabled (decay=0.9999)
- Condition dropout: 0.1 (for classifier-free guidance)

**Loss**:
- MSE loss: weight 1.0
- L1 loss: weight 0.5

**Checkpointing**:
- Save every epoch
- Keep top 5 checkpoints based on val_psnr
- Monitor: validation PSNR (higher is better)

## Training

### Quick Start

```bash
cd /localhome/dya78/code/lightgen/TEXGen
conda activate texgen
bash train_full.sh
```

### Manual Training

```bash
cd /localhome/dya78/code/lightgen/TEXGen
conda activate texgen

python launch.py \
    --config configs/lightgen_full.yaml \
    --gpu 0 \
    --train
```

### Training with TensorBoard

```bash
# Terminal 1: Start TensorBoard
cd /localhome/dya78/code/lightgen/TEXGen
conda activate texgen
tensorboard --logdir outputs/lightgen/full_training --port 6006

# Terminal 2: Run training
bash train_full.sh
```

Then open http://localhost:6006 in your browser.

## Monitoring Training

### Validation Frequency

The model validates **after every epoch**:
- Runs through all 115 validation samples
- Computes metrics: PSNR, SSIM, LPIPS
- Saves visualization outputs
- Updates checkpoints based on val_psnr

### Logged Metrics

**Training metrics** (logged every 50 steps):
- `train/loss`: Total loss
- `train/mse`: MSE loss component
- `train/l1`: L1 loss component
- `train/lr`: Learning rate

**Validation metrics** (logged every epoch):
- `val_psnr`: Peak Signal-to-Noise Ratio (higher is better)
- `val_ssim`: Structural Similarity Index (higher is better)
- `val_lpips`: Learned Perceptual Image Patch Similarity (lower is better)

### Output Structure

```
outputs/lightgen/full_training@TIMESTAMP/
├── ckpts/                    # Checkpoints
│   ├── last.ckpt            # Latest checkpoint
│   ├── epoch=X-step=Y.ckpt  # Top-k checkpoints
│   └── ...
├── save/                     # Visualization outputs
│   ├── it100-train.jpg      # Training visualizations
│   ├── it1000-test/         # Validation outputs
│   │   ├── pred_x0_*.jpg
│   │   ├── gt_x0_*.jpg
│   │   └── render_*.jpg
│   └── ...
├── configs/                  # Config snapshots
└── logs/                     # Training logs
```

## Testing the Dataloader

Before training, you can test the dataloader:

```bash
cd /localhome/dya78/code/lightgen/TEXGen
conda activate texgen
python test_full_data.py
```

Expected output:
```
Train dataset: 923 samples
Val dataset:   115 samples
Train dataloader: 462 batches (batch_size=2)
Val dataloader:   115 batches (batch_size=1)
```

## Resuming Training

To resume from a checkpoint:

```bash
python launch.py \
    --config configs/lightgen_full.yaml \
    --gpu 0 \
    --train \
    --resume outputs/lightgen/full_training@TIMESTAMP/ckpts/last.ckpt
```

## Evaluation

To evaluate on the test set:

```bash
python launch.py \
    --config configs/lightgen_full.yaml \
    --gpu 0 \
    --test \
    --resume outputs/lightgen/full_training@TIMESTAMP/ckpts/epoch=X-step=Y.ckpt
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce batch size in config: `data.batch_size: 2` or `1`
2. Reduce model capacity: `system.backbone.base_channels: 32`
3. Use gradient checkpointing (if implemented)

### Slow Data Loading

If data loading is slow:
1. Increase num_workers: `data.num_workers: 8`
2. Use faster storage (SSD instead of HDD)
3. Preload data to RAM if possible

### NaN Loss

If loss becomes NaN:
1. Reduce learning rate: `system.optimizer.args.lr: 5e-5`
2. Enable gradient clipping (already enabled at 1.0)
3. Check for invalid data samples

## Dataset Statistics

The annotated dataset contains various emissive object categories:
- Lamps: 444 samples
- Flashlights: 83 samples
- Swords with lighting: 51 samples
- Guns: 45 samples
- Light bulbs: 22 samples
- And more...

See `data_processing/annotation/annotation_stats.txt` for full statistics.

## Next Steps

After training:
1. Evaluate on test set (116 samples)
2. Visualize predictions vs ground truth
3. Analyze failure cases
4. Fine-tune hyperparameters if needed
5. Scale to larger dataset if performance is good

## References

- Config: `configs/lightgen_full.yaml`
- Data splits: `data_processing/annotation/data_splits.json`
- Dataloader: `spuv/data/lightgen_uv.py`
- Model: `spuv/models/simple_uv_unet.py`
- System: `spuv/systems/lightgen_system.py`



