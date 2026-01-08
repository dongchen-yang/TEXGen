#!/usr/bin/env python3
"""
Quick test to verify 1-sample overfitting setup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("Testing 1-Sample Overfitting Setup")
print("=" * 80)

from spuv.data.lightgen_uv import LightGenDataModule

# Create config matching the yaml
cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet",
    'batch_size': 1,
    'num_workers': 0,
    'train_indices': [0, 1],
    'val_indices': [0, 1],
    'test_indices': [0, 1],
    'cond_views': 1,
    'sup_views': 4,
    'camera_strategy': 'strategy_1',
    'eval_camera_strategy': 'strategy_1',
    'eval_cond_views': 1,
    'eval_sup_views': 4,
    'height': 512,
    'width': 512,
    'eval_height': 512,
    'eval_width': 512,
    'eval_batch_size': 1,
    'uv_height': 512,
    'uv_width': 512,
    'vertex_transformation': False,
    'repeat': 1,
}

print("\n1. Creating data module...")
dm = LightGenDataModule(cfg_dict)
dm.setup("fit")

print("2. Checking dataset sizes...")
print(f"   Train dataset: {len(dm.train_dataset)} samples")
print(f"   Val dataset: {len(dm.val_dataset)} samples")

print("\n3. Loading the single training sample...")
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))

if batch is None:
    print("   ❌ Failed to load batch!")
    sys.exit(1)

print(f"   ✓ Sample ID: {batch['scene_id'][0]}")
print(f"   ✓ Input shape: {batch['input_tensor'].shape}")
print(f"   ✓ Target emission shape: {batch['gt_emission'].shape}")
print(f"   ✓ Mask coverage: {batch['mask_map'].sum() / batch['mask_map'].numel() * 100:.1f}%")

# Check the emission map
gt_emission = batch['gt_emission'][0]  # [3, 512, 512]
mask = batch['mask_map'][0, 0] > 0.5  # [512, 512]
emission_range = gt_emission[:, mask]  # [3, N] where N is number of valid pixels
emission_range = emission_range.flatten()  # Flatten to 1D
print(f"\n4. Ground truth emission statistics (within mask):")
print(f"   Min: {emission_range.min():.3f}")
print(f"   Max: {emission_range.max():.3f}")
print(f"   Mean: {emission_range.mean():.3f}")
print(f"   Std: {emission_range.std():.3f}")
print(f"   Non-zero pixels: {(emission_range.abs() > 0.01).sum().item()} / {emission_range.numel()}")

print("\n" + "=" * 80)
print("✓ Ready for overfitting test!")
print("=" * 80)
print("\nConfiguration:")
print("  - 1 training sample (repeated every epoch)")
print("  - Batch size: 1")
print("  - Max epochs: 100")
print("  - LR: 5e-4 (higher for faster overfitting)")
print("  - No dropout, no EMA, no weight decay")
print("  - Validation every 10 epochs")
print("\nExpected behavior:")
print("  - Training loss should decrease significantly")
print("  - PSNR should increase (target: >30 dB for perfect fit)")
print("  - Training images should match ground truth exactly")
print("\nStart training with:")
print("  cd /localhome/dya78/code/lightgen/TEXGen")
print("  python launch.py --config configs/lightgen.yaml --train --gpu 0")

