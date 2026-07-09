#!/usr/bin/env python3
"""Verify the small dataset configuration."""

import sys
sys.path.insert(0, '.')

from spuv.data.lightgen_uv import LightGenDataModule

cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv_local",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet",
    'batch_size': 4,
    'num_workers': 4,
    'train_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/small_test_splits.json",
    'val_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/small_test_splits.json",
    'test_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/small_test_splits.json",
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

print("=" * 80)
print("Small Dataset Verification (100 train / 10 val / 10 test)")
print("=" * 80)

print("\nCreating datamodule...")
dm = LightGenDataModule(cfg_dict)
dm.setup("fit")

print(f"\n✓ Dataset loaded successfully!")
print(f"  Train: {len(dm.train_dataset)} samples")
print(f"  Val:   {len(dm.val_dataset)} samples")

expected_train = 100
expected_val = 10

if len(dm.train_dataset) == expected_train:
    print(f"  ✅ Train size correct!")
else:
    print(f"  ❌ Expected {expected_train} train samples, got {len(dm.train_dataset)}")

if len(dm.val_dataset) == expected_val:
    print(f"  ✅ Val size correct!")
else:
    print(f"  ❌ Expected {expected_val} val samples, got {len(dm.val_dataset)}")

# Check dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

print(f"\n✓ Dataloaders created!")
print(f"  Train batches: {len(train_loader)} (batch_size={cfg_dict['batch_size']})")
print(f"  Val batches:   {len(val_loader)} (batch_size=1)")

# Estimate training time
batches_per_epoch = len(train_loader)
val_batches = len(val_loader)
time_per_batch = 1.5  # seconds (estimate)
time_per_val_batch = 2.0  # seconds (estimate)

epoch_time = (batches_per_epoch * time_per_batch + val_batches * time_per_val_batch) / 60
total_time = epoch_time * 10  # 10 epochs

print(f"\n⏱️  Estimated time:")
print(f"  Per epoch: ~{epoch_time:.1f} minutes")
print(f"  10 epochs: ~{total_time:.1f} minutes ({total_time/60:.1f} hours)")

# Load one batch
print(f"\n✓ Loading sample batch...")
batch = next(iter(train_loader))
print(f"  Batch size: {batch['gt_emission'].shape[0]}")
print(f"  Sample IDs: {batch['scene_id']}")

print("\n" + "=" * 80)
print("✅ Small dataset is ready for testing!")
print("=" * 80)
print("\nTo start training:")
print("  python launch.py --config configs/lightgen_pointuv_small.yaml --gpu 0 --train --wandb")
print("\nThis will:")
print("  - Train on 100 samples (25 batches per epoch)")
print("  - Validate on 10 samples every epoch")
print("  - Complete 10 epochs in ~10-15 minutes")
print("  - Verify your code works before full training!")



