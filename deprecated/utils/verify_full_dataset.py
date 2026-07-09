#!/usr/bin/env python3
"""Verify the full dataset configuration."""

import sys
sys.path.insert(0, '.')

from spuv.data.lightgen_uv import LightGenDataModule

cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv_local",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet",
    'batch_size': 2,
    'num_workers': 4,
    'train_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
    'val_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
    'test_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
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
print("Full Dataset Verification for LightGenPointUVNet")
print("=" * 80)

print("\nCreating datamodule...")
dm = LightGenDataModule(cfg_dict)
dm.setup("fit")

print(f"\n✓ Dataset loaded successfully!")
print(f"  Train: {len(dm.train_dataset)} samples")
print(f"  Val:   {len(dm.val_dataset)} samples")

expected_train = 923
expected_val = 115

if len(dm.train_dataset) == expected_train:
    print(f"  ✅ Train size correct!")
else:
    print(f"  ⚠️  Expected {expected_train} train samples")

if len(dm.val_dataset) == expected_val:
    print(f"  ✅ Val size correct!")
else:
    print(f"  ⚠️  Expected {expected_val} val samples")

# Check dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

print(f"\n✓ Dataloaders created!")
print(f"  Train batches: {len(train_loader)} (batch_size={cfg_dict['batch_size']})")
print(f"  Val batches:   {len(val_loader)} (batch_size=1)")

# Load one batch
print(f"\n✓ Loading sample batch...")
batch = next(iter(train_loader))
print(f"  Batch size: {batch['gt_emission'].shape[0]}")
print(f"  Sample IDs: {batch['scene_id']}")

print("\n" + "=" * 80)
print("✅ Full dataset is ready for training!")
print("=" * 80)
print("\nTo start training:")
print("  python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train")



