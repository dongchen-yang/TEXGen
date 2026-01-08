#!/usr/bin/env python3
"""
Test the full annotated dataset with train/val/test splits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing LightGen with full annotated dataset...\n")

from spuv.data.lightgen_uv import LightGenDataModule
import torch

# Create config - load from YAML or use dict
cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv_local",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet",
    'batch_size': 2,
    'num_workers': 0,  # Use 0 for debugging
    'train_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
    'val_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
    'test_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/data_splits.json",
    'cond_views': 1,
    'sup_views': 4,
    'camera_strategy': 'strategy_1',
    'eval_camera_strategy': 'strategy_1',
    'eval_cond_views': 1,
    'eval_sup_views': 4,
    'height': 128,
    'width': 128,
    'eval_height': 512,
    'eval_width': 512,
    'eval_batch_size': 1,
    'uv_height': 512,
    'uv_width': 512,
    'vertex_transformation': False,
    'repeat': 1,
}

print("1. Creating data module...")
dm = LightGenDataModule(cfg_dict)

print("2. Setting up datasets...")
dm.setup("fit")

print("3. Checking dataset sizes...")
print(f"   Train dataset: {len(dm.train_dataset)} samples")
print(f"   Val dataset:   {len(dm.val_dataset)} samples")

print("\n4. Creating dataloaders...")
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
print(f"   ✓ Train dataloader: {len(train_loader)} batches")
print(f"   ✓ Val dataloader:   {len(val_loader)} batches")

print("\n5. Loading first training batch...")
try:
    train_batch = next(iter(train_loader))
    if train_batch is None:
        print("   ✗ First train batch is None!")
    else:
        print(f"   ✓ Train batch loaded successfully")
        print(f"     Batch keys: {list(train_batch.keys())}")
        print(f"     gt_emission shape: {train_batch['gt_emission'].shape}")
        print(f"     albedo_map shape: {train_batch['albedo_map'].shape}")
        print(f"     mask_map shape: {train_batch['mask_map'].shape}")
except Exception as e:
    print(f"   ✗ Error loading train batch: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Loading first validation batch...")
try:
    val_batch = next(iter(val_loader))
    if val_batch is None:
        print("   ✗ First val batch is None!")
    else:
        print(f"   ✓ Val batch loaded successfully")
        print(f"     Batch keys: {list(val_batch.keys())}")
        print(f"     gt_emission shape: {val_batch['gt_emission'].shape}")
except Exception as e:
    print(f"   ✗ Error loading val batch: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Testing a few more batches...")
try:
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        if batch is not None:
            print(f"   ✓ Batch {i+1}: shape {batch['gt_emission'].shape}")
        else:
            print(f"   ✗ Batch {i+1}: None")
except Exception as e:
    print(f"   ✗ Error iterating batches: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Full dataset test completed!")
print(f"\nSummary:")
print(f"  Train: {len(dm.train_dataset)} samples in {len(train_loader)} batches")
print(f"  Val:   {len(dm.val_dataset)} samples in {len(val_loader)} batches")



