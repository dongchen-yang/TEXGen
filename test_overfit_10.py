#!/usr/bin/env python3
"""
Test overfitting on 10 samples.
This script verifies the setup and monitors overfitting progress.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("LightGen Overfitting Test - 10 Samples")
print("=" * 80)
print()

from spuv.data.lightgen_uv import LightGenDataModule
import torch

# Create config for 10 samples
cfg_dict = {
    'data_root': "../data/baked_uv_local",
    'parquet_file': "../data/baked_uv_local/df_SomgProc_filtered.parquet",
    'batch_size': 2,
    'num_workers': 0,  # 0 for debugging
    'train_indices': [0, 10],  # 10 samples
    'val_indices': [0, 10],    # Same 10 samples
    'test_indices': [0, 10],   # Same 10 samples
    'cond_views': 1,
    'sup_views': 4,
    'camera_strategy': 'strategy_1',
    'eval_camera_strategy': 'strategy_test_1_to_4_90deg',
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

print("1. Creating data module...")
try:
    dm = LightGenDataModule(cfg_dict)
    print("   ✓ Data module created successfully")
except Exception as e:
    print(f"   ✗ Error creating data module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Setting up datasets...")
try:
    dm.setup("fit")
    print(f"   ✓ Train dataset: {len(dm.train_dataset)} samples")
    print(f"   ✓ Val dataset:   {len(dm.val_dataset)} samples")
    
    # Verify same samples
    if len(dm.train_dataset) == len(dm.val_dataset):
        print(f"   ✓ Train and Val have same number of samples (for overfitting)")
    else:
        print(f"   ⚠ WARNING: Different sizes - train: {len(dm.train_dataset)}, val: {len(dm.val_dataset)}")
        
except Exception as e:
    print(f"   ✗ Error setting up datasets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Creating dataloaders...")
try:
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"   ✓ Train dataloader: {len(train_loader)} batches")
    print(f"   ✓ Val dataloader:   {len(val_loader)} batches")
except Exception as e:
    print(f"   ✗ Error creating dataloaders: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Loading first training batch...")
try:
    train_batch = next(iter(train_loader))
    print("   ✓ Batch loaded successfully")
    print(f"   - Batch size: {len(train_batch['scene_id'])}")
    print(f"   - Scene IDs: {train_batch['scene_id']}")
    print(f"   - GT emission shape: {train_batch['gt_emission'].shape}")
    print(f"   - Albedo map shape: {train_batch['albedo_map'].shape}")
    print(f"   - Normal map shape: {train_batch['normal_map'].shape}")
    print(f"   - Position map shape: {train_batch['position_map'].shape}")
    print(f"   - Mask map shape: {train_batch['mask_map'].shape}")
    
    # Check data ranges
    print(f"\n   Data ranges:")
    print(f"   - GT emission: [{train_batch['gt_emission'].min():.3f}, {train_batch['gt_emission'].max():.3f}]")
    print(f"   - Albedo: [{train_batch['albedo_map'].min():.3f}, {train_batch['albedo_map'].max():.3f}]")
    print(f"   - Normal: [{train_batch['normal_map'].min():.3f}, {train_batch['normal_map'].max():.3f}]")
    
except Exception as e:
    print(f"   ✗ Error loading batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Loading first validation batch...")
try:
    val_batch = next(iter(val_loader))
    print("   ✓ Validation batch loaded successfully")
    print(f"   - Batch size: {len(val_batch['scene_id'])}")
    print(f"   - Scene IDs: {val_batch['scene_id']}")
    
    # Check if validation uses same samples as training (for overfitting)
    train_ids = set(train_batch['scene_id'])
    val_ids = set(val_batch['scene_id'])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"   ✓ Validation samples overlap with training (good for overfitting)")
        print(f"     Overlapping IDs: {overlap}")
    else:
        print(f"   ⚠ No overlap between train and val (might not overfit as expected)")
        
except Exception as e:
    print(f"   ✗ Error loading validation batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All checks passed! Ready to overfit on 10 samples")
print("=" * 80)
print("\nTo start training:")
print("  python launch.py --config configs/lightgen_overfit_10.yaml --train")
print("\nExpected behavior:")
print("  - Training loss should decrease to near 0")
print("  - Validation loss should also be very low (same samples)")
print("  - Model should perfectly memorize these 10 samples")
print("\nMonitor with:")
print("  tensorboard --logdir outputs/lightgen/overfit_10_samples")
print()






