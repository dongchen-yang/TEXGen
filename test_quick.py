#!/usr/bin/env python3
"""
Quick test of the full data pipeline with 10 samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing LightGen data pipeline with 10 samples...\n")

from spuv.data.lightgen_uv import LightGenDataModule

# Create config
cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv_local",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet",
    'batch_size': 2,
    'num_workers': 0,  # Use 0 for debugging
    'train_indices': [0, 6],
    'val_indices': [6, 8],
    'test_indices': [8, 10],
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

print("3. Creating train dataloader...")
train_loader = dm.train_dataloader()
print(f"   ✓ Train dataloader has {len(train_loader)} batches")

print("4. Loading first batch...")
try:
    batch = next(iter(train_loader))
    
    if batch is None:
        print("   ❌ Batch is None!")
        sys.exit(1)
    
    print(f"   ✓ Batch loaded successfully!")
    print(f"\n   Batch contents:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"      {key}: list of {len(value)} items")
        else:
            print(f"      {key}: {type(value)}")
    
    # Check specific tensors
    print(f"\n   Value ranges:")
    print(f"      position_map: [{batch['position_map'].min():.3f}, {batch['position_map'].max():.3f}]")
    print(f"      albedo_map: [{batch['albedo_map'].min():.3f}, {batch['albedo_map'].max():.3f}]")
    print(f"      gt_emission: [{batch['gt_emission'].min():.3f}, {batch['gt_emission'].max():.3f}]")
    
    print("\n" + "=" * 80)
    print("✓ SUCCESS! Data pipeline is working correctly!")
    print("=" * 80)
    print("\nYou can now start training:")
    print("  cd /localhome/dya78/code/lightgen/TEXGen")
    print("  python launch.py --config configs/lightgen.yaml --train --gpu 0")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

