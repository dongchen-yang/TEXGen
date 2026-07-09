#!/usr/bin/env python3
"""Verify that the correct sample is being loaded."""

import sys
sys.path.insert(0, '.')

from spuv.data.lightgen_uv import LightGenDataModule

# Config matching lightgen_pointuv_overfit.yaml
cfg_dict = {
    'data_root': "/localhome/dya78/code/lightgen/data/baked_uv",
    'parquet_file': "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet",
    'batch_size': 1,
    'num_workers': 0,
    'train_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/overfit_split.json",
    'val_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/overfit_split.json",
    'test_indices': "/localhome/dya78/code/lightgen/data_processing/annotation/overfit_split.json",
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

print("Creating datamodule...")
dm = LightGenDataModule(cfg_dict)
dm.setup("fit")

print(f"Train dataset size: {len(dm.train_dataset)} samples")

if len(dm.train_dataset) == 1:
    print("✓ Correct! Only 1 sample loaded")
    
    # Load the sample
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    sample_id = batch['scene_id'][0] if isinstance(batch['scene_id'], list) else batch['scene_id']
    print(f"\nSample ID: {sample_id}")
    
    if sample_id == "416f4870df6449dfaf9533be8aa18701":
        print("✅ SUCCESS! Correct sample is loaded!")
    else:
        print(f"✗ Wrong sample! Expected: 416f4870df6449dfaf9533be8aa18701")
else:
    print(f"✗ ERROR! Loading {len(dm.train_dataset)} samples instead of 1")

