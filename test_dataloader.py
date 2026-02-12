#!/usr/bin/env python3
"""Test script to verify data loading works correctly."""

import sys
import yaml
from pathlib import Path
import torch

def test_dataloader(config_path):
    """Test data loading from config file."""
    print(f"Loading config from: {config_path}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("\n=== Config Summary ===")
    print(f"Name: {config['name']}")
    print(f"Tag: {config['tag']}")
    print(f"Data root: {config['data']['data_root']}")
    print(f"Parquet file: {config['data']['parquet_file']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Train indices: {config['data']['train_indices']}")
    
    # Import data module
    sys.path.insert(0, str(Path(__file__).parent))
    from spuv.data.lightgen_uv import LightGenDataModule
    
    print("\n=== Initializing Data Module ===")
    data_module = LightGenDataModule(config['data'])
    
    print("Calling setup('fit')...")
    data_module.setup('fit')
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    
    # Create dataloaders
    print("\n=== Creating Train Dataloader ===")
    train_loader = data_module.train_dataloader()
    print(f"Train loader batches: {len(train_loader)}")
    
    print("\n=== Testing First Batch ===")
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  Batch keys: {batch.keys()}")
        
        # Print shapes of all tensors in batch
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: (dict with {len(value)} items)")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
        
        # Test a few more batches
        if i >= 2:
            print(f"\n✓ Successfully loaded {i+1} batches from train set")
            break
    
    print("\n=== Testing Val Dataloader ===")
    val_loader = data_module.val_dataloader()
    print(f"Val loader batches: {len(val_loader)}")
    
    for i, batch in enumerate(val_loader):
        print(f"Val batch {i}: {list(batch.keys())}")
        if i >= 1:
            print(f"✓ Successfully loaded {i+1} batches from val set")
            break
    
    print("\n=== Data Loading Test Successful! ===")
    return True

if __name__ == "__main__":
    config_path = "configs/lightgen_pointuv_256_batch32_unfiltered.yaml"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        test_dataloader(config_path)
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
