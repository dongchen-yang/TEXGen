#!/usr/bin/env python3
"""
Test script for LightGen data loader.
This script verifies that the data loading works correctly.
"""

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from spuv.data.lightgen_uv import LightGenDataModule, LightGenDataModuleConfig


def test_data_scanning():
    """Test if we can find all data samples"""
    print("=" * 80)
    print("Testing data directory scanning...")
    print("=" * 80)
    
    data_root = "/localhome/dya78/code/lightgen/data/baked_uv"
    
    if not os.path.exists(data_root):
        print(f"ERROR: Data root does not exist: {data_root}")
        return False
    
    samples = []
    for subdir in sorted(os.listdir(data_root)):
        subdir_path = os.path.join(data_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        for sample_id in sorted(os.listdir(subdir_path)):
            sample_path = os.path.join(subdir_path, sample_id)
            if not os.path.isdir(sample_path):
                continue
                
            npz_file = os.path.join(sample_path, "somage.npz")
            if os.path.exists(npz_file):
                samples.append({
                    "sample_id": sample_id,
                    "path": sample_path,
                    "npz_file": npz_file,
                })
    
    print(f"Found {len(samples)} samples")
    if len(samples) > 0:
        print(f"First sample: {samples[0]['sample_id']}")
        print(f"Last sample: {samples[-1]['sample_id']}")
    else:
        print("WARNING: No samples found!")
        return False
    
    return True


def test_npz_loading():
    """Test loading a single NPZ file"""
    print("\n" + "=" * 80)
    print("Testing NPZ file loading...")
    print("=" * 80)
    
    data_root = "/localhome/dya78/code/lightgen/data/baked_uv"
    sample_path = os.path.join(data_root, "000-000", "00a0e4b937404e8aa28dd3daaf480edd")
    npz_file = os.path.join(sample_path, "somage.npz")
    
    if not os.path.exists(npz_file):
        print(f"ERROR: Sample NPZ file not found: {npz_file}")
        return False
    
    try:
        data = np.load(npz_file)
        print(f"Successfully loaded NPZ file: {npz_file}")
        print(f"Keys: {list(data.keys())[:10]}...")  # Show first 10 keys
        
        # Check required keys
        required_keys = ['occupancy', 'position', 'objnormal', 'color', 'metal', 'rough', 'emission_color']
        missing_keys = [k for k in required_keys if k not in data.keys()]
        
        if missing_keys:
            print(f"ERROR: Missing required keys: {missing_keys}")
            return False
        
        # Check shapes
        print("\nData shapes:")
        for key in required_keys:
            print(f"  {key}: {data[key].shape}, dtype={data[key].dtype}")
        
        # Check if shapes are correct
        expected_shape = (512, 512)
        for key in required_keys:
            if data[key].shape[:2] != expected_shape:
                print(f"ERROR: {key} has wrong spatial dimensions: {data[key].shape[:2]}, expected {expected_shape}")
                return False
        
        print("\nAll data shapes are correct!")
        return True
        
    except Exception as e:
        print(f"ERROR loading NPZ file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test the LightGen dataset"""
    print("\n" + "=" * 80)
    print("Testing LightGen dataset...")
    print("=" * 80)
    
    try:
        # Create config
        cfg = LightGenDataModuleConfig(
            data_root="/localhome/dya78/code/lightgen/data/baked_uv",
            batch_size=2,
            num_workers=0,  # Use 0 for debugging
            train_indices=(0, 5),
            val_indices=(5, 7),
            test_indices=(7, 10),
        )
        
        # Create dataset
        from spuv.data.lightgen_uv import LightGenDataset
        dataset = LightGenDataset(cfg, split="train")
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty!")
            return False
        
        # Load first sample
        print("\nLoading first sample...")
        sample = dataset[0]
        
        if sample is None:
            print("ERROR: First sample is None!")
            return False
        
        print("Sample keys:", list(sample.keys()))
        
        # Check tensor shapes
        print("\nTensor shapes:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, dtype={value.dtype}")
        
        # Check specific fields
        expected_tensors = {
            'input_tensor': (11, 512, 512),
            'position_map': (3, 512, 512),
            'normal_map': (3, 512, 512),
            'albedo_map': (3, 512, 512),
            'metal_map': (1, 512, 512),
            'rough_map': (1, 512, 512),
            'mask_map': (1, 512, 512),
            'gt_emission': (3, 512, 512),
        }
        
        for key, expected_shape in expected_tensors.items():
            if key not in sample:
                print(f"ERROR: Missing key: {key}")
                return False
            if sample[key].shape != expected_shape:
                print(f"ERROR: {key} has wrong shape: {sample[key].shape}, expected {expected_shape}")
                return False
        
        print("\nAll tensor shapes are correct!")
        
        # Check value ranges
        print("\nValue ranges:")
        print(f"  position_map: [{sample['position_map'].min():.3f}, {sample['position_map'].max():.3f}]")
        print(f"  normal_map: [{sample['normal_map'].min():.3f}, {sample['normal_map'].max():.3f}]")
        print(f"  albedo_map: [{sample['albedo_map'].min():.3f}, {sample['albedo_map'].max():.3f}]")
        print(f"  metal_map: [{sample['metal_map'].min():.3f}, {sample['metal_map'].max():.3f}]")
        print(f"  rough_map: [{sample['rough_map'].min():.3f}, {sample['rough_map'].max():.3f}]")
        print(f"  gt_emission: [{sample['gt_emission'].min():.3f}, {sample['gt_emission'].max():.3f}]")
        print(f"  mask_map: [{sample['mask_map'].min():.3f}, {sample['mask_map'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"ERROR testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datamodule():
    """Test the LightGen data module with DataLoader"""
    print("\n" + "=" * 80)
    print("Testing LightGen data module with DataLoader...")
    print("=" * 80)
    
    try:
        # Create config
        cfg_dict = {
            'data_root': "/localhome/dya78/code/lightgen/data/baked_uv",
            'batch_size': 2,
            'num_workers': 0,
            'train_indices': [0, 5],
            'val_indices': [5, 7],
            'test_indices': [7, 10],
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
        }
        
        # Create data module
        dm = LightGenDataModule(cfg_dict)
        dm.setup("fit")
        
        # Get train dataloader
        train_loader = dm.train_dataloader()
        print(f"Train dataloader created with {len(train_loader)} batches")
        
        # Load first batch
        print("\nLoading first batch...")
        batch = next(iter(train_loader))
        
        if batch is None:
            print("ERROR: First batch is None!")
            return False
        
        print("Batch keys:", list(batch.keys()))
        
        # Check batch tensor shapes
        print("\nBatch tensor shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, dtype={value.dtype}")
        
        # Expected batch shapes (batch_size=2)
        expected_batch_tensors = {
            'input_tensor': (2, 11, 512, 512),
            'position_map': (2, 3, 512, 512),
            'normal_map': (2, 3, 512, 512),
            'albedo_map': (2, 3, 512, 512),
            'metal_map': (2, 1, 512, 512),
            'rough_map': (2, 1, 512, 512),
            'mask_map': (2, 1, 512, 512),
            'gt_emission': (2, 3, 512, 512),
        }
        
        for key, expected_shape in expected_batch_tensors.items():
            if key not in batch:
                print(f"ERROR: Missing batch key: {key}")
                return False
            if batch[key].shape != expected_shape:
                print(f"ERROR: {key} has wrong batch shape: {batch[key].shape}, expected {expected_shape}")
                return False
        
        print("\nAll batch shapes are correct!")
        return True
        
    except Exception as e:
        print(f"ERROR testing data module: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("LightGen Data Loader Test Suite")
    print("=" * 80 + "\n")
    
    tests = [
        ("Data Scanning", test_data_scanning),
        ("NPZ Loading", test_npz_loading),
        ("Dataset", test_dataset),
        ("DataModule", test_datamodule),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

