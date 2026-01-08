#!/usr/bin/env python3
"""
Simple test to verify we can load 10 samples from the dataset.
"""

import os
import sys
import numpy as np

print("Testing data loading with 10 samples...")

data_root = "/localhome/dya78/code/lightgen/data/baked_uv"

# Scan for samples
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
                "npz_file": npz_file,
            })

print(f"Found {len(samples)} total samples")

# Test loading first 10
test_samples = samples[:10]
print(f"\nTesting with {len(test_samples)} samples:")

for i, sample in enumerate(test_samples):
    print(f"\n[{i+1}/{len(test_samples)}] Loading {sample['sample_id']}...")
    
    try:
        data = np.load(sample['npz_file'])
        
        # Check required keys
        required_keys = ['occupancy', 'position', 'objnormal', 'color', 'metal', 'rough', 'emission_color']
        missing = [k for k in required_keys if k not in data.keys()]
        
        if missing:
            print(f"  ❌ Missing keys: {missing}")
            continue
        
        # Check shapes
        all_good = True
        for key in required_keys:
            expected_h, expected_w = 512, 512
            if data[key].shape[0] != expected_h or data[key].shape[1] != expected_w:
                print(f"  ❌ {key}: wrong shape {data[key].shape}")
                all_good = False
        
        if all_good:
            print(f"  ✓ All checks passed!")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*60)
print("Test complete! If all samples loaded successfully, you're ready to train.")
print("\nTo start training with 10 samples:")
print("  conda activate texgen")
print("  cd /localhome/dya78/code/lightgen/TEXGen")
print("  python launch.py --config configs/lightgen.yaml --train --gpu 0")

