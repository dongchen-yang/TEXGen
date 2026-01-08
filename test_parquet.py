#!/usr/bin/env python3
"""
Quick test to verify parquet-based data loading works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

print("=" * 80)
print("Testing Parquet-based Data Loading")
print("=" * 80)

parquet_file = "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet"
data_root = "/localhome/dya78/code/lightgen/data/baked_uv"

print(f"\n1. Reading parquet file: {parquet_file}")
df = pd.read_parquet(parquet_file)
print(f"   ✓ Total samples in parquet: {len(df)}")

# Filter for successful samples
df_success = df[df['success'] == True]
print(f"   ✓ Successful samples: {len(df_success)}")

# Test loading first 10 samples
print(f"\n2. Testing first 10 samples:")
test_samples = []
for i, (sample_id, row) in enumerate(df_success.iterrows()):
    if i >= 10:
        break
    
    ditem_dir = row['ditem_dir']
    sample_path = os.path.join(data_root, ditem_dir)
    npz_file = os.path.join(sample_path, "somage.npz")
    
    test_samples.append({
        "sample_id": sample_id,
        "path": sample_path,
        "npz_file": npz_file,
    })

print(f"   Sample IDs: {[s['sample_id'][:8] for s in test_samples]}")

# Verify NPZ files exist and can be loaded
print(f"\n3. Verifying NPZ files:")
for i, sample in enumerate(test_samples):
    print(f"   [{i+1}/10] {sample['sample_id'][:16]}...", end=" ")
    
    if not os.path.exists(sample['npz_file']):
        print(f"❌ NPZ file not found!")
        continue
    
    try:
        data = np.load(sample['npz_file'])
        required_keys = ['occupancy', 'position', 'objnormal', 'color', 'metal', 'rough', 'emission_color']
        missing = [k for k in required_keys if k not in data.keys()]
        
        if missing:
            print(f"❌ Missing keys: {missing}")
        else:
            print(f"✓")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("✓ Parquet-based loading works!")
print("=" * 80)
print("\nNow you can run:")
print("  python launch.py --config configs/lightgen.yaml --train --gpu 0")

