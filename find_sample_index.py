#!/usr/bin/env python3
"""Find the index of a specific sample ID in the parquet file."""

import pandas as pd

# Load parquet
parquet_file = "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet"
df = pd.read_parquet(parquet_file)

# Filter for successful samples
df_success = df[df['success'] == True].copy()

# The sample ID we're looking for
target_id = "416f4870df6449dfaf9533be8aa18701"

print(f"Total samples in parquet: {len(df)}")
print(f"Successful samples: {len(df_success)}")
print(f"\nSearching for sample ID: {target_id}")

# Check if it's in the index
if target_id in df_success.index:
    # Get positional index (for dataloader)
    position = list(df_success.index).index(target_id)
    print(f"✓ Found in success-filtered dataframe!")
    print(f"  Positional index: {position}")
    print(f"  For config, use: train_indices: !!python/tuple [{position}, {position + 1}]")
else:
    print(f"✗ Not found in success-filtered samples")
    
    # Check if it exists in full dataframe
    if target_id in df.index:
        print(f"  Note: Sample exists but success={df.loc[target_id, 'success']}")
    else:
        print(f"  Sample does not exist in parquet at all")



