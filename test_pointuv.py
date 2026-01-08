#!/usr/bin/env python3
"""
Test the LightGenPointUVNet backbone to ensure it works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from spuv.models.sparse_networks.lightgen_pointuvnet import LightGenPointUVNet
from dataclasses import dataclass

print("Testing LightGenPointUVNet backbone...\n")

# Create model config
@dataclass
class TestConfig:
    in_channels: int = 10
    out_channels: int = 3
    num_layers: tuple = (1, 1, 1, 1, 1)
    point_block_num: tuple = (1, 1, 2, 4, 6)
    block_out_channels: tuple = (32, 256, 1024, 1024, 2048)
    dropout: tuple = (0.0, 0.0, 0.0, 0.1, 0.1)
    block_type: tuple = ("uv", "point_uv", "uv_dit", "uv_dit", "uv_dit")
    voxel_size: tuple = (0.01, 0.02, 0.05, 0.05, 0.05)
    window_size: tuple = (0, 256, 256, 512, 1024)
    num_heads: tuple = (4, 4, 16, 16, 16)
    skip_input: bool = True
    skip_type: str = "adaptive"
    use_uv_head: bool = True

config = TestConfig()

print("1. Creating LightGenPointUVNet model...")
try:
    model = LightGenPointUVNet(config)
    model = model.cuda()
    print(f"   ✓ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"   ✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating test inputs...")
batch_size = 2
H, W = 512, 512

# Create dummy inputs
x_dense = torch.randn(batch_size, 3, H, W).cuda()  # noisy emission
mask_map = torch.ones(batch_size, 1, H, W).cuda()  # occupancy mask
position_map = torch.randn(batch_size, 3, H, W).cuda() * 0.5  # 3D positions
timestep = torch.randint(0, 1000, (batch_size,)).cuda()  # timesteps

# CLIP embeddings (text and image)
clip_embeddings = [
    torch.randn(batch_size, 1024).cuda(),  # text embedding
    torch.randn(batch_size, 768).cuda(),   # image embedding
]

# Pre-baked material properties (already in UV space)
baked_texture = torch.randn(batch_size, 3, H, W).cuda() * 0.5 + 0.5  # albedo map
baked_weights = mask_map  # same as occupancy

image_info = {
    'mvp_mtx_cond': None,
    'rgb_cond': None,
    'baked_texture': baked_texture,  # Pre-baked material
    'baked_weights': baked_weights,  # Occupancy mask
}

mesh = None  # Not needed for LightGen
data_normalization = True
condition_drop = torch.zeros(batch_size).cuda()

print(f"   ✓ Created test inputs:")
print(f"     - x_dense: {x_dense.shape}")
print(f"     - mask_map: {mask_map.shape}")
print(f"     - position_map: {position_map.shape}")
print(f"     - baked_texture: {baked_texture.shape}")
print(f"     - baked_weights: {baked_weights.shape}")

print("\n3. Running forward pass...")
try:
    with torch.no_grad():
        output, addition_info = model(
            x_dense,
            mask_map,
            position_map,
            timestep,
            clip_embeddings,
            mesh,
            image_info,
            data_normalization,
            condition_drop,
        )
    print(f"   ✓ Forward pass successful!")
    print(f"     Output shape: {output.shape}")
    print(f"     Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"     Additional info keys: {list(addition_info.keys())}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Checking output properties...")
assert output.shape == (batch_size, 3, H, W), f"Wrong output shape: {output.shape}"
assert not torch.isnan(output).any(), "Output contains NaN!"
assert not torch.isinf(output).any(), "Output contains Inf!"
print(f"   ✓ Output shape and values are valid")

print("\n5. Testing backward pass...")
try:
    model.train()
    output, addition_info = model(
        x_dense,
        mask_map,
        position_map,
        timestep,
        clip_embeddings,
        mesh,
        image_info,
        data_normalization,
        condition_drop,
    )
    
    # Compute dummy loss and backward
    loss = output.mean()
    loss.backward()
    print(f"   ✓ Backward pass successful!")
    print(f"     Loss: {loss.item():.6f}")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. Memory usage:")
if torch.cuda.is_available():
    print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n✓ All tests passed! LightGenPointUVNet is working correctly.")
print("\nYou can now train with:")
print("  python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train")



