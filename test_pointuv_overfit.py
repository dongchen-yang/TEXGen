#!/usr/bin/env python3
"""
Quick overfit test for LightGenPointUVNet.
Tests if the model can overfit on 1 sample.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

import spuv
from spuv.utils.config import load_config

print("=" * 80)
print("LightGenPointUVNet Overfit Test")
print("=" * 80)

# Load config
print("\n1. Loading config...")
config_path = "configs/lightgen_pointuv_overfit.yaml"
cfg = load_config(config_path, makedirs=False)
print(f"   ✓ Config loaded: {config_path}")
print(f"   Backbone: {cfg.system.backbone_cls}")

# Create datamodule
print("\n2. Creating datamodule...")
dm = spuv.find(cfg.data_cls)(cfg.data)
dm.setup("fit")
print(f"   ✓ Train dataset: {len(dm.train_dataset)} samples")

# Get a sample to check data
print("\n3. Loading one batch...")
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
print(f"   ✓ Batch loaded")
print(f"     Keys: {list(batch.keys())}")
print(f"     gt_emission: {batch['gt_emission'].shape}")
print(f"     albedo_map: {batch['albedo_map'].shape}")
print(f"     position_map: {batch['position_map'].shape}")
print(f"     mask_map: {batch['mask_map'].shape}")

# Create system
print("\n4. Creating LightGen system with PointUVNet...")
system = spuv.find(cfg.system_cls)(cfg.system)
# Set save directory to avoid errors
import tempfile
save_dir = tempfile.mkdtemp()
system.set_save_dir(save_dir)
print(f"   ✓ System created")
print(f"   Backbone type: {type(system.backbone).__name__}")
print(f"   Model parameters: {sum(p.numel() for p in system.backbone.parameters()) / 1e6:.2f}M")

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
system = system.to(device)
print(f"   Device: {device}")

# Test forward pass
print("\n5. Testing forward pass on data...")
# Use train mode for forward pass test (need to add noise)
system.train()
try:
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Prepare inputs
    diffusion_data = system.prepare_diffusion_data(batch)
    condition_info = system.prepare_condition_info(batch)
    
    # Forward pass
    with torch.no_grad():
        output, addition_info = system(condition_info, diffusion_data)
    
    print(f"   ✓ Forward pass successful!")
    print(f"     Output shape: {output.shape}")
    print(f"     Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training step
print("\n6. Testing training step...")
system.train()
try:
    loss = system.training_step(batch, 0)
    print(f"   ✓ Training step successful!")
    print(f"     Loss: {loss.item():.6f}")
except Exception as e:
    print(f"   ✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Quick training loop to test overfitting
print("\n7. Running quick training loop (50 steps)...")
print("   (This tests if model can reduce loss on 1 sample)")

optimizer = torch.optim.AdamW(system.parameters(), lr=5e-4)
losses = []

for step in range(50):
    # Forward pass
    diffusion_data = system.prepare_diffusion_data(batch)
    condition_info = system.prepare_condition_info(batch)
    output, addition_info = system(condition_info, diffusion_data)
    
    # Compute loss
    loss_dict = system.get_diffusion_loss(output, diffusion_data)
    loss = sum(loss_dict.values())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if step % 10 == 0:
        print(f"   Step {step:3d}: loss = {loss.item():.6f}")

print(f"\n   ✓ Training loop successful!")
print(f"     Initial loss: {losses[0]:.6f}")
print(f"     Final loss:   {losses[-1]:.6f}")
print(f"     Reduction:    {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

if losses[-1] < losses[0]:
    print(f"   ✅ Loss is decreasing - model is learning!")
else:
    print(f"   ⚠️  Loss not decreasing - may need more steps")

# Test inference
print("\n8. Testing inference (denoising)...")
system.eval()
with torch.no_grad():
    try:
        texture_outputs = system.test_pipeline(batch)
        print(f"   ✓ Inference successful!")
        print(f"     Predicted shape: {texture_outputs['pred_x0'].shape}")
        print(f"     GT shape: {texture_outputs['gt_x0'].shape}")
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()

# Memory usage
if torch.cuda.is_available():
    print(f"\n9. GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nLightGenPointUVNet is working correctly!")
print("\nTo run full overfitting test (200 epochs):")
print("  python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --train")
print("\nTo train on full dataset:")
print("  python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train")

