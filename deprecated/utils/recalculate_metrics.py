#!/usr/bin/env python3
"""
Recalculate metrics only over masked regions.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

def calculate_psnr_masked(pred, gt, mask, data_range=1.0):
    """Calculate PSNR only over masked regions."""
    # Expand mask to match channels [C, H, W]
    mask_expanded = mask.unsqueeze(0).expand_as(pred)
    mask_bool = mask_expanded > 0.5
    
    if mask_bool.sum() == 0:
        return 0.0
    
    pred_masked = pred[mask_bool]
    gt_masked = gt[mask_bool]
    
    mse = torch.mean((pred_masked - gt_masked) ** 2)
    if mse == 0:
        return 100.0
    
    psnr = 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim_masked(pred, gt, mask):
    """Calculate SSIM only over masked regions (simplified)."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    
    # Apply mask
    pred_masked = pred * mask
    gt_masked = gt * mask
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim_metric(pred_masked.unsqueeze(0), gt_masked.unsqueeze(0))
    return ssim.item()

# Sample information
samples = [
    {
        'id': 'c85c7a1dbb724e9ea1d90abd6445fad4',
        'name': 'Sample 1',
        'split': 'UNSEEN',
        'split_detail': 'Not in train/val/test',
        'split_color': '#FF6B6B',
    },
    {
        'id': '5fa9a65e7b0141ee877ed18f4f42d953',
        'name': 'Sample 2',
        'split': 'TRAIN',
        'split_detail': 'Training set',
        'split_color': '#4CAF50',
    }
]

print("=" * 60)
print("Recalculating Metrics Over Masked Regions Only")
print("=" * 60)

for sample_info in samples:
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    print(f"\n{sample_info['name']} ({sample_info['split']}):")
    print(f"  ID: {sample_id}")
    
    # Load images
    gt = np.array(Image.open(base_path / 'gt_emission.png')).astype(np.float32) / 255.0
    pred = np.array(Image.open(base_path / 'pred_emission.png')).astype(np.float32) / 255.0
    mask = np.array(Image.open(base_path / 'mask.png')).astype(np.float32) / 255.0
    
    # Convert to torch tensors [C, H, W]
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1)
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1)
    mask_tensor = torch.from_numpy(mask)
    
    # Calculate metrics over masked region
    psnr_masked = calculate_psnr_masked(pred_tensor, gt_tensor, mask_tensor, data_range=1.0)
    ssim_masked = calculate_ssim_masked(pred_tensor, gt_tensor, mask_tensor)
    
    # Calculate metrics over full image (unmasked)
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric_full = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    psnr_full = psnr_metric(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
    ssim_full = ssim_metric_full(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
    
    # Update sample info with both metrics
    sample_info['psnr_masked'] = psnr_masked
    sample_info['ssim_masked'] = ssim_masked
    sample_info['psnr_full'] = psnr_full
    sample_info['ssim_full'] = ssim_full
    
    # Calculate percentage of masked area
    mask_coverage = (mask_tensor > 0.5).float().mean().item() * 100
    sample_info['mask_coverage'] = mask_coverage
    
    print(f"  Mask coverage: {mask_coverage:.1f}%")
    print(f"  PSNR (full image): {psnr_full:.2f} dB")
    print(f"  PSNR (masked only): {psnr_masked:.2f} dB")
    print(f"  SSIM (full image): {ssim_full:.4f}")
    print(f"  SSIM (masked only): {ssim_masked:.4f}")

# Save updated metrics
import json
metrics_file = 'inference_outputs/metrics_masked.json'
with open(metrics_file, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"\n✓ Updated metrics saved to: {metrics_file}")
print("\n" + "=" * 60)
