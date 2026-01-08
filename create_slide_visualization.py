#!/usr/bin/env python3
"""
Create a professional slide visualization for LightGen inference results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# Sample information
samples = [
    {
        'id': 'c85c7a1dbb724e9ea1d90abd6445fad4',
        'name': 'Sample 1 (Unseen)',
        'psnr': 19.23,
        'ssim': 0.5191,
        'split': 'Not in train/val/test'
    },
    {
        'id': '5fa9a65e7b0141ee877ed18f4f42d953',
        'name': 'Sample 2 (Training)',
        'psnr': 23.90,
        'ssim': 0.6172,
        'split': 'Training set'
    }
]

# Create figure
fig = plt.figure(figsize=(16, 9))  # 16:9 aspect ratio for slides
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.15, 
                      left=0.05, right=0.95, top=0.92, bottom=0.08)

# Add main title
fig.suptitle('LightGen: Emission Map Generation Results\n' + 
             'Checkpoint: epoch=57 | Resolution: 256×256', 
             fontsize=18, fontweight='bold', y=0.97)

# Column headers
col_labels = ['Input (Albedo)', 'Ground Truth', 'Prediction', 'Overlay']
for col_idx, label in enumerate(col_labels):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.text(0.5, 1.15, label, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', ha='center')
    ax.axis('off')

# Process each sample
for row_idx, sample_info in enumerate(samples):
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    # Load images
    images = {
        'input': Image.open(base_path / 'input_albedo.png'),
        'gt': Image.open(base_path / 'gt_emission.png'),
        'pred': Image.open(base_path / 'pred_emission.png'),
        'mask': Image.open(base_path / 'mask.png')
    }
    
    # Convert to numpy arrays
    for key in ['input', 'gt', 'pred']:
        images[key] = np.array(images[key])
    mask = np.array(images['mask'])[:, :, None] / 255.0
    
    # Create overlay (prediction over input)
    overlay = images['input'].copy()
    overlay = (overlay * (1 - mask * 0.7) + images['pred'] * mask * 0.7).astype(np.uint8)
    
    # Plot images
    image_data = [images['input'], images['gt'], images['pred'], overlay]
    
    for col_idx, (img_data, col_label) in enumerate(zip(image_data, col_labels)):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(img_data)
        ax.axis('off')
        
        # Add sample name on the first column
        if col_idx == 0:
            # Add background box for text
            bbox_props = dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            edgecolor='black', 
                            alpha=0.9, 
                            linewidth=2)
            ax.text(0.5, -0.12, sample_info['name'], 
                   transform=ax.transAxes,
                   fontsize=12, fontweight='bold', 
                   ha='center', va='top',
                   bbox=bbox_props)
    
    # Add metrics box on the right
    metrics_text = (
        f"PSNR: {sample_info['psnr']:.2f} dB\n"
        f"SSIM: {sample_info['ssim']:.4f}\n"
        f"Split: {sample_info['split']}"
    )
    
    # Determine color based on performance
    color = '#4CAF50' if sample_info['psnr'] > 20 else '#FF9800'
    
    fig.text(0.965, 0.73 - row_idx * 0.47, metrics_text,
            fontsize=10,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor=color,
                     edgecolor='black',
                     alpha=0.3,
                     linewidth=1.5))

# Add footer with model info
footer_text = (
    'Model: LightGenPointUVNet (epoch 57, step 6728) | '
    'Architecture: PointUV-DiT with 3D-aware processing | '
    'Inference: 50 diffusion steps with CFG scale 2.0'
)
fig.text(0.5, 0.02, footer_text, 
         fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='lightgray',
                  alpha=0.5))

# Save figure
output_path = 'inference_outputs/slide_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✓ Slide visualization saved to: {output_path}")

# Also save a high-res version for printing
output_path_hires = 'inference_outputs/slide_visualization_hires.png'
plt.savefig(output_path_hires, dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"✓ High-res version saved to: {output_path_hires}")

plt.close()

# Create a simpler comparison without overlay
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('LightGen: Emission Map Generation\n' +
              'Input Albedo → Ground Truth → Prediction',
              fontsize=18, fontweight='bold')

for row_idx, sample_info in enumerate(samples):
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    # Load images
    images = {
        'input': np.array(Image.open(base_path / 'input_albedo.png')),
        'gt': np.array(Image.open(base_path / 'gt_emission.png')),
        'pred': np.array(Image.open(base_path / 'pred_emission.png'))
    }
    
    for col_idx, (key, title) in enumerate([('input', 'Input'), ('gt', 'GT'), ('pred', 'Prediction')]):
        ax = axes[row_idx, col_idx]
        ax.imshow(images[key])
        ax.axis('off')
        
        if col_idx == 0:
            ax.set_title(f"{sample_info['name']}\n{title}", 
                        fontsize=12, fontweight='bold', pad=10)
        else:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add metrics text below the row
    metrics_text = f"PSNR: {sample_info['psnr']:.2f} dB | SSIM: {sample_info['ssim']:.4f}"
    fig2.text(0.5, 0.47 - row_idx * 0.47, metrics_text,
             fontsize=11, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='lightblue',
                      alpha=0.3))

plt.tight_layout()
simple_output = 'inference_outputs/slide_visualization_simple.png'
plt.savefig(simple_output, dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"✓ Simple visualization saved to: {simple_output}")

plt.close()

print("\n✅ All visualizations created successfully!")
print(f"\nFiles created:")
print(f"  1. slide_visualization.png (300 DPI, with overlay)")
print(f"  2. slide_visualization_hires.png (600 DPI, publication quality)")
print(f"  3. slide_visualization_simple.png (300 DPI, 3-column layout)")
