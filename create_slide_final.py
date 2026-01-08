#!/usr/bin/env python3
"""
Create a clean slide visualization without heading and overlay.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# Sample information - BOTH masked and full metrics
samples = [
    {
        'id': 'c85c7a1dbb724e9ea1d90abd6445fad4',
        'name': 'Sample 1',
        'psnr_masked': 16.12,
        'psnr_full': 19.23,
        'ssim_masked': 0.5344,
        'ssim_full': 0.5344,
        'mask_coverage': 48.9,
        'split': 'UNSEEN',
        'split_detail': 'Not in train/val/test',
        'split_color': '#FF6B6B',
        'thumbnail': '/localhome/dya78/code/lightgen/data/TexVerse/thumbnails/thumbnails_batch/batch_00067/c85c7a1dbb724e9ea1d90abd6445fad4.png'
    },
    {
        'id': '5fa9a65e7b0141ee877ed18f4f42d953',
        'name': 'Sample 2',
        'psnr_masked': 20.50,
        'psnr_full': 23.89,
        'ssim_masked': 0.6336,
        'ssim_full': 0.6336,
        'mask_coverage': 45.8,
        'split': 'TRAIN',
        'split_detail': 'Training set',
        'split_color': '#4CAF50',
        'thumbnail': '/localhome/dya78/code/lightgen/data/TexVerse/thumbnails/thumbnails_batch/batch_00031/5fa9a65e7b0141ee877ed18f4f42d953.png'
    }
]

# Create figure - 5 columns (added mask)
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.18, 
                      left=0.05, right=0.88, top=0.95, bottom=0.08)

# Column headers (no main title)
col_labels = ['Thumbnail\n(SketchFab)', 'Input\n(Albedo)', 'UV Mask', 'Ground Truth\n(Emission)', 'Prediction\n(Emission)']
for col_idx, label in enumerate(col_labels):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.text(0.5, 1.08, label, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', ha='center', va='bottom')
    ax.axis('off')

# Process each sample
for row_idx, sample_info in enumerate(samples):
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    # Load images
    images = {
        'thumbnail': Image.open(sample_info['thumbnail']),
        'input': Image.open(base_path / 'input_albedo.png'),
        'mask': Image.open(base_path / 'mask.png'),
        'gt': Image.open(base_path / 'gt_emission.png'),
        'pred': Image.open(base_path / 'pred_emission.png'),
    }
    
    # Convert to numpy arrays
    thumbnail_img = np.array(images['thumbnail'])
    for key in ['input', 'mask', 'gt', 'pred']:
        images[key] = np.array(images[key])
    
    # Convert mask to RGB for visualization (white=valid, black=invalid)
    mask_vis = np.stack([images['mask']] * 3, axis=-1)
    
    # Plot images (with mask)
    image_data = [
        thumbnail_img,
        images['input'],
        mask_vis,
        images['gt'],
        images['pred']
    ]
    
    for col_idx, img_data in enumerate(image_data):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(img_data)
        ax.axis('off')
        
        # Add colored border for mask and emission maps
        if col_idx == 2:  # Mask - green border
            for spine in ax.spines.values():
                spine.set_edgecolor('#2ECC71')
                spine.set_linewidth(3)
                spine.set_visible(True)
        elif col_idx in [3, 4]:  # GT and Pred emission
            for spine in ax.spines.values():
                spine.set_edgecolor('#E74C3C' if col_idx == 3 else '#3498DB')
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        # Add split badge on first column
        if col_idx == 0:
            split_bbox = dict(boxstyle='round,pad=0.4', 
                            facecolor=sample_info['split_color'], 
                            edgecolor='black', 
                            alpha=0.9, 
                            linewidth=2.5)
            ax.text(0.5, -0.10, sample_info['split'], 
                   transform=ax.transAxes,
                   fontsize=12, fontweight='bold', 
                   ha='center', va='top',
                   color='white',
                   bbox=split_bbox)
    
    # Add metrics on the right - show both PSNR values only
    metrics_text = (
        f"PSNR (masked):\n"
        f"{sample_info['psnr_masked']:.2f} dB\n\n"
        f"PSNR (full):\n"
        f"{sample_info['psnr_full']:.2f} dB\n\n"
        f"{sample_info['split_detail']}"
    )
    
    fig.text(0.90, 0.72 - row_idx * 0.47, metrics_text,
            fontsize=10.5,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor=sample_info['split_color'],
                     edgecolor='black',
                     alpha=0.25,
                     linewidth=1.5))

# Add footnote about metrics
fig.text(0.5, 0.02, 'PSNR shown for both masked regions (valid UV area only) and full image (including background)',
         fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='lightgray',
                  alpha=0.5))

# Save
output_path = 'inference_outputs/slide_final_clean.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✓ Clean slide (no heading, no overlay) saved to: {output_path}")

# High-res version
output_hires = 'inference_outputs/slide_final_clean_hires.png'
plt.savefig(output_hires, dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"✓ High-res version saved to: {output_hires}")

plt.close()

print("\n✅ Final clean visualizations created!")
print(f"\nFiles:")
print(f"  → slide_final_clean.png (300 DPI)")
print(f"  → slide_final_clean_hires.png (600 DPI)")
print(f"\nLayout: 5 columns (3D | Albedo | Mask | GT | Prediction)")
print(f"Features:")
print(f"  - UV mask shown (white = valid region)")
print(f"  - Metrics computed over masked regions only")
print(f"  - No title, clean and minimal")
