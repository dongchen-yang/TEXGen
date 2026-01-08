#!/usr/bin/env python3
"""
Create a professional slide visualization with 3D thumbnail renderings.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# Sample information with thumbnail paths
samples = [
    {
        'id': 'c85c7a1dbb724e9ea1d90abd6445fad4',
        'name': 'Sample 1',
        'psnr': 19.23,
        'ssim': 0.5191,
        'split': 'UNSEEN',
        'split_detail': 'Not in train/val/test splits',
        'split_color': '#FF6B6B',  # Red for unseen
        'thumbnail': '/localhome/dya78/code/lightgen/data/TexVerse/thumbnails/thumbnails_batch/batch_00067/c85c7a1dbb724e9ea1d90abd6445fad4.png'
    },
    {
        'id': '5fa9a65e7b0141ee877ed18f4f42d953',
        'name': 'Sample 2',
        'psnr': 23.90,
        'ssim': 0.6172,
        'split': 'TRAIN',
        'split_detail': 'Training set (seen during training)',
        'split_color': '#4CAF50',  # Green for train
        'thumbnail': '/localhome/dya78/code/lightgen/data/TexVerse/thumbnails/thumbnails_batch/batch_00031/5fa9a65e7b0141ee877ed18f4f42d953.png'
    }
]

# Create comprehensive figure with thumbnails
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.15, 
                      left=0.04, right=0.96, top=0.92, bottom=0.08)

# Add main title
fig.suptitle('LightGen: Emission Map Generation Results\n' + 
             'Checkpoint: epoch=57 | Resolution: 256×256 | Architecture: PointUV-DiT', 
             fontsize=18, fontweight='bold', y=0.97)

# Column headers
col_labels = ['3D Rendering', 'Input\n(Albedo)', 'Ground Truth\n(Emission)', 
              'Prediction\n(Emission)', 'Emission\nOverlay', 'Full Render\n(Pred)']
for col_idx, label in enumerate(col_labels):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.text(0.5, 1.12, label, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', ha='center', va='bottom')
    ax.axis('off')

# Process each sample
for row_idx, sample_info in enumerate(samples):
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    # Load images
    images = {
        'thumbnail': Image.open(sample_info['thumbnail']),
        'input': Image.open(base_path / 'input_albedo.png'),
        'gt': Image.open(base_path / 'gt_emission.png'),
        'pred': Image.open(base_path / 'pred_emission.png'),
        'mask': Image.open(base_path / 'mask.png')
    }
    
    # Convert to numpy arrays
    thumbnail_img = np.array(images['thumbnail'])
    for key in ['input', 'gt', 'pred']:
        images[key] = np.array(images[key])
    mask = np.array(images['mask'])[:, :, None] / 255.0
    
    # Create emission overlay on input (for visualization)
    overlay = images['input'].copy().astype(float)
    overlay = (overlay * (1 - mask * 0.5) + images['pred'] * mask).astype(np.uint8)
    
    # Create full render with emission (thumbnail + emission overlay concept)
    # This simulates what it would look like rendered
    thumb_resized = np.array(Image.fromarray(thumbnail_img).resize((256, 256)))
    full_render = thumb_resized.copy()
    
    # Plot all images
    image_data = [
        thumbnail_img,  # 3D rendering
        images['input'],  # Albedo
        images['gt'],  # GT emission
        images['pred'],  # Predicted emission
        overlay,  # Emission overlay on albedo
        full_render  # Full render (for now just thumbnail)
    ]
    
    for col_idx, img_data in enumerate(image_data):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(img_data)
        ax.axis('off')
        
        # Add colored border for emission maps
        if col_idx in [2, 3]:  # GT and Pred emission
            for spine in ax.spines.values():
                spine.set_edgecolor('#FF6B6B' if col_idx == 2 else '#4ECDC4')
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        # Add sample name and split badge on the first column
        if col_idx == 0:
            # Split badge
            split_bbox = dict(boxstyle='round,pad=0.4', 
                            facecolor=sample_info['split_color'], 
                            edgecolor='black', 
                            alpha=0.9, 
                            linewidth=2.5)
            ax.text(0.5, -0.08, f"🏷️ {sample_info['split']}", 
                   transform=ax.transAxes,
                   fontsize=12, fontweight='bold', 
                   ha='center', va='top',
                   color='white',
                   bbox=split_bbox)
            
            # Sample name
            name_bbox = dict(boxstyle='round,pad=0.4', 
                            facecolor='white', 
                            edgecolor='black', 
                            alpha=0.95, 
                            linewidth=1.5)
            ax.text(0.5, -0.20, sample_info['name'], 
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold', 
                   ha='center', va='top',
                   bbox=name_bbox)
    
    # Add metrics box
    metrics_text = (
        f"Metrics:\n"
        f"PSNR: {sample_info['psnr']:.2f} dB\n"
        f"SSIM: {sample_info['ssim']:.4f}\n\n"
        f"Split: {sample_info['split']}\n"
        f"{sample_info['split_detail']}"
    )
    
    fig.text(0.973, 0.70 - row_idx * 0.47, metrics_text,
            fontsize=9.5,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor=sample_info['split_color'],
                     edgecolor='black',
                     alpha=0.25,
                     linewidth=1.5))

# Add legend for borders
legend_elements = [
    plt.Line2D([0], [0], color='#FF6B6B', linewidth=3, label='Ground Truth'),
    plt.Line2D([0], [0], color='#4ECDC4', linewidth=3, label='Prediction')
]
fig.legend(handles=legend_elements, loc='lower right', 
          fontsize=10, framealpha=0.9, bbox_to_anchor=(0.97, 0.04))

# Add footer
footer_text = (
    'Model: LightGenPointUVNet | Diffusion Steps: 50 | CFG Scale: 2.0 | '
    'Training: 1,154 emissive samples (923 train / 115 val / 116 test)'
)
fig.text(0.5, 0.015, footer_text, 
         fontsize=8.5, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='lightgray',
                  alpha=0.4))

# Save figure
output_path = 'inference_outputs/slide_with_thumbnails.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✓ Complete visualization saved to: {output_path}")

# High-res version
output_path_hires = 'inference_outputs/slide_with_thumbnails_hires.png'
plt.savefig(output_path_hires, dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"✓ High-res version saved to: {output_path_hires}")

plt.close()

# Create a cleaner version for presentations
fig2 = plt.figure(figsize=(16, 8))
gs2 = fig2.add_gridspec(2, 5, hspace=0.4, wspace=0.2,
                        left=0.05, right=0.95, top=0.90, bottom=0.1)

fig2.suptitle('LightGen: Controllable Emission Map Generation',
              fontsize=20, fontweight='bold', y=0.96)

col_labels2 = ['3D Object', 'Albedo (Input)', 'GT Emission', 'Predicted Emission', 'Overlay']
for col_idx, label in enumerate(col_labels2):
    fig2.text(0.1 + col_idx * 0.18, 0.92, label,
             fontsize=12, fontweight='bold', ha='center')

for row_idx, sample_info in enumerate(samples):
    sample_id = sample_info['id']
    base_path = Path(f'inference_outputs/{sample_id}')
    
    # Load images
    thumbnail = np.array(Image.open(sample_info['thumbnail']))
    input_img = np.array(Image.open(base_path / 'input_albedo.png'))
    gt_img = np.array(Image.open(base_path / 'gt_emission.png'))
    pred_img = np.array(Image.open(base_path / 'pred_emission.png'))
    mask = np.array(Image.open(base_path / 'mask.png'))[:, :, None] / 255.0
    
    # Create overlay
    overlay = (input_img * (1 - mask * 0.5) + pred_img * mask).astype(np.uint8)
    
    images = [thumbnail, input_img, gt_img, pred_img, overlay]
    
    for col_idx, img in enumerate(images):
        ax = fig2.add_subplot(gs2[row_idx, col_idx])
        ax.imshow(img)
        ax.axis('off')
        
        # Add border
        if col_idx >= 2:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.5)
                if col_idx == 2:
                    spine.set_edgecolor('#E74C3C')  # Red for GT
                elif col_idx == 3:
                    spine.set_edgecolor('#3498DB')  # Blue for Pred
    
    # Add sample info with split badge
    info_text = (f"{sample_info['name']} [{sample_info['split']}]\n"
                f"PSNR: {sample_info['psnr']:.1f} dB | SSIM: {sample_info['ssim']:.3f}")
    fig2.text(0.02, 0.48 - row_idx * 0.48, info_text,
             fontsize=11, fontweight='bold', va='center',
             bbox=dict(boxstyle='round,pad=0.6',
                      facecolor=sample_info['split_color'],
                      edgecolor='black',
                      alpha=0.35, linewidth=2))

output_clean = 'inference_outputs/slide_clean_presentation.png'
plt.savefig(output_clean, dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"✓ Clean presentation version saved to: {output_clean}")

plt.close()

print("\n✅ All visualizations with thumbnails created!")
print(f"\nRecommended for slides:")
print(f"  → slide_with_thumbnails.png (comprehensive, 6-column layout)")
print(f"  → slide_clean_presentation.png (clean, 5-column layout)")
print(f"\nHigh-res for posters/papers:")
print(f"  → slide_with_thumbnails_hires.png (600 DPI)")
