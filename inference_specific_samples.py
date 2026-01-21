#!/usr/bin/env python3
"""
Inference script for specific sample IDs using a trained checkpoint.
Generates input, ground truth, and predicted emission maps.

This script matches the validation process from training:
- Uses EMA weights if enabled in config
- Disables autocast during inference
- Respects data_normalization setting from config
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

# Add TEXGen to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_sample_index(sample_id, parquet_file):
    """Find the positional index of a sample ID in the parquet file."""
    df = pd.read_parquet(parquet_file)
    df_success = df[df['success'] == True].copy()
    
    if sample_id in df_success.index:
        position = list(df_success.index).index(sample_id)
        return position
    else:
        return None

def inference_samples(checkpoint_path, sample_ids, output_dir):
    """Run inference on specific samples and save results."""
    
    print("=" * 80)
    print("LightGen Inference for Specific Samples")
    print("=" * 80)
    
    # Load checkpoint and config
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    
    # First try to load config from checkpoint directory
    # The structure is: .../run_dir/ckpts/checkpoint.ckpt or .../run_dir/nested/ckpts/checkpoint.ckpt
    # Try both locations
    ckpt_path_obj = Path(checkpoint_path)
    config_path = ckpt_path_obj.parent.parent / "configs" / "parsed.yaml"
    if not config_path.exists():
        # Try going up one more level for nested structure
        config_path = ckpt_path_obj.parent.parent.parent / "configs" / "parsed.yaml"
    
    if config_path.exists():
        print(f"   Loading config from: {config_path}")
        full_cfg = OmegaConf.load(config_path)
        # Extract only the keys needed for the model (avoid metadata keys)
        cfg = OmegaConf.create({
            'data_cls': full_cfg.get('data_cls'),
            'data': full_cfg.get('data'),
            'system_cls': full_cfg.get('system_cls'),
            'system': full_cfg.get('system'),
            'trainer': full_cfg.get('trainer', {}),
            'checkpoint': full_cfg.get('checkpoint', {}),
        })
    else:
        print(f"   Warning: Config not found at {config_path}, using default")
        cfg = OmegaConf.load("configs/lightgen_pointuv_256_batch2.yaml")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"   ✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load data module
    print("\n2. Setting up data module...")
    from spuv.data.lightgen_uv import LightGenDataModule, LightGenDataset
    
    # Override to use full dataset (not filtered) to find all samples
    cfg_data = cfg.data.copy()
    # Use full dataset paths
    cfg_data.parquet_file = "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet"
    cfg_data.data_root = "/localhome/dya78/code/lightgen/data/baked_uv"
    print(f"   Using full dataset: {cfg_data.parquet_file}")
    
    # Don't apply train/val/test filters
    cfg_data.test_indices = None
    
    # Create a custom dataset that only loads specific samples
    data_module = LightGenDataModule(cfg_data)
    data_module.setup('test')
    
    # Find indices for our sample IDs
    parquet_file = cfg_data.parquet_file
    print(f"\n3. Finding sample indices in: {parquet_file}")
    
    sample_indices = {}
    for sample_id in sample_ids:
        idx = find_sample_index(sample_id, parquet_file)
        if idx is not None:
            sample_indices[sample_id] = idx
            print(f"   ✓ Found {sample_id} at index {idx}")
        else:
            print(f"   ✗ Sample {sample_id} not found in dataset")
    
    if not sample_indices:
        print("\nError: No valid samples found!")
        return
    
    # Load the model
    print("\n4. Loading model...")
    from spuv.systems.lightgen_system import LightGenSystem
    
    # Create model from config first (avoids config merge issues)
    # Pass only the system config, not the full config
    model = LightGenSystem(cfg.system)
    
    # Load weights from checkpoint
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   ✓ Model loaded on {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n5. Output directory: {output_dir}")
    
    # Check config settings
    use_ema = cfg.system.get('use_ema', True)
    val_with_ema = cfg.system.get('val_with_ema', True)
    data_normalization = cfg.system.get('data_normalization', True)
    
    print(f"\n6. Config settings:")
    print(f"   - use_ema: {use_ema}")
    print(f"   - val_with_ema: {val_with_ema}")
    print(f"   - data_normalization: {data_normalization}")
    
    # Run inference for each sample
    print("\n7. Running inference...")
    
    for sample_id, idx in sample_indices.items():
        print(f"\n   Processing: {sample_id} (index {idx})")
        
        # Get the sample
        dataset = data_module.test_dataset
        sample = dataset[idx]
        
        # Create batch (add batch dimension)
        batch = {}
        for key, value in sample.items():
            if key == 'thumbnail' and value is None:
                # Skip thumbnail if it's None - system will use albedo fallback
                continue
            elif isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(device)
            elif isinstance(value, dict):
                batch[key] = value
            elif key == 'thumbnail' and isinstance(value, torch.Tensor):
                # Thumbnail should be moved to device but keep its shape
                batch[key] = value.to(device)
            elif isinstance(value, (list, tuple)):
                batch[key] = [value]
            else:
                batch[key] = [value]
        
        # Run inference - matching validation process exactly
        with torch.no_grad():
            # Disable autocast (matching validation)
            with torch.cuda.amp.autocast(enabled=False):
                # Use EMA weights if enabled (matching validation)
                if use_ema and val_with_ema:
                    print(f"      Using EMA weights for inference")
                    with model.ema_scope("Inference with ema weights"):
                        texture_map_outputs = model.test_pipeline(batch)
                else:
                    print(f"      Using regular weights for inference")
                    texture_map_outputs = model.test_pipeline(batch)
            
            # Extract results
            pred_emission = texture_map_outputs["pred_x0"][0]  # [3, H, W]
            gt_emission = texture_map_outputs["gt_x0"][0]  # [3, H, W]
            mask_map = texture_map_outputs["mask_map"][0, 0]  # [H, W]
            
            # Get input (albedo map)
            albedo_map = batch["albedo_map"][0]  # [3, H, W]
            
            # Denormalize based on config (matching validation)
            if data_normalization:
                pred_emission = (pred_emission * 0.5 + 0.5).clamp(0, 1)
                gt_emission = (gt_emission * 0.5 + 0.5).clamp(0, 1)
            else:
                pred_emission = pred_emission.clamp(0, 1)
                gt_emission = gt_emission.clamp(0, 1)
            # albedo is already in [0, 1] range
            
            # Apply mask
            pred_emission = pred_emission * mask_map.unsqueeze(0)
            gt_emission = gt_emission * mask_map.unsqueeze(0)
            albedo_map = albedo_map * mask_map.unsqueeze(0)
            
            # Convert to numpy and transpose to HWC
            pred_img = pred_emission.cpu().permute(1, 2, 0).numpy()
            gt_img = gt_emission.cpu().permute(1, 2, 0).numpy()
            albedo_img = albedo_map.cpu().permute(1, 2, 0).numpy()
            mask_img = mask_map.cpu().numpy()
            
            # Save individual images
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(exist_ok=True)
            
            Image.fromarray((albedo_img * 255).astype(np.uint8)).save(sample_dir / "input_albedo.png")
            Image.fromarray((gt_img * 255).astype(np.uint8)).save(sample_dir / "gt_emission.png")
            Image.fromarray((pred_img * 255).astype(np.uint8)).save(sample_dir / "pred_emission.png")
            Image.fromarray((mask_img * 255).astype(np.uint8)).save(sample_dir / "mask.png")
            
            # Create comparison image (3 columns: input, gt, pred)
            h, w = pred_img.shape[:2]
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison[:, :w] = (albedo_img * 255).astype(np.uint8)
            comparison[:, w:2*w] = (gt_img * 255).astype(np.uint8)
            comparison[:, 2*w:] = (pred_img * 255).astype(np.uint8)
            
            Image.fromarray(comparison).save(sample_dir / "comparison.png")
            
            print(f"      ✓ Saved to {sample_dir}/")
            print(f"         - input_albedo.png")
            print(f"         - gt_emission.png")
            print(f"         - pred_emission.png")
            print(f"         - comparison.png (input | gt | pred)")
            
            # Compute metrics
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            
            # Expand dims for metrics
            pred_batch = pred_emission.unsqueeze(0)
            gt_batch = gt_emission.unsqueeze(0)
            mask_batch = mask_map.unsqueeze(0)
            
            # Compute metrics in masked regions
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            
            # Apply mask for metrics
            pred_masked = pred_batch * mask_batch.unsqueeze(1)
            gt_masked = gt_batch * mask_batch.unsqueeze(1)
            
            psnr = psnr_metric(pred_masked, gt_masked)
            ssim = ssim_metric(pred_masked, gt_masked)
            
            print(f"         PSNR: {psnr.item():.2f} dB")
            print(f"         SSIM: {ssim.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Inference Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Settings: EMA={use_ema and val_with_ema}, data_normalization={data_normalization}")
    print("=" * 80)


if __name__ == "__main__":
    checkpoint_path = "/localhome/dya78/code/lightgen/TEXGen/outputs/pointuv_256res_batch8@20260115-235332/ckpts/last.ckpt"
    
    sample_ids = [
        "c85c7a1dbb724e9ea1d90abd6445fad4",
        "5fa9a65e7b0141ee877ed18f4f42d953"
    ]
    
    output_dir = "inference_outputs"
    
    inference_samples(checkpoint_path, sample_ids, output_dir)
