from dataclasses import dataclass, field
import math
import random
from contextlib import contextmanager
import colorsys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import profiler
from torchvision.transforms import v2 as transform

import spuv
from spuv.utils.misc import get_device
from spuv.systems.texgen_test import TEXGenDiffusion
from spuv.systems.texgen_base import LossConfig
from spuv.utils.ops import binary_cross_entropy, get_plucker_rays
from spuv.utils.typing import *
from spuv.models.lpips import LPIPS
from spuv.utils.misc import time_recorder as tr
from spuv.utils.snr_utils import compute_snr_from_scheduler, get_weights_from_timesteps
from spuv.models.perceptual_loss import VGGPerceptualLoss
from spuv.models.renderers.rasterize import NVDiffRasterizerContext
from spuv.utils.mesh_utils import uv_padding
from spuv.utils.nvdiffrast_utils import *
from spuv.utils.lit_ema import LitEma
from spuv.utils.image_metrics import SSIM, PSNR


class LightGenSystem(TEXGenDiffusion):
    """
    LightGen System for generating emission maps from geometry and material properties.
    Inherits from texgen_test.TEXGenDiffusion to match original TEXGen exactly (with Flow Matching).
    
    Input: geometry (position, normal), material (albedo, roughness, metallic)
    Output: emission map in UV space
    """
    
    @dataclass
    class Config(TEXGenDiffusion.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()
        # Initialize image tokenizer for conditioning
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        
    def prepare_condition_info(self, batch):
        """
        Prepare conditioning information from batch.
        For LightGen, we condition on material properties and geometry.
        """
        # Extract material and geometry info
        albedo_map = batch['albedo_map']  # [B, 3, H, W]
        metal_map = batch['metal_map']  # [B, 1, H, W]
        rough_map = batch['rough_map']  # [B, 1, H, W]
        position_map = batch['position_map']  # [B, 3, H, W]
        normal_map = batch['normal_map']  # [B, 3, H, W]
        
        # Pack all material properties into a single tensor for conditioning
        # This gives us: normal(3) + albedo(3) + metal(1) + rough(1) = 8 channels
        material_cond = torch.cat([normal_map, albedo_map, metal_map, rough_map], dim=1)  # [B, 8, H, W]
        
        # Convert to expected format: [B, V, H, W, C]
        B, C, H, W = material_cond.shape
        rgb_cond = material_cond.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 8]
        
        # Create dummy prompt (not used but expected by tokenizer)
        if isinstance(batch['scene_id'], list):
            prompt = batch['scene_id']  # Use scene IDs as prompts
        else:
            prompt = ["emission generation"] * rgb_cond.shape[0]
        
        # Generate text embeddings
        text_embeddings = self.image_tokenizer.process_text(prompt).to(dtype=self.dtype)
        
        # Use pre-rendered thumbnail for CLIP image embedding (from batch)
        # These are pre-computed rendered views stored in processed_data/emissive_thumbnails/
        if 'thumbnail' in batch:
            rendered_thumbnail = batch['thumbnail']  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(rendered_thumbnail).to(dtype=self.dtype)
        else:
            # Fallback: use albedo UV map if thumbnail not available
            spuv.warn("Thumbnail not found in batch, using albedo UV map as fallback")
            albedo_for_clip = albedo_map.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(albedo_for_clip).to(dtype=self.dtype)
        
        condition_info = {
            'mesh': batch['mesh'],
            'mvp_mtx_cond': batch['mvp_mtx_cond'],
            'rgb_cond': rgb_cond,  # Contains all material properties [B, 1, H, W, 8]
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'prompt': prompt,
            'albedo_map': albedo_map,
            'metal_map': metal_map,
            'rough_map': rough_map,
            'normal_map': normal_map,
        }
        
        return condition_info
    
    def forward(self, condition: Dict[str, Any], diffusion_data: Dict[str, Any], condition_drop=None) -> Dict[str, Any]:
        """
        Override forward to work directly in UV space without 3D-to-UV baking.
        Since our data is already in UV space, we pass pre-baked material properties directly.
        
        This works with both SimpleUVUNet and LightGenPointUVNet.
        """
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        # Get CLIP embeddings (for PointUVNet compatibility)
        text_embeddings = condition.get("text_embeddings", None)
        image_embeddings = condition.get("image_embeddings", None)
        # PointUVNet expects clip_embeddings as a list [text, image]
        if text_embeddings is not None and image_embeddings is not None:
            clip_embeddings = [text_embeddings, image_embeddings]
        else:
            clip_embeddings = None
        
        mesh = condition.get("mesh", None)
        
        # Get material properties (already in UV space)
        albedo_map = condition.get("albedo_map")  # [B, 3, H, W]
        normal_map = condition.get("normal_map")  # [B, 3, H, W]
        metal_map = condition.get("metal_map")    # [B, 1, H, W]
        rough_map = condition.get("rough_map")    # [B, 1, H, W]
        
        # Prepare baked_texture: use albedo as the main material representation
        # You could also concatenate multiple channels: torch.cat([albedo, normal], dim=1)
        # For PointUVNet, baked_texture should match the expected input channels
        baked_texture = albedo_map  # [B, 3, H, W]
        baked_weights = mask_map     # [B, 1, H, W]
        
        # Prepare image info with conditioning
        image_info = {
            'mvp_mtx_cond': condition.get("mvp_mtx_cond"),
            'rgb_cond': condition.get("rgb_cond"),  # Contains all material properties [B, 1, H, W, 8]
            'baked_texture': baked_texture,  # For PointUVNet: pre-baked material in UV space
            'baked_weights': baked_weights,  # For PointUVNet: occupancy mask
        }

        # Get batch size from mask_map if input_tensor is None (eval mode)
        if input_tensor is not None:
            batch_size = input_tensor.shape[0]
            device = input_tensor.device
        else:
            batch_size = mask_map.shape[0]
            device = mask_map.device
        
        if condition_drop is None and self.training:
            condition_drop = torch.rand(batch_size, device=device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(batch_size, device=device)
        
        # Call the backbone (works with both SimpleUVUNet and LightGenPointUVNet)
        # For PointUVNet: pass clip_embeddings list
        # For SimpleUVUNet: it expects image_embeddings but doesn't use it, so clip_embeddings works too
        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,
           clip_embeddings,  # Changed from image_embeddings to support PointUVNet
           mesh,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info
    
    def prepare_diffusion_data(self, batch, noisy_images=None):
        """
        Prepare diffusion data from batch.
        Uses Flow Matching from parent class (matches original TEXGen).
        """
        device = get_device()
        B = batch['gt_emission'].shape[0]
        
        # Ground truth emission map (already normalized to [-1, 1])
        sample_images = batch['gt_emission']  # [B, 3, H, W]
        
        # Mask and position
        mask_map = batch['mask_map']  # [B, 1, H, W]
        position_map = batch['position_map']  # [B, 3, H, W]
        
        # Sample timesteps uniformly in [0, 1] with power transformation (like original TEXGen)
        uniform_samples = torch.rand(B, device=device)
        power = 2  # Skew towards smaller t (more noise)
        timesteps = uniform_samples ** power
        
        # Add noise using Flow Matching (inherited from parent)
        if noisy_images is not None:
            noisy_images = noisy_images.to(dtype=self.dtype)
        else:
            noise = torch.randn_like(sample_images, dtype=self.dtype)
            if sample_images is not None:
                noisy_images = self.get_conditional_flow(noise, sample_images, timesteps)
            else:
                noisy_images = noise
        
        noisy_images *= mask_map
        
        loss_weights = torch.ones_like(timesteps, device=device, dtype=self.dtype)
        
        diffusion_data = {
            'sample_images': sample_images,
            'noisy_images': noisy_images,
            'mask_map': mask_map,
            'position_map': position_map,
            'timesteps': timesteps,
            'noise': noise,
            'batch_loss_weights': loss_weights,
        }
        
        return diffusion_data
    
    def training_step(self, batch, batch_idx):
        """Training step for LightGen"""
        if batch is None:
            return None
        
        # Prepare data
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)
        
        # Forward pass
        out, addition_info = self(condition_info, diffusion_data)
        
        # Compute loss
        loss_dict = self.get_diffusion_loss(out, diffusion_data)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Total loss
        total_loss = sum(loss_dict.values())
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Store outputs for visualization
        outputs = {
            'texture_map_outputs': {
                'pred': out,
                'gt': diffusion_data['sample_images'],
            },
            'mask_map': diffusion_data['mask_map'],
            'render_out': None,
            'render_gt': None,
            'rgb_cond': condition_info['rgb_cond'],
        }
        
        # Visualization with wandb logging
        self.on_check_train(batch, outputs)
        
        # Log to wandb every N steps
        if hasattr(self, '_wandb_logger') and self._wandb_logger is not None:
            if self.global_step % 50 == 0:  # Log every 50 steps
                # Get the denoised prediction (x0) from the model output
                # Use the proper method to convert v-prediction/noise to x0
                pred_x0 = self.get_batched_pred_x0(
                    out[0:1], 
                    diffusion_data['timesteps'][0:1], 
                    diffusion_data['noisy_images'][0:1]
                )
                
                # Denormalize from [-1, 1] to [0, 1] for proper visualization
                pred_img = (pred_x0 * 0.5 + 0.5) * diffusion_data['mask_map'][0:1]
                gt_img = (diffusion_data['sample_images'][0:1] * 0.5 + 0.5) * diffusion_data['mask_map'][0:1]
                
                # Clamp to valid range
                pred_img = torch.clamp(pred_img, 0, 1)
                gt_img = torch.clamp(gt_img, 0, 1)
                
                # Get input albedo condition (baked_texture from UV space)
                albedo_img = None
                if 'albedo_map' in batch and batch['albedo_map'] is not None:
                    albedo = batch['albedo_map'][0:1]  # [1, 3, H, W]
                    # Denormalize if needed
                    if self.cfg.data_normalization:
                        albedo_vis = (albedo * 0.5 + 0.5) * diffusion_data['mask_map'][0:1]
                    else:
                        albedo_vis = albedo * diffusion_data['mask_map'][0:1]
                    albedo_vis = torch.clamp(albedo_vis, 0, 1)
                    albedo_img = albedo_vis[0].cpu().permute(1, 2, 0).detach().numpy()
                
                # Get thumbnail image condition
                thumbnail_img = None
                if 'thumbnail' in batch and batch['thumbnail'] is not None:
                    thumbnail = batch['thumbnail'][0:1]  # [1, 1, H, W, 3]
                    if thumbnail.dim() == 5:
                        thumbnail_img = thumbnail[0, 0]  # [H, W, 3]
                    else:
                        thumbnail_img = thumbnail[0]  # [H, W, 3]
                
                # Convert to wandb format
                import wandb
                images = []
                
                # Add albedo (UV space input condition) first if available
                if albedo_img is not None:
                    images.append(wandb.Image(albedo_img, caption="Input Albedo (UV)"))
                
                # Add thumbnail (3D rendered input condition) if available
                if thumbnail_img is not None:
                    images.append(wandb.Image(thumbnail_img.cpu().detach().numpy(), caption="Input Rendering"))
                
                # Add prediction and ground truth
                images.extend([
                    wandb.Image(pred_img[0].cpu().permute(1, 2, 0).detach().numpy(), caption="Predicted Emission"),
                    wandb.Image(gt_img[0].cpu().permute(1, 2, 0).detach().numpy(), caption="Ground Truth"),
                ])
                
                self._wandb_logger.log_image(
                    key="train/predictions",
                    images=images,
                    step=self.global_step
                )
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - images are logged via test tab composite views"""
        # Call parent validation (computes metrics and saves images)
        # The parent's test_step already logs composite images to test/validation/{object_id}
        result = super().validation_step(batch, batch_idx)
        return result
    
    def get_diffusion_loss(self, out, diffusion_data):
        """
        Compute diffusion loss with Flow Matching (inherited from parent).
        Target: velocity field v = x0 - noise
        """
        sample_images = diffusion_data['sample_images']
        noise = diffusion_data['noise']
        mask = diffusion_data['mask_map']
        
        # Flow Matching velocity target (same as original TEXGen)
        target = sample_images - noise
        
        # MSE loss
        mse_loss = F.mse_loss(out * mask, target * mask, reduction='mean')
        
        # L1 loss  
        l1_loss = F.l1_loss(out * mask, target * mask, reduction='mean')
        
        loss_dict = {}
        
        if self.cfg.loss.diffusion_loss_dict.get('lambda_mse', 0.0) > 0:
            loss_dict['mse'] = mse_loss * self.cfg.loss.diffusion_loss_dict['lambda_mse']
        
        if self.cfg.loss.diffusion_loss_dict.get('lambda_l1', 0.0) > 0:
            loss_dict['l1'] = l1_loss * self.cfg.loss.diffusion_loss_dict['lambda_l1']
        
        # Emissive region loss - stronger supervision on bright areas
        lambda_emissive = self.cfg.loss.diffusion_loss_dict.get('lambda_emissive', 0.0)
        if lambda_emissive > 0:
            # Ground truth emission in normalized space [-1, 1]
            # Convert to [0, 1] for thresholding
            gt_emission = (sample_images + 1.0) / 2.0
            
            # Create emissive mask: where any RGB channel > threshold
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.1)
            emissive_mask = (gt_emission.max(dim=1, keepdim=True)[0] > emissive_threshold).float()
            
            # Combine with UV mask
            emissive_mask = emissive_mask * mask
            
            # Only compute loss if there are emissive regions
            if emissive_mask.sum() > 0:
                # MSE loss on emissive regions only
                emissive_mse = F.mse_loss(
                    out * emissive_mask, 
                    target * emissive_mask, 
                    reduction='sum'
                ) / (emissive_mask.sum() + 1e-8)
                
                loss_dict['emissive_mse'] = emissive_mse * lambda_emissive
                
                # Log statistics
                if self.training and self.global_step % 100 == 0:
                    emissive_ratio = emissive_mask.sum() / (mask.sum() + 1e-8)
                    self.log('train/emissive_ratio', emissive_ratio, prog_bar=False)
        
        return loss_dict
    
    def get_batched_pred_x0(self, out, timesteps, noisy_input):
        """
        Get predicted x0 from model output based on prediction type.
        """
        # Ensure alphas_cumprod is on the same device as timesteps
        device = timesteps.device
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        
        # Convert timesteps to long type for indexing
        timesteps = timesteps.long()
        
        if self.prediction_type == "epsilon":
            # out is predicted noise
            alpha_prod_t = alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = (noisy_input - beta_prod_t.sqrt().view(-1, 1, 1, 1) * out) / alpha_prod_t.sqrt().view(-1, 1, 1, 1)
        elif self.prediction_type == "sample":
            # out is directly predicted x0
            pred_x0 = out
        elif self.prediction_type == "v_prediction":
            # out is v-prediction
            alpha_prod_t = alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = alpha_prod_t.sqrt().view(-1, 1, 1, 1) * noisy_input - beta_prod_t.sqrt().view(-1, 1, 1, 1) * out
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return pred_x0
    
    # test_pipeline method removed - using parent's Flow Matching-compatible version
    # Parent class (texgen_test.py) uses simple Euler integration: x += Δt*v
    # This matches the Flow Matching training: x_t = (1-t)*ε + t*x_0
    #
    # The old DDIM-based implementation below did NOT match training:
    # - Used DDIM scheduler with discrete timesteps [0, 999]
    # - Used complex DDIM.step() instead of simple Euler
    # - Training was on continuous t ∈ [0, 1] with linear interpolation
    #
    # To re-enable DDIM (not recommended), uncomment the method below

