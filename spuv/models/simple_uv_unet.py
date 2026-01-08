"""
Simple U-Net for UV space emission map generation.
Works directly in UV space without 3D rendering or feature baking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleUVUNet(nn.Module):
    """
    Simple U-Net that works directly in UV space.
    Input: noisy emission map + position + material properties (albedo, metal, rough)
    Output: denoised emission map
    """
    
    def __init__(
        self,
        cfg=None,  # Can accept config dict
        in_channels=11,  # position(3) + normal(3) + albedo(3) + metal(1) + rough(1)
        out_channels=3,  # emission RGB
        base_channels=64,
        time_emb_dim=256,
    ):
        super().__init__()
        
        # Handle config dict if provided
        if cfg is not None:
            if hasattr(cfg, 'in_channels'):
                in_channels = cfg.in_channels
            if hasattr(cfg, 'out_channels'):
                out_channels = cfg.out_channels
            if hasattr(cfg, 'base_channels'):
                base_channels = cfg.base_channels
            if hasattr(cfg, 'time_emb_dim'):
                time_emb_dim = cfg.time_emb_dim
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels + 3, base_channels, 3, padding=1)  # +3 for noisy emission
        
        # Encoder
        self.enc1 = self._make_layer(base_channels, base_channels * 2, time_emb_dim)
        self.enc2 = self._make_layer(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc3 = self._make_layer(base_channels * 4, base_channels * 8, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # Decoder
        self.dec3 = self._make_layer(base_channels * 16, base_channels * 4, time_emb_dim)
        self.dec2 = self._make_layer(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec1 = self._make_layer(base_channels * 4, base_channels, time_emb_dim)
        
        # Output
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def _make_layer(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'time_proj': nn.Linear(time_emb_dim, out_ch),
            'norm1': nn.GroupNorm(8, out_ch),
            'norm2': nn.GroupNorm(8, out_ch),
        })
    
    def _apply_layer(self, x, layer, time_emb, mask):
        # First conv + norm + activation
        h = layer['conv1'](x)
        h = layer['norm1'](h)
        h = F.silu(h)
        
        # Add time embedding
        time_proj = layer['time_proj'](time_emb)
        h = h + time_proj[:, :, None, None]
        
        # Second conv + norm + activation
        h = layer['conv2'](h)
        h = layer['norm2'](h)
        h = F.silu(h)
        
        # Apply mask
        h = h * mask
        
        return h
    
    def forward(
        self,
        noisy_emission,  # [B, 3, H, W]
        mask_map,  # [B, 1, H, W]
        position_map,  # [B, 3, H, W]
        timesteps,  # [B]
        image_embeddings,  # Not used in simple version
        mesh,  # Not used in simple version
        image_info,  # Contains conditioning info
        data_normalization=True,
        condition_drop=None,
    ):
        """
        Forward pass for simple UV U-Net.
        
        Args:
            noisy_emission: Noisy emission map [B, 3, H, W]
            mask_map: Valid UV mask [B, 1, H, W]
            position_map: 3D position in UV space [B, 3, H, W]
            timesteps: Diffusion timesteps [B]
            image_embeddings: CLIP embeddings (unused)
            mesh: Mesh data (unused)
            image_info: Dict with 'rgb_cond' (albedo/material)
            data_normalization: Whether data is normalized
            condition_drop: Dropout mask for classifier-free guidance
        """
        B, _, H, W = noisy_emission.shape
        device = noisy_emission.device
        
        # Get conditioning from image_info (this contains albedo/material from rgb_cond)
        rgb_cond = image_info.get('rgb_cond')  # [B, V, H, W, C]
        
        # Convert conditioning to [B, C, H, W]
        if rgb_cond.dim() == 5:
            B_cond, V, H_cond, W_cond, C = rgb_cond.shape
            # Take first view and permute
            cond_features = rgb_cond[:, 0, :, :, :].permute(0, 3, 1, 2)  # [B, C, H, W]
            # Resize to match input size if needed
            if H_cond != H or W_cond != W:
                cond_features = F.interpolate(cond_features, size=(H, W), mode='bilinear', align_corners=False)
        else:
            cond_features = rgb_cond
        
        # Normalize conditioning if data is normalized
        if data_normalization and cond_features.max() > 1.0:
            cond_features = cond_features / 255.0
        if data_normalization:
            cond_features = cond_features * 2.0 - 1.0
        
        # Apply condition dropout for classifier-free guidance
        if condition_drop is not None:
            cond_features = cond_features * (1 - condition_drop.view(B, 1, 1, 1))
        
        # The position_map actually contains the full material info in this setup
        # We just need: noisy_emission (3) + position (3) + albedo (3) = 9 channels
        # But config expects 11 input channels + 3 noisy = 14 total
        # Let me use position_map as-is (3 channels) since that's what we have
        # The cond_features from rgb_cond should have all materials
        
        # Prepare input: concatenate noisy emission with position and conditioning
        x = torch.cat([noisy_emission, position_map, cond_features], dim=1)  # [B, 3+3+C, H, W]
        
        # Time embedding
        timesteps_norm = timesteps.float() / 1000.0  # Normalize timesteps
        time_emb = self.time_mlp(timesteps_norm.view(-1, 1))  # [B, time_emb_dim]
        
        # Input projection
        x = self.input_conv(x) * mask_map
        
        # Encoder with skip connections
        e1 = self._apply_layer(x, self.enc1, time_emb, mask_map)
        e1_pool = self.pool(e1)
        
        # Downsample mask
        mask_2 = F.max_pool2d(mask_map, 2)
        
        e2 = self._apply_layer(e1_pool, self.enc2, time_emb, mask_2)
        e2_pool = self.pool(e2)
        mask_4 = F.max_pool2d(mask_2, 2)
        
        e3 = self._apply_layer(e2_pool, self.enc3, time_emb, mask_4)
        e3_pool = self.pool(e3)
        mask_8 = F.max_pool2d(mask_4, 2)
        
        # Bottleneck
        b = self._apply_layer(e3_pool, self.bottleneck, time_emb, mask_8)
        
        # Decoder with skip connections
        d3 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self._apply_layer(d3, self.dec3, time_emb, mask_4)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self._apply_layer(d2, self.dec2, time_emb, mask_2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self._apply_layer(d1, self.dec1, time_emb, mask_map)
        
        # Output
        output = self.output_conv(d1) * mask_map
        
        # Additional info for compatibility
        addition_info = {
            'baked_texture': cond_features,
            'baked_weights': mask_map,
        }
        
        return output, addition_info

