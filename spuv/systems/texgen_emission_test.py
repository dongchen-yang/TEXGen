"""TEXGenDiffusion for TEXGen-Emission: flow matching over a 13-channel UV input (noisy
emission, position, albedo, metallic, roughness, alpha, occupancy) with CLIP conditioning on
the shape's thumbnail, MSE+L1 on the velocity over the UV islands, UV-space validation, and
the Euler sampler the published inference runs.

Upstream's texgen_test.py, edited directly; the class the config names. The conditioning,
input assembly, loss and training step that the fork kept in lightgen_system.py live here now,
and test_step and validation_step are upstream's methods, edited in place. The originals are
at tag pre-trim-2026-09-16.
"""
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
from einops import rearrange

import spuv
from spuv.systems.texgen_emission_base import TEXGenBaseSystem
from spuv.utils.memory_tracker import init_tracker, log_memory, logged_cleanup
from spuv.utils.misc import get_device
from spuv.utils.uv_metrics import denorm_masked, emission_mask, fit_height, flip_for_view, rgb_panel, uv_mse_psnr
from spuv.utils.wandb_utils import wandb_image_chw, wandb_image_hwc


class TEXGenDiffusion(TEXGenBaseSystem):
    @dataclass
    class Config(TEXGenBaseSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        self.check_loss_config(self.cfg.loss)    # before anything is built
        super().configure()
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.sigma_min=0.000001
        self.memory_tracker = init_tracker(enabled=True, log_interval=1)
        spuv.info("[MEMORY] Memory tracking enabled")

    def get_conditional_flow(self, noise, sample, t):
        t = t[:, None, None, None]
        return (1 - (1 - self.sigma_min) * t) * noise + t * sample

    def prepare_diffusion_data(self, batch):
        device = get_device()
        batch_size = batch["gt_emission"].shape[0]
        sample_images = batch["gt_emission"]      # [B, 3, H, W] in [-1, 1], with the UV maps below from the dataloader
        mask_map = batch["mask_map"]
        position_map = batch["position_map"]

        # timesteps = torch.rand(batch_size, device=device)
        # Sample uniformly
        uniform_samples = torch.rand(batch_size, device=device)
        # Apply power transformation to skew towards smaller t
        power = 2  # >1 to skew towards 0
        timesteps = uniform_samples ** power

        noise = torch.randn_like(sample_images, dtype=self.dtype)
        noisy_images = self.get_conditional_flow(noise, sample_images, timesteps)

        noisy_images *= mask_map

        diffusion_data = {
            "sample_images": sample_images,
            "position_map": position_map,
            "mask_map": mask_map,
            "timesteps": timesteps,
            "noise": noise,
            "noisy_images": noisy_images,
        }

        return diffusion_data

    def forward(self,
                condition: Dict[str, Any],
                diffusion_data: Dict[str, Any],
                condition_drop=None,
                ) -> Dict[str, Any]:
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        text_embeddings = condition["text_embeddings"]
        image_embeddings = condition["image_embeddings"]
        clip_embeddings = [text_embeddings, image_embeddings]

        mesh = condition["mesh"]

        # The material maps come pre-baked in UV space. Alpha goes into baked_texture, before the
        # occupancy mask in the backbone's input order, with the same [0, 1] -> [-1, 1] remap.
        albedo_map = condition["albedo_map"]
        metal_map = condition["metal_map"]
        rough_map = condition["rough_map"]
        alpha_map = condition["alpha_map"]
        if alpha_map is not None:
            baked_texture = torch.cat([albedo_map, metal_map, rough_map, alpha_map], dim=1)  # [B, 6, H, W]
        else:
            baked_texture = torch.cat([albedo_map, metal_map, rough_map], dim=1)  # [B, 5, H, W]

        image_info = {
            'mvp_mtx_cond': condition["mvp_mtx_cond"],
            'baked_texture': baked_texture,
            'baked_weights': mask_map,
        }

        if condition_drop is None and self.training:
            condition_drop = torch.rand(input_tensor.shape[0], device=input_tensor.device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device)

        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,           # t in [0, 1], as the published checkpoints trained; upstream passed timesteps*1000
           clip_embeddings,
           mesh,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info

    def prepare_condition_info(self, batch):
        mesh = batch["mesh"]
        mvp_mtx_cond = batch["mvp_mtx_cond"]
        albedo_map = batch["albedo_map"]  # [B, 3, H, W]
        B = albedo_map.shape[0]

        # One fixed prompt (the dataset has no text prompts), its embedding computed once and cached
        prompt = ["emission generation"] * B
        if not hasattr(self, '_cached_text_embedding'):
            self._cached_text_embedding = self.image_tokenizer.process_text(["emission generation"]).to(dtype=self.dtype)  # [1, 768]
        text_embeddings = self._cached_text_embedding.expand(B, -1)

        # The shape's thumbnail, encoded online. The datamodule's collate_fn leaves it None when the batch's
        # first shape has none, and the inference script leaves the key out; then the albedo UV map stands in.
        if isinstance(batch.get('thumbnail'), torch.Tensor):
            rendered_thumbnail = batch['thumbnail']  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(rendered_thumbnail).to(dtype=self.dtype)
        else:
            spuv.warn("Thumbnail not found in batch, using albedo UV map as fallback")
            albedo_for_clip = albedo_map.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(albedo_for_clip).to(dtype=self.dtype)

        condition_info = {
            "mesh": mesh,
            "mvp_mtx_cond": mvp_mtx_cond,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "prompt": prompt,
            "albedo_map": albedo_map,
            "metal_map": batch["metal_map"],  # [B, 1, H, W]
            "rough_map": batch["rough_map"],  # [B, 1, H, W]
            "alpha_map": batch.get("alpha_map", None),  # [B, 1, H, W], or None without data.use_alpha
        }

        return condition_info

    def training_step(self, batch, batch_idx):
        """Training step"""
        if batch is None:
            return None

        # Memory tracking: log at start of training step
        if batch_idx == 0 or batch_idx % 10 == 0:
            log_memory(f"train_step_start (epoch={self.current_epoch}, batch={batch_idx})", self.global_step)

        # Prepare data
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        if batch_idx % 10 == 0:
            log_memory(f"after_data_prep (batch={batch_idx})", self.global_step)

        # Forward pass
        out, addition_info = self(condition_info, diffusion_data)

        if batch_idx % 10 == 0:
            log_memory(f"after_forward (batch={batch_idx})", self.global_step)

        # Compute loss
        loss_dict = self.get_diffusion_loss(out, diffusion_data)

        if batch_idx % 10 == 0:
            log_memory(f"after_loss_compute (batch={batch_idx})", self.global_step)

        # Log losses per-step only (x-axis is trainer/global_step via define_metric)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Total loss
        total_loss = sum(loss_dict.values())
        self.log('train/loss', total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Store outputs for visualization
        outputs = {
            'texture_map_outputs': {
                'pred': out,
                'gt': diffusion_data['sample_images'],
            },
            'mask_map': diffusion_data['mask_map'],
        }

        # Train previews saved to disk every check_train_every_n_steps*10 steps
        self.on_check_train(batch, outputs)

        # Log to wandb at the same frequency as check_train_every_n_steps
        if hasattr(self, '_wandb_logger') and self._wandb_logger is not None:
            n = self.cfg.check_train_every_n_steps
            if n > 0 and self.global_step % n == 0:
                import wandb as _wandb
                if _wandb.run is not None:
                    batch_size = out.shape[0]
                    emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
                    images = []

                    for i in range(batch_size):
                        s = f"S{i}"

                        # Denoised prediction for sample i
                        pred_x0_i = self.get_batched_pred_x0(
                            out[i:i+1],
                            diffusion_data['timesteps'][i:i+1],
                            diffusion_data['noisy_images'][i:i+1]
                        )

                        mask_i = diffusion_data['mask_map'][i:i+1]
                        pred_img_i = torch.clamp(denorm_masked(pred_x0_i, mask_i), 0, 1)
                        gt_img_i   = torch.clamp(denorm_masked(diffusion_data['sample_images'][i:i+1], mask_i), 0, 1)

                        # Albedo condition
                        if 'albedo_map' in batch and batch['albedo_map'] is not None:
                            albedo_vis_i = torch.clamp(batch['albedo_map'][i:i+1] * mask_i, 0, 1)
                            images.append(wandb_image_chw(albedo_vis_i, f"{s} Input Albedo (UV)"))

                        # Thumbnail condition
                        if 'thumbnail' in batch and batch['thumbnail'] is not None:
                            thumb = batch['thumbnail'][i]  # [1, H, W, 3] or [H, W, 3]
                            if thumb.dim() == 4:
                                thumb = thumb[0]
                            images.append(wandb_image_hwc(thumb, f"{s} Input Rendering"))

                        gt_emask_i   = emission_mask(gt_img_i, emissive_threshold).repeat(1, 3, 1, 1)
                        pred_emask_i = emission_mask(pred_img_i, emissive_threshold).repeat(1, 3, 1, 1)
                        images.extend([
                            wandb_image_chw(pred_img_i, f"{s} Predicted Emission"),
                            wandb_image_chw(gt_img_i, f"{s} Ground Truth"),
                            wandb_image_chw(gt_emask_i, f"{s} GT Emission Mask (>{emissive_threshold})"),
                            wandb_image_chw(pred_emask_i, f"{s} Pred Emission Mask (>{emissive_threshold})"),
                        ])

                    _wandb.log({"train/predictions": images, "trainer/global_step": self.global_step, "epoch": self.current_epoch})
                    del images

        # Explicit cleanup of large intermediate tensors after every training step
        del out, addition_info, outputs
        del diffusion_data, condition_info

        # Memory tracking: log after cleanup
        if batch_idx % 10 == 0:
            log_memory(f"train_step_end (batch={batch_idx})", self.global_step)

        return total_loss

    @staticmethod
    def check_loss_config(loss):
        """Raise on a loss setting get_diffusion_loss ignores, rather than train without it.

        get_diffusion_loss reads diffusion_loss_dict's lambda_mse and lambda_l1 only. Any other lambda in
        diffusion_loss_dict or render_loss_dict, the top-level lambdas and the use_min_snr_weight and use_vgg
        flags must be zero or false; the published configs set them so.
        """
        settings = {f"diffusion_loss_dict.{key}": value for key, value in loss.diffusion_loss_dict.items()
                    if key.startswith("lambda_") and key not in ("lambda_mse", "lambda_l1")}
        settings.update({f"render_loss_dict.{key}": value for key, value in loss.render_loss_dict.items()
                         if key.startswith("lambda_")})
        for key in ("lambda_mse", "lambda_l1", "lambda_render_lpips", "lambda_render_mse", "lambda_render_l1",
                    "use_min_snr_weight", "use_vgg"):
            settings[key] = getattr(loss, key)
        for key, value in settings.items():
            if value != 0:
                raise ValueError(
                    f"system.loss.{key} = {value}: get_diffusion_loss implements only diffusion_loss_dict's "
                    "lambda_mse and lambda_l1; the other loss terms are at tag pre-trim-2026-09-16"
                )

    def get_diffusion_loss(self, out, diffusion_data):
        """Flow-matching velocity target v = x0 - noise, MSE and L1 over the UV islands.

        The lambda keys live in cfg.loss.diffusion_loss_dict; a zero lambda drops its term
        from the dict, so the total loss is the sum of what is present.
        """
        target = diffusion_data['sample_images'] - diffusion_data['noise']
        mask = diffusion_data['mask_map']
        weights = self.cfg.loss.diffusion_loss_dict
        loss_dict = {}
        if weights.get('lambda_mse', 0.0) > 0:
            loss_dict['mse'] = F.mse_loss(out * mask, target * mask, reduction='mean') * weights['lambda_mse']
        if weights.get('lambda_l1', 0.0) > 0:
            loss_dict['l1'] = F.l1_loss(out * mask, target * mask, reduction='mean') * weights['lambda_l1']
        return loss_dict

    def get_batched_pred_x0(self, out, timesteps, noisy_input):
        """x0 from the predicted velocity, for the training panels.

        Training noises x_t = (1 - (1 - sigma_min) t) noise + t x0 (get_conditional_flow) and regresses
        v = x0 - noise (get_diffusion_loss). So x_t = (1 + sigma_min t) noise + t v, which gives
        noise = (x_t - t v) / (1 + sigma_min t) and x0 = noise + v. Exact inside the UV islands;
        prepare_diffusion_data zeroes x_t outside them.
        """
        t = timesteps[:, None, None, None]
        noise = (noisy_input - t * out) / (1 + self.sigma_min * t)
        return noise + out

    def on_check_train(self, batch, outputs):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if (
                self.global_rank == 0
                and self.cfg.check_train_every_n_steps > 0
                and self.true_global_step % (self.cfg.check_train_every_n_steps*10) == 0
        ):
            images = []
            texture_map_outputs = outputs["texture_map_outputs"]

            for key, value in texture_map_outputs.items():
                img = denorm_masked(value, outputs["mask_map"], self.cfg.data_normalization)
                images.append(rgb_panel(rearrange(img, "B C H W -> (B H) W C")))

            # Save to disk only, don't log to WandB
            self.save_image_grid(
                f"it{self.true_global_step}-train.jpg",
                images,
            )

    @torch.no_grad()
    def on_validation_epoch_start(self):
        """Switch to EMA weights once at the start of validation"""
        if self.use_ema and self.val_with_ema:
            spuv.info("Validation with EMA weights: Switching to EMA weights")
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            self._ema_switched = True
        else:
            self._ema_switched = False

    def on_validation_epoch_end(self):
        """Restore training weights once at the end of validation"""
        if self._ema_switched:
            spuv.info("Validation with EMA weights: Restoring training weights")
            self.backbone_ema.restore(self.backbone.parameters())
            self._ema_switched = False

    # trainer.test runs on the EMA weights too, as upstream's test_step did through ema_scope
    on_test_epoch_start = on_validation_epoch_start
    on_test_epoch_end = on_validation_epoch_end

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            log_memory(f"validation_start (epoch={self.current_epoch})", self.global_step, force=True)
        self.test_step(batch, batch_idx)
        logged_cleanup(f"after_validation (batch={batch_idx})", f"after_val_cleanup (batch={batch_idx})",
                       self.global_step, log=batch_idx % 10 == 0)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None
        try:
            with torch.cuda.amp.autocast(enabled=False):
                # EMA weights are already switched at epoch level, just run inference
                texture_map_outputs = self.test_pipeline(batch)
        except Exception as e:
            spuv.info(f"Error in test pipeline: {e}")
            spuv.info(f"Full traceback:\n{traceback.format_exc()}")
            return None

        # Get batch size (support batched validation)
        batch_size = len(batch["scene_id"])

        # Compute and log validation metrics (batched computation for efficiency)
        normalized = self.cfg.data_normalization
        pred_x0, gt_x0, mask_map = texture_map_outputs["pred_x0"], texture_map_outputs["gt_x0"], texture_map_outputs["mask_map"]
        mse, psnr = uv_mse_psnr(denorm_masked(pred_x0, mask_map, normalized), denorm_masked(gt_x0, mask_map, normalized))

        # Log metrics (aggregated per epoch and averaged over ranks, x-axis is trainer/global_step via define_metric)
        self.log('val/mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        def uv_preview(x, mask, normalized):
            """One shape's UV map as the [H, W, C] image every preview saves."""
            # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
            return rearrange(flip_for_view(denorm_masked(x, mask, normalized)), "B C H W-> (B H) W C")

        has_thumbnail = 'thumbnail' in batch and batch['thumbnail'] is not None
        has_albedo = 'albedo_map' in batch and batch['albedo_map'] is not None

        for b_idx in range(batch_size):
            save_str = batch["scene_id"][b_idx]
            mask = mask_map[b_idx:b_idx+1]
            pred_vis = uv_preview(pred_x0[b_idx:b_idx+1], mask, normalized)
            gt_vis = uv_preview(gt_x0[b_idx:b_idx+1], mask, normalized)

            # save prediction to png file (not logged to WandB; the composite below is)
            self.save_image_grid(
                f"it{self.true_global_step}-test/{save_str}.png",
                [rgb_panel(pred_vis)],
            )

            # Input condition (thumbnail), padded or cropped to the UV map's height for the side-by-side preview
            if has_thumbnail:
                thumbnail = batch['thumbnail'][b_idx:b_idx+1]  # [1, 1, H, W, 3]
                if thumbnail.dim() == 5:
                    thumbnail_img = thumbnail[0, 0]  # [H, W, 3]
                else:
                    thumbnail_img = thumbnail[0]  # [H, W, 3]
                thumbnail_img = fit_height(thumbnail_img, pred_vis.shape[0])

            # Input albedo UV map, already in [0, 1]
            if has_albedo:
                albedo_vis = uv_preview(batch['albedo_map'][b_idx:b_idx+1], mask, False)

            # Composite image: [Thumbnail | Albedo | Prediction | Ground Truth], logged under the object id
            composite_imgs = []
            if has_thumbnail:
                composite_imgs.append(rgb_panel(thumbnail_img))
            if has_albedo:
                composite_imgs.append(rgb_panel(albedo_vis))
            composite_imgs.append(rgb_panel(pred_vis))
            composite_imgs.append(rgb_panel(gt_vis))
            self.save_image_grid(
                f"it{self.true_global_step}-test/preview/composite_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                composite_imgs,
                name=f"test/validation/{save_str}",
            )

            # Also save individual images to disk (but not to WandB) for reference
            for vis, suffix in [(pred_vis, "prediction"), (gt_vis, "ground_truth")]:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/{suffix}_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    [rgb_panel(vis)],
                )

            if has_thumbnail:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/thumbnail_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    [rgb_panel(thumbnail_img)],
                )

            if has_albedo:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/albedo_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    [rgb_panel(albedo_vis)],
                )

        # Explicit cleanup of large tensors to prevent memory leaks
        del texture_map_outputs
        del pred_x0, gt_x0, mask_map

    def test_pipeline(self, batch):
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        device = get_device()
        test_num_steps = self.cfg.test_num_steps

        B, C, H, W = diffusion_data["mask_map"].shape
        noise = torch.randn((B, 3, H, W), device=device, dtype=self.dtype)
        noisy_images = noise

        t_span=torch.linspace(0, 1, test_num_steps, device=device, dtype=self.dtype)
        delta = 1.0 / test_num_steps

        for i, t in enumerate(t_span):
            timestep = t.repeat(B)
            diffusion_data["timesteps"] = timestep
            diffusion_data["noisy_images"] = noisy_images
            cond_step_out, addition_info = self(condition_info, diffusion_data)

            if (
                    self.cfg.test_cfg_scale != 0.0
                    and self.cfg.guidance_interval[0] <= t <= self.cfg.guidance_interval[1]
            ):
                uncond_step_out, _ = self(condition_info, diffusion_data, condition_drop=torch.ones(B, device=device))
                step_out = uncond_step_out + self.cfg.test_cfg_scale * (cond_step_out - uncond_step_out)
                # Apply guidance rescale. From paper [Common Diffusion Noise Schedules
                # and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) section 3.4.
                if self.cfg.guidance_rescale != 0:
                    std_pos = cond_step_out.std(dim=list(range(1, cond_step_out.ndim)), keepdim=True)
                    std_cfg = step_out.std(dim=list(range(1, step_out.ndim)), keepdim=True)
                    # Fuse equation 15,16 for more efficient computation.
                    step_out *= self.cfg.guidance_rescale * (std_pos / std_cfg) + (1 - self.cfg.guidance_rescale)
            else:
                step_out = cond_step_out

            noisy_images = noisy_images + delta * step_out

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
        }

        return texture_map_outputs
