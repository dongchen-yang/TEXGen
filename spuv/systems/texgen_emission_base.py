"""The base system for TEXGen-Emission: construction (the backbone, its EMA, the DDPM noise
schedule), the EMA scope, the checkpoint/resume/memory hooks and the wandb image-log override.

Upstream's texgen_base.py, edited directly. Its class was also named TEXGenDiffusion and was
overridden by texgen_test.py's class of the same name; here it carries the name upstream's own
texgen_test.py imported it under, TEXGenBaseSystem, and only TEXGenDiffusion in
texgen_emission_test.py keeps the name. The DDIM sampler, the render-based validation and the
render losses upstream kept here are at tag pre-trim-2026-09-16. spuv/systems/texgen_base.py
re-exports LossConfig for the published checkpoints.
"""
import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import DDPMScheduler

import spuv
from spuv.systems.base import BaseLossConfig, BaseSystem
from spuv.utils.lit_ema import LitEma
from spuv.utils.memory_tracker import log_memory, set_baseline
from spuv.utils.wandb_utils import log_image_to_wandb


@dataclass
class LossConfig(BaseLossConfig):
    # Every field the published parsed.yaml sets stays, used or not: parse_structured rejects unknown keys.
    diffusion_loss_dict: dict = field(default_factory=dict)
    render_loss_dict: dict = field(default_factory=dict)
    lambda_mse: Any = 0.0
    lambda_l1: Any = 0.0
    lambda_render_lpips: Any = 0.0
    lambda_render_mse: Any = 0.0
    lambda_render_l1: Any = 0.0
    use_min_snr_weight: bool = False
    use_vgg: bool = False
    p_loss_type: str = "lpips"
    lpips_resize: bool = False


class TEXGenBaseSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        loss: LossConfig = field(default_factory=LossConfig)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        data_normalization: bool = True
        render_background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        random_background_color: bool = False

        rescale_betas_zero_snr: bool = False
        train_regression: bool = False
        prediction_type: str = "sample"

        use_ema: bool = True
        ema_decay: float = 0.9999
        val_with_ema: bool = True

        test_num_steps: int = 50
        test_save_json: bool = False

        recon_warm_up_steps: int = 0
        test_scheduler_type: str = "ddim"
        test_save_mid_result: bool = False

        # see On the Importance of Noise Scheduling for Diffusion Models
        # http://arxiv.org/abs/2301.10972
        train_image_scaling: float = 1.0
        condition_drop_rate: float = 0.0
        test_cfg_scale: float = 0.0
        guidance_rescale: float = 0.0
        guidance_interval: Tuple[float, float] = (0.0, 1.0)

        # Cond image augmentation
        cond_rgb_perturb: bool = False
        cond_rgb_perturb_scale: Dict[str, Any] = field(default_factory=lambda: {})

    cfg: Config
    _wandb_run_id: Optional[str] = None      # saved into and restored from checkpoints
    _saved_scheduler_states = None           # filled by on_load_checkpoint, consumed by on_train_start

    def configure(self):
        super().configure()
        self.train_regression = self.cfg.train_regression      # on_check_train (the test file) still assigns it
        # The backbone and its EMA, as in upstream texgen_base.py's configure
        self.backbone = spuv.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.use_ema = self.cfg.use_ema
        self.ema_decay = self.cfg.ema_decay
        self.val_with_ema = self.cfg.val_with_ema
        if self.use_ema:
            self.backbone_ema = LitEma(self.backbone, decay=self.ema_decay)
            spuv.info(f"Keeping EMAs of {len(list(self.backbone_ema.buffers()))}.")
        # The DDPM noise schedule, as in upstream texgen_base.py's configure; its alphas_cumprod
        # back the training-panel x0 estimate
        self.prediction_type = self.cfg.prediction_type
        temp_noise_scheduler = DDPMScheduler.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", subfolder="scheduler",
            prediction_type=self.prediction_type,
            rescale_betas_zero_snr=self.cfg.rescale_betas_zero_snr
        )
        betas = temp_noise_scheduler.betas
        betas[-1] = 0.9999 if betas[-1] == 1.0 else betas[-1]    # avoid nan during inference
        self.betas = betas
        self.noise_scheduler = DDPMScheduler(
            prediction_type=self.prediction_type,
            trained_betas=self.betas.numpy(),
        )
        self.num_train_timesteps = self.noise_scheduler.num_train_timesteps

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            if context is not None:
                spuv.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.backbone_ema.restore(self.backbone.parameters())
                if context is not None:
                    spuv.info(f"{context}: Restored training weights")

    # ---- wandb run id and scheduler state ride in the checkpoint ----
    # Moved from spuv/systems/base.py, the fork's patch at tag pre-trim-2026-09-16.
    def set_wandb_run_id(self, run_id: Optional[str]):
        self._wandb_run_id = run_id

    def get_wandb_run_id(self) -> Optional[str]:
        return self._wandb_run_id

    def on_save_checkpoint(self, checkpoint):
        """Save wandb run ID and critical training state to checkpoint"""
        if self._wandb_logger is not None and hasattr(self._wandb_logger.experiment, 'id'):
            checkpoint['wandb_run_id'] = self._wandb_logger.experiment.id
            spuv.info(f"Saved wandb run ID to checkpoint: {checkpoint['wandb_run_id']}")

        # CRITICAL: Ensure scheduler state is explicitly saved
        # PyTorch Lightning 2.x sometimes fails to restore scheduler state properly
        if hasattr(self, 'lr_schedulers'):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

            checkpoint['_scheduler_states'] = []
            for i, scheduler in enumerate(schedulers):
                state = {
                    'last_epoch': scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else 0,
                    '_last_lr': scheduler._last_lr if hasattr(scheduler, '_last_lr') else None,
                    'state_dict': scheduler.state_dict(),
                }
                checkpoint['_scheduler_states'].append(state)
                spuv.info(f"Explicitly saved scheduler {i} state: last_epoch={state['last_epoch']}, last_lr={state['_last_lr']}")

    def on_load_checkpoint(self, checkpoint):
        """Load wandb run ID and restore scheduler state from checkpoint"""
        if 'wandb_run_id' in checkpoint:
            self._wandb_run_id = checkpoint['wandb_run_id']
            spuv.info(f"Loaded wandb run ID from checkpoint: {self._wandb_run_id}")

        # Log checkpoint state for debugging resume issues
        if 'epoch' in checkpoint and 'global_step' in checkpoint:
            spuv.info(f"Loading checkpoint from epoch {checkpoint['epoch']}, global_step {checkpoint['global_step']}")

        # Store scheduler state for restoration in on_fit_start
        # (can't restore here because schedulers aren't created yet)
        self._saved_scheduler_states = checkpoint.get('_scheduler_states', None)
        if self._saved_scheduler_states:
            spuv.info(f"Found explicitly saved scheduler states in checkpoint")
            for i, state in enumerate(self._saved_scheduler_states):
                spuv.info(f"  Scheduler {i}: last_epoch={state['last_epoch']}, last_lr={state['_last_lr']}")

        # Also check Lightning's built-in scheduler state
        if 'lr_schedulers' in checkpoint:
            spuv.info(f"Found {len(checkpoint['lr_schedulers'])} Lightning scheduler(s) in checkpoint")
            for i, sched_state in enumerate(checkpoint['lr_schedulers']):
                if 'last_epoch' in sched_state:
                    spuv.info(f"  Lightning Scheduler {i} last_epoch: {sched_state['last_epoch']}")
                if '_last_lr' in sched_state:
                    spuv.info(f"  Lightning Scheduler {i} last_lr: {sched_state['_last_lr']}")

    # ---- resume verification and scheduler restore ----
    # Moved from spuv/systems/base.py, the fork's patch at tag pre-trim-2026-09-16.
    def on_train_start(self):
        """Called at the start of training, AFTER checkpoint is loaded."""
        # CRITICAL: Verify checkpoint was actually loaded when resuming
        # This is called AFTER PyTorch Lightning loads the checkpoint
        if self._resumed:
            current_epoch = self.current_epoch
            current_step = self.global_step
            spuv.info(f"=" * 80)
            spuv.info(f"CHECKPOINT RESUME VERIFICATION:")
            spuv.info(f"  Current epoch: {current_epoch}")
            spuv.info(f"  Current global_step: {current_step}")

            if current_step == 0 and current_epoch == 0:
                spuv.warn(
                    "=" * 80 + "\n" +
                    "CRITICAL ERROR: Checkpoint resume FAILED!\n" +
                    "global_step and epoch are both 0, but resumed=True.\n" +
                    "This means PyTorch Lightning did not restore the checkpoint state.\n" +
                    "Training will start from scratch instead of resuming!\n" +
                    "=" * 80
                )
            else:
                spuv.info(f"  ✓ Checkpoint state successfully restored!")
                spuv.info(f"  Logging will continue from step {current_step}")

            spuv.info(f"=" * 80)

        # CRITICAL: Restore scheduler state when resuming training
        if self._resumed and hasattr(self, 'lr_schedulers'):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

            # Check if we have explicitly saved scheduler states
            if hasattr(self, '_saved_scheduler_states') and self._saved_scheduler_states:
                spuv.info("Restoring scheduler states from explicit checkpoint save...")
                for i, (scheduler, saved_state) in enumerate(zip(schedulers, self._saved_scheduler_states)):
                    # Restore scheduler state
                    scheduler.load_state_dict(saved_state['state_dict'])
                    if saved_state['last_epoch'] is not None:
                        scheduler.last_epoch = saved_state['last_epoch']
                    if saved_state['_last_lr'] is not None:
                        scheduler._last_lr = saved_state['_last_lr']
                    spuv.info(f"Restored scheduler {i}: last_epoch={scheduler.last_epoch}, last_lr={scheduler._last_lr}")

            # Verify and correct scheduler state
            for i, scheduler in enumerate(schedulers):
                if hasattr(scheduler, 'last_epoch'):
                    expected_step = self.global_step
                    actual_step = scheduler.last_epoch

                    # Allow small mismatch due to logging intervals
                    if abs(expected_step - actual_step) > 10:
                        spuv.warn(
                            f"Scheduler {i} last_epoch ({actual_step}) does not match global_step ({expected_step}). "
                            f"This may cause LR discontinuities. Correcting scheduler state..."
                        )
                        # Fix the scheduler's last_epoch to match global_step
                        scheduler.last_epoch = expected_step
                        # Step the scheduler to recalculate LR
                        scheduler.step()

                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 'unknown'
                    spuv.info(f"Scheduler {i} final state: step={scheduler.last_epoch}, LR={current_lr}")

    # ---- memory logging around the training loop ----
    # Moved from the fork's patches to spuv/systems/base.py and texgen_base.py at tag pre-trim-2026-09-16.
    def on_fit_start(self) -> None:
        super().on_fit_start()
        log_memory("fit_start (after model init)", force=True)
        set_baseline()
        spuv.info("[MEMORY] Baseline memory set after model initialization")

    def backward(self, loss, *args: Any, **kwargs: Any) -> None:
        if self.global_step % 50 == 0:
            log_memory("before_backward", self.global_step, force=True)
        super().backward(loss, *args, **kwargs)
        if self.global_step % 50 == 0:
            log_memory("after_backward", self.global_step, force=True)

    def on_train_epoch_start(self):
        log_memory(f"epoch_start (epoch={self.current_epoch})", force=True)

    def on_train_epoch_end(self):
        log_memory(f"epoch_end_before_cleanup (epoch={self.current_epoch})", force=True)
        if self.current_epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            log_memory(f"epoch_end_after_cleanup (epoch={self.current_epoch})", force=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # The fork ran BaseSystem's dataset hooks, a 10-batch cleanup, the EMA update, a 5-step cleanup, in that order.
        super().on_train_batch_end(outputs, batch, batch_idx)
        if batch_idx % 10 == 0:
            log_memory(f"before_batch_cleanup (batch={batch_idx})", self.true_global_step)
            gc.collect()
            torch.cuda.empty_cache()
            log_memory(f"after_batch_cleanup (batch={batch_idx})", self.true_global_step)
        if self.use_ema:
            self.backbone_ema(self.backbone)
        if self.global_step % 5 == 0:
            log_memory(f"before_ema_cleanup (step={self.global_step})", self.global_step)
            gc.collect()
            torch.cuda.empty_cache()
            log_memory(f"after_ema_cleanup (step={self.global_step})", self.global_step)

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        if self.global_step % 50 == 0:
            log_memory("before_optimizer_step", self.global_step, force=True)

    # ---- image saving: upstream logs with step=, we log with our x-axis convention (saving.py) ----
    def save_image_grid(self, filename, imgs, align=None, name=None, step=None, texts=None):
        kwargs = {} if align is None else {"align": align}
        save_path = super().save_image_grid(filename, imgs, name=None, step=None, texts=texts, **kwargs)
        if name and self._wandb_logger:
            log_image_to_wandb(name, save_path, step if step is not None else self.global_step, self.current_epoch)
        return save_path
