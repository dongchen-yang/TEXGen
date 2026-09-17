"""wandb helpers for TEXGen-Emission: image logging with our x-axis convention, plus (Task 8)
the logger setup, the stale-cache guard, the step logger and the shutdown handlers launch.py
installs. (Ours, not upstream's.)

The convention: never pass step= to wandb.log(). Every call carries trainer/global_step and
epoch as plain metrics, and define_metric() makes trainer/global_step the chart x-axis.
"""
import os


def local_rank():
    """LOCAL_RANK first (Lightning's subprocess launcher sets it per child), then SLURM_LOCALID, then 0."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def wandb_image_chw(img, caption):
    """A wandb.Image from a [1, C, H, W] tensor in [0, 1]."""
    import wandb
    return wandb.Image(img[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=caption)


def wandb_image_hwc(img, caption):
    """A wandb.Image from an [H, W, 3] tensor in [0, 1]."""
    import wandb
    return wandb.Image(img.cpu().detach().numpy().copy(), caption=caption)


def log_image_to_wandb(name, path, global_step, epoch):
    """Log one saved image under `name` with our x-axis metrics; a no-op without a live run."""
    import wandb
    if wandb.run is not None:
        wandb.log({name: wandb.Image(path), "trainer/global_step": global_step, "epoch": epoch})
