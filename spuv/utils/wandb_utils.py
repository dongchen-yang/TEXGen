"""wandb helpers for TEXGen-Emission: image logging with our x-axis convention, plus
the logger setup, the stale-cache guard, the step logger and the shutdown handlers launch.py
installs. (Ours, not upstream's.)

The convention: never pass step= to wandb.log(). Every call carries trainer/global_step and
epoch as plain metrics, and define_metric() makes trainer/global_step the chart x-axis.
"""
import atexit
import glob
import os
import shutil
import signal
import sys

import spuv


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


def wandb_kwargs(cfg, run_id):
    """The WandbLogger kwargs: project LightGen and name/tag by default, overridden by cfg.wandb,
    every other cfg.wandb key passed through, and id/resume set when resuming a run."""
    kw = {"project": "LightGen", "name": f"{cfg.name}-{cfg.tag}", "config": dict(cfg)}
    custom = dict(getattr(cfg, "wandb", {}) or {})
    for key in ("project", "name", "entity", "dir"):
        if key in custom:
            kw[key] = custom[key]
    if custom and "dir" not in custom and getattr(cfg, "custom_output_dir", None):
        kw["dir"] = os.path.join(cfg.trial_dir, "wandb")      # the fork's default: wandb beside the run's outputs
    if run_id is not None:
        kw["id"] = run_id
        kw["resume"] = "allow"
    for key, value in custom.items():
        if key not in ("project", "name", "entity", "dir", "config", "id", "resume"):
            kw[key] = value
    return kw


def clean_stale_wandb_cache(wandb_dir, run_id):
    """Before resuming a run online on rank 0, drop the local cache dirs of that run id: a
    killed session leaves un-synced metrics that would be flushed at the wrong steps.
    Offline mode keeps them (the cache is the data); other ranks never touch them."""
    if os.environ.get("WANDB_MODE", "").lower() in ("offline", "dryrun"):
        spuv.info("wandb offline: keeping the local cache — it is the data, not a stale artifact")
        return
    if local_rank() != 0:
        return
    stale = glob.glob(os.path.join(wandb_dir, "wandb", f"*-{run_id}"))
    if wandb_dir != ".":
        stale += glob.glob(os.path.join(".", "wandb", f"*-{run_id}"))
    for d in stale:
        if os.path.isdir(d):
            spuv.info(f"Cleaning stale wandb cache to prevent data oscillation: {d}")
            shutil.rmtree(d, ignore_errors=True)


def build_wandb_logger(cfg, run_id, system):
    """The logger, attached to the system, with the x-axis convention and the step logger installed."""
    from pytorch_lightning.loggers import WandbLogger
    kw = wandb_kwargs(cfg, run_id)
    if run_id is not None:
        spuv.info(f"Configuring wandb to resume run ID: {run_id}")
        clean_stale_wandb_cache(kw.get("dir", "."), run_id)
    # WandbLogger builds wandb.init's "dir" as `save_dir or dir` and save_dir defaults to ".",
    # so the dir has to go in as save_dir or every run writes ./wandb in the cwd.
    save_dir = kw.pop("dir", ".")
    logger = WandbLogger(save_dir=save_dir, **kw)
    system._wandb_logger = logger
    if run_id is not None:
        system.set_wandb_run_id(run_id)
    if local_rank() == 0:
        import wandb
        _ = logger.experiment          # wandb.init() on rank 0 only
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")
        spuv.info("Set wandb x-axes: all metrics use trainer/global_step (via define_metric)")
    install_step_logger(logger, system)
    install_shutdown_handlers()
    return logger


def install_step_logger(wandb_logger, system):
    """Replace WandbLogger.log_metrics so every call carries trainer/global_step from the system
    (never Lightning's step= argument, which is the epoch for on_epoch metrics)."""
    from pytorch_lightning.loggers.wandb import _add_prefix
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    @rank_zero_only
    def _log_metrics(metrics, step=None):
        metrics = _add_prefix(metrics, wandb_logger._prefix, wandb_logger.LOGGER_JOIN_CHAR)
        wandb_logger.experiment.log(
            dict(metrics, **{"trainer/global_step": system.global_step, "epoch": system.current_epoch})
        )

    wandb_logger.log_metrics = _log_metrics
    spuv.info("Patched WandbLogger.log_metrics → trainer/global_step always = system.global_step")


def ensure_wandb_finish():
    """Finish the live run, ignoring Ctrl+C while it syncs, and hand Ctrl+C back afterwards
    however that goes. Best effort; the one place that finishes a run (launch_ext calls it too)."""
    try:
        import wandb
        if wandb.run is None:
            return
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            spuv.info("Finishing wandb run (syncing data, please wait -- DO NOT press Ctrl+C again)...")
            wandb.finish()
        except Exception as e:
            spuv.warn(f"wandb.finish() error: {e}")
        finally:
            signal.signal(signal.SIGINT, prev)
    except Exception:
        pass


def install_shutdown_handlers():
    """atexit, SIGTERM (Slurm, scancel, kill) and SIGUSR1 (preemption warning) all finish wandb first."""
    atexit.register(ensure_wandb_finish)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm(signum, frame):
        spuv.info("Received SIGTERM, finishing wandb run before exit...")
        ensure_wandb_finish()
        if callable(original_sigterm):
            original_sigterm(signum, frame)
        else:
            sys.exit(128 + signum)

    def _sigusr1(signum, frame):
        spuv.info("Received SIGUSR1 (preemption warning), finishing wandb run...")
        ensure_wandb_finish()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGUSR1, _sigusr1)
