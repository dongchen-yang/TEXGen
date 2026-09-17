"""What TEXGen-Emission adds around upstream TEXGen's launch.py: a torch.load patch,
auto-resume from the newest last*.ckpt, the resume metadata, the checkpoint dir, and a
fit() that finishes the wandb run on any exit. launch.py calls these; the bodies are the
fork's patches, moved here so launch.py can stay upstream's file. (Ours, not upstream's.)
"""
import glob
import os
import signal
from typing import Optional

import torch

import spuv

_original_torch_load = torch.load


def patch_torch_load():
    """Force weights_only=False: our checkpoints hold OmegaConf objects and Lightning passes True."""
    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched


def register_safe_globals():
    """Let torch 2.6+ unpickle the OmegaConf nodes inside our checkpoints."""
    try:
        import omegaconf
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig,
            omegaconf.nodes.StringNode, omegaconf.nodes.IntegerNode,
            omegaconf.nodes.FloatNode, omegaconf.nodes.BooleanNode,
        ])
    except Exception:
        pass  # older torch: no add_safe_globals, nothing to register


def checkpoint_dirpath(cfg) -> str:
    return cfg.checkpoint["dirpath"] if "dirpath" in cfg.checkpoint else os.path.join(cfg.trial_dir, "ckpts")


def checkpoint_kwargs(cfg) -> dict:
    return {k: v for k, v in cfg.checkpoint.items() if k != "dirpath"}


def resolve_auto_resume(cfg, train: bool) -> Optional[str]:
    """When auto_resume (or resume="last"/"latest") and training, point cfg.resume at the newest
    last*.ckpt in the checkpoint dir, or at None when there is none. Returns cfg.resume."""
    if not ((getattr(cfg, "auto_resume", False) or cfg.resume in ["last", "latest"]) and train):
        return cfg.resume
    ckpt_dir = checkpoint_dirpath(cfg)
    if not os.path.exists(ckpt_dir):
        cfg.resume = None
        spuv.info(f"Auto-resume enabled but checkpoint directory {ckpt_dir} does not exist, starting fresh")
        return None
    last_ckpts = glob.glob(os.path.join(ckpt_dir, "last*.ckpt"))
    last_ckpt = max(last_ckpts, key=os.path.getmtime) if last_ckpts else None
    if not last_ckpt:
        cfg.resume = None
        spuv.info(f"Auto-resume enabled but no last*.ckpt found in {ckpt_dir}, starting fresh")
        return None
    try:
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        cfg.resume = last_ckpt
        spuv.info(f"Auto-resume: Found {os.path.basename(last_ckpt)}")
        spuv.info(f"  Epoch: {ckpt.get('epoch', -1)}, Global Step: {ckpt.get('global_step', -1)}")
    except Exception as e:
        spuv.warn(f"Failed to load {os.path.basename(last_ckpt)}: {e}")
        cfg.resume = None
    return cfg.resume


def read_resume_info(path: Optional[str], want_wandb: bool):
    """(wandb_run_id, global_step, epoch) from a checkpoint, each None when absent."""
    if path is None:
        return None, None, None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        spuv.warn(f"Failed to load checkpoint info: {e}")
        return None, None, None
    run_id = None
    if want_wandb and "wandb_run_id" in ckpt:
        run_id = ckpt["wandb_run_id"]
        spuv.info(f"Resuming wandb run with ID: {run_id}")
    elif want_wandb:
        spuv.warn("No wandb run ID found in checkpoint, starting new wandb run")
    step, epoch = ckpt.get("global_step", None), ckpt.get("epoch", None)
    if step is not None:
        spuv.info(f"Resuming from epoch {epoch}, global_step {step}")
    return run_id, step, epoch


def fit_with_graceful_exit(trainer, system, dm, ckpt_path, use_wandb, resume_epoch=None, resume_step=None):
    """trainer.fit + trainer.test, finishing the wandb run on Ctrl+C, SystemExit or normal exit
    so an interrupted run leaves no un-synced cache to pollute a later resume."""
    if ckpt_path is not None:
        spuv.info("=" * 80)
        spuv.info(f"RESUMING TRAINING FROM CHECKPOINT: {ckpt_path}")
        spuv.info(f"Expected resume from epoch: {resume_epoch}, global_step: {resume_step}")
        spuv.info("=" * 80)
    try:
        trainer.fit(system, datamodule=dm, ckpt_path=ckpt_path)
        trainer.test(system, datamodule=dm)
    except (KeyboardInterrupt, SystemExit):
        spuv.info("Detected interrupt, attempting graceful shutdown ...")
    finally:
        if use_wandb:
            import wandb
            if wandb.run is not None:
                prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
                spuv.info("Syncing wandb data (please wait, do NOT press Ctrl+C)...")
                try:
                    wandb.finish()
                except Exception as e:
                    spuv.warn(f"wandb.finish() error: {e}")
                finally:
                    signal.signal(signal.SIGINT, prev)
