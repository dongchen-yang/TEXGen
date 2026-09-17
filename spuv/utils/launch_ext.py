"""What TEXGen-Emission adds around upstream TEXGen's launch.py: a torch.load patch, the resume
choice and its metadata in one read of the checkpoint, the checkpoint dir, and a fit() that
finishes the wandb run on any exit. launch.py calls these; the bodies are the fork's patches,
moved here so launch.py can stay upstream's file. (Ours, not upstream's.)
"""
import glob
import os
from typing import Optional

import torch

import spuv
from spuv.utils import wandb_utils

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


def newest_last_ckpt(cfg) -> Optional[str]:
    """The run's own newest last*.ckpt in its checkpoint dir, or None with the reason logged."""
    ckpt_dir = checkpoint_dirpath(cfg)
    if not os.path.exists(ckpt_dir):
        spuv.info(f"Auto-resume enabled but checkpoint directory {ckpt_dir} does not exist")
        return None
    last_ckpts = glob.glob(os.path.join(ckpt_dir, "last*.ckpt"))
    if not last_ckpts:
        spuv.info(f"Auto-resume enabled but no last*.ckpt found in {ckpt_dir}")
        return None
    return max(last_ckpts, key=os.path.getmtime)


def resolve_resume(cfg, train: bool, want_wandb: bool):
    """Pick the checkpoint to resume from and read it once: (path, wandb run id, global step, epoch),
    each None when the checkpoint does not carry it. This function is the only writer of cfg.resume.

    On a --train launch with auto_resume (or resume "last"/"latest") the run's own newest last*.ckpt
    wins, so a requeued run continues where it stopped instead of restarting from the weights on the
    command line; an explicit resume=<path> is taken only when the run has no checkpoint of its own.
    Sets cfg.resume to the choice. Lightning reads the file again, so this is the only extra read
    (the published checkpoints are ~12 GB, and every rank runs this)."""
    if not train:
        return cfg.resume, None, None, None
    given = cfg.resume if cfg.resume not in (None, "last", "latest") else None
    from_this_run = False
    if getattr(cfg, "auto_resume", False) or cfg.resume in ["last", "latest"]:
        path = newest_last_ckpt(cfg)
        from_this_run = path is not None
        if path is None and given is not None:
            path = given
            spuv.info(f"This run has no checkpoint yet, resuming from the path given on the command line: {path}")
    else:
        path = given
    cfg.resume = path
    if path is None:
        if getattr(cfg, "auto_resume", False) or given is not None:
            spuv.info("Starting fresh: this run has no checkpoint and no resume path was given")
        else:
            spuv.info("Starting fresh: no resume path was given and auto_resume is off")
        return None, None, None, None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        if from_this_run:                     # an unreadable last*.ckpt: start fresh, do not crash a requeue
            spuv.warn(f"Failed to load {os.path.basename(path)}: {e}")
            cfg.resume = None
            return None, None, None, None
        spuv.warn(f"Failed to load checkpoint info: {e}")   # the given path stays: Lightning reports it loudly
        return path, None, None, None
    if from_this_run:
        spuv.info(f"Auto-resume: Found {os.path.basename(path)}")
        spuv.info(f"  Epoch: {ckpt.get('epoch', -1)}, Global Step: {ckpt.get('global_step', -1)}")
    run_id = None
    if want_wandb and "wandb_run_id" in ckpt:
        run_id = ckpt["wandb_run_id"]
        spuv.info(f"Resuming wandb run with ID: {run_id}")
    elif want_wandb:
        spuv.warn("No wandb run ID found in checkpoint, starting new wandb run")
    step, epoch = ckpt.get("global_step", None), ckpt.get("epoch", None)
    if step is not None:
        spuv.info(f"Resuming from epoch {epoch}, global_step {step}")
    return path, run_id, step, epoch


def fit_with_graceful_exit(trainer, system, dm, ckpt_path, use_wandb, resume_epoch=None, resume_step=None):
    """trainer.fit + trainer.test, finishing the wandb run on Ctrl+C, SystemExit or normal exit
    so an interrupted run leaves no un-synced cache to pollute a later resume, then re-raising
    the interrupt so the exit status reports it and trainer.test does not run."""
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
        raise                       # the finally below still syncs wandb first
    finally:
        if use_wandb:
            wandb_utils.ensure_wandb_finish()
