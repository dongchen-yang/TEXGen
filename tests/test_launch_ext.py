import signal
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from spuv.utils import launch_ext, wandb_utils


def _ckpt(path, epoch, step, run_id=None):
    d = {"epoch": epoch, "global_step": step, "state_dict": {}}
    if run_id:
        d["wandb_run_id"] = run_id
    torch.save(d, path)


def test_resolve_auto_resume_picks_newest_last_ckpt(tmp_path):
    _ckpt(tmp_path / "last.ckpt", 3, 300)
    time.sleep(0.05)
    _ckpt(tmp_path / "last-v1.ckpt", 5, 500)
    cfg = SimpleNamespace(auto_resume=True, resume=None, checkpoint={"dirpath": str(tmp_path)}, trial_dir="unused")
    assert launch_ext.resolve_auto_resume(cfg, train=True) == str(tmp_path / "last-v1.ckpt")
    assert cfg.resume == str(tmp_path / "last-v1.ckpt")


def test_resolve_auto_resume_empty_dir_and_not_training(tmp_path):
    cfg = SimpleNamespace(auto_resume=True, resume="last", checkpoint={}, trial_dir=str(tmp_path))
    assert launch_ext.resolve_auto_resume(cfg, train=True) is None and cfg.resume is None
    cfg = SimpleNamespace(auto_resume=True, resume="x.ckpt", checkpoint={}, trial_dir=str(tmp_path))
    assert launch_ext.resolve_auto_resume(cfg, train=False) == "x.ckpt"


def test_read_resume_info(tmp_path):
    p = tmp_path / "last.ckpt"
    _ckpt(p, 7, 700, run_id="abc123")
    assert launch_ext.read_resume_info(str(p), want_wandb=True) == ("abc123", 700, 7)
    assert launch_ext.read_resume_info(str(p), want_wandb=False) == (None, 700, 7)
    assert launch_ext.read_resume_info(None, want_wandb=True) == (None, None, None)


def test_checkpoint_dirpath_and_kwargs():
    cfg = SimpleNamespace(checkpoint={"dirpath": "/x/ckpts", "save_last": True}, trial_dir="/t")
    assert launch_ext.checkpoint_dirpath(cfg) == "/x/ckpts"
    assert launch_ext.checkpoint_kwargs(cfg) == {"save_last": True}
    cfg = SimpleNamespace(checkpoint={"save_last": True}, trial_dir="/t")
    assert launch_ext.checkpoint_dirpath(cfg) == "/t/ckpts"


def test_wandb_kwargs_defaults_overrides_and_resume():
    cfg = OmegaConf.create({"name": "lightgen", "tag": "run", "seed": 42,
                            "wandb": {"project": "LightGen", "name": "n", "tags": ["a"]}})
    kw = wandb_utils.wandb_kwargs(cfg, run_id=None)
    assert kw["project"] == "LightGen" and kw["name"] == "n" and list(kw["tags"]) == ["a"] and "id" not in kw
    assert "dir" not in kw
    kw = wandb_utils.wandb_kwargs(cfg, run_id="r1")
    assert kw["id"] == "r1" and kw["resume"] == "allow"
    cfg = OmegaConf.create({"name": "lightgen", "tag": "run", "seed": 42, "wandb": {}})
    assert wandb_utils.wandb_kwargs(cfg, None)["name"] == "lightgen-run"
    # custom_output_dir puts the wandb dir beside the run's outputs unless wandb.dir says otherwise (the fork's launch.py)
    cfg = OmegaConf.create({"name": "lightgen", "tag": "run", "seed": 42, "custom_output_dir": "/o",
                            "trial_dir": "/o/lightgen/run", "wandb": {"project": "LightGen"}})
    assert wandb_utils.wandb_kwargs(cfg, None)["dir"] == "/o/lightgen/run/wandb"
    cfg.wandb.dir = "/w"
    assert wandb_utils.wandb_kwargs(cfg, None)["dir"] == "/w"


def test_clean_stale_wandb_cache_respects_offline_and_rank(tmp_path, monkeypatch):
    stale = tmp_path / "wandb" / "run-20260101-r1"
    stale.mkdir(parents=True)
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb_utils.clean_stale_wandb_cache(str(tmp_path), "r1")
    assert stale.exists()
    monkeypatch.setenv("WANDB_MODE", "online"); monkeypatch.setenv("LOCAL_RANK", "1")
    wandb_utils.clean_stale_wandb_cache(str(tmp_path), "r1")
    assert stale.exists()
    monkeypatch.setenv("LOCAL_RANK", "0")
    wandb_utils.clean_stale_wandb_cache(str(tmp_path), "r1")
    assert not stale.exists()


def test_local_rank_prefers_LOCAL_RANK(monkeypatch):
    monkeypatch.setenv("SLURM_LOCALID", "3"); monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert wandb_utils.local_rank() == 3
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert wandb_utils.local_rank() == 2


def test_install_shutdown_handlers_registers_sigterm_and_sigusr1():
    before_term, before_usr1 = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGUSR1)
    try:
        wandb_utils.install_shutdown_handlers()
        assert signal.getsignal(signal.SIGTERM) is not before_term
        assert signal.getsignal(signal.SIGUSR1) is not before_usr1
    finally:
        signal.signal(signal.SIGTERM, before_term); signal.signal(signal.SIGUSR1, before_usr1)


def test_patch_torch_load_forces_weights_only_false(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "load", torch.load)   # restore the real torch.load after this test
    p = tmp_path / "o.pt"
    torch.save({"a": SimpleNamespace(x=1)}, p)     # a non-tensor object: weights_only=True would refuse it
    launch_ext.patch_torch_load()
    assert torch.load(p, weights_only=True)["a"].x == 1
