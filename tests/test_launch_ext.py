import os
import signal
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spuv.utils import launch_ext, wandb_utils


def _ckpt(path, epoch, step, run_id=None):
    d = {"epoch": epoch, "global_step": step, "state_dict": {}}
    if run_id:
        d["wandb_run_id"] = run_id
    torch.save(d, path)


def _cfg(ckpt_dir, resume=None, auto_resume=True, trial_dir="unused"):
    return SimpleNamespace(auto_resume=auto_resume, resume=resume,
                           checkpoint={"dirpath": str(ckpt_dir)}, trial_dir=trial_dir)


def test_resolve_resume_takes_the_runs_own_newest_last_ckpt_over_a_given_path(tmp_path):
    # A requeued fine-tune must continue from its own newest checkpoint; taking the path on the
    # command line instead would restart it from those weights at every requeue.
    _ckpt(tmp_path / "last.ckpt", 3, 300, run_id="old")
    _ckpt(tmp_path / "last-v1.ckpt", 5, 500, run_id="abc123")
    os.utime(tmp_path / "last.ckpt", (1000, 1000))
    os.utime(tmp_path / "last-v1.ckpt", (2000, 2000))
    given = tmp_path / "given.ckpt"
    _ckpt(given, 9, 900, run_id="given")
    cfg = _cfg(tmp_path, resume=str(given))
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (
        str(tmp_path / "last-v1.ckpt"), "abc123", 500, 5)
    assert cfg.resume == str(tmp_path / "last-v1.ckpt")


def test_resolve_resume_takes_the_given_path_when_the_run_has_no_checkpoint(tmp_path):
    given = tmp_path / "given.ckpt"
    _ckpt(given, 9, 900, run_id="r9")
    empty = tmp_path / "ckpts"
    empty.mkdir()                                     # the dir exists but holds no last*.ckpt
    cfg = _cfg(empty, resume=str(given))
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=False) == (str(given), None, 900, 9)
    assert cfg.resume == str(given)
    cfg = _cfg(tmp_path / "no-such-dir", resume=str(given))      # and the same without the dir
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (str(given), "r9", 900, 9)


def test_resolve_resume_starts_fresh_without_a_checkpoint_or_a_given_path(tmp_path):
    cfg = _cfg(tmp_path / "ckpts")
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (None, None, None, None)
    assert cfg.resume is None
    (tmp_path / "ckpts").mkdir()
    cfg = _cfg(tmp_path / "ckpts")
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (None, None, None, None)


def test_resolve_resume_starts_fresh_when_the_runs_last_ckpt_is_unreadable(tmp_path):
    (tmp_path / "last.ckpt").write_bytes(b"not a checkpoint")
    cfg = _cfg(tmp_path)
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=False) == (None, None, None, None)
    assert cfg.resume is None


def test_resolve_resume_keeps_an_unreadable_given_path_for_lightning_to_report(tmp_path):
    given = tmp_path / "given.ckpt"
    given.write_bytes(b"not a checkpoint")
    cfg = _cfg(tmp_path / "ckpts", resume=str(given), auto_resume=False)
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (str(given), None, None, None)
    assert cfg.resume == str(given)


def test_resolve_resume_returns_nones_for_the_keys_a_checkpoint_lacks(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "last.ckpt")
    cfg = _cfg(tmp_path)
    assert launch_ext.resolve_resume(cfg, train=True, want_wandb=True) == (
        str(tmp_path / "last.ckpt"), None, None, None)
    assert cfg.resume == str(tmp_path / "last.ckpt")     # the checkpoint is still resumed from


def test_resolve_resume_reads_nothing_when_not_training(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("no checkpoint read outside --train"))
    cfg = _cfg(tmp_path, resume="x.ckpt")
    assert launch_ext.resolve_resume(cfg, train=False, want_wandb=True) == ("x.ckpt", None, None, None)
    assert cfg.resume == "x.ckpt"


def test_checkpoint_dirpath_and_kwargs():
    cfg = SimpleNamespace(checkpoint={"dirpath": "/x/ckpts", "save_last": True}, trial_dir="/t")
    assert launch_ext.checkpoint_dirpath(cfg) == "/x/ckpts"
    assert launch_ext.checkpoint_kwargs(cfg) == {"save_last": True}
    cfg = SimpleNamespace(checkpoint={"save_last": True}, trial_dir="/t")
    assert launch_ext.checkpoint_dirpath(cfg) == "/t/ckpts"


class _Trainer:
    def __init__(self, fit_raises=None):
        self.fit_raises, self.calls = fit_raises, []

    def fit(self, system, datamodule=None, ckpt_path=None):
        self.calls.append(("fit", ckpt_path))
        if self.fit_raises is not None:
            raise self.fit_raises

    def test(self, system, datamodule=None):
        self.calls.append(("test", None))


def test_fit_with_graceful_exit_reraises_an_interrupt_and_skips_the_test(monkeypatch):
    finished = []
    monkeypatch.setattr(wandb_utils, "ensure_wandb_finish", lambda: finished.append(True))
    trainer = _Trainer(fit_raises=SystemExit(1))
    with pytest.raises(SystemExit) as excinfo:
        launch_ext.fit_with_graceful_exit(trainer, None, None, None, use_wandb=True)
    assert excinfo.value.code == 1
    assert trainer.calls == [("fit", None)] and finished == [True]      # wandb finished before it propagated


def test_fit_with_graceful_exit_tests_after_a_normal_fit(monkeypatch):
    finished = []
    monkeypatch.setattr(wandb_utils, "ensure_wandb_finish", lambda: finished.append(True))
    trainer = _Trainer()
    launch_ext.fit_with_graceful_exit(trainer, None, None, "last.ckpt", use_wandb=False)
    assert trainer.calls == [("fit", "last.ckpt"), ("test", None)] and finished == []


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


def test_build_wandb_logger_passes_the_dir_to_wandb_init(tmp_path, monkeypatch):
    # WandbLogger drops a dir= kwarg (it builds "dir": save_dir or dir, save_dir defaults to ".")
    monkeypatch.setenv("LOCAL_RANK", "1")        # rank != 0: no wandb.init, no define_metric
    cfg = OmegaConf.create({"name": "texgen_emission", "tag": "t",
                            "wandb": {"project": "LightGen", "dir": str(tmp_path)}})
    before_term, before_usr1 = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGUSR1)
    try:
        logger = wandb_utils.build_wandb_logger(cfg, None, SimpleNamespace())
        assert logger._wandb_init["dir"] == str(tmp_path)
    finally:
        signal.signal(signal.SIGTERM, before_term)
        signal.signal(signal.SIGUSR1, before_usr1)


def test_clean_stale_wandb_cache_respects_offline_and_rank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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


def test_clean_stale_wandb_cache_in_the_cwd_removes_only_that_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "online"); monkeypatch.setenv("LOCAL_RANK", "0")
    mine = tmp_path / "wandb" / "run-20260101-r1"
    other = tmp_path / "wandb" / "run-20260101-r2"
    mine.mkdir(parents=True); other.mkdir(parents=True)
    wandb_utils.clean_stale_wandb_cache(".", "r1")
    assert not mine.exists() and other.exists()


def test_ensure_wandb_finish_restores_sigint_even_when_finish_raises(monkeypatch):
    class _Wandb:
        run = object()

        @staticmethod
        def finish():
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN    # Ctrl+C ignored while it syncs
            raise RuntimeError("sync failed")

    monkeypatch.setitem(sys.modules, "wandb", _Wandb)
    before = signal.getsignal(signal.SIGINT)
    wandb_utils.ensure_wandb_finish()
    assert signal.getsignal(signal.SIGINT) is before


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
