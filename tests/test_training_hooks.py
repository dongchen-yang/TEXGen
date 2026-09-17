"""The training hooks TEXGenBaseSystem owns, driven on the CPU.

Training is the one path with no gate: the byte gate covers inference only, and no other test
calls a hook. This runs the real system and datamodule over four synthetic 32x32 shapes with the
toy backbone and tokenizer from tests/toy_modules.py — fit, resume from last.ckpt, then
trainer.test — and asserts what the fork's hooks are for: one EMA update per optimizer step, the
EMA weights swapped in for validation and test and the training weights back afterwards, and the
scheduler state riding in the checkpoint. It is a synthetic unit test, not training on any dataset.
"""
import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import spuv.systems.texgen_emission_test as texgen_emission_test
import spuv.utils.base as spuv_utils_base
import spuv.utils.misc as spuv_utils_misc
from spuv.data.mesh_uv import MeshUVDataModule
from spuv.systems.texgen_emission_base import TEXGenBaseSystem
from spuv.systems.texgen_emission_test import TEXGenDiffusion

SHAS = [f"{i:02d}" + "0123456789abcdef0123456789abcd" for i in range(4)]
RES = 32

# Lightning's advice about this setup, not about the code under test: no dataloader workers, a
# batch size it cannot infer from our dict batch, and a checkpoint dir the resume run writes into.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*does not have many workers"),
    pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`"),
    pytest.mark.filterwarnings("ignore:Checkpoint directory .* exists and is not empty"),
    pytest.mark.filterwarnings("ignore:GPU available but not used"),      # CPU on purpose
]

# Every hook the fork moved into TEXGenBaseSystem. Lightning calls a hook through getattr, so a
# slip that turns one into a property keeps running (its body fires on every attribute read)
# without ever being called as a hook: `in __dict__` passes for a property, isfunction does not.
MOVED_HOOKS = ("on_save_checkpoint", "on_load_checkpoint", "on_train_start", "on_fit_start",
               "backward", "on_train_epoch_start", "on_train_epoch_end", "on_train_batch_end",
               "on_before_optimizer_step")


def test_the_moved_training_hooks_are_functions_on_TEXGenBaseSystem():
    for name in MOVED_HOOKS:
        assert name in TEXGenBaseSystem.__dict__, name
        assert inspect.isfunction(TEXGenBaseSystem.__dict__[name]), name


def _write_shapes(root):
    """Four shapes in the current bake's layout: atlas.npz with an alpha key, plus a thumbnail."""
    os.makedirs(os.path.join(root, "thumbnails"))
    rng = np.random.default_rng(0)
    for i, sha in enumerate(SHAS):
        d = os.path.join(root, sha)
        os.makedirs(d)
        occ = np.zeros((RES, RES, 1), bool)
        occ[4:28, 4:28] = True
        emission = np.zeros((RES, RES, 3), np.uint8)
        emission[8 + i:14 + i, 8:14] = 200
        np.savez(os.path.join(d, "atlas.npz"), occupancy=occ,
                 position=rng.integers(0, 65535, (RES, RES, 3), dtype=np.uint16),
                 objnormal=rng.integers(0, 65535, (RES, RES, 3), dtype=np.uint16),
                 color=rng.integers(0, 255, (RES, RES, 3), dtype=np.uint8),
                 metal=rng.integers(0, 255, (RES, RES, 1), dtype=np.uint8),
                 rough=rng.integers(0, 255, (RES, RES, 1), dtype=np.uint8),
                 emission_color=emission,
                 alpha=np.full((RES, RES, 1), 100 + i, np.uint8))
        Image.fromarray(np.full((64, 64, 3), 40 * i, np.uint8)).save(
            os.path.join(root, "thumbnails", f"{sha}.png"))
    pd.DataFrame({"ditem_dir": SHAS, "success": [True] * len(SHAS)},
                 index=pd.Index(SHAS, name="ditem_id")).to_parquet(os.path.join(root, "eval.parquet"))
    split = os.path.join(root, "split.json")
    with open(split, "w") as f:
        json.dump({"train": {"indices": [0, 1]}, "val": {"indices": [2]}, "test": {"indices": [3]}}, f)
    return dict(data_root=root, parquet_file=os.path.join(root, "eval.parquet"),
                train_indices=split, val_indices=split, test_indices=split,
                uv_height=16, uv_width=16, use_alpha=True,
                batch_size=1, eval_batch_size=1, num_workers=0)


SYSTEM_CFG = dict(
    condition_drop_rate=0.5, use_ema=True, ema_decay=0.9999, val_with_ema=True,
    train_regression=False, recon_warm_up_steps=0, check_train_every_n_steps=2,
    cleanup_after_validation_step=True, test_cfg_scale=2.0, test_num_steps=2,
    guidance_rescale=0.0, guidance_interval=[0.0, 1.0],
    image_tokenizer_cls="tests.toy_modules.ToyTok", image_tokenizer={},
    backbone_cls="tests.toy_modules.ToyNet", backbone={"in_channels": 13, "out_channels": 3},
    loss={"diffusion_loss_dict": {"lambda_mse": 1.0, "lambda_l1": 0.5, "lambda_dark_region": 0.0,
                                  "emissive_threshold": 0.001}},
    optimizer={"name": "AdamW", "args": {"lr": 5e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}},
    scheduler={"name": "CosineAnnealingLR", "interval": "step", "args": {"T_max": 8, "eta_min": 1e-6}},
)


def _param_sum(system):
    return float(sum(p.detach().double().sum() for p in system.backbone.parameters()))


class Recorder(Callback):
    """Callback hooks run around the module's, so these see the weights before and after its swap."""

    def __init__(self):
        self.train_batches, self.eval_epochs, self.tested = [], [], []

    def on_train_batch_start(self, trainer, system, batch, batch_idx):
        scheduler = system.lr_schedulers()
        self.train_batches.append(dict(global_step=int(system.global_step),
                                       scheduler_last_epoch=int(scheduler.last_epoch),
                                       ema_updates=int(system.backbone_ema.num_updates)))

    def _start(self, system, stage):
        self.eval_epochs.append(dict(stage=stage, before=_param_sum(system)))

    def _batch(self, system):
        self.eval_epochs[-1].setdefault("during", _param_sum(system))

    def _end(self, system):
        self.eval_epochs[-1]["after"] = _param_sum(system)

    def on_validation_epoch_start(self, trainer, system):
        self._start(system, "validation")

    def on_validation_batch_start(self, trainer, system, batch, batch_idx, dataloader_idx=0):
        self._batch(system)

    def on_validation_end(self, trainer, system):
        self._end(system)

    def on_test_epoch_start(self, trainer, system):
        self._start(system, "test")

    def on_test_batch_start(self, trainer, system, batch, batch_idx, dataloader_idx=0):
        self._batch(system)
        self.tested.append(batch_idx)

    def on_test_end(self, trainer, system):
        self._end(system)


def _run(data_cfg, out_dir, max_epochs, ckpt_path):
    pl.seed_everything(7, workers=True)
    system = TEXGenDiffusion(SYSTEM_CFG, resumed=ckpt_path is not None)
    system.set_save_dir(os.path.join(out_dir, "save"))
    recorder = Recorder()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs, check_val_every_n_epoch=1,
                      num_sanity_val_steps=0, precision=32, gradient_clip_val=1.0, log_every_n_steps=1,
                      inference_mode=False, enable_progress_bar=False, enable_model_summary=False,
                      default_root_dir=out_dir, logger=CSVLogger(out_dir, name="csv"),
                      callbacks=[ModelCheckpoint(dirpath=os.path.join(out_dir, "ckpts"), save_last=True,
                                                 save_top_k=1, monitor="val/psnr", mode="max",
                                                 save_on_train_epoch_end=False),
                                 recorder])
    trainer.fit(system, datamodule=MeshUVDataModule(data_cfg), ckpt_path=ckpt_path)
    return system, trainer, recorder


def test_the_training_hooks_keep_the_ema_the_weights_and_the_scheduler(tmp_path, monkeypatch):
    cpu = lambda: torch.device("cpu")
    for module in (spuv_utils_misc, spuv_utils_base, texgen_emission_test):
        monkeypatch.setattr(module, "get_device", cpu)
    data_cfg = _write_shapes(str(tmp_path / "data"))
    out = str(tmp_path / "run")

    system, trainer, recorder = _run(data_cfg, out, max_epochs=2, ckpt_path=None)

    # on_train_batch_end updates the EMA once per batch, and never before the first one.
    assert [b["ema_updates"] for b in recorder.train_batches] == [0, 1, 2, 3]
    assert int(system.backbone_ema.num_updates) == system.global_step == 4

    # on_validation_epoch_start/end swap the EMA weights in and the training weights back.
    assert [e["stage"] for e in recorder.eval_epochs] == ["validation", "validation"]
    for epoch in recorder.eval_epochs:
        assert epoch["during"] != epoch["before"], epoch      # the EMA weights really went in
        assert epoch["after"] == epoch["before"], epoch

    # on_save_checkpoint puts the scheduler state in the checkpoint (Lightning's own copy is there too).
    last = os.path.join(out, "ckpts", "last.ckpt")
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    assert [s["last_epoch"] for s in ckpt["_scheduler_states"]] == [4]
    assert ckpt["global_step"] == 4

    # Resume: on_load_checkpoint hands the state to on_train_start, which restores the scheduler.
    resumed, trainer2, recorder2 = _run(data_cfg, out, max_epochs=3, ckpt_path=last)
    first_resumed_batch = recorder2.train_batches[0]
    assert first_resumed_batch["global_step"] == 4 and first_resumed_batch["scheduler_last_epoch"] == 4
    assert first_resumed_batch["ema_updates"] == 4                    # the EMA count came back too
    assert int(resumed.backbone_ema.num_updates) == resumed.global_step == 6
    assert resumed.lr_schedulers().last_epoch == resumed.global_step

    # trainer.test runs, on the EMA weights, and leaves the training weights in place.
    trainer2.test(resumed, datamodule=MeshUVDataModule(data_cfg))
    assert recorder2.tested == [0]
    test_epoch = recorder2.eval_epochs[-1]
    assert test_epoch["stage"] == "test"
    assert test_epoch["during"] != test_epoch["before"]
    assert test_epoch["after"] == test_epoch["before"]
    assert "val/psnr" in trainer2.callback_metrics
