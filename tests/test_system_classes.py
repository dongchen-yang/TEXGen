"""The system classes the checkpoints and Lightning reach by name (CPU: nothing here imports the network)."""
import importlib

import pytest

import spuv.systems.texgen_emission_base as texgen_emission_base
from spuv.systems.texgen_emission_base import LossConfig, TEXGenBaseSystem
from spuv.systems.texgen_emission_test import TEXGenDiffusion
from spuv.utils.config import parse_structured
from spuv.utils.saving import SaverMixin

# diffusion_loss_dict as the published parsed.yaml and the live config set it.
PUBLISHED_DIFFUSION_LOSS_DICT = {
    "lambda_mse": 1.0, "lambda_l1": 0.5, "lambda_dark_region": 0.0, "emissive_threshold": 0.001,
}


def _unconfigured(cls):
    """A system object that skips __init__ and configure: no backbone, no CLIP, no scheduler download."""
    return cls.__new__(cls)


def test_the_pickled_loss_config_path_is_the_same_class():
    # Published checkpoints pickle spuv.systems.texgen_base.LossConfig inside optimizer_states
    # (an OmegaConf node's object type); torch.load of any of them imports this path.
    assert importlib.import_module("spuv.systems.texgen_base").LossConfig is LossConfig


def test_the_config_named_system_path_is_the_same_class():
    from spuv.systems.lightgen_system import LightGenSystem
    assert LightGenSystem is TEXGenDiffusion and issubclass(TEXGenDiffusion, TEXGenBaseSystem)


def test_texgen_diffusion_defines_its_own_step_methods():
    # upstream BaseSystem.test_step raises NotImplementedError, and trainer.test after fit and --test reach it
    for name in ("training_step", "validation_step", "test_step", "test_pipeline"):
        assert name in TEXGenDiffusion.__dict__, name


def test_the_published_loss_lambdas_pass_the_check():
    cfg = parse_structured(TEXGenDiffusion.Config, {"loss": {"diffusion_loss_dict": PUBLISHED_DIFFUSION_LOSS_DICT}})
    TEXGenDiffusion.check_diffusion_loss_dict(cfg.loss.diffusion_loss_dict)


@pytest.mark.parametrize("key", ["lambda_dark_region", "lambda_emission_mask", "lambda_pred_mask_cls"])
def test_configure_rejects_a_nonzero_lambda_the_loss_does_not_implement(key):
    # get_diffusion_loss builds MSE and L1 only; any other nonzero lambda_* would train without its term
    system = _unconfigured(TEXGenDiffusion)
    system.cfg = parse_structured(
        TEXGenDiffusion.Config, {"loss": {"diffusion_loss_dict": {**PUBLISHED_DIFFUSION_LOSS_DICT, key: 0.01}}}
    )
    with pytest.raises(ValueError, match=key):
        system.configure()      # the check runs before super().configure() builds anything


class _SavingProbe(TEXGenBaseSystem):
    global_step = 120           # shadow LightningModule's properties, which need a trainer
    current_epoch = 4


def test_save_image_grid_logs_once_on_our_x_axis_and_never_through_upstream(monkeypatch):
    # Upstream's save_image_grid logs with wandb's step=, off the trainer/global_step x-axis. The override
    # hands it name=None and step=None, leaves its align default alone, and logs once itself.
    upstream_calls, logged = [], []

    def upstream_save_image_grid(self, filename, imgs, **kwargs):
        upstream_calls.append((filename, kwargs))
        return f"/save/{filename}"

    monkeypatch.setattr(SaverMixin, "save_image_grid", upstream_save_image_grid)
    monkeypatch.setattr(texgen_emission_base, "log_image_to_wandb", lambda *args: logged.append(args))
    system = _unconfigured(_SavingProbe)
    system._wandb_logger = object()

    assert system.save_image_grid("a.jpg", [], name="test/a") == "/save/a.jpg"
    assert system.save_image_grid("b.jpg", [], align=256, name="test/b", step=7, texts=["t"]) == "/save/b.jpg"
    system.save_image_grid("c.jpg", [])                     # no name: saved, not logged
    system._wandb_logger = None
    system.save_image_grid("d.jpg", [], name="test/d")      # no logger: saved, not logged

    no_log = {"name": None, "step": None}
    assert upstream_calls == [
        ("a.jpg", {**no_log, "texts": None}),               # no align key: upstream's default applies
        ("b.jpg", {**no_log, "texts": ["t"], "align": 256}),
        ("c.jpg", {**no_log, "texts": None}),
        ("d.jpg", {**no_log, "texts": None}),
    ]
    assert logged == [("test/a", "/save/a.jpg", 120, 4), ("test/b", "/save/b.jpg", 7, 4)]
