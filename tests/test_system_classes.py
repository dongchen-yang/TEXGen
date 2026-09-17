"""The system classes the checkpoints and Lightning reach by name (CPU: nothing here imports the network)."""
import importlib

import pytest
import torch

import spuv.systems.texgen_emission_base as texgen_emission_base
import spuv.systems.texgen_emission_test as texgen_emission_test
from spuv.systems.texgen_emission_base import LossConfig, TEXGenBaseSystem
from spuv.systems.texgen_emission_test import TEXGenDiffusion
from spuv.utils.config import parse_structured
from spuv.utils.saving import SaverMixin

# system.loss as the published parsed.yaml files and the live config set it.
PUBLISHED_DIFFUSION_LOSS_DICT = {
    "lambda_mse": 1.0, "lambda_l1": 0.5, "lambda_dark_region": 0.0, "emissive_threshold": 0.001,
}
PUBLISHED_LOSS = {
    "diffusion_loss_dict": PUBLISHED_DIFFUSION_LOSS_DICT,
    "render_loss_dict": {"lambda_render_lpips": 0.0, "lambda_render_mse": 0.0, "lambda_render_l1": 0.0},
    "use_min_snr_weight": False,
    "use_vgg": False,
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


def test_the_test_hooks_are_the_validation_hooks():
    # trainer.test swaps in the EMA weights as validation does; upstream's test_step did it through ema_scope
    assert TEXGenDiffusion.on_test_epoch_start is TEXGenDiffusion.on_validation_epoch_start
    assert TEXGenDiffusion.on_test_epoch_end is TEXGenDiffusion.on_validation_epoch_end


def test_the_published_loss_config_passes_the_check():
    cfg = parse_structured(TEXGenDiffusion.Config, {"loss": PUBLISHED_LOSS})
    TEXGenDiffusion.check_loss_config(cfg.loss)


@pytest.mark.parametrize("key, override", [
    ("diffusion_loss_dict.lambda_dark_region", {"diffusion_loss_dict": {**PUBLISHED_DIFFUSION_LOSS_DICT, "lambda_dark_region": 0.01}}),
    ("diffusion_loss_dict.lambda_emission_mask", {"diffusion_loss_dict": {**PUBLISHED_DIFFUSION_LOSS_DICT, "lambda_emission_mask": 0.01}}),
    ("diffusion_loss_dict.lambda_pred_mask_cls", {"diffusion_loss_dict": {**PUBLISHED_DIFFUSION_LOSS_DICT, "lambda_pred_mask_cls": 0.01}}),
    ("render_loss_dict.lambda_render_lpips", {"render_loss_dict": {"lambda_render_lpips": 0.1}}),
    ("render_loss_dict.lambda_render_mse", {"render_loss_dict": {"lambda_render_mse": 1.0}}),
    ("lambda_mse", {"lambda_mse": 1.0}),
    ("lambda_l1", {"lambda_l1": 0.5}),
    ("lambda_render_lpips", {"lambda_render_lpips": 0.1}),
    ("lambda_render_mse", {"lambda_render_mse": 0.1}),
    ("lambda_render_l1", {"lambda_render_l1": 0.1}),
    ("use_min_snr_weight", {"use_min_snr_weight": True}),
    ("use_vgg", {"use_vgg": True}),
])
def test_configure_rejects_a_loss_setting_the_loss_ignores(key, override):
    # get_diffusion_loss reads diffusion_loss_dict's lambda_mse and lambda_l1 only; any other nonzero
    # lambda or true flag would train without its term
    system = _unconfigured(TEXGenDiffusion)
    system.cfg = parse_structured(TEXGenDiffusion.Config, {"loss": {**PUBLISHED_LOSS, **override}})
    with pytest.raises(ValueError, match=f"system.loss.{key} = "):
        system.configure()      # the check runs before super().configure() builds anything


class _FlowProbe(TEXGenDiffusion):
    dtype = torch.float32       # shadow LightningModule's property, which needs __init__


@pytest.mark.parametrize("sigma_min", [0.000001, 0.1])
def test_get_batched_pred_x0_recovers_x0_from_the_training_target(monkeypatch, sigma_min):
    # The training panel's x0: noise a known x0 with prepare_diffusion_data, take the target get_diffusion_loss
    # regresses (its loss is zero there) as the model output, and get x0 back inside the UV islands.
    monkeypatch.setattr(texgen_emission_test, "get_device", lambda: torch.device("cpu"))
    system = _unconfigured(_FlowProbe)
    system.cfg = parse_structured(TEXGenDiffusion.Config, {"loss": PUBLISHED_LOSS})
    system.sigma_min = sigma_min        # configure sets 0.000001; 0.1 makes the (1 + sigma_min * t) term visible
    generator = torch.Generator().manual_seed(0)
    x0 = torch.rand(4, 3, 16, 16, generator=generator) * 2 - 1
    mask = (torch.rand(4, 1, 16, 16, generator=generator) > 0.3).float()
    batch = {"gt_emission": x0, "mask_map": mask, "position_map": torch.zeros(4, 3, 16, 16)}

    data = system.prepare_diffusion_data(batch)
    target = data["sample_images"] - data["noise"]
    loss = system.get_diffusion_loss(target, data)
    assert set(loss) == {"mse", "l1"} and all(float(term) == 0.0 for term in loss.values())

    pred_x0 = system.get_batched_pred_x0(target, data["timesteps"], data["noisy_images"])
    torch.testing.assert_close(pred_x0 * mask, x0 * mask, rtol=0, atol=1e-5)


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
