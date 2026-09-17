"""The system classes the checkpoints and Lightning reach by name (CPU: nothing here imports the network)."""
import importlib

from spuv.systems.texgen_emission_base import LossConfig, TEXGenBaseSystem
from spuv.systems.texgen_emission_test import TEXGenDiffusion


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
