"""Kept for the published checkpoints: they pickle spuv.systems.texgen_base.LossConfig inside
optimizer_states (the object type of an OmegaConf node), so torch.load of any of them imports this
path. The code is spuv/systems/texgen_emission_base.py."""
from spuv.systems.texgen_emission_base import LossConfig  # noqa: F401
