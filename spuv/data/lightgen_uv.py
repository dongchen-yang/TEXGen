"""Kept for the published checkpoints: every configs/parsed.yaml names
spuv.data.lightgen_uv.LightGenDataModule. The code is spuv/data/mesh_uv.py."""
from spuv.data.mesh_uv import (  # noqa: F401
    AlphaUnavailable, decode_uint16_to_float, decode_uint8_to_float,
    MeshUVDataModule as LightGenDataModule,
    MeshUVDataModuleConfig as LightGenDataModuleConfig,
    MeshUVDataset as LightGenDataset,
)
