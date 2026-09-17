import os

import pytest
import torch
from omegaconf import OmegaConf

import spuv
from spuv.utils.config import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(ROOT, "configs", "texgen_emission.yaml")
PARSED = os.path.join(ROOT, "outputs", "texgen_alpha_74k_v2_agentic", "configs", "parsed.yaml")
# The data, system and tokenizer classes import on the CPU; the backbone imports torchsparse, which needs CUDA.
CPU_CLS_KEYS = [("data_cls",), ("system_cls",), ("system", "image_tokenizer_cls")]


def _get(cfg, keys):
    for k in keys:
        cfg = cfg[k]
    return cfg


def test_live_config_resolves_and_keeps_the_alpha_guard():
    cfg = OmegaConf.load(CONFIG)
    for keys in CPU_CLS_KEYS:
        assert spuv.find(_get(cfg, keys)) is not None, keys
    assert cfg.system.backbone.in_channels == 13 and cfg.data.use_alpha is True
    assert cfg.system.backbone.out_channels == 3
    assert "use_gt_emission_mask_cond" not in cfg.system.backbone
    assert cfg.data_cls == "spuv.data.mesh_uv.MeshUVDataModule"
    assert cfg.system_cls == "spuv.systems.texgen_emission_test.TEXGenDiffusion"
    assert cfg.system.backbone_cls == "spuv.models.sparse_networks.texgen_emission_network.PointUVNet"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="resolving the backbone imports torchsparse, which needs a CUDA device")
def test_backbone_class_resolves_and_the_reexport_names_it():
    live = spuv.find(OmegaConf.load(CONFIG).system.backbone_cls)
    assert live is spuv.find("spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet")


@pytest.mark.skipif(not os.path.exists(PARSED), reason="the published run's parsed.yaml is not mirrored here")
def test_published_parsed_yaml_resolves_through_the_reexports_to_the_same_classes():
    live, parsed = OmegaConf.load(CONFIG), OmegaConf.load(PARSED)
    for keys in CPU_CLS_KEYS:
        assert spuv.find(_get(live, keys)) is spuv.find(_get(parsed, keys)), keys
    for section in ("backbone", "loss"):
        assert OmegaConf.to_container(live.system[section]) == OmegaConf.to_container(parsed.system[section]), section
    for key in ("uv_height", "uv_width", "use_alpha", "batch_size"):     # data locations will change for the retrain
        assert live.data[key] == parsed.data[key], key


@pytest.mark.skipif(not os.path.exists(PARSED), reason="the published run's parsed.yaml is not mirrored here")
def test_published_parsed_yaml_still_loads_through_load_config():
    # Every published parsed.yaml carries custom_output_dir: null, and parse_structured rejects unknown keys.
    cfg = load_config(PARSED, makedirs=False)
    assert cfg.tag == "texgen_alpha_74k_v2_agentic" and cfg.custom_output_dir is None
