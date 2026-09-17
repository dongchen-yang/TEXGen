import importlib

import pytest
import torch

# The module imports torchsparse, which initializes CUDA at import (pytest.importorskip does not catch its RuntimeError).
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device (torchsparse initializes CUDA at import)", allow_module_level=True)


def _enabled(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TEXGEN_ENABLE_FLASH", raising=False)
    else:
        monkeypatch.setenv("TEXGEN_ENABLE_FLASH", value)
    import spuv.models.sparse_networks.texgen_emission_network as net
    return importlib.reload(net)._ENABLE_FLASH


def test_flash_gate_parses_the_env(monkeypatch):
    assert _enabled(monkeypatch, None) is True
    assert _enabled(monkeypatch, "1") is True
    assert _enabled(monkeypatch, "0") is False
    assert _enabled(monkeypatch, "false") is False
    assert _enabled(monkeypatch, "False") is False
