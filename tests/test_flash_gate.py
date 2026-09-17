import importlib.util

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
    # A fresh copy, not importlib.reload: reloading would leave the session's module on the last
    # value tested and hand every later test a new PointUVNet class object.
    spec = importlib.util.find_spec("spuv.models.sparse_networks.texgen_emission_network")
    net = importlib.util.module_from_spec(spec)      # module_from_spec sets __package__, so the relative imports resolve
    spec.loader.exec_module(net)
    return net._ENABLE_FLASH


def test_flash_gate_parses_the_env(monkeypatch):
    assert _enabled(monkeypatch, None) is True
    assert _enabled(monkeypatch, "1") is True
    assert _enabled(monkeypatch, "0") is False
    assert _enabled(monkeypatch, "false") is False
    assert _enabled(monkeypatch, "False") is False
