import os

import pytest
import torch

# torchsparse initializes CUDA when it is imported and raises RuntimeError without a device, which
# pytest.importorskip does not catch; check for the device before importing the network.
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device (torchsparse initializes CUDA at import)", allow_module_level=True)

os.environ["TEXGEN_ENABLE_FLASH"] = "0"      # the dense branch the published checkpoint trained and infers on

from spuv.utils.config import parse_structured  # noqa: E402
from spuv.models.sparse_networks.texgen_emission_network import PointUVNet  # noqa: E402

TINY = dict(
    in_channels=13, out_channels=3, num_layers=[1, 1, 1, 1, 1], point_block_num=[1, 1, 1, 1, 1],
    block_out_channels=[8, 16, 32, 32, 64], dropout=[0.0, 0.0, 0.0, 0.0, 0.0],
    block_type=["uv", "point_uv", "uv_dit", "uv_dit", "uv_dit"],
    voxel_size=[0.01, 0.02, 0.05, 0.05, 0.05], window_size=[0, 16, 16, 32, 32],
    num_heads=[1, 1, 2, 2, 2], skip_input=True, skip_type="adaptive", use_uv_head=True,
)


def _batch(B=1, H=64, W=64, material_channels=6, device="cuda"):
    g = torch.Generator().manual_seed(0)
    mask = torch.zeros(B, 1, H, W); mask[:, :, 8:56, 8:56] = 1
    return dict(
        x_dense=torch.randn(B, 3, H, W, generator=g), mask_map=mask,
        position_map=torch.rand(B, 3, H, W, generator=g) * 2 - 1, timestep=torch.rand(B, generator=g),
        clip_embeddings=[torch.zeros(B, 768), torch.zeros(B, 768)], mesh=None,
        image_info={"baked_texture": torch.rand(B, material_channels, H, W, generator=g), "baked_weights": mask,
                    "rgb_cond": None, "mvp_mtx_cond": None},
        data_normalization=True, condition_drop=torch.zeros(B),
    ), device


def _to(d, device):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v): out[k] = v.to(device)
        elif isinstance(v, list): out[k] = [t.to(device) for t in v]
        elif isinstance(v, dict): out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else: out[k] = v
    return out


def test_thirteen_channel_input_gives_three_channel_velocity():
    net = PointUVNet(parse_structured(PointUVNet.Config, TINY)).cuda().eval()
    batch, device = _batch()
    with torch.no_grad():
        out, _ = net(**_to(batch, device))
    assert out.shape == (1, 3, 64, 64)


def test_twelve_channel_batch_against_thirteen_channel_config_fails_loudly():
    net = PointUVNet(parse_structured(PointUVNet.Config, TINY)).cuda().eval()
    batch, device = _batch(material_channels=5)
    with pytest.raises(AssertionError, match="12ch but cfg.in_channels=13"):
        with torch.no_grad():
            net(**_to(batch, device))
