import torch

from spuv.utils.uv_metrics import denorm_masked, emission_mask, fit_height, flip_for_view, rgb_panel, uv_mse_psnr


def test_denorm_masked_maps_minus1_1_to_0_1_and_zeroes_outside_mask():
    x = torch.tensor([[[[-1.0, 1.0], [0.0, 0.5]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    out = denorm_masked(x, mask)
    assert torch.allclose(out, torch.tensor([[[[0.0, 1.0], [0.0, 0.75]]]]))
    assert torch.allclose(denorm_masked(x, mask, normalized=False), x * mask)


def test_emission_mask_thresholds_the_channel_max():
    img = torch.zeros(1, 3, 2, 2)
    img[0, 2, 0, 1] = 0.002
    m = emission_mask(img, 0.001)
    assert m.shape == (1, 1, 2, 2) and m[0, 0, 0, 1] == 1.0 and m.sum() == 1.0


def test_flip_for_view_flips_rows():
    img = torch.arange(4.0).reshape(1, 1, 2, 2)
    assert torch.equal(flip_for_view(img)[0, 0, 0], img[0, 0, 1])


def test_uv_mse_psnr_identical_and_known():
    a = torch.rand(1, 3, 8, 8)
    mse, psnr = uv_mse_psnr(a, a)
    assert mse == 0 and psnr == 80.0                    # -10*log10(1e-8)
    mse, psnr = uv_mse_psnr(torch.zeros(1, 3, 4, 4), torch.full((1, 3, 4, 4), 0.1))
    assert abs(mse.item() - 0.01) < 1e-7 and abs(psnr.item() - 20.0) < 1e-4


def test_rgb_panel_shape_of_entry():
    img = torch.zeros(4, 4, 3)
    p = rgb_panel(img)
    assert p == {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}


def test_fit_height_pads_and_crops():
    short = torch.ones(224, 224, 3)
    tall = fit_height(short, 256)
    assert tall.shape == (256, 224, 3) and tall[:16].sum() == 0 and tall[16:240].sum() == 224 * 224 * 3
    cropped = fit_height(torch.ones(300, 10, 3), 256)
    assert cropped.shape == (256, 10, 3)
    assert fit_height(short, 224) is short
