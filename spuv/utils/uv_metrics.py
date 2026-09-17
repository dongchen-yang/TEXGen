"""Small tensor helpers shared by TEXGen-Emission's training, validation and previews.

Each of these was an inline copy, repeated in the fork's patches to upstream TEXGen; one
function each, so the numerics are defined once. (Ours, not upstream's.)
"""
import torch


def denorm_masked(x, mask, normalized=True):
    """Map a [-1, 1] map to [0, 1] (or leave a [0, 1] map alone) and zero it outside the UV islands."""
    img = (x * 0.5 + 0.5) if normalized else x
    return img * mask


def emission_mask(img01, threshold):
    """[B, 1, H, W] float mask: 1 where any channel is above `threshold`."""
    return (img01.max(dim=1, keepdim=True)[0] > threshold).float()


def flip_for_view(img):
    """Flip the V axis of a [B, C, H, W] UV map for viewing; never feed the flipped map to a rasterizer."""
    return torch.flip(img, dims=[2])


def uv_mse_psnr(pred01, gt01):
    """Mean squared error over every pixel and the PSNR of it (1e-8 floor, so identical maps give 80 dB)."""
    mse = torch.mean((pred01 - gt01) ** 2)
    psnr = -10 * torch.log10(mse + 1e-8)
    return mse, psnr


def rgb_panel(img_hwc):
    """One entry of the list `SaverMixin.save_image_grid` consumes."""
    return {"type": "rgb", "img": img_hwc, "kwargs": {"data_format": "HWC"}}


def fit_height(img_hwc, target_h):
    """Pad rows (zeros, centered) or center-crop rows so an [H, W, C] image has `target_h` rows."""
    h = img_hwc.shape[0]
    if h == target_h:
        return img_hwc
    if h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        return torch.nn.functional.pad(
            img_hwc.permute(2, 0, 1), (0, 0, pad_top, pad_bottom), mode="constant", value=0
        ).permute(1, 2, 0)
    start = (h - target_h) // 2
    return img_hwc[start:start + target_h]
