#!/usr/bin/env python3
"""
Per-sample inference with an optional custom CLIP conditioning thumbnail.

Loads a checkpoint + the training config explicitly (both as CLI args), pulls a
single sample from the dataset by sample_id, optionally replaces the
`thumbnail` tensor with a user-supplied PNG, runs `model.test_pipeline` under
the EMA scope, and saves pred/gt/albedo/thumbnail/comparison PNGs.

Example:
  cd TEXGen
  python inference_custom_thumbnail.py \
      --checkpoint outputs/output_emission_filtered/last.ckpt \
      --config configs/lightgen_pointuv_256_batch32_emission_filtered.yaml \
      --sample-id fff48e914c4847a08660b9e08b1b733c \
      --thumbnail ../experiments/green_emission/edited_green_thumbnail.png \
      --output-dir ../experiments/green_emission/green
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_custom_thumbnail(path: str, device: torch.device) -> torch.Tensor:
    """Load a PNG and format it exactly like LightGenDataset returns it."""
    with Image.open(path) as pil:
        img = pil.convert("RGB")
        img = TF.resize(img, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [224, 224, 3]
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224, 3]
    return tensor.to(device)


def find_sample_in_test_split(dataset, sample_id: str) -> int:
    for i, s in enumerate(dataset.all_samples):
        if s["sample_id"] == sample_id:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True,
                        help="Path to the training YAML (e.g. emission_filtered vanilla).")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--thumbnail", default=None,
                        help="Optional custom PNG; if omitted, uses the dataset's original thumbnail.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    print("=" * 80)
    print("LightGen Custom-Thumbnail Inference")
    print("=" * 80)

    print(f"\n1. Loading config: {args.config}")
    full_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.create({
        "data_cls": full_cfg.get("data_cls"),
        "data": full_cfg.get("data"),
        "system_cls": full_cfg.get("system_cls"),
        "system": full_cfg.get("system"),
    })

    print(f"\n2. Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"   epoch={checkpoint.get('epoch', '?')} step={checkpoint.get('global_step', '?')}")

    print(f"\n3. Building dataset ({args.split} split)")
    from spuv.data.lightgen_uv import LightGenDataModule
    data_module = LightGenDataModule(cfg.data)
    data_module.setup(args.split if args.split != "train" else "fit")

    dataset = {
        "train": getattr(data_module, "train_dataset", None),
        "val": getattr(data_module, "val_dataset", None),
        "test": getattr(data_module, "test_dataset", None),
    }[args.split]
    if dataset is None:
        raise RuntimeError(f"Dataset for split '{args.split}' not initialized.")

    local_idx = find_sample_in_test_split(dataset, args.sample_id)
    if local_idx < 0:
        raise RuntimeError(
            f"Sample '{args.sample_id}' not found in the {args.split} split of the "
            f"filtered dataset ({len(dataset)} samples)."
        )
    print(f"   sample found at local index {local_idx}/{len(dataset)}")

    print(f"\n4. Building model")
    from spuv.systems.lightgen_system import LightGenSystem
    model = LightGenSystem(cfg.system)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_ema = cfg.system.get("use_ema", True)
    val_with_ema = cfg.system.get("val_with_ema", True)
    data_normalization = cfg.system.get("data_normalization", True)
    print(f"   use_ema={use_ema}  val_with_ema={val_with_ema}  data_normalization={data_normalization}")

    print(f"\n5. Collating batch")
    sample = dataset[local_idx]
    if sample is None:
        raise RuntimeError(f"Dataset returned None for index {local_idx}.")

    batch = {}
    for k, v in sample.items():
        if v is None:
            batch[k] = None
        elif isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, dict):
            batch[k] = [v]
        else:
            batch[k] = [v]

    if args.thumbnail is not None:
        print(f"   overriding thumbnail with: {args.thumbnail}")
        batch["thumbnail"] = load_custom_thumbnail(args.thumbnail, device)
        batch["clip_image_embedding"] = None  # ensure thumbnail path is used in prepare_condition_info
    else:
        print(f"   using dataset's original thumbnail")
    print(f"   batch['thumbnail'].shape = {tuple(batch['thumbnail'].shape)}")

    print(f"\n6. Running test_pipeline")
    output_dir = Path(args.output_dir) / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        if use_ema and val_with_ema:
            print("   Using EMA weights for inference")
            with model.ema_scope("Inference with ema weights"):
                outputs = model.test_pipeline(batch)
        else:
            outputs = model.test_pipeline(batch)

    pred = outputs["pred_x0"][0]   # [3, H, W]
    gt = outputs["gt_x0"][0]       # [3, H, W]
    mask = outputs["mask_map"][0, 0]  # [H, W]
    albedo = batch["albedo_map"][0]  # [3, H, W]

    if data_normalization:
        pred = (pred * 0.5 + 0.5).clamp(0, 1)
        gt = (gt * 0.5 + 0.5).clamp(0, 1)
    else:
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)

    m3 = mask.unsqueeze(0)
    pred = pred * m3
    gt = gt * m3
    albedo_masked = albedo * m3

    pred_img = pred.cpu().permute(1, 2, 0).numpy()
    gt_img = gt.cpu().permute(1, 2, 0).numpy()
    albedo_img = albedo_masked.cpu().permute(1, 2, 0).numpy()

    thumb_used = batch["thumbnail"][0, 0].cpu().numpy()  # [224, 224, 3]

    print(f"\n7. Saving outputs to {output_dir}/")
    Image.fromarray((pred_img * 255).astype(np.uint8)).save(output_dir / "pred_emission.png")
    Image.fromarray((gt_img * 255).astype(np.uint8)).save(output_dir / "gt_emission.png")
    Image.fromarray((albedo_img * 255).astype(np.uint8)).save(output_dir / "input_albedo.png")
    Image.fromarray((thumb_used * 255).astype(np.uint8)).save(output_dir / "thumbnail_used.png")

    h, w = pred_img.shape[:2]
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = (albedo_img * 255).astype(np.uint8)
    comparison[:, w:2 * w] = (gt_img * 255).astype(np.uint8)
    comparison[:, 2 * w:] = (pred_img * 255).astype(np.uint8)
    Image.fromarray(comparison).save(output_dir / "comparison.png")

    print("   wrote pred_emission.png, gt_emission.png, input_albedo.png, "
          "thumbnail_used.png, comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
