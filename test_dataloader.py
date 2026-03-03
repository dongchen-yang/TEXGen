#!/usr/bin/env python3
"""
Quick smoke-test for LightGenDataModule with the emission-filtered config.

Run from the TEXGen directory:
    conda run -n texgen python test_dataloader.py [--config configs/lightgen_pointuv_256_batch32_emission_filtered.yaml]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

# Make sure spuv is importable when run from TEXGen/
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/lightgen_pointuv_256_batch32_emission_filtered.yaml",
        help="Path to config YAML (relative to TEXGen/ dir).",
    )
    p.add_argument("--num-batches", type=int, default=3, help="Batches to load per split.")
    p.add_argument("--batch-size", type=int, default=2, help="Override batch size for fast testing.")
    p.add_argument("--num-workers", type=int, default=2, help="Override num_workers.")
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to test.")
    return p.parse_args()


def load_data_cfg(config_path: str, batch_size: int, num_workers: int) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    data_cfg["batch_size"] = batch_size
    data_cfg["eval_batch_size"] = batch_size
    data_cfg["num_workers"] = num_workers
    return data_cfg


def check_batch(batch: dict, split: str, batch_idx: int) -> None:
    """Print shapes and basic stats for one batch."""
    if batch is None:
        print(f"  [batch {batch_idx}] SKIPPED (collate returned None)")
        return

    print(f"  [batch {batch_idx}]  scene_ids: {batch['scene_id'][:2]} ...")
    for key in ["input_tensor", "gt_emission", "position_map", "albedo_map", "mask_map"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            t = batch[key]
            print(f"    {key:20s}  shape={tuple(t.shape)}  "
                  f"min={t.min():.3f}  max={t.max():.3f}  "
                  f"nan={t.isnan().any().item()}")

    # Sanity check: gt_emission should be in [-1, 1]
    em = batch.get("gt_emission")
    if em is not None:
        out_of_range = ((em < -1.01) | (em > 1.01)).sum().item()
        if out_of_range:
            print(f"    WARNING: {out_of_range} gt_emission values outside [-1, 1]")
        else:
            print(f"    gt_emission range OK (all within [-1, 1])")

    # Check thumbnail presence
    thumbs = batch.get("thumbnail")
    if isinstance(thumbs, list):
        n_none = sum(1 for t in thumbs if t is None)
        print(f"    thumbnail: {len(thumbs) - n_none}/{len(thumbs)} loaded")


def test_split(dm, split: str, num_batches: int) -> bool:
    print(f"\n{'='*60}")
    print(f"Testing split: {split.upper()}")
    print(f"{'='*60}")

    dm.setup(stage="fit" if split in ("train", "val") else "test")

    if split == "train":
        dataset = dm.train_dataset
        loader = dm.train_dataloader()
    elif split == "val":
        dataset = dm.val_dataset
        loader = dm.val_dataloader()
    else:
        dataset = dm.test_dataset
        loader = dm.test_dataloader()

    print(f"  Dataset size : {len(dataset)}")
    print(f"  Num batches  : {len(loader)}")

    ok = True
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        try:
            check_batch(batch, split, i)
        except Exception as e:
            print(f"  ERROR checking batch {i}: {e}")
            ok = False

    elapsed = time.time() - t0
    print(f"\n  Loaded {min(num_batches, len(loader))} batches in {elapsed:.1f}s")
    return ok


def main():
    args = parse_args()

    # Resolve config path relative to TEXGen/
    texgen_dir = Path(__file__).resolve().parent
    config_path = texgen_dir / args.config
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    print(f"Config : {config_path}")
    data_cfg = load_data_cfg(config_path, args.batch_size, args.num_workers)
    print(f"Data root  : {data_cfg['data_root']}")
    print(f"Parquet    : {data_cfg['parquet_file']}")
    print(f"Train idx  : {data_cfg['train_indices']}")
    print(f"Batch size : {args.batch_size}  |  Workers: {args.num_workers}")

    from spuv.data.lightgen_uv import LightGenDataModule

    dm = LightGenDataModule(data_cfg)

    all_ok = True
    for split in args.splits:
        all_ok &= test_split(dm, split, args.num_batches)

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - see output above")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
