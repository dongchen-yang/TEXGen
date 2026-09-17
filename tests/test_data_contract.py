import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spuv.data.mesh_uv import AlphaUnavailable, MeshUVDataModuleConfig, MeshUVDataset

SHA = "0123456789abcdef0123456789abcdef"


def _make_root(tmp_path, alpha_key=False, sidecar=True, res=32):
    root = tmp_path / "root"
    d = root / SHA
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    occ = np.zeros((res, res, 1), bool); occ[4:28, 4:28] = True
    arrays = dict(
        occupancy=occ,
        position=rng.integers(0, 65535, (res, res, 3), dtype=np.uint16),
        objnormal=rng.integers(0, 65535, (res, res, 3), dtype=np.uint16),
        color=rng.integers(0, 255, (res, res, 3), dtype=np.uint8),
        metal=rng.integers(0, 255, (res, res, 1), dtype=np.uint8),
        rough=rng.integers(0, 255, (res, res, 1), dtype=np.uint8),
        emission_color=np.zeros((res, res, 3), np.uint8),
    )
    arrays["emission_color"][10:14, 10:14] = 200
    alpha = np.full((res, res, 1), 128, np.uint8)
    if alpha_key:
        arrays["alpha"] = alpha
    np.savez(d / "somage.npz", **arrays)
    if sidecar:
        np.save(d / "alpha.npy", alpha)
    (root / "thumbnails").mkdir()
    Image.fromarray(np.full((64, 64, 3), 255, np.uint8)).save(root / "thumbnails" / f"{SHA}.png")
    # indexed by sha, as the real eval parquets are (index name ditem_id): the loader takes the sample id from the index
    pd.DataFrame({"ditem_dir": [SHA], "success": [True]}, index=pd.Index([SHA], name="ditem_id")).to_parquet(root / "eval.parquet")
    return root


def _cfg(root, use_alpha=True, uv=16):
    return MeshUVDataModuleConfig(
        data_root=str(root), parquet_file=str(root / "eval.parquet"),
        uv_height=uv, uv_width=uv, use_alpha=use_alpha, num_workers=0,
    )


def test_item_keys_shapes_and_ranges(tmp_path):
    ds = MeshUVDataset(_cfg(_make_root(tmp_path)), split="test")
    item = ds.try_get_item(0)
    assert item["scene_id"] == SHA
    for key, c in [("position_map", 3), ("normal_map", 3), ("albedo_map", 3), ("metal_map", 1),
                   ("rough_map", 1), ("alpha_map", 1), ("mask_map", 1), ("gt_emission", 3)]:
        assert item[key].shape == (c, 16, 16), key
    assert item["gt_emission"].min() >= -1 and item["gt_emission"].max() <= 1
    assert item["gt_emission"].max() > 0.5                       # the lit patch survived the resize
    assert item["alpha_map"].min() >= 0 and item["alpha_map"].max() <= 1
    assert abs(item["alpha_map"].mean().item() - 128 / 255) < 1e-3   # bilinear of a constant plane
    assert set(item["mask_map"].unique().tolist()) <= {0.0, 1.0}
    assert item["thumbnail"].shape == (1, 224, 224, 3)
    assert "gt_emission_mask" not in item and "clip_image_embedding" not in item
    assert item["position_map"].min() >= -2 and item["position_map"].max() <= 2


def test_no_alpha_key_when_use_alpha_false(tmp_path):
    ds = MeshUVDataset(_cfg(_make_root(tmp_path), use_alpha=False), split="test")
    assert "alpha_map" not in ds.try_get_item(0)


def test_missing_alpha_raises_at_construction(tmp_path):
    root = _make_root(tmp_path, sidecar=False)
    with pytest.raises(AlphaUnavailable):
        MeshUVDataset(_cfg(root), split="test")


def test_npz_alpha_key_wins_over_sidecar(tmp_path):
    root = _make_root(tmp_path, alpha_key=True, sidecar=True)
    np.save(root / SHA / "alpha.npy", np.zeros((32, 32, 1), np.uint8))   # a sidecar that would read as 0
    item = MeshUVDataset(_cfg(root), split="test").try_get_item(0)
    assert abs(item["alpha_map"].mean().item() - 128 / 255) < 1e-3


def test_split_json_indices_are_positional_into_the_success_rows(tmp_path):
    # Index 1 is the second success row, not the second parquet row: the failed row is dropped first.
    root = _make_root(tmp_path)
    pd.DataFrame({"ditem_dir": ["a" * 32, "b" * 32, SHA], "success": [True, False, True]},
                 index=pd.Index(["a" * 32, "b" * 32, SHA], name="ditem_id")).to_parquet(root / "eval.parquet")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": {"indices": [1]}, "train": {"indices": []}}))
    cfg = _cfg(root, use_alpha=False); cfg.test_indices = str(split)
    assert [s["sample_id"] for s in MeshUVDataset(cfg, split="test").all_samples] == [SHA]


def test_current_bake_layout_atlas_npz_with_alpha_key_and_no_sidecar(tmp_path):
    root = _make_root(tmp_path, alpha_key=True, sidecar=False)
    (root / SHA / "somage.npz").rename(root / SHA / "atlas.npz")       # the datasets/dataset_73k layout
    item = MeshUVDataset(_cfg(root), split="test").try_get_item(0)
    assert abs(item["alpha_map"].mean().item() - 128 / 255) < 1e-3


def test_atlas_npz_wins_over_somage_npz(tmp_path):
    # Both files side by side: the current bake's atlas.npz is the one the loader reads.
    root = _make_root(tmp_path, alpha_key=True, sidecar=False)
    arrays = dict(np.load(root / SHA / "somage.npz"))
    arrays["alpha"] = np.full((32, 32, 1), 200, np.uint8)
    np.savez(root / SHA / "atlas.npz", **arrays)
    ds = MeshUVDataset(_cfg(root), split="test")
    assert ds.all_samples[0]["npz_file"].endswith("atlas.npz")
    assert abs(ds.try_get_item(0)["alpha_map"].mean().item() - 200 / 255) < 1e-3
