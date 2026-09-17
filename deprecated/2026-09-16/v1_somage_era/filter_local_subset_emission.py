#!/usr/bin/env python3
"""
Filter the baked_uv_local_subset dataset by removing samples whose emission map
is entirely zero, then update data_splits_filtered.json to reflect the change.

Outputs (all inside --data-root):
  df_SomgProc_emission_filtered.parquet  - parquet with zero-emission rows removed
  data_splits_emission_filtered.json     - updated split config (indices remapped)
  emission_filter_kept.txt               - one ditem_id per line for kept samples
  emission_filter_removed.txt            - one ditem_id per line for removed samples
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "baked_uv_local_subset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of the local subset (default: data/baked_uv_local_subset).",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Input parquet (default: <data-root>/df_SomgProc_filtered.parquet).",
    )
    p.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Input splits JSON (default: <data-root>/data_splits_filtered.json).",
    )
    p.add_argument(
        "--verify-npz",
        action="store_true",
        help=(
            "For samples where #stat_emission_color_max is missing, "
            "fall back to loading the actual NPZ file."
        ),
    )
    return p.parse_args()


def _has_nonzero_emission_npz(npz_path: Path) -> bool:
    """Fallback: check emission key inside a somage.npz."""
    try:
        with np.load(npz_path, allow_pickle=False, mmap_mode="r") as d:
            key = (
                "emission_color" if "emission_color" in d
                else "emission" if "emission" in d
                else None
            )
            if key is None:
                return False
            return bool(np.any(d[key]))
    except Exception as exc:
        print(f"  Warning: failed to read {npz_path}: {exc}")
        return False


def classify_samples(
    df: pd.DataFrame,
    data_root: Path,
    verify_npz: bool,
) -> tuple[list[str], list[str]]:
    """Return (keep_ids, remove_ids) lists based on emission stat."""
    keep_ids: list[str] = []
    remove_ids: list[str] = []

    for ditem_id, row in df.iterrows():
        stat = row.get("#stat_emission_color_max", None)

        if stat is not None and not (isinstance(stat, float) and np.isnan(stat)):
            has_emission = float(stat) > 0.0
        elif verify_npz:
            npz_path = data_root / row["ditem_dir"] / "somage.npz"
            has_emission = _has_nonzero_emission_npz(npz_path)
        else:
            # No stat and not verifying — treat as zero emission (filtered out)
            has_emission = False

        if has_emission:
            keep_ids.append(ditem_id)
        else:
            remove_ids.append(ditem_id)

    return keep_ids, remove_ids


def update_splits(
    splits: dict,
    old_index_to_new: dict[int, int],
    remove_set_old_indices: set[int],
) -> dict:
    """
    Return a new splits dict with index lists updated to reflect the filtered parquet.

    Old positional indices that were removed are dropped; the rest are remapped to
    their new positions in the filtered parquet.
    """
    updated = {k: v for k, v in splits.items() if not isinstance(v, dict)}

    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            continue
        entry = splits[split_name]
        old_indices = entry.get("indices", [])
        new_indices = sorted(
            old_index_to_new[i]
            for i in old_indices
            if i not in remove_set_old_indices
        )
        updated[split_name] = {
            "count": len(new_indices),
            "indices": new_indices,
        }

    return updated


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root

    parquet_path = args.parquet or data_root / "df_SomgProc_filtered.parquet"
    splits_path = args.splits_json or data_root / "data_splits_filtered.json"

    out_parquet = data_root / "df_SomgProc_emission_filtered.parquet"
    out_splits = data_root / "data_splits_emission_filtered.json"
    out_kept = data_root / "emission_filter_kept.txt"
    out_removed = data_root / "emission_filter_removed.txt"

    # ------------------------------------------------------------------ load
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total rows: {len(df)}")

    print(f"Loading splits: {splits_path}")
    with splits_path.open() as f:
        splits = json.load(f)

    # ------------------------------------------------------------------ classify
    print("Classifying samples by emission …")
    keep_ids, remove_ids = classify_samples(df, data_root, args.verify_npz)
    print(f"  Keep: {len(keep_ids)}  |  Remove (zero emission): {len(remove_ids)}")

    # ------------------------------------------------------------------ filtered parquet
    df_keep = df.loc[keep_ids]

    # Build old-positional-index → new-positional-index mapping
    # (old index = position in original df; new = position in df_keep)
    old_positions: dict[str, int] = {ditem_id: i for i, ditem_id in enumerate(df.index)}
    new_positions: dict[str, int] = {ditem_id: i for i, ditem_id in enumerate(df_keep.index)}

    remove_set_old = {old_positions[sid] for sid in remove_ids if sid in old_positions}
    old_index_to_new = {
        old_positions[sid]: new_positions[sid]
        for sid in keep_ids
        if sid in old_positions and sid in new_positions
    }

    # ------------------------------------------------------------------ save parquet
    df_keep.to_parquet(out_parquet)
    print(f"Filtered parquet saved → {out_parquet}")

    # ------------------------------------------------------------------ update splits JSON
    updated_splits = update_splits(splits, old_index_to_new, remove_set_old)
    # Update metadata counts
    updated_splits["total_annotated"] = len(df_keep)
    updated_splits["matched"] = len(df_keep)
    updated_splits["missing"] = 0
    updated_splits["note"] = (
        f"Emission-filtered subset of {splits_path.name}; "
        f"{len(remove_ids)} zero-emission samples removed."
    )

    with out_splits.open("w") as f:
        json.dump(updated_splits, f, indent=2)
    print(f"Updated splits JSON saved → {out_splits}")

    # ------------------------------------------------------------------ save id lists
    out_kept.write_text("\n".join(keep_ids) + "\n")
    out_removed.write_text("\n".join(remove_ids) + "\n")
    print(f"Kept IDs   → {out_kept}  ({len(keep_ids)} entries)")
    print(f"Removed IDs → {out_removed}  ({len(remove_ids)} entries)")

    # ------------------------------------------------------------------ summary
    print("\n=== Split summary ===")
    for split in ("train", "val", "test"):
        if split in updated_splits:
            print(f"  {split}: {updated_splits[split]['count']}")


if __name__ == "__main__":
    main()
