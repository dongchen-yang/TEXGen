#!/usr/bin/env python3
"""
Filter out samples whose emission map is entirely zero.

Default paths assume this script is run from the TEXGen directory. Use the CLI
flags to point at alternative parquet/data locations if needed.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Paths relative to repo root (../ from TEXGen/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "baked_uv"
DEFAULT_PARQUET = DEFAULT_DATA_ROOT / "df_SomgProc_final.parquet"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "df_SomgProc_final_emission_filtered.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove samples whose emission map is all zeros."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Input parquet file (df_SomgProc_final.parquet).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains the sample folders (e.g., data/baked_uv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the filtered parquet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging; process only the first N successful samples.",
    )
    parser.add_argument(
        "--write-removed",
        type=Path,
        default=None,
        help="Optional path to write the list of removed sample IDs (one per line).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for NPZ checks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Chunksize for executor.map to balance overhead vs. latency.",
    )
    return parser.parse_args()


def resolve_sample_paths(sample_id: str, row: pd.Series, data_root: Path) -> Tuple[Path, Path]:
    """Build sample directory and NPZ path from a parquet row."""
    ditem_dir = getattr(row, "ditem_dir", None)
    if ditem_dir is None or (isinstance(ditem_dir, float) and pd.isna(ditem_dir)):
        ditem_dir = f"{sample_id[:3]}-{sample_id[3:6]}/{sample_id}"

    sample_dir = data_root / ditem_dir
    npz_path = sample_dir / "somage.npz"
    return sample_dir, npz_path


def has_nonzero_emission(npz_path: Path) -> bool:
    """
    Check whether the emission map contains any non-zero values.

    Prefer the precomputed '#stat_emission_color_max' scalar when available
    (avoids loading the full 512x512x3 array). Fallback to checking the
    emission map itself if the stat is missing.
    """
    try:
        with np.load(npz_path, allow_pickle=False, mmap_mode="r") as data:
            if "#stat_emission_color_max" in data:
                max_val = float(np.asarray(data["#stat_emission_color_max"]).reshape(-1)[0])
                return max_val > 0.0

            key = (
                "emission_color"
                if "emission_color" in data
                else "emission"
                if "emission" in data
                else None
            )
            if key is None:
                return False

            emission = data[key]
            return bool(np.any(emission))
    except FileNotFoundError:
        # Missing NPZ is treated as invalid (filtered out)
        return False
    except Exception as exc:  # pragma: no cover - runtime diagnostics only
        print(f"Failed to read {npz_path}: {exc}")
        return False


def _check_sample(args: Tuple[str, Path]) -> Tuple[str, bool]:
    """Worker helper to keep process-pool pickling simple."""
    sample_id, npz_path = args
    return sample_id, has_nonzero_emission(npz_path)


def filter_samples(
    df: pd.DataFrame,
    data_root: Path,
    limit: Optional[int] = None,
    workers: int = 1,
    chunk_size: int = 64,
) -> Tuple[List[str], List[str]]:
    """Iterate over successful samples and split into keep/remove lists."""
    df_success = df[df["success"] == True] if "success" in df.columns else df
    if limit is not None:
        df_success = df_success.iloc[:limit]

    tasks = []
    for row in df_success.itertuples(index=True):
        sample_id = row.Index
        _, npz_path = resolve_sample_paths(sample_id, row, data_root)
        tasks.append((sample_id, npz_path))

    keep_ids: List[str] = []
    drop_ids: List[str] = []

    if workers <= 1:
        for sample_id, keep in tqdm(map(_check_sample, tasks), total=len(tasks), desc="Checking emission maps"):
            if keep:
                keep_ids.append(sample_id)
            else:
                drop_ids.append(sample_id)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for sample_id, keep in tqdm(
                executor.map(_check_sample, tasks, chunksize=chunk_size),
                total=len(tasks),
                desc="Checking emission maps",
            ):
                if keep:
                    keep_ids.append(sample_id)
                else:
                    drop_ids.append(sample_id)

    return keep_ids, drop_ids


def main() -> None:
    args = parse_args()

    # Safety: refuse to overwrite the original parquet
    if args.output.resolve() == args.parquet.resolve():
        raise SystemExit("Refusing to overwrite the input parquet; choose a different --output path.")

    print(f"Reading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"Total rows in parquet: {len(df)}")
    if "success" in df.columns:
        print(f"Rows with success=True: {len(df[df['success'] == True])}")

    keep_ids, drop_ids = filter_samples(
        df,
        args.data_root,
        limit=args.limit,
        workers=max(1, args.workers),
        chunk_size=max(1, args.chunk_size),
    )

    print(f"\nKeeping {len(keep_ids)} samples, removing {len(drop_ids)} samples.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_filtered = df.loc[keep_ids]
    df_filtered.to_parquet(args.output)
    print(f"Filtered parquet saved to: {args.output}")

    if args.write_removed:
        args.write_removed.parent.mkdir(parents=True, exist_ok=True)
        args.write_removed.write_text("\n".join(drop_ids))
        print(f"Removed sample IDs written to: {args.write_removed}")


if __name__ == "__main__":
    main()
