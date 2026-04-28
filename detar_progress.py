#!/usr/bin/env python3
"""Extract NPZ chunk tars in parallel with a single tqdm progress bar.

Usage:
    python detar_progress.py [TARS_DIR] [DEST] [PATTERN]

Defaults match fir layout: tars under ~/scratch/lightgen/data/tars/, dest /tmp/baked_uv,
pattern "npz_chunk_0?.tar".

Skips upfront `tar tf` counting (which reads each tar end-to-end and adds ~2 min for 8 chunks)
by using known entry counts for the npz_chunk_*.tar archives we shipped. Falls back to
counting if a tar's name isn't recognized.
"""
import subprocess
import sys
import threading
from pathlib import Path

from tqdm import tqdm

# Known entry counts for the 8 NPZ chunks (3 entries per sample dir × samples per chunk).
# 7 chunks × 9295 + 1 chunk × 9288 = 74353 samples; entries = samples × 3.
KNOWN_ENTRY_COUNTS = {
    "npz_chunk_00.tar": 27885,
    "npz_chunk_01.tar": 27885,
    "npz_chunk_02.tar": 27885,
    "npz_chunk_03.tar": 27885,
    "npz_chunk_04.tar": 27885,
    "npz_chunk_05.tar": 27885,
    "npz_chunk_06.tar": 27885,
    "npz_chunk_07.tar": 27864,
}


def main():
    tars_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/dya78/scratch/lightgen/data/tars")
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/baked_uv")
    pattern = sys.argv[3] if len(sys.argv) > 3 else "npz_chunk_0?.tar"

    dest.mkdir(parents=True, exist_ok=True)
    tars = sorted(tars_dir.glob(pattern))
    if not tars:
        sys.exit(f"[detar] no tars match {tars_dir}/{pattern}")

    print(f"[detar] {len(tars)} tar(s) -> {dest}", flush=True)
    total = 0
    for t in tars:
        n = KNOWN_ENTRY_COUNTS.get(t.name)
        if n is None:
            n = int(subprocess.check_output(["bash", "-c", f"tar tf {t} | wc -l"]).strip())
            print(f"[detar]   {t.name}: {n} (counted)")
        total += n
    print(f"[detar] total: {total} entries", flush=True)

    bar = tqdm(total=total, unit=" entries", desc="extract", smoothing=0.05)
    lock = threading.Lock()

    def extract(tar):
        p = subprocess.Popen(
            ["tar", "-xvf", str(tar), "-C", str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        for _ in p.stdout:
            with lock:
                bar.update(1)
        p.wait()

    threads = [threading.Thread(target=extract, args=(t,)) for t in tars]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bar.close()
    print(f"[detar] done -> {dest}")


if __name__ == "__main__":
    main()
