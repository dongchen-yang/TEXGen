"""Per-shape RNG seed. VERBATIM COPY of the parent repo's evaluation/newdata_eval/seedutil.py
(the canonical one, with the reasoning); the two repos cannot import each other on every host
this runs on, and tests/test_seed.py pins the two copies equal. (Ours, not upstream's.)"""
import hashlib


def sample_seed(sha: str, seed: int) -> int:
    """Per-shape RNG seed for `(sha, seed)`. Order-, batch- and skip-independent."""
    return int.from_bytes(hashlib.sha256(f"{sha}:{seed}".encode()).digest()[:8], "big")
