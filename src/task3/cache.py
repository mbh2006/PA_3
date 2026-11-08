"""On-disk cache for the expensive MATH pairwise equivalence matrices."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_VERSION = "math-equiv-v1"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(generations_path: Path) -> str:
    return f"{CACHE_VERSION}:{file_sha256(generations_path)}"


def cache_path(cache_dir: Path, dataset: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{dataset}_equivalence.npz"


def load(cache_dir: Path, dataset: str, key: str) -> dict[str, np.ndarray] | None:
    path = cache_path(cache_dir, dataset)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["key"]) != key:
            return None
        ids = data["sample_ids"]
        mats = data["matrices"]
        n = int(data["n"])
        return {str(sid): np.unpackbits(mats[i])[: n * n].reshape(n, n).astype(bool)
                for i, sid in enumerate(ids)}
    except Exception:
        return None


def save(cache_dir: Path, dataset: str, key: str, items: dict[str, np.ndarray]) -> Path:
    path = cache_path(cache_dir, dataset)
    ids = sorted(items)
    n = items[ids[0]].shape[0]
    packed = np.stack([np.packbits(items[sid].astype(np.uint8)) for sid in ids])
    np.savez_compressed(path, key=np.array(key), sample_ids=np.array(ids),
                        matrices=packed, n=np.array(n))
    return path
