"""Memorization check: fraction of each program's n-token windows that occur in train.bin.

  uv run python -m eval.overlap runs/gen/v0-trim.jsonl runs/gen/val-trim.jsonl --n 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MULT = np.uint64(0x9E3779B97F4A7C15)


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Polynomial hash of every length-n window (uint64, wrapping)."""
    x = ids.astype(np.uint64)
    if len(x) < n:
        return np.empty(0, np.uint64)
    h = np.zeros(len(x) - n + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * MULT + x[j: len(x) - n + 1 + j] + np.uint64(1)
    return h


def seen_fraction(ids: list[int], ref_sorted: np.ndarray, n: int) -> float | None:
    h = ngram_hashes(np.asarray(ids, np.uint16), n)
    if len(h) == 0:
        return None
    pos = np.searchsorted(ref_sorted, h).clip(max=len(ref_sorted) - 1)
    return float((ref_sorted[pos] == h).mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--train", type=Path, default=Path("data/tok/train.bin"))
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args(argv)
    ref = np.unique(ngram_hashes(np.fromfile(a.train, dtype="<u2"), a.n))
    for f in a.jsonl:
        fr = [seen_fraction(json.loads(l)["ids"][1:], ref, a.n) for l in f.read_text().splitlines()]
        fr = np.array([v for v in fr if v is not None])
        print(f"{f}: programs={len(fr)} mean_seen={fr.mean():.3f} median={np.median(fr):.3f} "
              f">=90%_seen={np.mean(fr >= 0.9):.3f} zero_seen={np.mean(fr == 0):.3f}")


if __name__ == "__main__":
    main()
