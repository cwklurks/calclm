"""Human reference row for G2a: val programs judged like generated ones.

    uv run python -m lab.reference --name REF-val-1024 --cap 1024

Each val program (<bos> .. <eos>) is cut at a line boundary only if it is longer than the cap,
run on emulated TI-OS with lab.score (same trimming, outcomes and metrics as samples), and
saved to runs/lab/NAME/result.json. distinct-4 depends on the number of programs (666 here,
1000 for samples).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lab.score import score
from model.data import load_split
from model.tokens import BOS

ROOT = Path(__file__).resolve().parents[1]


def programs(stream, limit: int = 0) -> list[list[int]]:
    starts = np.flatnonzero(np.asarray(stream) == BOS)
    ends = np.append(starts[1:], len(stream))
    out = [np.asarray(stream[lo:hi]).astype(int).tolist() for lo, hi in zip(starts, ends)]
    return out[: limit or None]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="REF-val-1024")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "tok")
    ap.add_argument("--vocab", type=Path, default=ROOT / "corpus" / "vocab.json")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "lab")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--budget-ms", type=int, default=30_000)
    ap.add_argument("--limit", type=int, default=0, help="first N val programs (0 = all)")
    a = ap.parse_args(argv)
    progs = programs(load_split(a.data / "val.bin"), a.limit)
    run_dir = a.out_root / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc = score(progs, run_dir / "eval", a.vocab, a.data / "train.bin", a.workers, a.budget_ms, a.cap)
    res = {"name": a.name, "cap": a.cap, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": str(a.data / "val.bin"), "score": sc, "score_s": round(time.time() - t0, 2)}
    (run_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
