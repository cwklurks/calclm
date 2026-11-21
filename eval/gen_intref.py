"""Generate programs with the integer reference (what the calculator computes) in parallel.

  uv run python -m eval.gen_intref --export runs/v0/export --out runs/gen/v0.jsonl --n 500
  uv run python -m eval.gen_intref --export runs/v2/export --out runs/gen/v2-1024.jsonl \
      --n 1000 --cap 1024 --masks                  # v2: sliding window, the tracker's masks

Writes one {"seed", "temp", "top_k", "cap", "masks", "ids"} record per line, in seed order,
for `python -m eval.run_programs --ids <out> --vocab corpus/vocab.json` or lab.score.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from model.intref import IntModel, SamplerConfig, generate

_model: IntModel | None = None
_mask = None


def _load(export: str, masks: bool) -> None:
    global _model, _mask
    _model = IntModel.load(export)
    if masks:
        from model.intmask import IntMask
        _mask = IntMask(_model.vocab)


def _one(seed: int, temp: float, top_k: int, max_new: int, cap: int | None) -> dict:
    cfg = SamplerConfig(round(temp * 256), top_k, False, seed)
    ids = generate(_model, [], max_new, cfg, cap=cap, mask=_mask)   # generate resets the mask
    return {"seed": seed, "temp": temp, "top_k": top_k, "cap": cap, "masks": _mask is not None,
            "ids": [int(t) for t in ids]}


def generate_jsonl(export: Path, out: Path, seeds: list[int], temp: float, top_k: int,
                   max_new: int, workers: int, cap: int | None = None,
                   masks: bool = False) -> Path:
    """cap: tokens in all, <bos> included (None: intref's default, the context for learned
    positions and 1024 for relative ones); masks: the tracker's logit masks (INTREF.md 13)."""
    job = partial(_one, temp=temp, top_k=top_k, max_new=max_new, cap=cap)
    with ProcessPoolExecutor(workers, initializer=_load, initargs=(str(export), masks)) as pool:
        records = list(pool.map(job, seeds))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=None,
                    help="default: cap - 1 (255 without --cap)")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    max_new = a.max_new if a.max_new is not None else (a.cap or 256) - 1
    print(generate_jsonl(a.export, a.out, seeds, a.temp, a.top_k, max_new, a.workers, a.cap,
                         a.masks))


if __name__ == "__main__":
    main()
