"""Acceptance check: val loss of the QAT model vs the integer reference on the same windows.

    uv run python -m model.evaluate --ckpt runs/v0/best.pt --export runs/v0/export \
        --data data/tok --windows 24
    uv run python -m model.evaluate --ckpt runs/lab/S0-s10/train/avg.pt --export runs/v2/export \
        --data data/tok --programs 0 --workers 24      # relative positions: whole val programs

The QAT reference runs the fake-quant torch model in float64 on CPU. The intref loss turns the
integer Q8 logits into log-probabilities with floats; that conversion is evaluation only.
`--programs N` (relative-position models) compares whole val programs (<bos> to <eos>, any
length) through the sliding window instead of fixed windows: the first N programs, 0 = all.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from model.data import IGNORE, load_split, val_windows
from model.intref import IntModel, expected_mults, sequence_logits
from model.model import load_checkpoint


def _nll(logits: np.ndarray, targets: np.ndarray) -> tuple[float, int]:
    lp = torch.log_softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()
    mask = targets != IGNORE
    picked = np.take_along_axis(lp, np.where(mask, targets, 0)[..., None], axis=-1)[..., 0]
    return float(-(picked * mask).sum()), int(mask.sum())


def compare(model, im: IntModel, xs: np.ndarray, ys: np.ndarray) -> dict:
    ref = model.double().eval()
    with torch.no_grad():
        want = ref(torch.as_tensor(xs, dtype=torch.long))[0].numpy()
    model.float()
    t0 = time.time()
    got = np.stack([sequence_logits(im, row) for row in xs])  # every window fed in full
    elapsed = time.time() - t0
    got = got / 256.0
    mask = ys != IGNORE
    q_sum, n = _nll(want, ys)
    i_sum, _ = _nll(got, ys)
    err = np.abs(want - got)[mask]
    qat, intl = q_sum / n, i_sum / n
    return {"qat_loss": qat, "intref_loss": intl, "rel_diff": (intl - qat) / qat,
            "top1_agree": float((want.argmax(-1) == got.argmax(-1))[mask].mean()),
            "max_logit_err": float(err.max()), "mean_logit_err": float(err.mean()),
            "n_targets": n, "intref_tokens_per_s": xs.size / elapsed}


_PROGRAM_MODELS: tuple = ()


def _program_init(ckpt: str, export: str) -> None:
    global _PROGRAM_MODELS
    torch.set_num_threads(1)
    _PROGRAM_MODELS = (load_checkpoint(ckpt).double().eval(), IntModel.load(export))


def _program_stats(ids: list[int]) -> dict:
    """QAT (float64) and intref logits over one whole program, fed from its <bos>."""
    model, im = _PROGRAM_MODELS
    xs, ys = np.asarray([ids[:-1]]), np.asarray([ids[1:]])
    with torch.no_grad():
        want = model(torch.as_tensor(xs, dtype=torch.long))[0].numpy()[0]
    got = sequence_logits(im, ids[:-1]) / 256.0
    q_sum, n = _nll(want[None], ys)
    i_sum, _ = _nll(got[None], ys)
    err = np.abs(want - got)
    return {"qat_sum": q_sum, "intref_sum": i_sum, "n": n,
            "agree": int((want.argmax(-1) == got.argmax(-1)).sum()),
            "err_sum": float(err.sum()), "err_count": int(err.size), "err_max": float(err.max())}


def compare_programs(ckpt, export, programs: list[list[int]], workers: int = 8) -> dict:
    """compare() for a relative-position model over whole programs (lists of ids from <bos> to
    <eos>, any length): every position sees its last ctx tokens, as on the calculator. Each
    worker runs both models on one program at a time (longest first)."""
    order = sorted(range(len(programs)), key=lambda i: -len(programs[i]))
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_program_init, initargs=(str(ckpt), str(export))) as ex:
        stats = list(ex.map(_program_stats, [programs[i] for i in order]))
    elapsed = time.time() - t0
    n = sum(s["n"] for s in stats)
    qat = sum(s["qat_sum"] for s in stats) / n
    intl = sum(s["intref_sum"] for s in stats) / n
    return {"qat_loss": qat, "intref_loss": intl, "rel_diff": (intl - qat) / qat,
            "top1_agree": sum(s["agree"] for s in stats) / n,
            "max_logit_err": max(s["err_max"] for s in stats),
            "mean_logit_err": sum(s["err_sum"] for s in stats) / sum(s["err_count"] for s in stats),
            "n_targets": n, "programs": len(programs),
            "longest_program": max(len(p) for p in programs), "wall_s": round(elapsed, 1)}


def val_programs(stream: np.ndarray) -> list[list[int]]:
    """Every program of a token stream, <bos> to <eos> inclusive."""
    starts = np.flatnonzero(stream == 1)
    ends = np.append(starts[1:], len(stream))
    return [stream[a:b].tolist() for a, b in zip(starts, ends)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--programs", type=int, default=None,
                    help="whole val programs through the sliding window (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    args = ap.parse_args(argv)
    if args.programs is not None:
        progs = val_programs(load_split(args.data / "val.bin"))[: args.programs or None]
        rep = compare_programs(args.ckpt, args.export, progs, args.workers)
        print(json.dumps(rep, indent=2))
        if args.out:
            args.out.write_text(json.dumps(rep, indent=2))
        return
    model = load_checkpoint(args.ckpt)
    im = IntModel.load(args.export)
    xs, ys = val_windows(load_split(args.data / "val.bin"), model.cfg.max_seq_len)
    n_all = len(xs)
    pick = np.linspace(0, n_all - 1, min(args.windows, n_all)).round().astype(int)
    rep = compare(model, im, xs[pick], ys[pick])
    last = im.cfg.max_seq_len - 1
    rep.update({"windows": len(pick), "of_windows": n_all,
                "mults_pos0": expected_mults(im.cfg, 0), f"mults_pos{last}": expected_mults(im.cfg, last)})
    print(json.dumps(rep, indent=2))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
