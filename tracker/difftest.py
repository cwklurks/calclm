"""T2 differential test: C tracker (bind.batch) vs Python spec (spec.batch).

    uv run python -m tracker.difftest            # every val and train program + fuzz
    uv run python -m tracker.difftest --train 0  # val + fuzz only

Programs are split at <bos>; chunks of whole programs run in worker processes (the spec is
pure Python). Exit status 1 on the first mismatch, which is printed with its context.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from . import bind, spec

ROOT = Path(__file__).resolve().parent.parent
TOK = ROOT / "data" / "tok"
BOS, EOS, PAD = 1, 2, 0


def split_programs(stream: np.ndarray) -> list[np.ndarray]:
    stream = np.asarray(stream, dtype=np.uint16)
    starts = np.flatnonzero(stream == BOS).tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    return [stream[a:b] for a, b in zip(starts, starts[1:] + [len(stream)])]


def sample_programs(stream: np.ndarray, n: int, seed: int) -> np.ndarray:
    progs = split_programs(stream)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(progs), size=min(n, len(progs)), replace=False)
    return np.concatenate([progs[i] for i in sorted(pick)])


def _vocab() -> list[str]:
    return list(spec.default_vocab())


def fuzz_stream(seed: int, n_programs: int, mean_len: int = 120) -> np.ndarray:
    """Random programs biased toward the tokens the tracker reacts to, plus mutated val
    programs when the corpus is available. Includes <pad>, ids past the vocab and programs
    without <eos>."""
    rng = np.random.default_rng(seed)
    vocab = _vocab()
    idx = {h: i for i, h in enumerate(vocab)}
    struct_hex = ["3F", "3E", "2A", "04", "2B", "10", "11", "08", "09", "06", "07", "CE", "CF",
                  "D0", "D4", "D3", "D1", "D2", "D6", "D7", "E6", "DA", "DB", "EB", "5F",
                  "BB54", "DC", "DD", "BB4A", "98", "99", "9A", "9B", "B5", "BB55", "BB3A",
                  "BB39", "FA", "DE", "BB68", "BB69", "6212", "6203", "6221", "F2", "BB3B"]
    struct_hex += [f"AA0{k}" for k in range(10)] + [f"5C0{k}" for k in range(10)]
    struct_hex += [f"600{k}" for k in range(10)] + [f"610{k}" for k in range(10)] + ["5D00"]
    names = [f"{c:02X}" for c in range(0x41, 0x5B)] + [f"{c:02X}" for c in range(0x30, 0x3A)]
    names.append("5B")
    struct = np.array([idx[h] for h in struct_hex if h in idx])
    name_ids = np.array([idx[h] for h in names])
    real = split_programs(np.fromfile(TOK / "val.bin", dtype="<u2")) \
        if (TOK / "val.bin").exists() else []
    out: list[int] = []
    for _ in range(n_programs):
        if real and rng.random() < 0.3:
            prog = real[rng.integers(len(real))][1:].tolist()[: rng.integers(10, 600)]
            for _ in range(rng.integers(1, 12)):          # point mutations
                j = int(rng.integers(len(prog))) if prog else 0
                tok = int(rng.choice(struct) if rng.random() < 0.7 else rng.integers(4, 781))
                if prog and rng.random() < 0.5:
                    prog[j] = tok
                else:
                    prog.insert(j, tok)
        else:
            n = int(rng.integers(1, 2 * mean_len))
            r = rng.random(n)
            prog = np.where(r < 0.45, rng.choice(struct, n),
                            np.where(r < 0.85, rng.choice(name_ids, n),
                                     rng.integers(3, 781, n))).tolist()
            for j in np.flatnonzero(rng.random(n) < 0.004):
                prog[j] = int(rng.choice([PAD, 900, 65535]))
        out += [BOS] + prog + ([EOS] if rng.random() < 0.8 else [])
    return np.array(out, dtype=np.uint16)


def compare(stream: np.ndarray) -> str | None:
    """None if C and spec agree on every token, else a description of the first difference."""
    stream = np.asarray(stream, dtype=np.uint16)
    c_cats, c_bags = bind.batch(stream)
    s_cats, s_bags = spec.batch(stream)
    diff = np.flatnonzero((c_cats != s_cats).any(axis=1) | (c_bags != s_bags))
    if not len(diff):
        return None
    i = int(diff[0])
    start = int(np.flatnonzero(stream[: i + 1] == BOS)[-1]) if (stream[: i + 1] == BOS).any() else 0
    vocab = _vocab()
    ctx = [vocab[t] if t < len(vocab) else str(t) for t in stream[max(start, i - 12): i + 1]]
    return (f"{len(diff)} differing tokens; first at {i} (program starting at {start}): "
            f"tokens ...{ctx} C={c_cats[i].tolist()} {int(c_bags[i]):#x} "
            f"spec={s_cats[i].tolist()} {int(s_bags[i]):#x}")


def spec_summaries(stream: np.ndarray) -> np.ndarray:
    rows = []
    for prog in split_programs(stream):
        t = spec.Tracker()
        for tok in prog.tolist():
            t.step(tok)
        rows.append([t.summary()[k] for k in bind.SUMMARY_KEYS])
    return np.array(rows, dtype=np.uint16).reshape(-1, 8)


def compare_summaries(stream: np.ndarray) -> str | None:
    c = bind.summaries(stream)
    s = spec_summaries(stream)
    if c.shape == s.shape and np.array_equal(c, s):
        return None
    bad = np.flatnonzero((c != s).any(axis=1)) if c.shape == s.shape else [0]
    i = int(bad[0])
    return f"summary differs for program {i}: C={c[i].tolist()} spec={s[i].tolist()}"


def _work(chunk: np.ndarray) -> tuple[int, str | None]:
    return len(chunk), compare(chunk) or compare_summaries(chunk)


def _chunks(stream: np.ndarray, size: int) -> list[np.ndarray]:
    progs = split_programs(stream)
    out, cur, n = [], [], 0
    for p in progs:
        cur.append(p)
        n += len(p)
        if n >= size:
            out.append(np.concatenate(cur))
            cur, n = [], 0
    if cur:
        out.append(np.concatenate(cur))
    return out


def run(stream: np.ndarray, label: str, workers: int) -> bool:
    t0 = time.perf_counter()
    chunks = _chunks(stream, 200_000)
    bind.build()
    with mp.get_context("fork").Pool(workers) as pool:
        for n, err in pool.imap_unordered(_work, chunks):
            if err:
                print(f"{label}: MISMATCH {err}")
                return False
    n_prog = len(split_programs(stream))
    print(f"{label}: {n_prog:,} programs, {len(stream):,} tokens identical "
          f"(features and summaries) in {time.perf_counter() - t0:.0f} s")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", type=int, default=1, help="1: also every train program")
    p.add_argument("--fuzz", type=int, default=20000, help="fuzzed programs")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args(argv)
    ok = run(fuzz_stream(seed=12345, n_programs=a.fuzz), "fuzz", a.workers)
    ok = ok and run(np.fromfile(TOK / "val.bin", dtype="<u2"), "val", a.workers)
    if a.train:
        ok = ok and run(np.fromfile(TOK / "train.bin", dtype="<u2"), "train", a.workers)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
