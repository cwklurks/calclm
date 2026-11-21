"""Token-stream data: the data/tok format (little-endian uint16 ids) and window sampling.

Windowing decision (documented in model/INTREF.md, "Training data"): programs are packed
back to back as <bos> ... <eos> and windows do not stop at program boundaries. A training
window starts at a program start (<bos> at position 0, exactly how the calculator generates)
with probability p_bos, otherwise at a uniformly random stream offset so the tails of
programs longer than the context are also trained on. Validation uses contiguous,
non-overlapping windows that cover every target token exactly once.

Per-token input features (v2): a directory with `<split>.cats.npy` (N, n_cats) uint8,
`<split>.bag.npy` (N,) uint64 and `feats.json` ({"cats": [values per feature], "bag": bits,
...}), N = len(<split>.bin). Row i is the state after stream token i, computed from the
start of its program (the step function resets at <bos>), so a window starting mid-program
gets the state built before it. Windows gather features at the same offsets as tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.tokens import BOS, PAD

IGNORE = -100


def write_split(path: str | Path, stream: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(stream, dtype="<u2").tofile(path)


def load_split(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2")


def program_starts(stream: np.ndarray) -> np.ndarray:
    return np.flatnonzero(stream == BOS)


def sample_offsets(stream, batch, ctx, rng, p_bos=0.5, starts=None):
    """Start offsets of `batch` random training windows (see the module doc)."""
    last = len(stream) - ctx - 1
    starts = program_starts(stream) if starts is None else starts
    starts = starts[starts <= last]
    random_off = rng.integers(0, last + 1, size=batch)
    bos_off = starts[rng.integers(0, len(starts), size=batch)] if len(starts) else random_off
    use_bos = rng.random(batch) < p_bos
    return np.minimum(np.where(use_bos, bos_off, random_off), last)


def windows(stream, offs, ctx):
    """(x, y) int64 windows of ctx tokens at the given offsets; y is x shifted left by one."""
    idx = np.asarray(offs)[:, None] + np.arange(ctx + 1)[None, :]
    win = stream[idx].astype(np.int64)
    return win[:, :-1], win[:, 1:]


def sample_batch(stream, batch, ctx, rng, p_bos=0.5, starts=None):
    """Random (x, y) int64 windows of ctx tokens; y is x shifted left by one."""
    return windows(stream, sample_offsets(stream, batch, ctx, rng, p_bos, starts), ctx)


# ---- per-token input features ----------------------------------------------------------------

def write_features(directory, split: str, cats, bag, meta: dict) -> None:
    """Save one split's features and (re)write feats.json with `meta`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{split}.cats.npy", np.asarray(cats, dtype=np.uint8))
    np.save(d / f"{split}.bag.npy", np.asarray(bag, dtype=np.uint64))
    (d / "feats.json").write_text(json.dumps(meta, indent=1))


def feature_meta(directory) -> dict:
    return json.loads((Path(directory) / "feats.json").read_text())


def load_features(directory, split: str, n: int | None = None):
    """(cats (N, n_cats) uint8, bag (N,) uint64), memory-mapped; n checks the stream length."""
    d = Path(directory)
    cats = np.load(d / f"{split}.cats.npy", mmap_mode="r")
    bag = np.load(d / f"{split}.bag.npy", mmap_mode="r")
    if len(cats) != len(bag) or (n is not None and len(cats) != n):
        raise ValueError(f"{d}/{split}: {len(cats)} cats / {len(bag)} bag rows for a "
                         f"{n}-token stream")
    return cats, bag


def stream_features(stream, tracker):
    """Features of every stream position from a step function: tracker.reset() once, then
    tracker.step(tok) -> (cats, bag) per token; the tracker resets itself at <bos>."""
    tracker.reset()
    rows = [tracker.step(int(t)) for t in stream]
    cats = np.array([r[0] for r in rows], dtype=np.uint8).reshape(len(rows), -1)
    return cats, np.array([int(r[1]) for r in rows], dtype=np.uint64)


def _feature_rows(feats, idx):
    cats, bag = feats
    return np.asarray(cats)[idx].astype(np.int64), np.asarray(bag)[idx].view(np.int64)


def feature_windows(feats, offs, ctx):
    """Features of the input tokens of windows(stream, offs, ctx): cats (B, ctx, n_cats) and
    bag (B, ctx), int64 (bag bit 63 is the sign bit)."""
    return _feature_rows(feats, np.asarray(offs)[:, None] + np.arange(ctx)[None, :])


def _spans_to_windows(feats, spans, ctx):
    cats, bag = feats
    c = np.zeros((len(spans), ctx, np.asarray(cats).shape[1]), dtype=np.int64)
    b = np.zeros((len(spans), ctx), dtype=np.int64)
    for w, (lo, n) in enumerate(spans):
        c[w, :n], b[w, :n] = _feature_rows(feats, np.arange(lo, lo + n))
    return c, b


def val_feature_windows(feats, n_stream: int, ctx):
    """Features aligned with val_windows(stream, ctx); padding rows are 0."""
    n_targets = n_stream - 1
    spans = [(lo, min(ctx, n_targets - lo)) for lo in range(0, n_targets, ctx)]
    return _spans_to_windows(feats, spans, ctx)


def bos_feature_windows(stream, feats, ctx):
    """Features aligned with bos_windows(stream, ctx); padding rows are 0."""
    starts = program_starts(stream)
    ends = np.append(starts[1:], len(stream))
    return _spans_to_windows(feats, [(lo, min(ctx, hi - lo - 1)) for lo, hi in zip(starts, ends)],
                             ctx)


def shuffle_features(cats, bag, shift: int | None = None):
    """A2 control: row i gets the features of row i - shift (mod n; default n//2), from another
    window or program, so the tables see the same kind of inputs without this row's
    information. Works on numpy arrays and torch tensors (B, ...); either may be None."""
    n = len(cats if cats is not None else bag)
    if n < 2:
        raise ValueError("shuffling features needs at least two rows")
    shift = n // 2 if shift is None else shift

    def roll(a):
        if a is None:
            return None
        if isinstance(a, np.ndarray):
            return np.roll(a, shift, axis=0)
        import torch

        return torch.roll(a, shift, 0)

    return roll(cats), roll(bag)


def val_windows(stream, ctx):
    """Contiguous non-overlapping windows; padding targets are IGNORE."""
    n_targets = len(stream) - 1
    n_win = -(-n_targets // ctx)
    xs = np.full((n_win, ctx), PAD, dtype=np.int64)
    ys = np.full((n_win, ctx), IGNORE, dtype=np.int64)
    for w in range(n_win):
        lo = w * ctx
        n = min(ctx, n_targets - lo)
        xs[w, :n] = stream[lo : lo + n]
        ys[w, :n] = stream[lo + 1 : lo + 1 + n]
    return xs, ys


def bos_windows(stream, ctx):
    """One window per program starting at its <bos>; targets after its <eos> are IGNORE.

    Measures loss in the on-calculator setting: generation from <bos> at position 0.
    """
    starts = program_starts(stream)
    ends = np.append(starts[1:], len(stream))  # one past each program's <eos>
    xs = np.full((len(starts), ctx), PAD, dtype=np.int64)
    ys = np.full((len(starts), ctx), IGNORE, dtype=np.int64)
    for w, (lo, hi) in enumerate(zip(starts, ends)):
        n = min(ctx, hi - lo - 1)
        xs[w, :n] = stream[lo : lo + n]
        ys[w, :n] = stream[lo + 1 : lo + 1 + n]
    return xs, ys
