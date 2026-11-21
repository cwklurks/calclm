"""ctypes binding to the C tracker (tracker/tracker.c) built as a host shared library.

    from tracker import bind
    cats, bags = bind.batch(np.fromfile("data/tok/train.bin", dtype="<u2"))
    # cats: (T, NCAT) uint8, bags: (T,) uint64; row i = state after token i (<bos> resets)
    t = bind.Tracker(); t.reset(); cats_i, bag_i = t.step(token_id); t.summary()

    uv run python -m tracker.bind data/tok/train.bin --out data/tok/train.trk   # cache file

The library is rebuilt (gcc -O2) when tracker.c, host_batch.c or a header changes; it lives in
tracker/build/ (git-ignored).
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import feats as F

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
SOURCES = [HERE / "tracker.c", HERE / "host_batch.c"]
HEADERS = [HERE / "tracker.h", HERE / "trk_tables.h"]
SUMMARY_KEYS = ("n_unres", "depth", "n_undef", "first_undef_pos", "n_bad", "first_bad_pos",
                "n_extra_end", "n_prgm")
BOS = 1


def build() -> Path:
    BUILD.mkdir(exist_ok=True)
    lib = BUILD / "libtracker.so"
    deps = SOURCES + HEADERS
    if not lib.exists() or max(p.stat().st_mtime for p in deps) > lib.stat().st_mtime:
        tmp = lib.with_suffix(".tmp.so")
        res = subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
                              "-shared", *map(str, SOURCES), "-o", str(tmp)],
                             capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f"tracker build failed:\n{res.stderr}")
        tmp.replace(lib)
    return lib


@lru_cache(maxsize=1)
def lib() -> ctypes.CDLL:
    so = ctypes.CDLL(str(build()))
    u16p = ctypes.POINTER(ctypes.c_uint16)
    so.tracker_reset.argtypes = [ctypes.c_void_p]
    so.tracker_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
    so.tracker_summary.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    so.trk_sizeof_state.restype = ctypes.c_size_t
    so.trk_sizeof_feat.restype = ctypes.c_size_t
    so.trk_offsetof_stmt.restype = ctypes.c_size_t
    so.tracker_classes.restype = ctypes.POINTER(ctypes.c_uint8)
    so.trk_batch.argtypes = [u16p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8),
                             ctypes.POINTER(ctypes.c_uint64)]
    so.trk_summaries.argtypes = [u16p, ctypes.c_size_t, ctypes.c_uint16, u16p, ctypes.c_size_t]
    so.trk_summaries.restype = ctypes.c_size_t
    return so


def state_size() -> int:
    return lib().trk_sizeof_state()


def feat_size() -> int:
    return lib().trk_sizeof_feat()


def _u16(stream) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(stream), dtype=np.uint16)


def batch(stream) -> tuple[np.ndarray, np.ndarray]:
    """Features of a uint16 token stream: cats (T, NCAT) uint8, bags (T,) uint64."""
    toks = _u16(stream)
    n = len(toks)
    cats = np.zeros((n, F.NCAT), dtype=np.uint8)
    bags = np.zeros(n, dtype=np.uint64)
    if n:
        lib().trk_batch(toks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), n,
                        cats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                        bags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
    return cats, bags


def summaries(stream) -> np.ndarray:
    """(programs, 8) uint16 end-of-program summaries, columns SUMMARY_KEYS."""
    toks = _u16(stream)
    if not len(toks):
        return np.zeros((0, 8), dtype=np.uint16)
    n_prog = int((toks == BOS).sum()) + (0 if toks[0] == BOS else 1)
    out = np.zeros((n_prog, 8), dtype=np.uint16)
    k = lib().trk_summaries(toks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), len(toks), BOS,
                            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), n_prog)
    assert k == n_prog
    return out


class Tracker:
    """Stepwise access to the C tracker (same interface as spec.Tracker)."""

    def __init__(self):
        self._lib = lib()
        self._state = ctypes.create_string_buffer(state_size())
        self._feat = ctypes.create_string_buffer(feat_size())
        self._sum = (ctypes.c_uint16 * 8)()
        self.reset()

    def reset(self) -> None:
        self._lib.tracker_reset(self._state)

    def step(self, tok: int) -> tuple[tuple[int, ...], int]:
        self._lib.tracker_step(self._state, tok, self._feat)
        raw = self._feat.raw
        hint = int.from_bytes(raw[F.NCAT:F.NCAT + 5], "little") & ((1 << 38) - 1)
        var = int.from_bytes(raw[F.NCAT + 5:F.NCAT + 9], "little")
        return tuple(raw[:F.NCAT]), hint | (var << 38)

    def summary(self) -> dict:
        self._lib.tracker_summary(self._state, self._sum)
        return dict(zip(SUMMARY_KEYS, list(self._sum)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="tracker features for a token .bin")
    p.add_argument("bin", help="uint16 token stream (corpus .bin)")
    p.add_argument("--out", help="write cats (T*NCAT uint8) then bags (T uint64) here")
    a = p.parse_args(argv)
    stream = np.fromfile(a.bin, dtype="<u2")
    t0 = time.perf_counter()
    cats, bags = batch(stream)
    dt = time.perf_counter() - t0
    print(f"{len(stream):,} tokens in {dt:.2f} s ({len(stream) / dt / 1e6:.1f} M tokens/s)")
    if a.out:
        with open(a.out, "wb") as fh:
            fh.write(cats.tobytes())
            fh.write(bags.astype("<u8").tobytes())
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
