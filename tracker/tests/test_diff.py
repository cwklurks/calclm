"""T2 differential tests: the C tracker (batch, via ctypes) and the Python spec produce
identical features (and end-of-program summaries) on every val program, a random sample of
train programs and fuzzed token streams. The full-train run is `python -m tracker.difftest`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tracker import bind, difftest, spec

ROOT = Path(__file__).resolve().parents[2]
TOK = ROOT / "data" / "tok"
need_data = pytest.mark.skipif(not (TOK / "val.bin").exists(), reason="corpus not built")


def _check(stream: np.ndarray) -> None:
    bad = difftest.compare(stream)
    assert bad is None, bad


@need_data
def test_every_val_program():
    stream = np.fromfile(TOK / "val.bin", dtype="<u2")
    _check(stream)
    assert difftest.compare_summaries(stream) is None


@need_data
def test_random_train_sample():
    stream = np.fromfile(TOK / "train.bin", dtype="<u2")
    sample = difftest.sample_programs(stream, n=1500, seed=7)
    _check(sample)
    assert difftest.compare_summaries(sample) is None


@pytest.mark.parametrize("seed", range(4))
def test_fuzzed_streams(seed):
    stream = difftest.fuzz_stream(seed=seed, n_programs=400)
    _check(stream)
    assert difftest.compare_summaries(stream) is None


def test_stepwise_binding_matches_batch():
    stream = difftest.fuzz_stream(seed=11, n_programs=30)
    cats, bags = bind.batch(stream)
    t = bind.Tracker()
    for i, tok in enumerate(stream.tolist()):
        c, b = t.step(tok)
        assert c == tuple(cats[i]) and b == int(bags[i]), i


def test_spec_batch_matches_spec_stepwise():
    stream = difftest.fuzz_stream(seed=3, n_programs=20)
    cats, bags = spec.batch(stream)
    t = spec.Tracker()
    for i, tok in enumerate(stream.tolist()):
        c, b = t.step(tok)
        assert c == tuple(cats[i]) and b == int(bags[i]), i
