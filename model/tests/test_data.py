import json

import numpy as np

from model.data import IGNORE, load_split, sample_batch, val_windows, write_split
from model.tokens import BOS, EOS


def _stream(lengths):
    """Programs as <bos> body <eos>, body tokens numbered from 10."""
    out, nxt = [], 10
    for n in lengths:
        out += [BOS, *range(nxt, nxt + n), EOS]
        nxt += n
    return np.array(out, dtype=np.uint16)


def test_write_and_load_little_endian_uint16(tmp_path):
    stream = _stream([3, 700])
    write_split(tmp_path / "train.bin", stream)
    raw = (tmp_path / "train.bin").read_bytes()
    assert raw[:2] == bytes([BOS, 0]) and len(raw) == 2 * len(stream)
    assert np.array_equal(load_split(tmp_path / "train.bin"), stream)


def test_sample_batch_targets_are_shifted_inputs():
    stream = _stream([50, 400, 30])
    rng = np.random.default_rng(0)
    x, y = sample_batch(stream, batch=8, ctx=64, rng=rng, p_bos=0.5)
    assert x.shape == y.shape == (8, 64)
    assert np.array_equal(x[:, 1:], y[:, :-1])


def test_bos_aligned_windows_start_at_program_start():
    stream = _stream([50, 400, 30])
    x, _ = sample_batch(stream, batch=16, ctx=32, rng=np.random.default_rng(1), p_bos=1.0)
    assert (x[:, 0] == BOS).all()


def test_val_windows_cover_every_target_once():
    stream = _stream([5, 300, 9])
    xs, ys = val_windows(stream, ctx=64)
    assert xs.shape[1] == 64
    targets = ys[ys != IGNORE]
    assert len(targets) == len(stream) - 1
    assert np.array_equal(targets, stream[1:].astype(np.int64))


def test_meta_roundtrip(tmp_path):
    meta = {"vocab_size": 812}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert json.loads((tmp_path / "meta.json").read_text())["vocab_size"] == 812


def test_bos_windows_one_per_program_masked_after_eos():
    from model.data import bos_windows

    stream = _stream([5, 300, 9])
    xs, ys = bos_windows(stream, ctx=64)
    assert xs.shape == (3, 64) and (xs[:, 0] == BOS).all()
    assert (ys[0] != IGNORE).sum() == 6 and ys[0, 5] == EOS  # 5 body tokens + <eos>
    assert (ys[1] != IGNORE).sum() == 64  # long program: full window
    assert (ys[2] != IGNORE).sum() == 10
