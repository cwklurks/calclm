"""lab.features: precomputed per-token tracker features for train/val (model/data.py format)."""
import json

import numpy as np

from lab import features as LF
from model.data import load_features, write_split
from model.tokens import BOS, EOS


def _toy_batch(stream):
    """A batch function like tracker.bind.batch: cats (T, 2) uint8, bags (T,) uint64."""
    stream = np.asarray(stream)
    pos = np.zeros(len(stream), dtype=np.int64)
    for i in range(1, len(stream)):
        pos[i] = 0 if stream[i] == BOS else pos[i - 1] + 1
    cats = np.stack([np.minimum(pos, 3), stream % 2], 1).astype(np.uint8)
    return cats, (stream.astype(np.uint64) << np.uint64(60))


def test_features_are_written_for_each_split_with_their_metadata(tmp_path):
    data = tmp_path / "tok"
    for split, n in (("train", 50), ("val", 20)):
        write_split(data / f"{split}.bin", np.array([BOS, *range(5, 5 + n), EOS] * 3))
    meta = LF.write_tracker_features(data, tmp_path / "f", batch=_toy_batch, groups=[4, 2],
                                     bag_bits=64, tracker="toy:Toy", identity={"toy": "1"})
    saved = json.loads((tmp_path / "f" / "feats.json").read_text())
    assert saved == meta and saved["cats"] == [4, 2] and saved["bag"] == 64
    assert saved["tracker"] == "toy:Toy" and saved["identity"] == {"toy": "1"}
    assert set(saved["stream_sha256"]) == {"train", "val"}
    c, b = load_features(tmp_path / "f", "val", n=3 * 22)
    want_c, want_b = _toy_batch(np.fromfile(data / "val.bin", dtype="<u2"))
    assert np.array_equal(c, want_c) and np.array_equal(b, want_b)
    assert saved["bag_bits_mean"]["val"] == np.mean(np.bitwise_count(want_b))
