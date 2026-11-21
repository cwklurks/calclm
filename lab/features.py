"""Precompute the tracker's per-token features for train/val (model/data.py feature format).

    uv run python -m lab.features --data data/tok --out data/feats/tracker

Row i of each split is the C tracker's state after stream token i (tracker.bind.batch, <bos>
resets), so windows starting mid-program carry the state from their program's start.
feats.json records the group sizes, the bag width, the step-function class the sampler must run
("tracker.bind:Tracker"), sha256s of the tracker sources and of the streams, and the mean
number of set bag bits (the data-dependent part of the calculator cost, lab/budget.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from model.data import load_split, write_features

ROOT = Path(__file__).resolve().parents[1]
TRACKER_SOURCES = ("tracker.c", "tracker.h", "trk_tables.h", "host_batch.c", "feats.py")


def tracker_identity() -> dict:
    d = ROOT / "tracker"
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in TRACKER_SOURCES}


def write_tracker_features(data: Path, out: Path, batch=None, groups=None, bag_bits=None,
                           tracker: str = "tracker.bind:Tracker", identity=None,
                           splits=("train", "val")) -> dict:
    """Features of every split in data/ -> out/; returns the feats.json contents."""
    if batch is None:
        from tracker import bind
        from tracker import feats as F

        batch, groups, bag_bits = bind.batch, [n for _, n in F.GROUPS], F.BAG_BITS
        identity = tracker_identity() if identity is None else identity
    meta = {"cats": list(groups), "bag": int(bag_bits), "tracker": tracker,
            "identity": identity or {}, "stream_sha256": {}, "bag_bits_mean": {}}
    computed = {}
    for split in splits:
        path = Path(data) / f"{split}.bin"
        stream = load_split(path)
        cats, bag = batch(stream)
        meta["stream_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        meta["bag_bits_mean"][split] = float(np.mean(np.bitwise_count(bag)))
        computed[split] = (cats, bag)
    for split, (cats, bag) in computed.items():
        write_features(out, split, cats, bag, meta)
    return json.loads((Path(out) / "feats.json").read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "tok")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "feats" / "tracker")
    a = ap.parse_args(argv)
    meta = write_tracker_features(a.data, a.out)
    print(json.dumps({k: meta[k] for k in ("cats", "bag", "tracker", "bag_bits_mean")}))


if __name__ == "__main__":
    main()
