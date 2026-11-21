"""Regression check on the trained v1 artifacts (the run that matches the current data/tok) (skipped when they are not on disk)."""
from pathlib import Path

import numpy as np
import pytest

from model.data import bos_windows, load_split

ROOT = Path(__file__).resolve().parents[2]
RUN, DATA = ROOT / "runs" / "v1", ROOT / "data" / "tok"

pytestmark = pytest.mark.skipif(
    not ((RUN / "best.pt").exists() and (RUN / "export" / "model.bin").exists()
         and (DATA / "val.bin").exists()),
    reason="trained v1 run or M1 data not present")


def test_intref_tracks_trained_model_on_val_programs():
    from model.evaluate import compare
    from model.intref import IntModel
    from model.model import load_checkpoint

    xs, ys = bos_windows(load_split(DATA / "val.bin"), 256)
    rep = compare(load_checkpoint(RUN / "best.pt"), IntModel.load(RUN / "export"), xs[:2, :128], ys[:2, :128])
    assert abs(rep["rel_diff"]) < 0.02
    assert rep["top1_agree"] > 0.95


V2_RUN = ROOT / "runs" / "lab" / "S0-s10" / "train" / "avg.pt"   # the shipped v2 model
V2_EXPORT = ROOT / "runs" / "v2" / "export"


@pytest.mark.skipif(not (V2_RUN.exists() and (V2_EXPORT / "model.bin").exists()),
                    reason="the v2 checkpoint (S0-s10) or runs/v2/export not present")
def test_intref_tracks_the_v2_model_on_whole_programs_past_the_window():
    """Format 2 (ALiBi, sliding window): whole val programs longer than the 256-token window."""
    from model.evaluate import compare_programs

    val = load_split(DATA / "val.bin")
    starts = np.flatnonzero(val == 1)
    ends = np.append(starts[1:], len(val))
    long = [val[a:b].tolist() for a, b in zip(starts, ends) if 300 <= b - a <= 420][:2]
    rep = compare_programs(V2_RUN, V2_EXPORT, long, workers=2)
    assert rep["programs"] == 2 and rep["n_targets"] == sum(len(p) - 1 for p in long)
    assert abs(rep["rel_diff"]) < 0.02
    assert rep["top1_agree"] > 0.95
