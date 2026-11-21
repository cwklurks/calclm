"""The tracker at sampling time: features for feature models, state for logit masks.

feats.json (model/data.py) names the step-function tracker class that computed a feature
directory ("tracker": "module:Class", reset() / step(tok) -> (cats, bag)), so sampling runs the
same code. Logit masks (lab/masks.py) need one-token look-ahead on the C tracker's state.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from lab.sample import BatchTracker


def load_class(spec: str):
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


def make_sampling_tracker(feat_meta: dict | None, masks: bool, vocab: Path):
    """(tracker, mask) for lab.sample: (None, None) for a v1 model without masks."""
    if masks:
        from lab.masks import TrackerMask

        mask = TrackerMask(vocab)
        if feat_meta is not None and feat_meta.get("tracker") != mask.tracker_name:
            raise ValueError(f"masks run the C tracker but the features came from "
                             f"{feat_meta.get('tracker')}")
        return mask.rows, mask
    if feat_meta is None:
        return None, None
    if feat_meta.get("tracker") == "tracker.bind:Tracker":
        from lab.masks import CTrackerRows

        return CTrackerRows(), None
    return BatchTracker(load_class(feat_meta["tracker"])), None
