"""Shared helpers for the tracker tests: TI-BASIC source -> vocab ids, and both
implementations (the Python spec and the C tracker via ctypes) behind one interface."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tivars import TIProgram

from eval.tokens import split_tokens

ROOT = Path(__file__).resolve().parents[2]
VOCAB = json.loads((ROOT / "corpus" / "vocab.json").read_text())
INDEX = {h: i for i, h in enumerate(VOCAB)}
BOS, EOS = 1, 2


def ids_of(src: str, bos: bool = True) -> list[int]:
    """TI-BASIC source (tivars load_string syntax) -> [<bos>] + vocab ids (no <eos>)."""
    p = TIProgram(name="X")
    p.load_string(src)
    out = [INDEX.get(t.hex().upper(), 3) for t in split_tokens(p.data)]
    return ([BOS] if bos else []) + out


def raw(*hexes: str) -> list[int]:
    """Vocab ids of explicit token hex strings."""
    return [INDEX[h.upper()] for h in hexes]


def make_impl(name: str):
    if name == "spec":
        from tracker import spec
        return spec.Tracker()
    from tracker import bind
    return bind.Tracker()


@pytest.fixture(params=["spec", "c"])
def impl(request):
    return make_impl(request.param)


def run(tracker, ids):
    """Reset, then step through ids; returns the list of (cats, bag) per token."""
    tracker.reset()
    return [tracker.step(i) for i in ids]
