"""Causality of the batch feature function (what the model may see at position i).

The model predicts token i+1 from inputs at positions <= i, so the features at position i
must be a function of tokens[program start .. i] only:
1. a prefix gets the same features whether or not the rest of the program follows;
2. features at position i are the state AFTER token i has been read (token i included);
3. <bos> resets the state: nothing carries over from the previous program in the stream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tracker import bind, difftest, feats as F, spec
from tracker.tests.conftest import BOS, EOS, ids_of

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "tok" / "val.bin"
G = F.GROUP_INDEX


def _programs(n: int, max_len: int) -> list[np.ndarray]:
    if VAL.exists():
        progs = difftest.split_programs(np.fromfile(VAL, dtype="<u2"))
    else:
        progs = difftest.split_programs(difftest.fuzz_stream(seed=5, n_programs=200))
    rng = np.random.default_rng(0)
    picked = [p[: rng.integers(20, max_len)] for p in progs if len(p) > 20]
    return [picked[i] for i in rng.permutation(len(picked))[:n]]


@pytest.mark.parametrize("backend", ["c", "spec"])
def test_prefix_features_do_not_depend_on_the_future(backend):
    batch = bind.batch if backend == "c" else spec.batch
    n, max_len = (40, 700) if backend == "c" else (6, 160)
    for prog in _programs(n, max_len):
        cats, bags = batch(prog)
        for k in range(1, len(prog) + 1):
            pc, pb = batch(prog[:k])
            assert np.array_equal(pc, cats[:k]) and np.array_equal(pb, bags[:k]), k


def test_prefix_features_do_not_depend_on_the_next_token():
    # every possible next token after a prefix leaves the prefix's features unchanged
    prog = np.array(ids_of('If A\nThen\nGoto Q\nLbl Q\nMenu("T","A",Q\n{1}→ʟAB\nDisp ʟA'),
                    dtype=np.uint16)
    for k in range(1, len(prog) + 1):
        ref_c, ref_b = bind.batch(prog[:k])
        for nxt in range(0, 781, 7):
            c, b = bind.batch(np.append(prog[:k], np.uint16(nxt)))
            assert np.array_equal(c[:k], ref_c) and np.array_equal(b[:k], ref_b)


@pytest.mark.parametrize("backend", ["c", "spec"])
def test_features_at_i_include_token_i(backend):
    batch = bind.batch if backend == "c" else spec.batch
    cases = [
        ("If A", -2, "if_state", F.IF_COND),       # the If token itself already counts
        ("Goto A\n", -1, "unres", 1),              # the newline that completes Goto A
        ("If A\nThen", -1, "blk_top", F.BLK_IF),   # Then pushes at its own position
        ("While 1\nEnd", -1, "blk_depth", 0),      # End pops at its own position
        ('Disp "', -1, "ctx", F.CTX_STRING),       # the opening quote is inside the string
        ("1→", -1, "ctx", F.CTX_STORE),
    ]
    for src, pos, group, want in cases:
        cats, _ = batch(np.array(ids_of(src), dtype=np.uint16))
        assert cats[pos][G[group]] == want, src
    # and stepwise: batch row i == tracker state after step(token i)
    prog = np.array(ids_of("For(I,1,3\nIf I=2:Goto B\nEnd\nLbl B"), dtype=np.uint16)
    t = bind.Tracker() if backend == "c" else spec.Tracker()
    cats, bags = batch(prog)
    t.reset()
    for i, tok in enumerate(prog.tolist()):
        c, b = t.step(tok)
        assert c == tuple(cats[i]) and b == int(bags[i])


def test_bos_position_is_the_reset_state():
    cats, bags = bind.batch(np.array(ids_of("If A\nThen\nGoto X") + [EOS, BOS], dtype=np.uint16))
    assert tuple(cats[-1]) == tuple(cats[0]) and bags[-1] == bags[0] == 0


PATHOLOGICAL_ENDINGS = [
    'Disp "OPEN STRING',            # open string
    "If A\nThen\nWhile 1\nFor(I,1,2",  # open blocks
    "Goto Q\nGoto R",               # unresolved labels
    "Lbl A",                        # name in progress
    '{1}→ʟABC\n"X"→Str1\n[[1]]→[A]\nDisp ʟAB',   # assigned vars, list name in progress
    'Menu("T","A",A',               # inside Menu(
    "If A",                         # pending single-line If
    "DelVar ",                      # DelVar operand pending
    "(((((",                        # parens
]


@pytest.mark.parametrize("backend", ["c", "spec"])
def test_bos_resets_and_nothing_carries_over(backend):
    batch = bind.batch if backend == "c" else spec.batch
    progs = _programs(25, 300)
    for prev_src in PATHOLOGICAL_ENDINGS:
        prev = ids_of(prev_src)
        for with_eos in (False, True):
            head = prev + ([EOS] if with_eos else [])
            for prog in progs[:8] if backend == "spec" else progs:
                alone_c, alone_b = batch(prog)
                both_c, both_b = batch(np.array(head + prog.tolist(), dtype=np.uint16))
                assert np.array_equal(both_c[len(head):], alone_c)
                assert np.array_equal(both_b[len(head):], alone_b)


def test_stream_equals_programs_one_by_one():
    stream = difftest.fuzz_stream(seed=21, n_programs=50)
    cats, bags = bind.batch(stream)
    starts = np.flatnonzero(stream == BOS).tolist() + [len(stream)]
    for a, b in zip(starts, starts[1:]):
        c1, b1 = bind.batch(stream[a:b])
        assert np.array_equal(c1, cats[a:b]) and np.array_equal(b1, bags[a:b])
