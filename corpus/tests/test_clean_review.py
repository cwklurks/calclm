"""Boundary tests added after the independent review of corpus/clean.py (M6, L1-L2). Made-up
programs only (D6)."""
from __future__ import annotations

import numpy as np

from corpus import clean as C
from corpus.tests.test_clean import VOCAB, _split_lines, _with_lines, ids, prog, shown


def test_a_store_ends_a_string_and_the_rest_is_code():
    t = ids('"BY ZORB"→Str1\n"A"→B')
    out, info = C.Scrubber(VOCAB).scrub(t)
    assert shown(out) == shown(ids('""→Str1\n"A"→B')) and info['edited_strings'] == 1


def test_x3_never_excises_any_seed_even_beyond_k():
    rng = np.random.default_rng(7)
    block = _with_lines(rng, 8)
    progs = [prog(i, _with_lines(rng, 10 + i) + block + _with_lines(rng, 10), seed=(i < 6))
             for i in range(8)]
    out, stats = C.x3(progs, K=4)
    by = {p['key']: p for p in out}
    for k in range(6):
        assert np.array_equal(by[k]['ids'], progs[k]['ids'])
    assert stats['touched'] == 2


def _contains(seq, sub):
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def test_x3_keeps_every_line_not_wholly_covered():
    rng = np.random.default_rng(8)
    nl = VOCAB.index('3F')
    block = _with_lines(rng, 6)
    # each program's own 6 tokens (no newline) share a line with the block's first line
    halves = [list(rng.integers(40, 700, size=6)) for _ in range(6)]
    progs = [prog(i, _with_lines(rng, 15 + i) + halves[i] + block + _with_lines(rng, 15))
             for i in range(6)]
    out, stats = C.x3(progs, K=4)
    by = {p['key']: p for p in out}
    assert stats['touched'] == 2
    for k in (0, 1):
        after = by[k]['ids'].tolist()
        lines = _split_lines(progs[k]['ids'].tolist(), nl)
        first_block_line = next(i for i, ln in enumerate(lines) if list(ln[:6]) == halves[k])
        assert _contains(after, list(lines[first_block_line]))      # partly covered: kept
        assert not _contains(after, list(lines[first_block_line + 1]))   # wholly covered: gone


def test_large_lsh_buckets_are_counted():
    sigs = np.zeros((60, C.NUM_PERM), dtype=np.uint64)       # one bucket of 60 in every band
    assert C.large_buckets(sigs, lambda i: i < 30) == C.BANDS
    assert C.large_buckets(sigs, lambda i: True) == 0
