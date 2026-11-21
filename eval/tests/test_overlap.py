import numpy as np

from eval.overlap import ngram_hashes, seen_fraction


def test_equal_windows_hash_equal_and_count_is_right():
    a = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.uint16)
    h = ngram_hashes(a, 3)
    assert len(h) == 5
    assert h[0] == h[4] and len(set(h.tolist())) == 4


def test_seen_fraction_against_reference():
    ref = np.unique(ngram_hashes(np.array([5, 6, 7, 8, 9], dtype=np.uint16), 2))
    assert seen_fraction([5, 6, 7, 1], ref, 2) == 2 / 3
    assert seen_fraction([1, 2], ref, 3) is None  # shorter than n
