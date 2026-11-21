"""Ternary weight packing (5 weights per byte, base 3) and the reference LUT matvec.

Contract (SPEC.md): inputs padded to a multiple of 5; group g holds inputs 5g..5g+4;
byte b = sum_j (w[5g+j] + 1) * 3**j, first weight least significant.
"""
from __future__ import annotations

import numpy as np

GROUP = 5
POW3 = np.array([3**j for j in range(GROUP)], dtype=np.int32)
# DIGITS[b, j] = (j-th base-3 digit of b) - 1, the weight it encodes
DIGITS = ((np.arange(243)[:, None] // POW3[None, :]) % 3 - 1).astype(np.int32)


def groups(n_in: int) -> int:
    return -(-n_in // GROUP)


def pack_ternary(w: np.ndarray) -> np.ndarray:
    """(rows, n_in) int weights in {-1, 0, 1} -> (rows, ceil(n_in/5)) uint8."""
    w = np.asarray(w)
    if w.size and not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("weights must be in {-1, 0, 1}")
    rows, n_in = w.shape
    padded = np.zeros((rows, groups(n_in) * GROUP), dtype=np.int32)
    padded[:, :n_in] = w
    digits = padded.reshape(rows, -1, GROUP) + 1
    return (digits * POW3).sum(axis=2).astype(np.uint8)


def unpack_ternary(packed: np.ndarray, n_in: int) -> np.ndarray:
    """Inverse of pack_ternary; n_in may exceed the original width (padding reads as 0)."""
    w = DIGITS[packed.astype(np.int32)].reshape(packed.shape[0], -1)
    return w[:, :n_in].astype(np.int8)


def lut_matvec(packed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y[row] = sum_j w[row][j] * x[j] via per-group 243-entry partial-sum tables."""
    n_groups = packed.shape[1]
    xp = np.zeros(n_groups * GROUP, dtype=np.int32)
    xp[: len(x)] = x
    tables = DIGITS @ xp.reshape(n_groups, GROUP).T  # (243, n_groups): T_g[b]
    return tables[packed.astype(np.int32), np.arange(n_groups)].sum(axis=1).astype(np.int32)
