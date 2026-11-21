"""Integer primitives of the CalcLM integer reference (see INTREF.md).

Only Python ints and numpy int64 arrays. Every per-element product fits in signed 32 bits
(asserted), every per-vector scalar step fits in unsigned 32 bits, and rounding is always
round-half-up via add-then-arithmetic-shift (`rsr`). No floats are used at inference time.

PF15 numbers are (m, e) pairs with m in [2**14, 2**15) meaning m * 2**e; PF8 scales are
(m, e) with m in [128, 256) (or m = 0 for an all-zero vector).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

I64 = np.int64
LIMIT31 = 1 << 31


def check_bits(x, bits: int, what: str):
    """Assert |x| < 2**bits (signed width bits+1); returns x unchanged."""
    peak = int(np.abs(np.asarray(x)).max(initial=0))
    if peak >= (1 << bits):
        raise OverflowError(f"{what}: |value| {peak} needs more than {bits}+1 bits")
    return x


def rsr(a, s: int):
    """Round-half-up shift right: floor((a + 2**(s-1)) / 2**s); s <= 0 is an exact left shift."""
    if s > 0:
        return (a + (1 << (s - 1))) >> s
    return a << (-s) if s < 0 else a


def rsr_vec(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """rsr with a per-element shift amount (int arrays of equal shape)."""
    a, s = np.asarray(a, dtype=I64), np.asarray(s, dtype=I64)
    if (s < -32).any() or (s > 62).any():
        raise OverflowError("shift amount out of range")
    half = np.where(s > 0, np.left_shift(I64(1), np.maximum(s - 1, 0)), 0)
    right = (a + half) >> np.maximum(s, 0)
    return np.where(s >= 0, right, a << np.maximum(-s, 0))


def clamp(a, lo: int, hi: int):
    return np.clip(a, lo, hi)


def bitlen(n: int) -> int:
    return int(n).bit_length()


def isqrt(n: int) -> int:
    """floor(sqrt(n)) by the bit-by-bit method (what the calculator implements)."""
    if n < 0:
        raise ValueError("isqrt of negative")
    root, bit = 0, 1 << ((bitlen(n) - 1) & ~1) if n else 0
    while bit:
        if n >= root + bit:
            n -= root + bit
            root = (root >> 1) + bit
        else:
            root >>= 1
        bit >>= 2
    return root


def udiv(a: int, b: int) -> int:
    if a < 0 or b <= 0 or a >= (1 << 32):
        raise ValueError("udiv operands out of range")
    return a // b


# ---- PF (pseudo-float) scalars ---------------------------------------------------------

def pf15(n: int, e: int = 0) -> tuple[int, int]:
    n = int(n)
    if n <= 0:
        raise ValueError("pf15 needs a positive integer")
    sh = bitlen(n) - 15
    m = rsr(n, sh)
    if m == 1 << 15:
        m, sh = 1 << 14, sh + 1
    return m, e + sh


def pf_mul(a, b):
    return pf15(a[0] * b[0], a[1] + b[1])


def pf_div(a, b):
    return pf15(udiv(a[0] << 16, b[0]), a[1] - b[1] - 16)


def pf_shift(a, k: int):
    return a[0], a[1] + k


def pf8(a) -> tuple[int, int]:
    m, e = rsr(a[0], 7), a[1] + 7
    return (128, e + 1) if m == 256 else (m, e)


def pf15_of_pf8(s) -> tuple[int, int]:
    return s[0] << 7, s[1] - 7


def pf_to_float(a) -> float:
    """For tests and reporting only (never used by inference)."""
    return a[0] * 2.0 ** a[1]


def sqrt_pf15(n: int) -> tuple[int, int]:
    """sqrt(n) as PF15, from an integer square root of n * 2**28."""
    return pf15(isqrt(n << 28), -14)


INV127 = pf_div((1 << 14, -14), pf15(127))


# ---- vector operations -------------------------------------------------------------------

class MulCounter:
    """Tally of per-token multiplies outside the ternary matvecs, by operation kind."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def add(self, kind: str, n: int) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + int(n)

    def total(self) -> int:
        return sum(self.counts.values())

    def merged(self, other: "MulCounter") -> dict:
        return dict(Counter(self.counts) + Counter(other.counts))


NULL_COUNTER = MulCounter()


def normalize16(v: np.ndarray) -> tuple[np.ndarray, int]:
    """Shift a vector so its peak lands in [2**14, 2**15]; saturate to int16 (+-32767).

    rsr can round the peak up to exactly 2**15, which int16 cannot hold: saturating keeps
    every normalized vector in int16 for the calculator. Returns (v', c) with v ~ v' * 2**c.
    """
    v = np.asarray(v, dtype=I64)
    c = bitlen(int(np.abs(v).max(initial=0))) - 15
    return clamp(rsr(v, c), -32767, 32767), c


def quant_vec(v: np.ndarray, unit, counter: MulCounter = NULL_COUNTER):
    """int8 absmax quantization of v * unit (unit PF15). Returns (x8, sx) with sx PF8.

    v' = v normalized to <= 16 bits; sx = PF8(max|v| * unit / 127);
    x8 = clamp(rsr(v' * G.m, -(G.e + c)), +-127) with G = unit / sx (PF15).
    """
    v = np.asarray(v, dtype=I64)
    peak = int(np.abs(v).max(initial=0))
    if peak == 0:
        return np.zeros_like(v), (0, 0)
    vp, c = normalize16(v)
    sx = pf8(pf_mul(pf_mul(pf15(peak), unit), INV127))
    g = pf_div(unit, pf15_of_pf8(sx))
    prod = check_bits(vp * g[0], 31, "quant_vec product")
    counter.add("requant", v.size)
    return clamp(rsr(prod, -(g[1] + c)), -127, 127), sx


def norm_quant(r: np.ndarray, g16: np.ndarray, eg: int, counter: MulCounter = NULL_COUNTER,
               sqrtd=None):
    """Fused RMSNorm (gain g16 * 2**-eg) + int8 absmax quantization of a residual vector.

    The residual's fixed-point format cancels out, so it is not a parameter. sqrtd is
    sqrt(len(r)) as PF15; the export header carries it (computing it needs a 36-bit isqrt).
    """
    r = np.asarray(r, dtype=I64)
    peak = int(np.abs(r).max(initial=0))
    if peak == 0:
        return np.zeros_like(r), (0, 0)
    rp, _ = normalize16(r)  # |rp| <= 32767
    r8 = rsr(rp, 8)  # |r8| <= 128: squares are 8x8 unsigned products
    ss = int((r8 * r8).sum())
    b = (30 - bitlen(ss)) >> 1
    s = isqrt(ss << (2 * b))  # in [2**14, 2**15): sqrt(ss) * 2**b
    z = check_bits(rp * np.asarray(g16, dtype=I64), 30, "norm gain product")
    counter.add("norm_square", r.size)
    counter.add("norm_gain", r.size)
    sqrtd = sqrt_pf15(r.size) if sqrtd is None else sqrtd
    unit = pf_shift(pf_div(sqrtd, pf15(s)), b - eg - 8)
    return quant_vec(z, unit, counter)


def scale_out(acc, bm, be, sx, f_out: int, lo: int, hi: int, counter=NULL_COUNTER):
    """Dequantize a ternary matvec output to fixed point with f_out fractional bits.

    y = clamp(rsr(acc * bm * sx.m, -(be + sx.e + f_out)), lo, hi), all exact integers;
    acc * bm * sx.m < 2**31 because |acc| < 2**16, bm < 2**7, sx.m < 2**8.
    """
    acc = np.asarray(acc, dtype=I64)
    if sx[0] == 0:
        return np.zeros_like(acc)
    check_bits(acc, 16, "matvec accumulator")
    prod = check_bits(acc * np.asarray(bm, dtype=I64) * sx[0], 31, "scale_out product")
    counter.add("scale_out", 2 * acc.size)
    shift = -(np.asarray(be, dtype=I64) + sx[1] + f_out)
    return clamp(rsr_vec(prod, shift), lo, hi)


def xorshift32(state: int) -> int:
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x
