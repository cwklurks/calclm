"""One-sided Welch t-test for G2a (PLAN-v2.md), without scipy.

    welch_greater(a, b) -> {"t", "df", "p", "diff"}: H1 mean(a) > mean(b)
    paired_greater(a, b, margin) -> same keys: H1 mean(a - b) > margin, paired by index (PLAN-v2.1.md)

The Student t upper tail uses the regularized incomplete beta function (continued fraction,
modified Lentz), valid for fractional degrees of freedom as Welch's test needs.
"""
from __future__ import annotations

import math
import statistics


def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def welch_greater(a, b) -> dict:
    """One-sided Welch test of H1: mean(a) > mean(b)."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    na, nb = len(a), len(b)
    diff = statistics.fmean(a) - statistics.fmean(b)
    va, vb = statistics.variance(a) / na, statistics.variance(b) / nb
    se2 = va + vb
    if se2 == 0.0:
        return {"t": math.copysign(math.inf, diff) if diff else 0.0, "df": float(na + nb - 2),
                "p": 0.0 if diff > 0 else (1.0 if diff < 0 else 0.5), "diff": diff}
    t = diff / math.sqrt(se2)
    df = se2 * se2 / ((va * va / (na - 1) if va else 0.0) + (vb * vb / (nb - 1) if vb else 0.0))
    return {"t": t, "df": df, "p": t_sf(t, df), "diff": diff}


def paired_greater(a, b, margin: float = 0.0) -> dict:
    """One-sided paired t-test of H1: mean(a - b) > margin (df = n - 1). "diff" is the raw mean
    difference; a negative margin makes it a non-inferiority test."""
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")
    d = [float(x) - float(y) for x, y in zip(a, b)]
    n, diff = len(d), statistics.fmean(d)
    var = statistics.variance(d) / n
    shift = diff - margin
    if var == 0.0:
        return {"t": math.copysign(math.inf, shift) if shift else 0.0, "df": float(n - 1),
                "p": 0.0 if shift > 0 else (1.0 if shift < 0 else 0.5), "diff": diff}
    t = shift / math.sqrt(var)
    return {"t": t, "df": n - 1, "p": t_sf(t, n - 1), "diff": diff}
