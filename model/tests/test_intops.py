import math

import numpy as np
import pytest
import torch

from model.intops import (
    INV127, MulCounter, isqrt, norm_quant, pf8, pf15, pf_div, pf_mul, pf_to_float, quant_vec,
    rsr, rsr_vec, sqrt_pf15, xorshift32,
)
from model.quant import act_quant


def test_rsr_round_half_up_and_left_shift():
    assert [rsr(v, 1) for v in (1, 2, 3, -1, -2, -3)] == [1, 1, 2, 0, -1, -1]
    assert rsr(5, 0) == 5 and rsr(5, -3) == 40
    a = np.array([1, 3, -3, 7], dtype=np.int64)
    assert rsr(a, 1).tolist() == [1, 2, -1, 4]
    assert rsr_vec(a, np.array([1, 0, -2, 2])).tolist() == [1, 3, -12, 2]


def test_isqrt_is_floor_sqrt():
    for n in [0, 1, 2, 3, 4, 15, 16, 17, 2**30 - 1, 2**30, 123456789]:
        assert isqrt(n) == math.isqrt(n)


@pytest.mark.parametrize("n", [1, 2, 3, 127, 32767, 32768, 65535, 2**30 + 12345])
def test_pf15_normalized_and_close(n):
    m, e = pf15(n)
    assert 2**14 <= m < 2**15
    assert abs(m * 2.0**e - n) / n <= 2.0**-15


def test_pf_mul_div_and_constants():
    a, b = pf15(12345, -3), pf15(777, 5)
    assert math.isclose(pf_to_float(pf_mul(a, b)), 12345 / 8 * 777 * 32, rel_tol=2**-13)
    assert math.isclose(pf_to_float(pf_div(a, b)), (12345 / 8) / (777 * 32), rel_tol=2**-13)
    assert math.isclose(pf_to_float(INV127), 1 / 127, rel_tol=2**-14)
    assert math.isclose(pf_to_float(sqrt_pf15(192)), math.sqrt(192), rel_tol=2**-14)
    m8, _ = pf8(a)
    assert 128 <= m8 < 256


def test_xorshift32_reference_sequence():
    # Marsaglia xorshift32 (13, 17, 5) from state 1
    s, out = 1, []
    for _ in range(3):
        s = xorshift32(s)
        out.append(s)
    assert out == [270369, 67634689, 2647435461]


def test_quant_vec_matches_float_act_quant():
    rng = np.random.default_rng(0)
    agree, total = 0, 0
    for _ in range(50):
        v = rng.integers(-(2**20), 2**20, size=192).astype(np.int64)
        unit = pf15(rng.integers(2**14, 2**15), int(rng.integers(-40, -20)))
        x8, sx = quant_vec(v, unit)
        assert x8.dtype == np.int64 and np.abs(x8).max() <= 127
        real = v * pf_to_float(unit)
        f8, fsx = act_quant(torch.tensor(real, dtype=torch.float64))
        assert math.isclose(sx[0] * 2.0 ** sx[1], float(fsx), rel_tol=2**-7)
        agree += int((x8 == f8.round().long().numpy()).sum())
        total += 192
        assert np.abs(x8 * sx[0] * 2.0 ** sx[1] - real).max() <= 0.51 * sx[0] * 2.0 ** sx[1] + 1e-9
    assert agree / total > 0.99


def test_norm_quant_matches_float_rmsnorm():
    rng = np.random.default_rng(1)
    g = rng.uniform(0.3, 2.0, size=192)
    eg = 13
    g16 = np.round(g * 2**eg).astype(np.int64)
    for scale in (2**4, 2**12, 2**20):
        r = rng.normal(0, 1, size=192)
        r[3] = 9.0  # an outlier channel
        r_int = np.round(r * scale).astype(np.int64)
        x8, sx = norm_quant(r_int, g16, eg)
        y = (g16 / 2**eg) * r_int / np.sqrt(np.mean(r_int.astype(float) ** 2))
        assert np.abs(x8 * sx[0] * 2.0 ** sx[1] - y).max() < 0.02 * np.abs(y).max()


def test_mul_counter_tallies_by_kind():
    c = MulCounter()
    c.add("attn_score", 32 * 5)
    c.add("attn_score", 32)
    c.add("requant", 192)
    assert c.counts == {"attn_score": 192, "requant": 192} and c.total() == 384


def test_normalized_vectors_saturate_to_int16():
    # rsr(65535, 1) rounds up to 32768, which does not fit int16; the contract saturates
    v = np.array([65535, -65535, 3], dtype=np.int64)
    from model.intops import normalize16

    vp, c = normalize16(v)
    assert c == 1 and vp.tolist() == [32767, -32767, 2]
    x8, _ = quant_vec(v, pf15(20000, -20))
    assert x8.tolist()[:2] == [127, -127]


def test_norm_quant_takes_sqrt_d_from_caller():
    rng = np.random.default_rng(5)
    r = rng.integers(-5000, 5000, 192)
    g = np.full(192, 16384)
    assert np.array_equal(norm_quant(r, g, 14)[0], norm_quant(r, g, 14, sqrtd=sqrt_pf15(192))[0])
