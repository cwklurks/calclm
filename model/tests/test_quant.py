import math

import pytest
import torch

from model.quant import (
    act_quant, attn_lut_shift, exp_lut_values, pf_round, round_half_up, weight_quant,
)


def test_round_half_up_is_not_bankers():
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, 0.49])
    assert round_half_up(x).tolist() == [1.0, 2.0, 3.0, 0.0, -1.0, 0.0]


@pytest.mark.parametrize("bits", [7, 8, 15])
def test_pf_round_mantissa_range_and_error(bits):
    x = torch.exp(torch.linspace(-20, 20, 4001, dtype=torch.float64))
    m, e = pf_round(x, bits)
    assert ((m >= 2 ** (bits - 1)) & (m < 2**bits)).all()
    y = m * torch.pow(2.0, e)
    assert ((y - x).abs() / x).max() <= 2.0 ** (-bits)
    exact = torch.tensor([3.0 * 2**-10, 2.0**5], dtype=torch.float64)
    m2, e2 = pf_round(exact, bits)
    assert torch.equal(m2 * torch.pow(2.0, e2), exact)


def test_pf_round_zero():
    m, e = pf_round(torch.zeros(3), 8)
    assert (m == 0).all()


def test_act_quant_int8_and_pf8_scale():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 192) * 3
    x8, sx = act_quant(x)
    assert x8.abs().max() <= 127 and torch.equal(x8, x8.round())
    m, e = pf_round(sx.double(), 8)
    assert torch.equal(m * torch.pow(2.0, e), sx.double())  # scale is exactly PF8
    assert (x8.abs().amax(-1) >= 126).all()
    err = (x8 * sx - x).abs()
    assert (err <= sx * 0.5 + 1e-5 * x.abs()).all()


def test_act_quant_straight_through_gradient():
    x = torch.randn(3, 16, requires_grad=True)
    x8, sx = act_quant(x)
    (x8 * sx).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_weight_quant_ternary_rows_and_pf7_scales():
    torch.manual_seed(1)
    w = torch.randn(48, 40) * torch.linspace(0.2, 2.0, 48)[:, None]
    wt, bm, be = weight_quant(w)
    assert set(wt.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert bm.dtype == torch.float32 and (bm == bm.round()).all()
    assert bm.max() <= 127 and bm.min() >= 64  # normalized 7-bit mantissa per row
    assert be.shape == (48,) and (be == be.round()).all()  # one exponent per row
    beta = w.abs().mean(1)
    assert torch.allclose(bm * 2.0**be, beta, rtol=2**-7, atol=0)
    # post-scale folds a constant into the row scales before rounding
    _, bm2, be2 = weight_quant(w, post=10.0)
    assert torch.allclose(bm2 * 2.0**be2, beta * 10, rtol=2**-7, atol=0)


def test_weight_quant_threshold_uses_absmean():
    w = torch.tensor([[0.1, -0.4, 2.0, 0.0, -1.0]])  # absmean 0.7
    wt, _, _ = weight_quant(w)
    assert wt.tolist() == [[0.0, -1.0, 1.0, 0.0, -1.0]]


def test_attention_lut_covers_until_zero():
    u = 3e-4
    sh = attn_lut_shift(u, lut_max=512)
    lut = exp_lut_values(u * 2**sh, lut_max=512)
    assert lut[0] == 255 and lut[-1] == 0 and len(lut) <= 512
    assert all(a >= b for a, b in zip(lut, lut[1:]))
    assert sh == 0 or exp_lut_values(u * 2 ** (sh - 1), lut_max=10**6).__len__() > 512
    assert lut[1] == math.floor(255 * math.exp(-u * 2**sh) + 0.5)
