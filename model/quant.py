"""Fake-quantization primitives shared by QAT (model.py) and the exporter (export.py).

Every function here mirrors an integer operation in intref.py (see INTREF.md):
- PF<bits> numbers: value = m * 2**e with integer m in [2**(bits-1), 2**bits) (or m = 0).
- Activations: int8 in [-127, 127] with one PF8 scale per vector (absmax / 127, rounded).
- Weights: ternary with a per-row absmean scale stored as PF7 (mantissa 64..127, exponent per
  row), so acc * weight-mantissa * activation-mantissa fits in 31 bits.
- Attention: exp lookup table of uint8 weights indexed by rounded score differences.
Rounding is always round-half-up (floor(x + 0.5)), matching the integer reference.
"""
from __future__ import annotations

import math

import torch

LN510 = math.log(510.0)  # exp(-x) * 255 < 0.5 beyond this


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5)


def pf_round(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round positive x to m * 2**e with m in [2**(bits-1), 2**bits); zero stays zero."""
    x = torch.as_tensor(x)
    f, ex = torch.frexp(x)  # x = f * 2**ex, f in [0.5, 1)
    m = round_half_up(f * 2.0**bits)
    e = (ex - bits).to(x.dtype)
    carry = m >= 2.0**bits
    return torch.where(carry, m / 2, m), torch.where(carry, e + 1, e)


def pf_value(x: torch.Tensor, bits: int) -> torch.Tensor:
    m, e = pf_round(x, bits)
    return m * torch.pow(2.0, e)


def act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector absmax int8 quantization with a PF8 scale.

    Returns (x8, sx): x8 holds integer values in [-127, 127] (straight-through gradient,
    d(x8 * sx)/dx = 1) and sx has shape (..., 1); x8 * sx approximates x.
    """
    with torch.no_grad():
        amax = x.detach().abs().amax(dim=-1, keepdim=True).float()
        sx = pf_value(amax / 127.0, 8)
        sx = torch.where(sx > 0, sx, torch.ones_like(sx)).to(x.dtype)
    scaled = x / sx
    x8 = torch.clamp(round_half_up(scaled.detach()), -127, 127)
    return scaled + (x8 - scaled).detach(), sx


@torch.no_grad()
def weight_quant(w: torch.Tensor, post=None):
    """Ternary weights with per-row absmean scales.

    Returns (wt, bm, be): wt in {-1, 0, 1}; row scale = bm * 2**be per row, bm an integer
    in [64, 127] (PF7) and be an integer exponent per row. The ternary threshold uses the
    raw absmean; `post` (a positive scalar) is folded into the scale before rounding.
    """
    w = w.float()
    beta = w.abs().mean(dim=1).clamp(min=1e-8)
    wt = torch.clamp(round_half_up(w / beta[:, None]), -1, 1)
    scaled = beta if post is None else beta * post
    bm, be = pf_round(scaled, 7)
    return wt, bm, be


def attn_lut_shift(u, lut_max: int):
    """Smallest shift sh >= 0 so that (lut_max - 1) table steps of u * 2**sh pass ln(510).

    Works on floats and tensors; u is the real value of one integer score unit.
    """
    if isinstance(u, torch.Tensor):
        sh = torch.ceil(torch.log2(LN510 / ((lut_max - 1) * u)))
        return torch.clamp(sh, min=0)
    return max(0, math.ceil(math.log2(LN510 / ((lut_max - 1) * u))))


def exp_lut_index(delta: torch.Tensor, sh) -> torch.Tensor:
    """Rounded table index of a non-negative integer score difference."""
    half = torch.where(torch.as_tensor(sh) > 0, torch.pow(2.0, sh - 1), torch.zeros(()))
    return torch.floor((delta + half.to(delta.device)) / torch.pow(2.0, sh))


def exp_lut_weight(idx: torch.Tensor, step, lut_max: int) -> torch.Tensor:
    """uint8 softmax weight: round(255 * exp(-idx * step)), forced to 0 at idx >= lut_max - 1."""
    vals = round_half_up(255.0 * torch.exp(-idx * step))
    return torch.where(idx >= lut_max - 1, torch.zeros_like(vals), vals)


def exp_lut_values(step: float, lut_max: int) -> list[int]:
    """The attention table as exported: entries up to and including the first zero."""
    idx = torch.arange(lut_max, dtype=torch.float64)
    vals = exp_lut_weight(idx, torch.tensor(step, dtype=torch.float64), lut_max)
    first_zero = int(torch.nonzero(vals == 0)[0])
    return [int(v) for v in vals[: first_zero + 1]]
