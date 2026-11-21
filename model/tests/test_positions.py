"""Position schemes (G2a): learned (v1 default), ALiBi with power-of-two slopes, NoPE; and the
sliding window that lets ALiBi/NoPE models run past max_seq_len."""
import math

import pytest
import torch

from model.model import ModelConfig, Transformer, alibi_exponents, attention_core
from model.quant import attn_lut_shift, exp_lut_index, exp_lut_weight

TINY = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, n_kv_heads=1,
                   hidden_dim=40, max_seq_len=16)


def _cfg(pos, **kw):
    return ModelConfig(**{**TINY.to_dict(), "pos": pos, **kw})


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = Transformer(cfg)
    m.calibrate(torch.randint(4, cfg.vocab_size, (2, cfg.max_seq_len)))
    return m.eval()


def test_learned_positions_are_the_default_and_serialize_like_v1():
    assert TINY.pos == "learned" and "pos" not in TINY.to_dict()
    assert ModelConfig(**_cfg("alibi").to_dict()).pos == "alibi"
    with pytest.raises(ValueError):
        _cfg("rope")


def test_alibi_slopes_are_powers_of_two_halving_per_head():
    assert alibi_exponents(6) == [5, 4, 3, 2, 1, 0]
    assert alibi_exponents(1) == [0]


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_relative_schemes_have_no_position_table(pos):
    m = _model(_cfg(pos))
    assert not hasattr(m, "pos_emb") and m.embedding_param_count() == 40 * 24
    assert not [k for k in m.state_dict() if k.startswith("pos_emb")]


def test_learned_positions_cannot_run_past_the_table():
    with pytest.raises(ValueError):
        _model(TINY)(torch.randint(4, 40, (1, 17)))


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_long_sequences_see_a_sliding_window_of_max_seq_len(pos):
    """One layer: changing token j changes exactly the logits at positions j .. j + W - 1."""
    m = _model(_cfg(pos, n_layers=1))
    x = torch.randint(4, 40, (1, 50))
    x2 = x.clone()
    x2[0, 20] = (x[0, 20] + 1 - 4) % 36 + 4
    with torch.no_grad():
        a, b = m(x)[0][0], m(x2)[0][0]
    changed = (a != b).any(-1).nonzero().flatten().tolist()
    assert changed[0] == 20 and changed[-1] == 20 + 16 - 1      # windows that contain token 20


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_blockwise_attention_equals_one_banded_mask(pos):
    """The query-block path for long sequences equals a full T x T banded computation."""
    torch.manual_seed(1)
    b, t, h, hd, w, lut = 2, 37, 3, 8, 16, 512
    q8 = torch.randint(-127, 128, (b, t, h * hd)).float()
    k8 = torch.randint(-127, 128, (b, t, hd)).float()
    v8 = torch.randint(-127, 128, (b, t, hd)).float()
    u, sv = torch.tensor(0.0004), torch.tensor(0.01)
    ex = torch.tensor(alibi_exponents(h)) if pos == "alibi" else None
    o, _ = attention_core(q8, k8, v8, u, sv, h, lut, alibi=ex, window=w)
    # reference: everything at once with an explicit band and the integer bias
    q = q8.view(b, t, h, hd).transpose(1, 2)
    s = q @ k8[:, None].transpose(-1, -2)
    i, j = torch.arange(t)[:, None], torch.arange(t)[None, :]
    sh = attn_lut_shift(u, lut)
    if ex is not None:
        bias = torch.clamp((i - j).clamp(min=0)[None] * 2.0 ** ex[:, None, None], max=lut) * 2.0 ** sh
        s = s - bias
    s = s.masked_fill(~((j <= i) & (j > i - w)), float("-inf"))
    delta = s.amax(-1, keepdim=True) - s
    e = exp_lut_weight(exp_lut_index(delta, sh), u * 2.0 ** sh, lut)
    want = ((e @ v8[:, None]) / e.sum(-1, keepdim=True) * sv).transpose(1, 2).reshape(b, t, -1)
    assert torch.allclose(o, want, atol=1e-6)


def test_alibi_bias_is_integer_shifts_of_the_distance():
    """Weights = LUT[rsr(max z - z, sh)], z = s - min(dist << e_h, L) << sh (INTREF.md 12)."""
    torch.manual_seed(2)
    t, h, hd, lut = 12, 2, 4, 64
    q8 = torch.randint(-20, 21, (1, t, h * hd)).float()
    k8 = torch.randint(-20, 21, (1, t, hd)).float()
    v8 = torch.randint(-127, 128, (1, t, hd)).float()
    u = torch.tensor(0.05)
    _, e = attention_core(q8, k8, v8, u, torch.tensor(1.0), h, lut,
                          alibi=torch.tensor([1, 0]), window=16)
    sh = int(attn_lut_shift(u, lut))
    step = float(u) * 2 ** sh
    for head, ex in enumerate((1, 0)):
        for i in range(t):
            z = [int(q8[0, i, head * hd:(head + 1) * hd] @ k8[0, j]) - (min((i - j) << ex, lut) << sh)
                 for j in range(i + 1)]
            idx = [min((max(z) - zj + (1 << (sh - 1) if sh else 0)) >> sh, lut - 1) for zj in z]
            want = [0.0 if ix >= lut - 1 else math.floor(255 * math.exp(-ix * step) + 0.5) for ix in idx]
            assert e[0, head, i, : i + 1].tolist() == want


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_relative_schemes_train(pos):
    m = _model(_cfg(pos))
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(4, 40, (4, 17))
    first = None
    for _ in range(60):
        _, loss = m(idx[:, :-1], idx[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * first
