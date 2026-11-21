import torch

from model.model import BitLinear, ModelConfig, Transformer, attention_weights
from model.quant import act_quant, weight_quant

TINY = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, n_kv_heads=1,
                   hidden_dim=40, max_seq_len=16)


def _model(cfg=TINY, seed=0):
    torch.manual_seed(seed)
    m = Transformer(cfg)
    m.calibrate(torch.randint(4, cfg.vocab_size, (2, cfg.max_seq_len)))
    return m


def test_bitlinear_equals_integer_accumulate_times_scales():
    torch.manual_seed(0)
    lin = BitLinear(23, 9).double()
    x = torch.randn(5, 23, dtype=torch.float64)
    x8, sx = act_quant(x)
    wt, bm, be = weight_quant(lin.weight)
    acc = x8.round() @ wt.double().T  # exact integers
    want = acc * (bm * 2.0**be).double() * sx
    assert torch.allclose(lin(x8, sx), want, rtol=1e-12, atol=0)
    # post folds a constant into the row scales: the output is in units of 1/post,
    # exactly what the integer scale_out computes with the exported (folded) mantissas
    post = torch.tensor(3.0, dtype=torch.float64)
    _, bm3, be3 = weight_quant(lin.weight, post)
    want3 = acc * (bm3 * 2.0**be3).double() * sx
    assert torch.allclose(lin(x8, sx, post=post), want3, rtol=1e-12, atol=0)
    assert torch.allclose(want3 / post, want, rtol=2**-6)


def test_bitlinear_ste_gradients_reach_latent_weights_and_inputs():
    lin = BitLinear(10, 4)
    x = torch.randn(3, 10, requires_grad=True)
    lin(*act_quant(x)).pow(2).sum().backward()
    assert lin.weight.grad.abs().sum() > 0 and x.grad.abs().sum() > 0


def test_forward_shapes_and_loss():
    m = _model()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, TINY.vocab_size) and loss.ndim == 0


def test_causal():
    m = _model().eval()
    idx = torch.randint(4, TINY.vocab_size, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10] = (idx[0, 10] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, b = m(idx)[0], m(idx2)[0]
    assert torch.equal(a[:, :10], b[:, :10]) and not torch.equal(a[:, 10:], b[:, 10:])


def test_static_scales_calibrated_and_frozen_in_eval():
    m = _model()
    amax = m.layers[0].attn.amax.clone()
    assert (amax > 0).all()
    m.eval()
    with torch.no_grad():
        m(torch.randint(0, TINY.vocab_size, (2, 16)))
    assert torch.equal(m.layers[0].attn.amax, amax)
    m.train()
    m(torch.randint(0, TINY.vocab_size, (2, 16)))
    assert not torch.equal(m.layers[0].attn.amax, amax)  # EMA moves in training


def test_attention_weights_are_uint8_lut_values():
    m = _model().eval()
    with torch.no_grad():
        e = attention_weights(m, torch.randint(4, TINY.vocab_size, (1, 16)), layer=0)
    assert torch.equal(e, e.round()) and e.min() >= 0 and e.max() == 255
    assert (e.amax(-1) == 255).all()  # every query row has its max at 255
    assert (torch.triu(e[0, 0], diagonal=1) == 0).all()  # causal mask


def test_v0_shape_counts():
    cfg = ModelConfig(vocab_size=812)
    m = Transformer(cfg)
    assert cfg.head_dim == 32
    per_layer = 192 * 192 * 2 + 192 * 32 * 2 + 192 * 512 * 3
    assert m.ternary_param_count() == 6 * per_layer + 192 * 812
    assert m.embedding_param_count() == 812 * 192 + 256 * 192


def test_overfits_tiny_sequence():
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(4, TINY.vocab_size, (4, 16))
    first = None
    for _ in range(60):
        _, loss = m(idx[:, :-1], idx[:, 1:])
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.6 * first


def test_float_attention_caps_alibi_penalties_and_respects_causality_and_window():
    from model import model as mm
    from dataclasses import replace
    cfg = replace(TINY, n_layers=1, max_seq_len=8, lut_max=4, pos='alibi')
    m = _model(cfg).train()
    with torch.no_grad():
        m.layers[0].attn.wq.weight.zero_()
        m.layers[0].attn.wk.weight.zero_()  # isolate the ALiBi distance penalty
    try:
        mm.set_quant_enabled(False)
        e = mm.attention_weights(m, torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11]]), 0)
        assert (torch.triu(e[0, 0], diagonal=1) == 0).all()
        # Head 0 saturates at distance 1 with lut_max=4 and exponent=2.
        assert torch.equal(e[0, 0, -1, :7], e[0, 0, -1, :1].expand(7))
        assert torch.allclose(e.sum(-1), torch.ones_like(e.sum(-1)))
        x = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
        changed = x.clone()
        changed[0, 0] = 6
        a, b = m(x)[0], m(changed)[0]
        assert torch.equal(a[:, 8:], b[:, 8:])  # changed token left the one-layer window
        m(x, x)[1].backward()
        assert m.output.weight.grad.abs().sum() > 0
    finally:
        mm.set_quant_enabled(True)
