"""ALiBi horizons in the integer reference (PLAN-v2.1.md E1). With `horizons`, head h attends
only to the keys at distance d with (d << e_h) < lut_max: a key whose penalty sits at the cap
gets weight exactly 0 (out of the weight sum and of P.V), and is not multiplied at all. Off,
intref stays bit-identical to the shipped v2 contract."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from model.export import export_model, load_export
from model.intops import NULL_COUNTER, MulCounter, norm_quant, scale_out
from model.intref import (
    IntConfig, IntModel, KVCache, SamplerConfig, alibi_exponents, alibi_horizons, expected_mults,
    generate, ranked_candidates, sequence_logits,
)
from model.model import ModelConfig, Transformer
from model.pack import lut_matvec
from model.tests.conftest import pattern_batch
from model.tokens import BOS, EOS

ROOT = Path(__file__).resolve().parents[2]
V2_EXPORT = ROOT / "runs" / "v2" / "export"
have_v2 = pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(),
                             reason="runs/v2/export not present")

# Window 16, lut_max 128, exponents 5..0: horizons 4, 8, 16, 32, 64, 128, so heads 0 and 1 drop
# keys inside the window and heads 2-5 keep all 16.
HZ = ModelConfig(vocab_size=24, dim=48, n_layers=2, n_heads=6, n_kv_heads=1, hidden_dim=96,
                 max_seq_len=16, lut_max=128, pos="alibi")

# A fixed 300-token prompt for the v2 export, and the sha256 of its int64 logits at positions
# 0..299 from intref as it was before horizons existed (commit 4bcaeed).
PROMPT = [BOS] + [(37 * p) % 777 + 4 for p in range(1, 300)]
V2_DIGEST = "988b176fb1d9b3f4e5d3f9d1111b21c2cf7f195379bf4279f1c7f12ef8263cc7"


def _digest(logits) -> str:
    return hashlib.sha256(np.ascontiguousarray(logits, dtype="<i8").tobytes()).hexdigest()


def _export(cfg, path, steps=0, seed=0):
    """Export a small model after `steps` of training on repeating patterns. (Untrained, the
    attention shift is so large that lut_max << sh overflows the 24-bit ALiBi scores.)"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w, v = cfg.max_seq_len, cfg.vocab_size
    m = Transformer(cfg)
    m.calibrate(torch.from_numpy(pattern_batch(rng, 8, w, v)[:, :-1]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        seq = torch.from_numpy(pattern_batch(rng, 16, w, v))
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    calib = pattern_batch(np.random.default_rng(seed + 1), 8, w, v)[:, :-1]
    export_model(m.eval(), calib, path)
    return load_export(path)


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    """Export of a small pattern-trained ALiBi model (window 16, horizons 4, 8, 16, ...)."""
    return _export(HZ, tmp_path_factory.mktemp("hz"), steps=300)


@pytest.fixture(scope="module")
def v2():
    return load_export(V2_EXPORT)


@pytest.fixture(scope="module")
def v2_off_logits(v2):
    return sequence_logits(IntModel(v2), PROMPT)


def _seq(n, seed=4):
    return pattern_batch(np.random.default_rng(seed), 1, n, 24)[0, :n].tolist()


# ---- the cutoff -------------------------------------------------------------------------------

@pytest.mark.parametrize("lut_max", [512, 128, 100, 7, 1])
def test_horizon_is_the_first_distance_whose_penalty_is_capped(lut_max):
    exps = alibi_exponents(6)
    d = np.arange(4096)
    hz = alibi_horizons(exps, lut_max)
    assert len(hz) == 6
    for e, h in zip(exps, hz):
        assert np.array_equal((d << e) >= lut_max, d >= h), (e, h)


@have_v2
def test_v2_horizons_come_from_the_export(v2):
    im = IntModel(v2, horizons=True)
    assert im.horizons.tolist() == [16, 32, 64, 128, 256, 512]       # lut_max 512, e_h 5..0
    assert IntModel(v2).horizons is None
    assert IntModel.load(V2_EXPORT, horizons=True).horizons.tolist() == im.horizons.tolist()


def test_horizons_need_alibi(tmp_path):
    for pos in ("learned", "nope"):
        cfg = ModelConfig(**{**HZ.to_dict(), "pos": pos})
        e = _export(cfg, tmp_path / pos)
        IntModel(e)                                                      # still loads off
        with pytest.raises(ValueError):
            IntModel(e, horizons=True)
        with pytest.raises(ValueError):
            expected_mults(IntModel(e).cfg, 3, horizons=True)


# ---- off: the shipped contract ------------------------------------------------------------------

@have_v2
def test_off_is_bit_identical_to_intref_before_horizons(v2, v2_off_logits):
    assert v2_off_logits.shape == (300, 781)
    assert _digest(v2_off_logits) == V2_DIGEST
    assert _digest(sequence_logits(IntModel(v2, horizons=False), PROMPT)) == V2_DIGEST


# ---- on: exact exclusion ------------------------------------------------------------------------

def _step_recording(im, tok, pos, cache):
    """Step one token; return each layer's attention input r and per-head outputs (the
    _head_avg results), in layer order."""
    rs, heads = [], []
    attention = im.attention

    def attend(i, lp, r, pos, cache, counter):
        rs.append(r.copy())
        return attention(i, lp, r, pos, cache, counter)

    def head_avg(a, z, counter):
        o = IntModel._head_avg(a, z, counter)
        heads.append(o.copy())
        return o

    im.attention, im._head_avg = attend, head_avg
    try:
        im.step(tok, pos, cache)
    finally:
        del im.attention, im._head_avg
    h = im.cfg.n_heads
    return rs, [heads[i * h:(i + 1) * h] for i in range(len(im.layers))]


def _copy(cache):
    return KVCache(cache.k.copy(), cache.v.copy())


def _perturbed(im, seq, d, layer=0):
    """Feed seq[:-1], then the last token twice: once as is, and once with the layer's K/V of
    the key at distance d overwritten (key aligned with head 0's query, value flipped). Returns
    the layer's per-head outputs for both."""
    P, hd, win = len(seq) - 1, im.cfg.head_dim, im.cfg.max_seq_len
    cache = im.new_cache()
    for p, t in enumerate(seq[:P]):
        im.step(t, p, cache)
    rs, base = _step_recording(im, seq[P], P, _copy(cache))
    lp = im.layers[layer]
    x8, sx = norm_quant(rs[layer], lp.attn_g, lp.attn_eg, NULL_COUNTER, im.sqrtd)
    q0 = scale_out(lut_matvec(lp.wq.w, x8), lp.wq.m, lp.wq.e, sx, 0, -127, 127)[:hd]
    moved = _copy(cache)
    slot = (P - d) % win
    moved.k[layer, slot] = np.where(q0 >= 0, 127, -127)
    moved.v[layer, slot] = np.where(cache.v[layer, slot] >= 0, -127, 127)
    _, got = _step_recording(im, seq[P], P, moved)
    return base[layer], got[layer]


@pytest.mark.parametrize("horizons", [True, False])
def test_a_key_past_head_0s_horizon_reaches_head_0_only_without_horizons(ex, horizons):
    """Distance 5 is past head 0's horizon (4) and inside every other head's. In layer 1 the
    aligned key wins weight in head 0 without horizons, so the check is not vacuous."""
    im = IntModel(ex, horizons=horizons)
    seq = _seq(42)                                    # past two windows: the ring has wrapped
    base, got = _perturbed(im, seq, 5, layer=1)
    assert np.array_equal(base[0], got[0]) == horizons
    assert any(not np.array_equal(base[h], got[h]) for h in range(1, im.cfg.n_heads))
    base, got = _perturbed(im, seq, 3, layer=1)       # inside head 0's horizon: it reads it
    assert not np.array_equal(base[0], got[0])


@have_v2
@pytest.mark.parametrize("horizons", [True, False])
def test_v2_head_0_reads_a_key_at_distance_20_only_without_horizons(v2, horizons):
    """Without horizons a capped key can still win weight in head 0 (here: a key aligned with
    its query, in layer 3, whose 506-entry LUT nearly reaches lut_max), so excluding it is a
    real change, not a no-op."""
    base, got = _perturbed(IntModel(v2, horizons=horizons), PROMPT[:271], 20, layer=3)
    assert np.array_equal(base[0], got[0]) == horizons
    assert not np.array_equal(base[1], got[1])        # head 1 (horizon 32) reads it either way


def test_on_matches_off_until_a_key_reaches_a_horizon(ex):
    off, on = IntModel(ex), IntModel(ex, horizons=True)
    seq = _seq(60)
    a, b = sequence_logits(off, seq), sequence_logits(on, seq)
    first = int(on.horizons.min())                    # 4: the first position with a capped key
    assert first == 4
    assert np.array_equal(a[:first], b[:first])
    assert not np.array_equal(a[first:], b[first:])


@have_v2
def test_v2_on_matches_off_below_16_then_differs(v2, v2_off_logits):
    on = sequence_logits(IntModel(v2, horizons=True), PROMPT)
    assert np.array_equal(on[:16], v2_off_logits[:16])
    assert not np.array_equal(on[16:], v2_off_logits[16:])


# ---- multiply count -----------------------------------------------------------------------------

def test_multiply_count_with_horizons_counts_only_the_keys_used(ex):
    im = IntModel(ex, horizons=True)
    cache = im.new_cache()
    for pos, tok in enumerate([1, 5, 6, 7] * 10):
        counter = MulCounter()
        im.step(tok, pos, cache, counter)
        assert counter.counts == expected_mults(im.cfg, pos, horizons=True), pos
    full, cut = expected_mults(im.cfg, 40), expected_mults(im.cfg, 40, horizons=True)
    keys = 4 + 8 + 16 * 4                             # per-head keys at a full 16-token window
    assert cut["attn_score"] == cut["attn_pv"] == im.cfg.n_layers * im.cfg.head_dim * keys
    assert full["attn_score"] == im.cfg.n_layers * im.cfg.head_dim * 16 * 6
    assert {k: v for k, v in cut.items() if k not in ("attn_score", "attn_pv")} == \
        {k: v for k, v in full.items() if k not in ("attn_score", "attn_pv")}
    assert expected_mults(im.cfg, 2, horizons=True) == expected_mults(im.cfg, 2)   # nothing cut


def test_v2_shape_attention_multiplies_with_horizons():
    cfg = IntConfig(vocab_size=781, pos="alibi")      # v2: 6 heads of 32, window 256, lut_max 512
    keys = 16 + 32 + 64 + 128 + 256 + 256
    assert expected_mults(cfg, 1023, horizons=True)["attn_score"] == 6 * 32 * keys
    assert expected_mults(cfg, 1023)["attn_score"] == 6 * 32 * 6 * 256
    assert expected_mults(cfg, 20, horizons=True)["attn_pv"] == 6 * 32 * (16 + 21 * 5)


# ---- generation ---------------------------------------------------------------------------------

def test_generation_with_horizons_is_deterministic(ex):
    cfg = SamplerConfig(256, 8, False, 7)
    a = generate(IntModel(ex, horizons=True), [], 10_000, cfg, cap=60)
    b = generate(IntModel(ex, horizons=True), [], 10_000, cfg, cap=60)
    assert a == b and a[0] == BOS and (len(a) == 60 or a[-1] == EOS)
    on = IntModel(ex, horizons=True)                  # greedy follows the horizon logits
    ids = generate(on, [], 10_000, SamplerConfig(greedy=True), cap=50)
    assert ranked_candidates(sequence_logits(on, ids[:-1])[-1], 1) == [ids[-1]]


@have_v2
def test_v2_generation_with_horizons_is_deterministic(v2):
    cfg = SamplerConfig(205, 20, False, 1)
    a = generate(IntModel(v2, horizons=True), [], 10_000, cfg, cap=300)
    assert a == generate(IntModel(v2, horizons=True), [], 10_000, cfg, cap=300)
    assert len(a) == 300 or a[-1] == EOS


def test_the_cli_samples_with_horizons(capsys):
    """INSTALL.md names `python -m model.intref ... --horizons` as the v2.1 check program."""
    from pathlib import Path

    from model.intref import main
    export = Path(__file__).resolve().parents[2] / "runs" / "v2" / "export"
    if not (export / "model.json").exists():
        pytest.skip("needs runs/v2/export")
    main(["--export", str(export), "--seed", "623", "--n", "1", "--max-new", "40", "--masks",
          "--horizons"])
    on = capsys.readouterr().out
    main(["--export", str(export), "--seed", "623", "--n", "1", "--max-new", "40", "--masks"])
    off = capsys.readouterr().out
    assert on.startswith("--- sample 0: seed=623") and on != off
