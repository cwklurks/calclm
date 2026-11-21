"""Integer reference v2 (INTREF.md 6, 12, 13): relative-position exports (format 2), ALiBi in
the integer attention, the ring K/V cache past the window, generation up to a cap, and the
tracker's logit masks (model/intmask.py) making lab/masks.py's decisions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.export import export_model, load_export
from model.intops import NULL_COUNTER, MulCounter, norm_quant, quant_vec, rsr, scale_out
from model.intref import (
    IntModel, SamplerConfig, alibi_bias, choose, expected_mults, generate, sequence_logits,
)
from model.model import ModelConfig, Transformer
from model.pack import lut_matvec
from model.tests.conftest import pattern_batch
from model.tokens import BOS, EOS, PAD, UNK

ROOT = Path(__file__).resolve().parents[2]
VOCAB = ROOT / "corpus" / "vocab.json"
VAL = ROOT / "data" / "tok" / "val.bin"
V2_EXPORT = ROOT / "runs" / "v2" / "export"
S0_SAMPLES = ROOT / "runs" / "lab" / "S0-s1" / "samples.jsonl"

# 6 heads (ALiBi exponents 5..0) and a short exp table, so the lut_max cap is active
REL = ModelConfig(vocab_size=24, dim=48, n_layers=2, n_heads=6, n_kv_heads=1, hidden_dim=96,
                  max_seq_len=16, lut_max=128, pos="alibi")


def _train(cfg, steps, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Transformer(cfg)
    w = cfg.max_seq_len
    m.calibrate(torch.from_numpy(pattern_batch(rng, 8, w, cfg.vocab_size)[:, :-1]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        seq = torch.from_numpy(pattern_batch(rng, 16, w, cfg.vocab_size))
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


@pytest.fixture(scope="module")
def rel(tmp_path_factory):
    """(model, IntModel) of a small pattern-trained ALiBi model with a 16-token window."""
    m = _train(REL, 300)
    out = tmp_path_factory.mktemp("rel")
    calib = pattern_batch(np.random.default_rng(1), 8, 16, 24)[:, :-1]
    export_model(m, calib, out)
    return m, IntModel.load(out)


# ---- export format 2 ----------------------------------------------------------------------

@pytest.mark.parametrize("pos,code", [("alibi", 1), ("nope", 2)])
def test_relative_exports_are_format_2_with_the_scheme_in_the_header(pos, code, tmp_path):
    cfg = ModelConfig(**{**REL.to_dict(), "pos": pos})
    torch.manual_seed(0)
    m = Transformer(cfg)
    calib = pattern_batch(np.random.default_rng(0), 4, 16, 24)[:, :-1]
    m.calibrate(torch.from_numpy(calib))
    meta = export_model(m.eval(), calib, tmp_path)
    hdr = load_export(tmp_path).header()
    assert meta["version"] == hdr["version"] == 2 and hdr["pos_scheme"] == code
    assert meta["tail_fields"] == ["pos_scheme"]
    assert meta["tensors"]["header"]["shape"] == [16 + 6 * 2 + 1]
    assert not [t for t in meta["tensors"] if t.startswith("pos_emb")]


def test_learned_position_exports_keep_format_1(tmp_path):
    cfg = ModelConfig(**{**REL.to_dict(), "pos": "learned"})
    torch.manual_seed(0)
    m = Transformer(cfg)
    calib = pattern_batch(np.random.default_rng(0), 4, 16, 24)[:, :-1]
    m.calibrate(torch.from_numpy(calib))
    meta = export_model(m.eval(), calib, tmp_path)
    hdr = load_export(tmp_path).header()
    assert meta["version"] == hdr["version"] == 1 and "pos_scheme" not in hdr
    assert meta["tensors"]["header"]["shape"] == [16 + 6 * 2]


def test_v1_shipping_export_still_loads_unchanged():
    path = ROOT / "runs" / "v1" / "export"
    if not (path / "model.bin").exists():
        pytest.skip("runs/v1/export not present")
    im = IntModel.load(path)
    assert im.pos_scheme == "learned" and im.cfg.max_seq_len == 256


def test_feature_exports_are_still_refused(tmp_path):
    cfg = ModelConfig(**{**REL.to_dict(), "feat_cats": (3,), "feat_bag": 2})
    torch.manual_seed(0)
    m = Transformer(cfg)
    calib = pattern_batch(np.random.default_rng(0), 4, 16, 24)[:, :-1]
    cats = np.zeros((4, 16, 1), dtype=np.int64)
    bag = np.zeros((4, 16), dtype=np.int64)
    m.calibrate(torch.from_numpy(calib), torch.from_numpy(cats), torch.from_numpy(bag))
    export_model(m.eval(), calib, tmp_path, calib_feats=(cats, bag))
    with pytest.raises(NotImplementedError):
        IntModel.load(tmp_path)


# ---- ALiBi bias and the ring cache ----------------------------------------------------------

@pytest.mark.parametrize("sh,lut_max", [(0, 512), (3, 512), (6, 512), (4, 64)])
def test_alibi_bias_is_the_qat_penalty_in_score_units(sh, lut_max):
    """b = min(d << e_h, lut_max) << sh, exactly model.model._windowed_attention's penalty."""
    exps = np.array([5, 4, 3, 2, 1, 0])
    dist = np.arange(256)
    got = alibi_bias(exps, dist, sh, lut_max)
    want = torch.clamp(torch.arange(256)[None] * torch.pow(2.0, torch.tensor(exps).float())[:, None],
                       max=lut_max) * 2.0 ** sh
    assert got.dtype == np.int64 and np.array_equal(got, want.numpy().astype(np.int64))


def _literal_attention(im, i, lp, r, pos, kv):
    """INTREF.md 12 written out per key, over an unbounded K/V history (kv: list of (k8, v8)
    per layer): the window is the last ctx positions, distances are positions."""
    cfg = im.cfg
    x8, sx = norm_quant(r, lp.attn_g, lp.attn_eg, NULL_COUNTER, im.sqrtd)
    q8, k8, v8 = (scale_out(lut_matvec(w.w, x8), w.m, w.e, sx, 0, -127, 127)
                  for w in (lp.wq, lp.wk, lp.wv))
    kv[i].append((k8, v8))
    held = range(max(0, pos - cfg.max_seq_len + 1), pos + 1)
    o = []
    for h in range(cfg.n_heads):
        q = q8[h * cfg.head_dim:(h + 1) * cfg.head_dim]
        e_h = cfg.n_heads - 1 - h
        z = {j: int(q @ kv[i][j][0]) - (min((pos - j) << e_h, cfg.lut_max) << lp.sh) for j in held}
        top = max(z.values())
        e = {j: int(lp.lut[min(rsr(top - z[j], lp.sh), len(lp.lut) - 1)]) for j in held}
        zsum = sum(e.values())
        a = sum(e[j] * kv[i][j][1] for j in held)
        o.append(im._head_avg(a, zsum, NULL_COUNTER))
    o8, so = quant_vec(np.concatenate(o), (lp.sv[0], lp.sv[1] - 8))
    delta = scale_out(lut_matvec(lp.wo.w, o8), lp.wo.m, lp.wo.e, so, im.F, -(1 << 23), (1 << 23) - 1)
    return np.clip(r + delta, -(1 << 23), (1 << 23) - 1)


def test_ring_cache_equals_the_literal_definition_past_the_window(rel):
    _, im = rel
    seq = pattern_batch(np.random.default_rng(4), 1, 52, 24)[0, :52].tolist()   # > 3 windows
    want, kv = [], [[] for _ in im.layers]
    for pos, tok in enumerate(seq):
        r = im.embed(tok, pos, NULL_COUNTER)
        for i, lp in enumerate(im.layers):
            r = _literal_attention(im, i, lp, r, pos, kv)
            r = im.ffn(lp, r, NULL_COUNTER)
        x8, sx = norm_quant(r, im.norm_g, im.norm_eg, NULL_COUNTER, im.sqrtd)
        o = im.output
        want.append(scale_out(lut_matvec(o.w, x8), o.m, o.e, sx, 8, -(1 << 23), (1 << 23) - 1))
    got = sequence_logits(im, seq)
    np.testing.assert_array_equal(got, np.stack(want))


def test_relative_models_have_no_position_row(rel):
    _, im = rel
    assert im.pos_scheme == "alibi" and not hasattr(im, "pos")
    assert im.alibi_exponents == [5, 4, 3, 2, 1, 0]


def test_intref_tracks_the_qat_model_past_the_window(rel):
    m, im = rel
    xs = pattern_batch(np.random.default_rng(9), 4, 64, 24)[:, :-1]          # 4 windows long
    with torch.no_grad():
        want = m.double()(torch.from_numpy(xs))[0].numpy()
    m.float()
    got = np.stack([sequence_logits(im, row) for row in xs]) / 256.0
    top1 = (want.argmax(-1) == got.argmax(-1)).mean()
    assert top1 >= 0.97
    assert np.abs(want - got).max() < 0.25 * np.abs(want).max()


def test_multiply_count_stops_growing_at_the_window(rel):
    _, im = rel
    cache = im.new_cache()
    for pos, tok in enumerate([1, 5, 6, 7] * 6):
        counter = MulCounter()
        im.step(tok, pos, cache, counter)
        assert counter.counts == expected_mults(im.cfg, pos)
    assert expected_mults(im.cfg, 40) == expected_mults(im.cfg, 15)
    assert expected_mults(im.cfg, 0)["emb"] == im.cfg.dim                   # token row only


def test_learned_positions_stop_at_the_window(tmp_path):
    cfg = ModelConfig(**{**REL.to_dict(), "pos": "learned"})
    torch.manual_seed(0)
    m = Transformer(cfg)
    calib = pattern_batch(np.random.default_rng(0), 4, 16, 24)[:, :-1]
    m.calibrate(torch.from_numpy(calib))
    export_model(m.eval(), calib, tmp_path)
    im = IntModel.load(tmp_path)
    with pytest.raises(ValueError):
        sequence_logits(im, [1] + [5] * 16)
    with pytest.raises(ValueError):
        generate(im, [], 40, SamplerConfig(seed=1), cap=17)


# ---- generation up to a cap -------------------------------------------------------------------

def test_generation_slides_past_the_window_up_to_the_cap(rel):
    _, im = rel
    cfg = SamplerConfig(256, 4, False, 3)
    out = generate(im, [], 10_000, cfg, cap=60)
    assert out[0] == BOS and len(out) == 60 and EOS not in out          # patterns never end
    assert generate(im, [], 10_000, cfg, cap=40) == out[:40]            # a smaller cap: prefix
    assert generate(im, [], 25, cfg, cap=60) == out[:26]                # max_new still counts
    ids = generate(im, [], 10_000, SamplerConfig(greedy=True), cap=50)
    np.testing.assert_array_equal(sequence_logits(im, ids[:-1]).argmax(-1)[-1], ids[-1])


def test_relative_default_cap_is_1024(rel):
    _, im = rel
    assert len(generate(im, [], 10_000, SamplerConfig(256, 2, False, 5))) == 1024


def test_v1_generation_is_unchanged_by_the_cap_argument(tmp_path):
    """Learned positions: the default cap is the context, as before."""
    cfg = ModelConfig(**{**REL.to_dict(), "pos": "learned"})
    m = _train(cfg, 50)
    calib = pattern_batch(np.random.default_rng(0), 4, 16, 24)[:, :-1]
    export_model(m, calib, tmp_path)
    im = IntModel.load(tmp_path)
    out = generate(im, [], 255, SamplerConfig(256, 8, False, 2))
    assert len(out) == 16 and out == generate(im, [], 255, SamplerConfig(256, 8, False, 2), cap=16)


# ---- logit masks ----------------------------------------------------------------------------

masks_lab = pytest.importorskip("lab.masks")


def _tracker_available() -> bool:
    try:
        from tracker import bind
        bind.lib()
        return True
    except Exception:  # noqa: BLE001 - the C tracker needs gcc
        return False


needs_tracker = pytest.mark.skipif(not _tracker_available(), reason="C tracker not available")


@pytest.fixture(scope="module")
def vocab():
    return json.loads(VOCAB.read_text())


@needs_tracker
def test_mask_constants_are_lab_masks_constants():
    from model import intmask
    for name in ("SAFETY_HEX", "END_HEX", "ELSE_HEX", "VAR_HEX", "COMMA_HEX", "LPAREN_HEX",
                 "NEWLINE_HEX", "LISTL_HEX", "NAME_HEX", "N_UNDEF", "N_BAD", "N_EXTRA_END",
                 "CTX_LIST", "HINT_BITS", "HINT_END", "BAG_LIST"):
        assert getattr(intmask, name) == getattr(masks_lab, name), name


@needs_tracker
def test_mask_refuses_a_vocab_the_tracker_was_not_built_for(vocab):
    from model.intmask import IntMask
    with pytest.raises(ValueError):
        IntMask(vocab[:-1])
    swapped = list(vocab)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    with pytest.raises(ValueError):
        IntMask(swapped)


def _compare_on_prefixes(vocab, programs, length):
    """IntMask (one per program) vs lab TrackerMask (one row per program), position by
    position on the same token prefixes."""
    from model.intmask import IntMask
    lab = masks_lab.TrackerMask(VOCAB)
    n = len(programs)
    progs = [list(p)[:length] for p in programs]
    ours = [IntMask(vocab) for _ in range(n)]
    for o in ours:
        o.reset()
        o.step(BOS)
    lab.rows.reset(n)
    lab.rows.step(np.full(n, BOS))
    checked = 0
    for p in range(1, length):
        live = [r for r in range(n) if p < len(progs[r])]
        if not live:
            break
        want = lab.banned(lab.rows)
        for r in live:
            assert np.array_equal(ours[r].banned(), want[r]), (r, p, progs[r][:p])
            checked += 1
        toks = np.array([progs[r][p] if p < len(progs[r]) else EOS for r in range(n)])
        lab.rows.step(toks)
        for r in live:
            ours[r].step(int(toks[r]))
    return checked, lab.stats


# Pending reads (rule 4 of model/intmask.py): after these prefixes the token in the middle is
# banned or allowed; the first four ban `,` (Input Str1,A uses Str1 as a prompt) or `(` (->[A](
# stores into an element of [A]); the rest are siblings where nothing is pending any more.
WORDS = {"Input": "DC", "Prompt": "DD", "->": "04", '"': "2A", ",": "2B", "(": "10", "NL": "3F",
         ":": "3E", "Str1": "AA00", "Str0": "AA09", "[A]": "5C00", "[J]": "5C09", "[": "06",
         "]": "07"}
PENDING_CASES = [
    (("Input", "Str1"), ",", True), (("Input", "Str0"), ",", True),
    (("5", "->", "[A]"), "(", True), (("5", "->", "[J]"), "(", True),
    (("Input", "Str1"), "(", False), (("5", "->", "[A]"), ",", False),
    (('"', "A", '"', "->", "Str1", "NL", "Input", "Str1"), ",", False),       # Str1 assigned
    (("[", "[", "1", "]", "]", "->", "[A]", "NL", "5", "->", "[A]"), "(", False),
    (("Prompt", "Str1"), ",", False), (("Input", "Str1", "NL"), ",", False),
    (("5", "->", "[A]", ":"), "(", False),
]


def word_ids(vocab, *words) -> list[int]:
    idx = {h: i for i, h in enumerate(vocab)}
    out = []
    for w in words:
        out += [idx[WORDS[w]]] if w in WORDS else [idx["%02X" % ord(ch)] for ch in w]
    return out


def pending_programs(vocab) -> list[list[int]]:
    """<bos>, the case's prefix, then a newline (so the comparison covers the position right
    after the prefix)."""
    return [[BOS, *word_ids(vocab, *words), *word_ids(vocab, "NL")]
            for words, _, _ in PENDING_CASES]


@needs_tracker
def test_pending_read_masks_on_synthetic_prefixes(vocab):
    from model.intmask import IntMask
    m = IntMask(vocab)
    for words, tok, want in PENDING_CASES:
        m.reset()
        for t in [BOS, *word_ids(vocab, *words)]:
            m.step(t)
        assert bool(m.banned()[word_ids(vocab, tok)[0]]) == want, (words, tok)
    checked, stats = _compare_on_prefixes(vocab, pending_programs(vocab), 64)
    assert checked > 40 and stats["pending"] >= 4         # lab/masks.py banned all four too


@needs_tracker
@pytest.mark.skipif(not VAL.exists(), reason="data/tok/val.bin not present")
def test_masks_match_lab_masks_on_val_program_prefixes(vocab):
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)
    ends = np.append(starts[1:], len(stream))
    programs = [stream[a:b].tolist() for a, b in zip(starts[::10], ends[::10])]   # 67 programs
    checked, stats = _compare_on_prefixes(vocab, programs + pending_programs(vocab), 400)
    assert checked > 15_000
    for rule in ("end", "else", "var", "pending", "list_start", "list_name"):
        assert stats[rule] > 0, rule                   # every rule decided something


@needs_tracker
@pytest.mark.skipif(not S0_SAMPLES.exists(), reason="S0-s1 samples not present")
def test_masks_match_lab_masks_on_generated_programs(vocab):
    """Masked samples of G2a's S0 seed 1 (sliding window, 1024-token cap)."""
    rows = [json.loads(line)["ids"] for line in S0_SAMPLES.read_text().splitlines()[:40]]
    checked, stats = _compare_on_prefixes(vocab, rows, 1024)
    assert checked > 20_000 and stats["pending"] > 0


def test_masked_choose_matches_the_lab_sampler_bit_for_bit(rel):
    from lab import sample as S
    _, im = rel
    rng = np.random.default_rng(11)
    logits = rng.integers(-3000, 3000, size=(64, 24)).astype(np.int64)
    logits[:4, 4:9] = 1234
    banned = rng.random((64, 24)) < 0.6
    banned[:8, :] = True                           # rows with fewer allowed ids than top-k
    for r in range(8):
        banned[r, rng.choice(np.arange(4, 24), size=r + 1, replace=False)] = False
    banned[[0, 5], EOS] = False
    seeds = np.arange(1, 65, dtype=np.int64)
    for tq8, k, greedy in [(205, 20, False), (33, 64, False), (65535, 5, False), (256, 1, False),
                           (256, 20, True)]:
        want = [choose(im, logits[i], SamplerConfig(tq8, k, greedy, int(seeds[i])), int(seeds[i]),
                       banned[i]) for i in range(64)]
        tok, state = S.choose(torch.as_tensor(logits), torch.as_tensor(seeds), tq8, k, greedy,
                              torch.as_tensor(banned))
        assert tok.tolist() == [w[0] for w in want]
        assert state.tolist() == [w[1] for w in want]
        assert all(not banned[i, w[0]] for i, w in enumerate(want))


class _IntrefStepper:
    """intref behind lab.sample.QATDecoder's interface (ring cache included)."""

    def __init__(self, im):
        self.im, self.cfg, self.device = im, im.cfg, "cpu"
        self.cfg = type("cfg", (), {"max_seq_len": im.cfg.max_seq_len, "pos": im.pos_scheme,
                                    "feat_rows": 0})

    def new_cache(self, b):
        return [self.im.new_cache() for _ in range(b)]

    def step(self, tok, pos, cache):
        rows = [self.im.step(int(t), pos, c) for t, c in zip(tok.tolist(), cache)]
        return torch.as_tensor(np.stack(rows), dtype=torch.float64) / 256.0


def _masked_loop_matches_lab(im, vocab, seeds, tq8, k, greedy, cap):
    from lab import sample as S
    from model.intmask import IntMask
    lab_mask = masks_lab.TrackerMask(VOCAB)
    got = S.generate(_IntrefStepper(im), seeds, tq8, k, greedy, cap - 1, tracker=lab_mask.rows,
                     mask=lab_mask, cap=cap)
    want = [generate(im, [], cap - 1, SamplerConfig(tq8, k, greedy, s), cap=cap,
                     mask=IntMask(vocab)) for s in seeds]
    assert got == want
    return want


@needs_tracker
def test_masked_generation_loop_matches_the_lab_sampler(vocab, tmp_path):
    """An untrained vocab-781 ALiBi model writes near-random token soup, which the masks act on
    at many positions; lab.sample.generate fed intref's logits must give the same tokens."""
    cfg = ModelConfig(vocab_size=781, dim=20, n_layers=1, n_heads=2, hidden_dim=32,
                      max_seq_len=16, pos="alibi")
    torch.manual_seed(3)
    m = Transformer(cfg)
    calib = np.random.default_rng(3).integers(4, 781, size=(4, 16))
    m.calibrate(torch.from_numpy(calib))
    export_model(m.eval(), calib, tmp_path, vocab=vocab)
    im = IntModel.load(tmp_path)
    out = _masked_loop_matches_lab(im, vocab, [1, 2, 3, 0, 2**32 - 1], 512, 64, False, 70)
    assert max(len(o) for o in out) == 70 > cfg.max_seq_len
    _masked_loop_matches_lab(im, vocab, [1], 256, 1, True, 40)


@needs_tracker
@pytest.mark.slow
@pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(), reason="runs/v2/export not present")
def test_masked_generation_of_the_v2_export_matches_the_lab_sampler(vocab):
    im = IntModel.load(V2_EXPORT)
    out = _masked_loop_matches_lab(im, vocab, [1, 2], 205, 20, False, 300)
    assert len(out[0]) > 256 or out[0][-1] == EOS


def test_mask_bans_are_respected_by_generate(rel):
    """A mask object is any reset/step/banned trio; banned ids are never generated."""
    _, im = rel

    class EvenOnly:
        def reset(self):
            self.fed = []

        def step(self, tok):
            self.fed.append(tok)

        def banned(self):
            b = np.zeros(24, dtype=bool)
            b[4::2] = True
            return b

    mk = EvenOnly()
    out = generate(im, [], 100, SamplerConfig(1024, 64, False, 1), cap=30, mask=mk)
    assert all(t % 2 == 1 for t in out[1:] if t != EOS)
    assert mk.fed == out[:len(mk.fed)] and len(mk.fed) >= len(out) - 1
