"""lab.sample: batched KV-cached sampler for a QAT checkpoint, with intref's sampler semantics."""
import json
import time

import numpy as np
import pytest
import torch

from lab import sample as S
from lab.tests.conftest import DATA, V1_CKPT, V1_EXPORT, have_cuda, have_v1, pattern_batch
from model.data import bos_windows, load_split
from model.intref import IntModel, SamplerConfig, choose, generate
from model.model import load_checkpoint
from model.tokens import BOS, EOS, PAD, UNK

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---- sampler semantics: identical to intref.choose for identical integer logits --------------

@pytest.mark.parametrize("tq8, top_k", [(205, 20), (256, 1), (33, 64), (65535, 7), (512, 40)])
def test_choose_matches_intref_choose_bit_for_bit(tiny_run, tq8, top_k):
    _, export, _ = tiny_run
    im = IntModel.load(export)
    rng = np.random.default_rng(tq8 + top_k)
    logits = rng.integers(-3000, 3000, size=(64, 24)).astype(np.int64)
    logits[:5, 4:9] = 1234                      # ties: the smaller id ranks first
    logits[5, :] = 0                            # all equal
    seeds = np.arange(1, 65, dtype=np.int64)
    seeds[0] = 2463534242
    seeds[1] = 2**32 - 1
    want = [choose(im, logits[i], SamplerConfig(tq8, top_k, False, int(seeds[i])), int(seeds[i]))
            for i in range(64)]
    tok, state = S.choose(torch.as_tensor(logits), torch.as_tensor(seeds), tq8, top_k)
    assert tok.tolist() == [w[0] for w in want]
    assert state.tolist() == [w[1] for w in want]


def test_greedy_choose_matches_intref_and_keeps_state(tiny_run):
    _, export, _ = tiny_run
    im = IntModel.load(export)
    logits = np.random.default_rng(3).integers(-500, 500, size=(16, 24)).astype(np.int64)
    logits[:, [PAD, BOS, UNK]] = 10_000         # banned ids never win
    seeds = torch.arange(1, 17, dtype=torch.int64)
    tok, state = S.choose(torch.as_tensor(logits), seeds, 256, 20, greedy=True)
    assert tok.tolist() == [choose(im, row, SamplerConfig(greedy=True), 1)[0] for row in logits]
    assert state.tolist() == seeds.tolist()


def test_logits_to_q8_rounds_half_up():
    x = torch.tensor([0.0, 1.0, -1.0, 0.5 / 256, -0.5 / 256, 2.3])
    assert S.logits_q8(x).tolist() == [0, 256, -256, 1, 0, 589]


# ---- the cached decoder reproduces the QAT model ---------------------------------------------

def test_cached_decoder_matches_full_sequence_forward(tiny_run):
    _, _, model = tiny_run
    dec = S.QATDecoder(model, "cpu")
    x = torch.from_numpy(pattern_batch(np.random.default_rng(5), 6, 32, 24)[:, :-1])
    with torch.no_grad():
        want = model(x)[0]
    got = dec.sequence_logits(x)
    assert got.shape == want.shape
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.99
    assert (got - want).abs().mean() < 1e-3


class IntrefStepper:
    """The integer reference behind QATDecoder's interface: generate() driven by intref's own
    logits must reproduce intref.generate token for token."""

    def __init__(self, im):
        self.im, self.cfg, self.device = im, im.cfg, "cpu"

    def new_cache(self, b):
        return [self.im.new_cache() for _ in range(b)]

    def step(self, tok, pos, cache):
        rows = [self.im.step(int(t), pos, c) for t, c in zip(tok.tolist(), cache)]
        return torch.as_tensor(np.stack(rows), dtype=torch.float64) / 256.0


@pytest.mark.parametrize("tq8, top_k, max_new", [(205, 8, 31), (256, 20, 10), (33, 3, 31),
                                                 (1024, 64, 31)])
def test_generation_loop_reproduces_intref_given_intref_logits(tiny_run, tq8, top_k, max_new):
    _, export, _ = tiny_run
    im = IntModel.load(export)
    seeds = [0, 1, 2, 7, 42, 2**32 - 1, *range(100, 110)]
    got = S.generate(IntrefStepper(im), seeds, tq8, top_k, False, max_new)
    want = [generate(im, [], max_new, SamplerConfig(tq8, top_k, False, s)) for s in seeds]
    assert got == want


class FakeStepper:
    """Row r makes <eos> the top token at position r + 1, token 5 otherwise; rows >= 8 never end."""

    class cfg:
        max_seq_len = 10

    device = "cpu"

    def new_cache(self, b):
        return None

    def step(self, tok, pos, cache):
        logits = torch.zeros(len(tok), 12)
        logits[:, 5] = 10.0
        rows = torch.arange(len(tok))
        logits[rows == pos - 1, EOS] = 20.0
        return logits


def test_generation_stops_each_row_at_eos_or_the_context():
    out = S.generate(FakeStepper(), list(range(1, 11)), 256, 1, True, 255)
    assert [len(o) for o in out] == [3, 4, 5, 6, 7, 8, 9, 10, 10, 10]
    assert out[0] == [BOS, 5, EOS] and out[9] == [BOS] + [5] * 9
    capped = S.generate(FakeStepper(), list(range(1, 11)), 256, 1, True, 4)   # max_new = 4
    assert [len(o) for o in capped] == [3, 4, 5, 5, 5, 5, 5, 5, 5, 5]


def test_why_coupled_samples_diverge():
    """intref draws target = state % total: weights that differ by one unit in their total
    give unrelated targets, so QAT and intref samples with one seed decorrelate at the first
    logit difference (only their distributions can agree)."""
    state = S.xorshift32(torch.tensor([3]))
    assert int(state % 202_362) != int(state % 203_011)


def test_sample_output_contract(tiny_run):
    ckpt, _, _ = tiny_run
    out = S.sample(ckpt, n=12, temperature=1.0, top_k=20, device="cpu", batch=5)
    assert len(out) == 12
    for ids in out:
        assert ids[0] == BOS and 1 < len(ids) <= 32
        assert not {PAD, BOS, UNK} & set(ids[1:])
        assert ids[-1] == EOS or len(ids) == 32
        assert EOS not in ids[:-1]
    # seeds 1..n by default and deterministic
    assert out == S.sample(ckpt, seeds=range(1, 13), temperature=1.0, top_k=20, device="cpu")


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seeds_outside_32_bits_are_rejected_like_intref(tiny_run, seed):
    ckpt, _, _ = tiny_run
    with pytest.raises(ValueError):
        S.sample(ckpt, seeds=[seed], device="cpu")


def test_max_new_limits_generated_tokens(tiny_run):
    ckpt, _, _ = tiny_run
    out = S.sample(ckpt, n=4, max_new=5, device="cpu")
    assert all(len(ids) <= 6 for ids in out)


# ---- the shipping model --------------------------------------------------------------------

@have_v1
@pytest.mark.skipif(not (DATA / "val.bin").exists(), reason="data/tok/val.bin not present")
def test_v1_top1_agreement_with_intref_on_fixed_contexts():
    xs, _ = bos_windows(load_split(DATA / "val.bin"), 256)
    contexts = [row[:96] for row in xs[:8]]
    rep = S.intref_agreement(load_checkpoint(V1_CKPT), V1_EXPORT, contexts, device=DEV, workers=8)
    assert rep["n_positions"] == 8 * 96
    assert rep["top1_agree"] > 0.96, rep


@have_v1
@have_cuda
def test_v1_thousand_samples_well_under_a_minute():
    t0 = time.time()
    out = S.sample(V1_CKPT, n=1000, temperature=0.8, top_k=20, device="cuda")
    elapsed = time.time() - t0
    assert len(out) == 1000 and elapsed < 30, elapsed
    assert all(o[0] == BOS and len(o) <= 256 for o in out)


@have_v1
@have_cuda
def test_v1_samples_match_the_intref_generations_of_record_in_distribution():
    """runs/gen/v1.jsonl holds the intref samples behind RESULTS.md (seeds 1..1000, T 0.8,
    top-k 20). Same seeds, GPU sampler: the <eos> rate and mean length agree within noise."""
    path = V1_CKPT.parents[1] / "gen" / "v1.jsonl"
    if not path.exists():
        pytest.skip("runs/gen/v1.jsonl not present")
    rec = [json.loads(line) for line in path.read_text().splitlines()]
    got = S.sample(V1_CKPT, seeds=[r["seed"] for r in rec], temperature=0.8, top_k=20,
                   device="cuda")
    eos = lambda seqs: np.mean([EOS in s for s in seqs])  # noqa: E731
    want = [r["ids"] for r in rec]
    assert abs(eos(got) - eos(want)) < 0.03, (eos(got), eos(want))
    assert abs(np.mean([len(s) for s in got]) / np.mean([len(s) for s in want]) - 1) < 0.02


# ---- v2: masks in the sampler ------------------------------------------------------------------

def _choose_reference(lq8, state, tq8, top_k, banned):
    """INTREF.md section 5 with masked ids excluded from the ranking like <pad>/<bos>/<unk>."""
    expw = S._EXPW.tolist()
    allowed = [i for i in range(len(lq8)) if i not in S.BANNED and not banned[i]]
    ranked = sorted(allowed, key=lambda i: (-lq8[i], i))[:top_k]
    inv_t = (32 << 16) // tq8
    w = [expw[min((min(lq8[ranked[0]] - lq8[c], (1 << 15) - 1) * inv_t + (1 << 15)) >> 16,
                  len(expw) - 1)] for c in ranked]
    state = int(S.xorshift32(torch.tensor([state]))[0])
    target, acc = state % sum(w), 0
    for c, wi in zip(ranked, w):
        acc += wi
        if acc > target:
            return c, state


@pytest.mark.parametrize("tq8, top_k", [(205, 20), (256, 20), (33, 5), (65535, 64)])
def test_masked_ids_are_excluded_from_the_ranking(tq8, top_k):
    g = np.random.default_rng(top_k)
    lq8 = g.integers(-3000, 3000, size=(40, 30)).astype(np.int64)
    banned = g.random((40, 30)) < 0.3
    banned[:5, :] = True                       # rows with fewer allowed ids than top_k
    banned[:5, [2, 7, 9]] = False
    banned[5:8, :] = False                     # unmasked rows behave as before
    banned[8, np.argsort(-lq8[8])[:3]] = True  # the best ids are masked
    seeds = np.arange(1, 41, dtype=np.int64)
    tok, state = S.choose(torch.as_tensor(lq8), torch.as_tensor(seeds), tq8, top_k,
                          banned=torch.as_tensor(banned))
    want = [_choose_reference(lq8[i].tolist(), int(seeds[i]), tq8, top_k, banned[i]) for i in range(40)]
    assert tok.tolist() == [w[0] for w in want] and state.tolist() == [w[1] for w in want]
    plain = S.choose(torch.as_tensor(lq8[5:8]), torch.as_tensor(seeds[5:8]), tq8, top_k)
    assert tok[5:8].tolist() == plain[0].tolist()


def test_greedy_takes_the_best_allowed_id():
    lq8 = torch.tensor([[0, 0, 0, 0, 90, 80, 70]])
    banned = torch.tensor([[False] * 4 + [True, False, False]])
    assert S.choose(lq8, torch.tensor([1]), 256, 20, greedy=True, banned=banned)[0].tolist() == [5]


def test_a_row_with_nothing_allowed_is_an_error():
    lq8 = torch.zeros(2, 8, dtype=torch.int64)
    banned = torch.zeros(2, 8, dtype=torch.bool)
    banned[1, :] = True
    with pytest.raises(ValueError):
        S.choose(lq8, torch.tensor([1, 2]), 256, 20, banned=banned)


class BanMask:
    """Bans token 5 everywhere and every even id >= 4 in odd rows (not <eos>)."""

    def __init__(self, v):
        self.v, self.calls = v, 0

    def banned(self, tracker):
        self.calls += 1
        b = np.zeros((tracker.batch, self.v), dtype=bool)
        b[:, 5] = True
        b[1::2, 4::2] = True
        return b


def test_masked_tokens_are_never_sampled(tiny_run):
    ckpt, _, model = tiny_run
    seeds = list(range(1, 17))
    mask = BanMask(model.cfg.vocab_size)
    out = S.sample(ckpt, seeds=seeds, temperature=1.0, device="cpu", mask=mask,
                   tracker=S.BatchTracker(lambda: Toy()))
    assert mask.calls > 0
    for r, ids in enumerate(out):
        assert 5 not in ids[1:]
        if r % 2:
            assert not [t for t in ids[1:] if t >= 4 and t % 2 == 0]
    plain = S.sample(ckpt, seeds=seeds, temperature=1.0, device="cpu")
    assert out != plain


def test_a_mask_that_bans_nothing_changes_nothing(tiny_run):
    ckpt, _, model = tiny_run

    class Nothing:
        def banned(self, tracker):
            return np.zeros((tracker.batch, model.cfg.vocab_size), dtype=bool)

    kw = dict(seeds=range(1, 13), temperature=0.8, top_k=20, device="cpu")
    assert S.sample(ckpt, mask=Nothing(), tracker=S.BatchTracker(Toy), **kw) == S.sample(ckpt, **kw)


# ---- v2: feature inputs in the sampler ---------------------------------------------------------

class Toy:
    """A step-function tracker: cats = (tokens since <bos> capped at 7, last token % 3),
    bag = bits of the last token id."""

    def reset(self):
        self.n = 0

    def step(self, tok):
        self.n = 0 if tok == BOS else min(self.n + 1, 7)
        return (self.n, tok % 3), tok


@pytest.fixture(scope="module")
def feat_run(tmp_path_factory):
    """A tiny feature model trained on patterns with Toy features; (ckpt, model)."""
    from model.data import stream_features
    from model.model import ModelConfig, Transformer
    from lab.tests.conftest import TINY
    cfg = ModelConfig(**{**TINY.to_dict(), "feat_cats": (8, 3), "feat_bag": 5})
    torch.manual_seed(0)
    g = np.random.default_rng(0)
    model = Transformer(cfg)

    def batch(n):
        seq = pattern_batch(g, n, 32, 24)
        f = [stream_features(row, Toy()) for row in seq[:, :-1]]
        cats = torch.as_tensor(np.stack([c for c, _ in f]).astype(np.int64))
        bag = torch.as_tensor(np.stack([b for _, b in f]).astype(np.int64))
        return torch.as_tensor(seq), cats, bag

    seq, cats, bag = batch(8)
    model.calibrate(seq[:, :-1], cats, bag)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(200):
        seq, cats, bag = batch(16)
        _, loss = model(seq[:, :-1], seq[:, 1:], cats, bag)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    ckpt = tmp_path_factory.mktemp("feat") / "best.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 200,
                "val_loss": float(loss.detach()), "args": {}}, ckpt)
    return ckpt, model


def _features_of(seqs):
    from model.data import stream_features
    f = [stream_features(np.asarray(s), Toy()) for s in seqs]
    return (torch.as_tensor(np.stack([c for c, _ in f]).astype(np.int64)),
            torch.as_tensor(np.stack([b for _, b in f]).astype(np.int64)))


def test_cached_decoder_with_features_matches_the_full_forward(feat_run):
    _, model = feat_run
    x = torch.from_numpy(pattern_batch(np.random.default_rng(5), 6, 32, 24)[:, :-1])
    cats, bag = _features_of(x.numpy())
    with torch.no_grad():
        want = model(x, cats=cats, bag=bag)[0]
    got = S.QATDecoder(model, "cpu").sequence_logits(x, cats, bag)
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.99
    assert (got - want).abs().mean() < 1e-3


class Recorder:
    """Wraps a decoder and records the features fed at each position."""

    def __init__(self, dec):
        self.dec, self.cfg, self.device, self.seen = dec, dec.cfg, dec.device, {}

    def new_cache(self, b):
        return self.dec.new_cache(b)

    def step(self, tok, pos, cache, cats=None, bag=None):
        self.seen[pos] = (cats.clone(), bag.clone())
        return self.dec.step(tok, pos, cache, cats, bag)


def test_sampled_programs_get_the_features_they_would_get_in_training(feat_run):
    _, model = feat_run
    rec = Recorder(S.QATDecoder(model, "cpu"))
    out = S.generate(rec, list(range(1, 9)), 256, 20, False, 31, tracker=S.BatchTracker(Toy))
    for r, ids in enumerate(out):
        cats, bag = _features_of([ids])        # what training would compute for this program
        for pos in range(min(len(ids), 31)):      # the token at the last position is not fed
            assert rec.seen[pos][0][r].tolist() == cats[0, pos].tolist()
            assert int(rec.seen[pos][1][r]) == int(bag[0, pos])


def test_shuffled_feature_models_get_another_rows_features_at_sampling(feat_run):
    from model.model import ModelConfig, Transformer
    _, model = feat_run
    shuffled = Transformer(ModelConfig(**{**model.cfg.to_dict(), "feat_shuffle": True}))
    shuffled.load_state_dict(model.state_dict())
    rec = Recorder(S.QATDecoder(shuffled.eval(), "cpu"))
    out = S.generate(rec, list(range(1, 9)), 256, 20, False, 6, tracker=S.BatchTracker(Toy))
    pos = 5
    own = [_features_of([ids[: pos + 1]])[1][0, pos] for ids in out]
    # rolled by one row: a row's own state would need 8 > 6 steps to come back to it
    assert [int(rec.seen[pos][1][r]) for r in range(8)] == [int(own[(r - 1) % 8]) for r in range(8)]
    with pytest.raises(ValueError):
        S.generate(rec, list(range(1, 9)), 256, 20, False, 8, tracker=S.BatchTracker(Toy))


def test_a_feature_model_needs_a_tracker(feat_run):
    ckpt, _ = feat_run
    with pytest.raises(ValueError):
        S.sample(ckpt, n=2, device="cpu")
    out = S.sample(ckpt, n=4, device="cpu", tracker=S.BatchTracker(Toy))
    assert all(o[0] == BOS for o in out)


# ---- G2a: ring K/V cache, generation past the window -------------------------------------------

def _relative(pos, feat=False, shuffle=False, train_steps=150):
    from model.model import ModelConfig, Transformer
    from lab.tests.conftest import TINY
    cfg = ModelConfig(**{**TINY.to_dict(), "pos": pos, "max_seq_len": 16,
                         **({"feat_cats": (8, 3), "feat_bag": 5, "feat_shuffle": shuffle}
                            if feat else {})})
    torch.manual_seed(0)
    g = np.random.default_rng(0)
    m = Transformer(cfg)
    seq = torch.from_numpy(pattern_batch(g, 8, 16, 24))
    f = _features_of(seq[:, :-1].numpy()) if feat else ()
    m.calibrate(seq[:, :-1], *f)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(train_steps):
        seq = torch.from_numpy(pattern_batch(g, 16, 16, 24))
        f = _features_of(seq[:, :-1].numpy()) if feat else ()
        _, loss = m(seq[:, :-1], seq[:, 1:], *f)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_ring_cache_decoder_matches_the_banded_forward_past_the_window(pos):
    m = _relative(pos)
    x = torch.from_numpy(pattern_batch(np.random.default_rng(7), 4, 70, 24)[:, :-1])
    with torch.no_grad():
        want = m(x)[0]
    got = S.QATDecoder(m, "cpu").sequence_logits(x)
    assert got.shape == want.shape == (4, 70, 24)
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.99
    assert (got - want).abs().mean() < 1e-3


def test_learned_positions_cannot_generate_past_their_table(tiny_run):
    ckpt, _, _ = tiny_run
    with pytest.raises(ValueError):
        S.sample(ckpt, n=2, device="cpu", cap=40)


def test_generation_stops_at_eos_or_at_the_cap():
    class Fake(FakeStepper):
        class cfg:
            max_seq_len, pos, feat_rows = 10, "nope", 0

    out = S.generate(Fake(), list(range(1, 11)), 256, 1, True, 10_000, cap=25)
    assert [len(o) for o in out] == list(range(3, 13))       # past the 10-token window
    out = S.generate(Fake(), list(range(1, 11)), 256, 1, True, 10_000, cap=11)
    assert [len(o) for o in out] == [3, 4, 5, 6, 7, 8, 9, 10, 11, 11]


def test_relative_models_sample_past_the_window_with_features_from_the_program_start():
    m = _relative("alibi", feat=True, train_steps=0)
    rec = Recorder(S.QATDecoder(m, "cpu"))
    out = S.generate(rec, list(range(1, 9)), 256, 20, False, 47, tracker=S.BatchTracker(Toy), cap=48)
    assert max(len(o) for o in out) == 48 > m.cfg.max_seq_len
    for r, ids in enumerate(out):
        cats, bag = _features_of([ids])
        for pos in range(min(len(ids), 47)):
            assert int(rec.seen[pos][1][r]) == int(bag[0, pos])


def test_shuffled_models_pad_the_batch_so_no_row_feeds_itself():
    m = _relative("nope", feat=True, shuffle=True, train_steps=0)
    out = S.sample(m, n=6, device="cpu", tracker=S.BatchTracker(Toy), cap=40)
    assert len(out) == 6 and all(o[0] == BOS and len(o) <= 40 for o in out)


class FakeFeatDecoder:
    """Row 0 ends at position 2; records the tokens and features fed at each position."""

    def __init__(self, shuffle):
        self.cfg = type("cfg", (), {"max_seq_len": 8, "pos": "alibi", "feat_rows": 16,
                                    "feat_cats": (8, 3), "feat_bag": 5, "feat_shuffle": shuffle})
        self.device, self.fed = "cpu", {}

    def new_cache(self, b):
        return None

    def step(self, tok, pos, cache, cats=None, bag=None):
        self.fed[pos] = (tok.clone(), cats.clone())
        logits = torch.zeros(len(tok), 12)
        logits[:, 5] = 10.0
        if pos == 1:
            logits[0, EOS] = 20.0
        return logits


def test_finished_rows_of_the_shuffled_control_start_a_new_program():
    """S2 (shuffled, sliding window): after <eos> a row is fed <bos> and writes a new program, as
    in the packed training stream, so its neighbour never reads a finished program's state."""
    dec = FakeFeatDecoder(shuffle=True)
    out = S.generate(dec, list(range(1, 7)), 256, 1, True, 5, tracker=S.BatchTracker(Toy), cap=6)
    assert out[0] == [BOS, 5, EOS]                          # the sample itself still ends
    assert int(dec.fed[2][0][0]) == EOS and int(dec.fed[3][0][0]) == BOS
    assert int(dec.fed[3][1][1, 0]) == 0                    # row 1 reads row 0's reset tracker
    plain = FakeFeatDecoder(shuffle=False)                   # other models: unchanged
    S.generate(plain, list(range(1, 7)), 256, 1, True, 5, tracker=S.BatchTracker(Toy), cap=6)
    assert int(plain.fed[3][0][0]) == 5


@pytest.mark.parametrize("pos", ["alibi"])
def test_ring_cache_matches_with_the_lut_max_cap_active(pos):
    """6 heads and a short exp table: distance penalties hit lut_max for most heads."""
    from model.model import ModelConfig, Transformer
    cfg = ModelConfig(vocab_size=24, dim=48, n_layers=2, n_heads=6, hidden_dim=96, max_seq_len=16,
                      lut_max=64, pos=pos)
    torch.manual_seed(0)
    m = Transformer(cfg)
    x = torch.from_numpy(pattern_batch(np.random.default_rng(3), 3, 60, 24)[:, :-1])
    m.calibrate(x[:, :16])
    m.eval()
    with torch.no_grad():
        want = m(x)[0]
    got = S.QATDecoder(m, "cpu").sequence_logits(x)
    assert (got - want).abs().max() < 1e-3
