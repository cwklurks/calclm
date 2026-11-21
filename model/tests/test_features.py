"""Extra per-token input features (v2 tracker inputs): config, embedding, data alignment, shuffle.

Contract: the features at stream position i describe the program state after token i; the model
adds their embedding rows to token i's input, which predicts token i+1.
"""
import hashlib
import json

import numpy as np
import pytest
import torch

from model import data as D
from model.model import ModelConfig, Transformer, load_checkpoint
from model.tokens import BOS, EOS

TINY = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, n_kv_heads=1,
                   hidden_dim=40, max_seq_len=16)
FEAT = ModelConfig(**{**TINY.to_dict(), "feat_cats": (3, 4), "feat_bag": 5})


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = Transformer(cfg)
    b = 2
    m.calibrate(torch.randint(4, cfg.vocab_size, (b, cfg.max_seq_len)), *_feats(cfg, b, seed))
    return m


def _feats(cfg, b, seed=0, t=None):
    if not cfg.feat_rows:
        return ()
    g = torch.Generator().manual_seed(seed)
    t = t or cfg.max_seq_len
    cats = torch.stack([torch.randint(0, n, (b, t), generator=g) for n in cfg.feat_cats], -1)
    bag = torch.randint(0, 1 << min(cfg.feat_bag, 62), (b, t), generator=g) if cfg.feat_bag else None
    return cats, bag


# ---- config ---------------------------------------------------------------------------------

def test_default_config_has_no_features_and_serializes_like_v1():
    assert TINY.feat_cats == () and TINY.feat_bag == 0 and not TINY.feat_shuffle
    assert TINY.feat_rows == 0
    assert not any(k.startswith("feat_") for k in TINY.to_dict())   # v1 config dicts unchanged


def test_feature_config_roundtrips_through_json():
    d = json.loads(json.dumps(FEAT.to_dict()))
    assert ModelConfig(**d) == FEAT and ModelConfig(**d).feat_cats == (3, 4)
    assert FEAT.feat_rows == 3 + 4 + 5 and FEAT.feat_offsets == (0, 3)


@pytest.mark.parametrize("bad", [{"feat_bag": 65}, {"feat_cats": (0,)}, {"feat_cats": (257,)},
                                 {"feat_shuffle": True}])
def test_invalid_feature_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        ModelConfig(**{**TINY.to_dict(), **bad})


# ---- backward compatibility ------------------------------------------------------------------

def test_a_model_without_features_is_bit_identical_to_v1_code():
    """Golden hash recorded with the code before feature inputs existed."""
    torch.manual_seed(0)
    m = Transformer(TINY)
    m.calibrate(torch.randint(4, 40, (2, 16)))
    m.eval()
    idx = torch.arange(16).remainder(36).add(4).view(1, 16)
    with torch.no_grad():
        lg = m(idx)[0]
    assert hashlib.sha256(lg.numpy().tobytes()).hexdigest() == \
        "baedc658881a34a96712e32b59de81cf98f2ca9c2dd46f83dae3eca6b23f5587"
    assert not [k for k in m.state_dict() if "feat" in k]


def test_v1_checkpoint_loads_and_gives_the_same_logits():
    from lab.tests.conftest import DATA, V1_CKPT
    if not (V1_CKPT.exists() and (DATA / "val.bin").exists()):
        pytest.skip("runs/v1 or data/tok not present")
    v1 = load_checkpoint(V1_CKPT)
    x = torch.from_numpy(D.load_split(DATA / "val.bin")[:64].astype(np.int64)).view(1, 64)
    with torch.no_grad():
        lg = v1(x)[0]
    assert hashlib.sha256(lg.numpy().tobytes()).hexdigest() == \
        "dfa2fdc370f78d3a4162afb90ce168a9d9f6305e2bd4b0139db86162f1ab95af"


def test_feature_tables_do_not_change_the_initialization_of_shared_weights():
    """Same seed, with and without features: identical starting weights except the new table,
    so arms with and without features differ only by the features."""
    torch.manual_seed(3)
    a = Transformer(TINY).state_dict()
    torch.manual_seed(3)
    b = Transformer(FEAT).state_dict()
    assert set(b) - set(a) == {"feat_emb.weight"}
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---- the embedding --------------------------------------------------------------------------

def test_embedding_adds_one_row_per_categorical_feature_and_one_per_set_bag_bit():
    m = _model(FEAT)
    idx = torch.randint(4, 40, (2, 16))
    cats, bag = _feats(FEAT, 2, seed=5)
    w8, s = m.feat_emb.quantized()
    rows = w8 * s
    t8, ts = m.tok_emb.quantized()
    p8, ps = m.pos_emb.quantized()
    want = (t8 * ts)[idx] + (p8 * ps)[:16]
    want = want + rows[cats[..., 0]] + rows[3 + cats[..., 1]]
    for b in range(5):
        want = want + ((bag >> b) & 1)[..., None] * rows[7 + b]
    with torch.no_grad():
        got = m.embed(idx, cats, bag)
    assert torch.allclose(got, want, atol=1e-6)


def test_bag_bit_63_is_read_from_a_negative_int64():
    cfg = ModelConfig(**{**TINY.to_dict(), "feat_cats": (2,), "feat_bag": 64})
    m = _model(cfg)
    idx = torch.full((1, 1), 5)
    cats = torch.zeros(1, 1, 1, dtype=torch.long)
    with torch.no_grad():
        top = m.embed(idx, cats, torch.tensor([[-(1 << 63)]]))     # only bit 63 set
        none = m.embed(idx, cats, torch.zeros(1, 1, dtype=torch.long))
    w8, s = m.feat_emb.quantized()
    assert torch.allclose(top - none, (w8 * s)[2 + 63], atol=1e-6)


def test_features_must_match_the_config():
    m, plain = _model(FEAT), _model(TINY)
    idx = torch.randint(4, 40, (2, 16))
    with pytest.raises(ValueError):
        m(idx)
    with pytest.raises(ValueError):
        plain(idx, cats=torch.zeros(2, 16, 2, dtype=torch.long))


def test_causal_with_features():
    m = _model(FEAT).eval()
    idx = torch.randint(4, 40, (1, 16))
    cats, bag = _feats(FEAT, 1, seed=1)
    cats2 = cats.clone()
    cats2[0, 10, 1] = (cats[0, 10, 1] + 1) % 4
    with torch.no_grad():
        a, b = m(idx, cats=cats, bag=bag)[0], m(idx, cats=cats2, bag=bag)[0]
    assert torch.equal(a[:, :10], b[:, :10]) and not torch.equal(a[:, 10:], b[:, 10:])


def test_gradients_reach_the_feature_table():
    m = _model(FEAT)
    idx = torch.randint(4, 40, (2, 16))
    _, loss = m(idx, idx, *_feats(FEAT, 2))
    loss.backward()
    assert m.feat_emb.weight.grad.abs().sum() > 0


def test_a_feature_at_position_i_informs_the_prediction_of_token_i_plus_1():
    """Tokens are random; the next token is 4 + the categorical feature of the current one.
    Only a model that reads features at the aligned position can learn it."""
    cfg = ModelConfig(**{**TINY.to_dict(), "feat_cats": (3,)})
    g = torch.Generator().manual_seed(0)

    def batch():
        c = torch.randint(0, 3, (16, 17), generator=g)
        x = torch.cat([torch.full((16, 1), BOS), 4 + c[:, :-1]], 1)
        return x[:, :16], 4 + c[:, :16], c[:, :16, None]

    losses = {}
    for use in (True, False):
        m = _model(cfg if use else TINY)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for _ in range(150):
            x, y, c = batch()
            _, loss = m(x, y, *((c, None) if use else ()))
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses[use] = loss.item()
    assert losses[True] < 0.3 < 0.8 < losses[False], losses    # ln 3 = 1.10 without features


# ---- data: per-token feature arrays aligned with the stream -----------------------------------

def _stream(lengths):
    out, nxt = [], 10
    for n in lengths:
        out += [BOS, *range(nxt, nxt + n), EOS]
        nxt += n
    return np.array(out, dtype=np.uint16)


def _indexed_feats(stream):
    """cats[:, 0] = position % 251, bag = position: identifies the stream position of a row."""
    n = len(stream)
    cats = np.stack([np.arange(n) % 251, np.arange(n) % 7], 1).astype(np.uint8)
    return cats, np.arange(n, dtype=np.uint64)


def test_feature_files_roundtrip_and_are_checked_against_the_stream(tmp_path):
    stream = _stream([5, 300, 9])
    cats, bag = _indexed_feats(stream)
    D.write_features(tmp_path, "val", cats, bag, {"cats": [251, 7], "bag": 64, "source": "test"})
    meta = D.feature_meta(tmp_path)
    assert meta["cats"] == [251, 7] and meta["bag"] == 64
    got = D.load_features(tmp_path, "val", n=len(stream))
    assert np.array_equal(got[0], cats) and np.array_equal(got[1], bag)
    with pytest.raises(ValueError):
        D.load_features(tmp_path, "val", n=len(stream) + 1)


def test_training_batches_draw_the_same_windows_as_before_and_gather_aligned_features():
    stream = _stream([50, 400, 30, 700])
    cats, bag = _indexed_feats(stream)
    x0, y0 = D.sample_batch(stream, 32, 64, np.random.default_rng(4), p_bos=0.5)
    rng = np.random.default_rng(4)
    offs = D.sample_offsets(stream, 32, 64, rng, p_bos=0.5)
    x, y = D.windows(stream, offs, 64)
    c, b = D.feature_windows((cats, bag), offs, 64)
    assert np.array_equal(x, x0) and np.array_equal(y, y0)       # v1 batches unchanged
    assert c.shape == (32, 64, 2) and b.shape == (32, 64) and c.dtype == b.dtype == np.int64
    assert np.array_equal(x, stream[b].astype(np.int64))           # features of input token i
    assert np.array_equal(c[..., 0], b % 251)
    assert (x[:, 0] != BOS).any()                                    # some windows start mid-program


def test_precomputed_features_carry_state_from_the_program_start():
    """Features come from running a step function over the whole stream (reset at <bos>), so a
    window that starts mid-program sees the state built before the window."""
    stream = _stream([50, 400, 30])

    class Counter:                       # toy tracker: tokens read since <bos>
        def reset(self):
            self.n = 0

        def step(self, tok):
            self.n = 0 if tok == BOS else self.n + 1
            return [min(self.n, 255)], self.n

    cats, bag = D.stream_features(stream, Counter())
    starts = D.program_starts(stream)
    assert cats.shape == (len(stream), 1) and bag.dtype == np.uint64
    assert (cats[starts, 0] == 0).all()
    off = starts[1] + 100                            # a window inside the 400-token program
    c, b = D.feature_windows((cats, bag), np.array([off]), 32)
    assert b[0, 0] == 100 and b[0, 31] == 131


def test_val_feature_windows_align_with_val_windows():
    stream = _stream([5, 300, 9])
    feats = _indexed_feats(stream)
    xs, ys = D.val_windows(stream, 64)
    c, b = D.val_feature_windows(feats, len(stream), 64)
    valid = ys != D.IGNORE
    assert c.shape == (len(xs), 64, 2)
    assert np.array_equal(xs[valid], stream[b[valid]].astype(np.int64))
    assert (b[~valid] == 0).all()


def test_bos_feature_windows_align_with_bos_windows():
    stream = _stream([5, 300, 9])
    feats = _indexed_feats(stream)
    xs, ys = D.bos_windows(stream, 64)
    c, b = D.bos_feature_windows(stream, feats, 64)
    valid = ys != D.IGNORE
    assert np.array_equal(xs[valid], stream[b[valid]].astype(np.int64))


def test_shuffled_features_come_from_another_row_far_away():
    c = np.arange(10)[:, None, None] * np.ones((1, 4, 2), dtype=np.int64)
    b = np.arange(10)[:, None] * np.ones((1, 4), dtype=np.int64)
    c2, b2 = D.shuffle_features(c, b)
    assert np.array_equal(b2[:, 0], (np.arange(10) - 5) % 10)       # row i gets row i - n/2
    assert np.array_equal(c2[..., 0], b2) and np.array_equal(c2[..., 1], b2)
    tc, tb = D.shuffle_features(torch.as_tensor(c), torch.as_tensor(b))
    assert np.array_equal(tb.numpy(), b2) and np.array_equal(tc.numpy(), c2)
    with pytest.raises(ValueError):
        D.shuffle_features(c[:1], b[:1])                             # one row cannot be shuffled
    c1, b1 = D.shuffle_features(c, b, shift=1)
    assert np.array_equal(b1[:, 0], (np.arange(10) - 1) % 10)
