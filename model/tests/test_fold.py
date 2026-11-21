"""Folded tags (E3, PLAN-v3-E3.md): one categorical feature folded into the token table.

With feat_fold, token t under tag c reads Q(tok[t] + tag[c]) (Q: QuantEmbedding's per-row
int8 / PF8 fake quantization, straight-through). fold_feature(model, c) is a plain model whose
token table is tok + tag[c]: its forward equals the source's under tag c, and it exports as an
ordinary format-2 model, so intref and the calculator run it unchanged.
"""
import json

import numpy as np
import pytest
import torch

from model.model import ModelConfig, QuantEmbedding, Transformer, fold_feature

TINY = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, n_kv_heads=1,
                   hidden_dim=40, max_seq_len=16, pos="alibi")
FOLD = ModelConfig(**{**TINY.to_dict(), "feat_cats": (4,), "feat_fold": True})


def _model(cfg=FOLD, seed=0):
    torch.manual_seed(seed)
    m = Transformer(cfg)
    with torch.no_grad():   # tag rows at the token rows' scale, so folding moves the grid
        m.feat_emb.weight.normal_(0.0, 0.02)
    m.calibrate(torch.randint(4, cfg.vocab_size, (2, 16)), cats=torch.randint(0, 4, (2, 16, 1)))
    return m.eval()


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 40, (2, 16), generator=g), torch.randint(0, 4, (2, 16, 1), generator=g)


# ---- config ---------------------------------------------------------------------------------

def test_fold_roundtrips_through_json_and_is_absent_when_off():
    assert "feat_fold" not in TINY.to_dict()
    d = json.loads(json.dumps(FOLD.to_dict()))
    assert d["feat_fold"] is True and ModelConfig(**d) == FOLD


@pytest.mark.parametrize("bad", [{"feat_cats": ()}, {"feat_cats": (4, 2)},
                                 {"feat_cats": (4,), "feat_bag": 3}])
def test_fold_needs_exactly_one_categorical_feature_and_no_bag(bad):
    with pytest.raises(ValueError):
        ModelConfig(**{**TINY.to_dict(), "feat_fold": True, **bad})


# ---- the embedding --------------------------------------------------------------------------

def test_input_row_is_the_quantized_sum_of_token_and_tag_rows():
    m = _model()
    idx, cats = _inputs()
    want = torch.empty(2, 16, 24)
    for c in range(4):
        q = QuantEmbedding(40, 24)
        with torch.no_grad():
            q.weight.copy_(m.tok_emb.weight + m.feat_emb.weight[c])
            rows = q.ste_weight()[idx]
        hit = cats[..., 0] == c
        want[hit] = rows[hit]
    with torch.no_grad():
        got = m.embed(idx, cats)
    assert torch.equal(got, want)


def test_folding_is_not_the_sum_of_separately_quantized_rows():
    m = _model()
    idx, cats = _inputs()
    with torch.no_grad():
        unfolded = m.tok_emb(idx) + m.feat_emb.ste_weight()[cats[..., 0]]
        assert not torch.allclose(m.embed(idx, cats), unfolded, atol=1e-7)


def test_gradients_reach_token_and_tag_rows():
    m = _model().train()
    idx, cats = _inputs()
    _, loss = m(idx, targets=idx, cats=cats)
    loss.backward()
    assert m.feat_emb.weight.grad.abs().sum() > 0
    assert m.tok_emb.weight.grad.abs().sum() > 0


# ---- fold_feature ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", range(4))
def test_folded_model_is_plain_and_matches_the_source_under_its_tag(c):
    m = _model()
    idx, _ = _inputs()
    p = fold_feature(m, c)
    assert p.cfg.feat_rows == 0 and not p.cfg.feat_fold and p.cfg.pos == "alibi"
    assert not hasattr(p, "feat_emb")
    with torch.no_grad():
        a, _ = m(idx, cats=torch.full((2, 16, 1), c))
        b, _ = p(idx)
    assert torch.equal(a, b)


def test_fold_feature_leaves_the_source_unchanged_and_checks_its_input():
    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    fold_feature(m, 1)
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items())
    with pytest.raises(ValueError):
        fold_feature(m, 4)
    with pytest.raises(ValueError):
        fold_feature(_model(ModelConfig(**{**TINY.to_dict(), "feat_cats": (4,)})), 0)


def test_folded_model_exports_as_an_ordinary_format_2_model(tmp_path):
    from model.export import export_model
    from model.intref import IntModel

    calib = np.random.default_rng(0).integers(4, 40, size=(3, 16))
    meta = export_model(fold_feature(_model(), 2), calib, tmp_path)
    assert meta["version"] == 2
    assert not any(k.startswith("feat_") for k in meta["config"])
    assert IntModel.load(tmp_path).pos_scheme == "alibi"


def test_uint8_tags_index_like_int64_tags():
    m = _model()
    idx, cats = _inputs()
    with torch.no_grad():
        assert torch.equal(m.embed(idx, cats.to(torch.uint8)), m.embed(idx, cats))


def test_an_unfolded_tag_model_cannot_be_exported(tmp_path):
    from model.export import export_model
    calib = np.random.default_rng(0).integers(4, 40, size=(3, 16))
    with pytest.raises(ValueError):
        export_model(_model(), calib, tmp_path, calib_feats=(np.zeros((3, 16, 1), np.uint8),
                                                             np.zeros((3, 16), np.uint64)))
