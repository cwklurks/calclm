import numpy as np
import torch

from model.intops import MulCounter
from model.intref import expected_mults, sequence_logits
from model.tests.conftest import SMALL, pattern_batch


def _float_logits(model, seqs):
    ref = model.double()
    with torch.no_grad():
        logits, _ = ref(torch.from_numpy(seqs))
    model.float()
    return logits.numpy()


def test_small_model_learned_patterns(trained_small):
    _, _, final_loss = trained_small
    assert final_loss < 1.6  # unigram baseline is ln(20) = 3.0


def test_intref_matches_float_fake_quant_model(trained_small):
    model, im, _ = trained_small
    seqs = pattern_batch(np.random.default_rng(7), 6, 32, 24)[:, :-1]
    want = _float_logits(model, seqs)
    got = np.stack([sequence_logits(im, s) for s in seqs]) / 256.0
    top1 = (want.argmax(-1) == got.argmax(-1)).mean()
    err = np.abs(want - got)
    print(f"top-1 agreement {top1:.4f}, max |logit err| {err.max():.4f}, mean {err.mean():.4f}")
    assert top1 >= 0.97
    assert err.max() < 0.25 * np.abs(want).max()


def test_intref_state_is_integer_only(trained_small):
    _, im, _ = trained_small

    def walk(obj, seen):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, float):
            raise AssertionError("float scalar in intref state")
        if isinstance(obj, np.ndarray):
            assert obj.dtype.kind in "iu", obj.dtype
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, seen)
        elif hasattr(obj, "__dict__"):
            walk(vars(obj), seen)

    walk(im, set())
    logits = sequence_logits(im, [1, 5, 6, 7])
    assert logits.dtype == np.int64


def test_multiply_count_matches_formula(trained_small):
    _, im, _ = trained_small
    cache = im.new_cache()
    for pos, tok in enumerate([1, 5, 6, 7, 8]):
        counter = MulCounter()
        im.step(tok, pos, cache, counter)
        assert counter.counts == expected_mults(im.cfg, pos)
    per_layer = 6 * 32 * 2  # attention products per cached position, v0 shape
    v0 = expected_mults(type(im.cfg)(vocab_size=812), pos=0)
    assert v0["attn_score"] + v0["attn_pv"] == 6 * per_layer
    assert sum(v0.values()) - 6 * per_layer < 40_000


def test_logits_match_incremental_and_batch(trained_small):
    _, im, _ = trained_small
    seq = [1, 9, 10, 11, 9, 10]
    full = sequence_logits(im, seq)
    prefix = sequence_logits(im, seq[:4])
    assert np.array_equal(full[:4], prefix)
    assert SMALL.max_seq_len == im.cfg.max_seq_len
