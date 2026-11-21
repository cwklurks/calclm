import numpy as np
import pytest

from model.intref import SamplerConfig, choose, generate, ranked_candidates
from model.tokens import BOS, EOS, PAD, UNK


def test_ranked_candidates_exclude_specials_and_break_ties_by_id():
    logits = np.array([900, 800, 50, 700, 10, 50, 50], dtype=np.int64)
    # pad(0), bos(1), unk(3) are banned even though they have the highest logits
    assert ranked_candidates(logits, 4) == [2, 5, 6, 4]


def test_same_seed_same_tokens(trained_small):
    _, im, _ = trained_small
    cfg = SamplerConfig(temperature_q8=307, top_k=8, seed=1234)
    a = generate(im, [5, 6], 20, cfg)
    b = generate(im, [5, 6], 20, cfg)
    assert a == b
    others = {tuple(generate(im, [], 20, SamplerConfig(512, 16, seed=s))) for s in range(1, 6)}
    assert len(others) > 1  # different seeds explore different continuations


def test_greedy_runs_end_to_end_and_never_emits_specials(trained_small):
    _, im, _ = trained_small
    out = generate(im, [], 64, SamplerConfig(greedy=True))
    assert out[0] == BOS and 1 < len(out) <= im.cfg.max_seq_len
    assert not {PAD, BOS, UNK} & set(out[1:])
    assert out[-1] == EOS or len(out) == im.cfg.max_seq_len
    assert out == generate(im, [], 64, SamplerConfig(top_k=1, seed=99))


def test_choose_respects_temperature_and_prng_contract(trained_small):
    _, im, _ = trained_small
    logits = np.zeros(24, dtype=np.int64)
    logits[7], logits[9] = 256 * 3, 256 * 2  # 1 nat apart at T = 1
    cold_logits = logits.copy()
    cold_logits[9] = 0  # 3 nats below the top; at T = 1/8 that is 24 nats: past the table, weight 0
    cold = [choose(im, cold_logits, SamplerConfig(33, 4), s)[0] for s in range(1, 200)]
    assert cold.count(7) == len(cold)
    hot = [choose(im, logits, SamplerConfig(256, 2), s)[0] for s in range(1, 400)]
    share = hot.count(9) / len(hot)
    assert 0.2 < share < 0.33  # exp(-1) / (1 + exp(-1)) = 0.269
    tok, state = choose(im, logits, SamplerConfig(greedy=True), 77)
    assert tok == 7 and state == 77  # greedy does not advance the PRNG


def test_invalid_sampler_settings_rejected(trained_small):
    _, im, _ = trained_small
    with pytest.raises(ValueError):
        choose(im, np.zeros(24, dtype=np.int64), SamplerConfig(temperature_q8=8), 1)
    with pytest.raises(ValueError):  # INV_T must fit 16 bits: Tq >= 33
        choose(im, np.zeros(24, dtype=np.int64), SamplerConfig(temperature_q8=32), 1)
    with pytest.raises(ValueError):
        generate(im, [], 5, SamplerConfig(seed=1 << 32))
    with pytest.raises(ValueError):
        generate(im, [5] * im.cfg.max_seq_len, 5, SamplerConfig())
