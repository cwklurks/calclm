"""model/kv4.py, E19's 4-bit K/V integer reference (PLAN-E19.md "The kv4 scheme", Control 7, 8).

Fast tests run on a tiny ALiBi export trained on repeating patterns. The slow tests
(CALCLM_E19_SLOW=1, CALCLM_RUNS pointing at the shared runs dir) are read-only checks against
the real E16 W8-s10 export: they calibrate it into tmp, run the exhaustive encode test on the
calibrated tables and the float oracle on a few NLL programs.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from model import kv4 as K
from model.evaluate import val_programs
from model.export import export_model, load_export
from model.intref import IntModel, KVCache, sequence_logits
from model.model import ModelConfig, Transformer
from model.tests.conftest import pattern_batch
from model.tokens import BOS

REAL = Path(os.environ.get('CALCLM_RUNS', Path(__file__).resolve().parents[2] / 'runs')).parent
have_real = pytest.mark.skipif(not (REAL / 'runs/e16/export/W8-s10/model.bin').exists(),
                               reason='the real runs/e16 exports are not present')
slow = pytest.mark.skipif(os.environ.get('CALCLM_E19_SLOW') != '1',
                          reason='slow: set CALCLM_E19_SLOW=1')

TINY = ModelConfig(vocab_size=24, dim=48, n_layers=2, n_heads=6, n_kv_heads=1, hidden_dim=96,
                   max_seq_len=16, lut_max=128, pos='alibi')


def _stream(seed: int, n: int = 40) -> np.ndarray:
    """A token stream of n programs: <bos>, a repeated motif, of random lengths 6..40."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        motif = rng.integers(4, TINY.vocab_size, size=rng.integers(3, 6))
        out += [BOS] + list(np.resize(motif, int(rng.integers(5, 40))))
    return np.asarray(out, dtype='<u2')


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """(export dir, train.bin, val.bin) of a tiny ALiBi model trained 150 steps."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    w, v = TINY.max_seq_len, TINY.vocab_size
    m = Transformer(TINY)
    m.calibrate(torch.from_numpy(pattern_batch(rng, 8, w, v)[:, :-1]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(150):
        seq = torch.from_numpy(pattern_batch(rng, 16, w, v))
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = tmp_path_factory.mktemp('kv4')
    export_model(m.eval(), pattern_batch(np.random.default_rng(1), 8, w, v)[:, :-1], out / 'ex')
    _stream(2, 90).tofile(out / 'train.bin')
    _stream(3, 140).tofile(out / 'val.bin')
    return out / 'ex', out / 'train.bin', out / 'val.bin'


@pytest.fixture(scope='module')
def tables(tiny, tmp_path_factory):
    ex, train, _ = tiny
    rec = K.calibrate(ex, train, workers=1)
    path = tmp_path_factory.mktemp('tables') / 'T-s1.json'
    K.write_tables(path, rec)
    return path


# ================================================================ the scheme's constants

def test_the_scheme_is_fixed_as_the_plan_states():
    assert K.LEVELS == 7 and K.PERCENTILE == 99.9 and K.STEP_FLOOR == 1.0
    assert K.FRAC_BITS == 16 and K.ONE == 65536 and K.M_MAX == 65535
    assert (K.CAL_RNG, K.CAL_N, K.CAL_LEN) == (1901, 64, 512)
    assert (K.NLL_RNG, K.NLL_N, K.NLL_CAP) == (1902, 128, 1024)
    assert K.KINDS == ('k', 'v')
    assert K.MODE_KINDS == {'identity': (), 'kv4': ('k', 'v'), 'k4': ('k',), 'v4': ('v',)}


# ================================================================ steps, reciprocals, tables

def test_the_step_is_a_over_7_floored_at_1():
    a = np.array([0., 3.5, 7., 14., 127.])
    assert K.step(a).tolist() == [1., 1., 1., 2., 127 / 7]


def test_the_reciprocal_is_round_65536_over_s_capped_at_65535():
    s = np.array([1., 1.0000001, 2., 3., 127 / 7])
    m = K.reciprocal(s)
    assert m.dtype == np.int64
    assert m.tolist() == [65535, 65535, 32768, 21845, 3612]
    assert m.min() >= 3612 and m.max() <= 65535       # the plan's range: fits uint16


def test_the_table_rounds_half_to_even_and_caps_at_127():
    assert K.table(np.array(1.)).tolist() == list(range(-7, 8))
    t = K.table(np.array(2.5)).tolist()               # 2.5 -> 2, 7.5 -> 8, 12.5 -> 12
    assert t[7 + 1] == 2 and t[7 + 3] == 8 and t[7 + 5] == 12 and t[7 - 1] == -2
    assert K.table(np.array(127 / 7)).tolist()[-1] == 127
    assert K.table(np.array(19.)).tolist()[0] == -127
    assert K.table(np.ones((2, 3))).shape == (2, 3, 15)


def test_tables_from_values_take_the_linear_99_9th_percentile_of_abs_per_channel():
    rng = np.random.default_rng(0)
    vals = rng.integers(-127, 128, size=(500, 2, 2, 3))     # (positions, layers, kinds, channels)
    rec = K.tables_from_values(vals)
    a = np.percentile(np.abs(vals.astype(np.float64)), 99.9, axis=0, method='linear')
    assert np.array_equal(rec['a'], a)
    assert np.array_equal(rec['s'], np.maximum(a / 7, 1.))
    assert np.array_equal(rec['M'], np.minimum(np.rint(65536 / rec['s']), 65535))
    assert rec['T'].shape == (2, 2, 3, 15)
    assert np.array_equal(rec['T'], K.table(rec['s']))


# ================================================================ encode and decode

def test_encode_rounds_ties_to_even_with_the_fixed_point_reciprocal():
    m = np.int64(32768)                                    # s = 2: every odd x is an exact tie
    assert K.encode(np.array([1, 3, 5, 7, -1, -3, -5]), m).tolist() == [0, 2, 2, 4, 0, -2, -2]
    assert K.encode(np.array([0, 2, 4, -4]), m).tolist() == [0, 1, 2, -2]


def test_encode_at_step_1_returns_x_clamped_for_every_int8():
    x = np.arange(-128, 128)
    assert K.encode(x, np.int64(65535)).tolist() == np.clip(x, -7, 7).tolist()


def test_encode_clamps_to_7_and_keeps_the_sign():
    assert K.encode(np.array([127, -127, 128, -128]), np.int64(3612)).tolist() == [7, -7, 7, -7]


def test_encode_matches_the_float_reference_for_every_input_and_many_reciprocals():
    x = np.arange(-128, 128)
    for m in [3612, 4096, 9362, 21845, 32768, 32769, 43690, 65535,
              *np.random.default_rng(5).integers(3612, 65536, size=300)]:
        assert np.array_equal(K.encode(x, np.int64(m)), K.ref_encode(x, np.int64(m))), m


def test_encode_is_integer_only_and_the_product_fits_24_bits():
    x = np.arange(-127, 128)
    assert K.encode(x, np.int64(65535)).dtype == np.int64
    assert 127 * 65535 < 2 ** 23
    with pytest.raises(TypeError):
        K.encode(np.array([1.5]), np.int64(65535))


def test_decode_reads_the_channels_table():
    t = np.stack([K.table(np.array(1.)), K.table(np.array(3.))])     # 2 channels
    j = np.array([2, -2])
    assert K.decode(j, t).tolist() == [2, -6]


def test_the_exhaustive_check_passes_on_good_tables_and_catches_a_wrong_rule(monkeypatch):
    s = np.array([[[1., 2., 2.5]], [[3.3, 7.77, 127 / 7]]])           # (L 2, kinds 1, hd 3)
    m, t = K.reciprocal(s), K.table(s)
    rec = K.exhaustive_check(m, t)
    assert rec['pass'] is True and rec['n_checked'] == 2 * 1 * 3 * 256
    real = K.encode

    def half_away(x, mm):
        x = np.asarray(x)
        out = real(x, mm)
        p = np.abs(x) * mm
        tie = (p & 0xFFFF) == 32768
        return np.where(tie, np.sign(x) * np.minimum((p >> 16) + 1, 7), out)
    monkeypatch.setattr(K, 'encode', half_away)
    bad = K.exhaustive_check(m, t)
    assert bad['pass'] is False and bad['mismatches']


# ================================================================ calibration programs

def test_select_programs_equals_val_programs_permuted_then_cut():
    st = np.concatenate([np.array([7, 8], dtype='<u2'), _stream(9, 30)])   # tokens before <bos>
    progs = val_programs(st)
    order = np.random.default_rng(1901).permutation(len(progs))
    want = [progs[i][:12] for i in order[:5]]
    assert K.select_programs(st, 1901, 5, 12) == want
    assert K.select_programs(st, 1901, 5) == [progs[i] for i in order[:5]]


def test_calibration_and_nll_programs_use_the_plans_rngs(tiny):
    _, train, val = tiny
    st = np.fromfile(train, dtype='<u2')
    assert K.calibration_programs(train) == K.select_programs(st, 1901, 64, 512)
    sv = np.fromfile(val, dtype='<u2')
    assert K.nll_programs(val) == K.select_programs(sv, 1902, 128, 1024)


# ================================================================ calibration and the tables file

def test_calibration_records_the_scheme_the_token_count_and_the_export(tiny, tables):
    ex, train, _ = tiny
    rec = json.loads(Path(tables).read_text())
    progs = K.calibration_programs(train)
    assert rec['n_tokens'] == sum(len(p) for p in progs)
    assert rec['export_sha256'] == hashlib.sha256((ex / 'model.bin').read_bytes()).hexdigest()
    assert rec['export_meta_sha256'] == hashlib.sha256((ex / 'model.json').read_bytes()).hexdigest()
    assert rec['scheme'] == {'levels': 7, 'percentile': 99.9, 'step_floor': 1.0, 'frac_bits': 16,
                             'm_max': 65535, 'rng': 1901, 'n_programs': 64, 'cut': 512,
                             'horizons': True, 'percentile_method': 'linear'}
    assert rec['programs_sha256'] == K.programs_sha256(progs)
    assert np.asarray(rec['M']).shape == (2, 2, 8) and np.asarray(rec['T']).shape == (2, 2, 8, 15)


def test_calibration_collects_every_k8_and_v8_the_int8_model_writes(tiny):
    """The collection is the unchanged int8 intref's cache writes: K row j of layer l is what
    IntModel wrote at that position."""
    ex, train, _ = tiny
    prog = K.calibration_programs(train)[0]
    vals = K.collect(ex, [prog])
    m = IntModel(load_export(ex), horizons=True)
    cache = m.new_cache()
    for p, t in enumerate(prog):
        m.step(int(t), p, cache)
        slot = p % m.cfg.max_seq_len
        assert np.array_equal(vals[p, :, 0], cache.k[:, slot])
        assert np.array_equal(vals[p, :, 1], cache.v[:, slot])
    assert vals.shape == (len(prog), 2, 2, 8)


def test_calibration_does_not_depend_on_the_worker_count(tiny, tables):
    ex, train, _ = tiny
    two = K.calibrate(ex, train, workers=2)
    one = json.loads(Path(tables).read_text())
    assert json.loads(json.dumps(two)) == one


def test_read_tables_refuses_another_export(tiny, tables, tmp_path):
    ex, _, _ = tiny
    t = K.read_tables(tables, ex)
    assert t.M.shape == (2, 2, 8) and t.T.shape == (2, 2, 8, 15)
    assert t.sha256 == hashlib.sha256(Path(tables).read_bytes()).hexdigest()
    rec = json.loads(Path(tables).read_text())
    rec['export_sha256'] = '0' * 64
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='export'):
        K.read_tables(bad, ex)
    with pytest.raises(ValueError, match='export'):
        K.Kv4Model.load(ex, bad, 'kv4')


# ================================================================ the 4-bit model (new_cache)

def test_the_hook_is_new_cache_and_attention_is_intrefs():
    assert K.Kv4Model.attention is IntModel.attention
    assert K.Kv4Model.step is IntModel.step
    assert K.Kv4Model.new_cache is not IntModel.new_cache


def test_the_identity_quantizer_reproduces_int8_logits_bit_for_bit(tiny, tables):
    ex, _, val = tiny
    ids = K.nll_programs(val)[0]
    base = sequence_logits(IntModel(load_export(ex), horizons=True), ids)
    ident = K.Kv4Model.load(ex, None, 'identity')
    assert ident.kv == 'identity'
    assert np.array_equal(sequence_logits(ident, ids), base)
    assert isinstance(ident.new_cache(), KVCache)


def test_the_4bit_cache_holds_table_values_only(tiny, tables):
    ex, _, val = tiny
    t = K.read_tables(tables, ex)
    m = K.Kv4Model.load(ex, tables, 'kv4')
    cache = m.new_cache()
    ids = K.nll_programs(val)[1]
    for p, tok in enumerate(ids[:16]):
        m.step(int(tok), p, cache)
    for kind, ring in ((0, cache.k), (1, cache.v)):
        held = ring[:, :]
        for layer in range(2):
            for c in range(8):
                assert set(held[layer, :, c].tolist()) <= set(t.T[layer, kind, c].tolist()) | {0}


def test_k_only_and_v_only_quantize_only_their_kind(tiny, tables):
    """At position 0 attention returns the one value whatever the key, so with K alone
    quantized every layer's V write is the raw one; with V alone layer 0's K write is raw."""
    ex, _, _ = tiny
    t = K.read_tables(tables, ex)
    raw = K.Kv4Model.load(ex, None, 'identity')
    rc = raw.new_cache()
    raw.step(5, 0, rc)
    q = lambda x, l, kind: K.decode(K.encode(x, t.M[l, kind]), t.T[l, kind])   # noqa: E731
    for name in ('k4', 'v4', 'kv4'):
        m = K.Kv4Model.load(ex, tables, name)
        c = m.new_cache()
        m.step(5, 0, c)
        if name == 'k4':
            assert np.array_equal(c.v[:, 0], rc.v[:, 0])
            assert all(np.array_equal(c.k[l, 0], q(rc.k[l, 0], l, 0)) for l in range(2))
        elif name == 'v4':
            assert np.array_equal(c.k[0, 0], rc.k[0, 0])
            assert np.array_equal(c.v[0, 0], q(rc.v[0, 0], 0, 1))
        else:
            assert np.array_equal(c.k[0, 0], q(rc.k[0, 0], 0, 0))
            assert np.array_equal(c.v[0, 0], q(rc.v[0, 0], 0, 1))


def test_the_cache_refuses_any_write_but_one_layer_slot_row(tiny):
    ex, _, _ = tiny
    c = K.Kv4Model.load(ex, None, 'identity').new_cache()
    with pytest.raises(TypeError):
        c.k[0] = np.zeros((16, 8), dtype=np.int64)
    with pytest.raises(TypeError):
        c.k[0, 0:2] = np.zeros((2, 8), dtype=np.int64)
    c.k[1, 3] = np.arange(8)
    assert c.k[1, 3].tolist() == list(range(8))


def test_the_model_needs_tables_unless_identity(tiny):
    ex, _, _ = tiny
    with pytest.raises(ValueError):
        K.Kv4Model.load(ex, None, 'kv4')
    with pytest.raises(ValueError):
        K.Kv4Model.load(ex, None, 'int8')


def test_clip_counts_saturated_codes_beyond_the_last_table_entry():
    s = np.array([[[2.]]])                                  # one layer, one kind, one channel
    q = K.Quantizer(K.reciprocal(s).repeat(2, 1), K.table(s).repeat(2, 1), ('k', 'v'), count=True)
    q(0, 'k', np.array([14]))      # code 7, |x| = T[7] = 14: saturated but not beyond
    q(0, 'k', np.array([15]))      # code 7 (rounds to 8, clamped), 15 > 14: a clip
    q(0, 'k', np.array([-100]))    # a clip
    q(0, 'v', np.array([3]))
    assert q.counts['k'][0].tolist() == [2, 3] and q.counts['v'][0].tolist() == [0, 1]


# ================================================================ the oracle (Control 7)

def test_the_oracle_is_a_separate_subclass_with_its_own_attention():
    assert issubclass(K.OracleModel, IntModel)
    assert K.OracleModel.attention is not IntModel.attention
    assert K.OracleModel.new_cache is IntModel.new_cache       # not E19's cache class


def test_the_oracle_agrees_bit_for_bit_with_the_4bit_model(tiny, tables):
    ex, _, val = tiny
    m = K.Kv4Model.load(ex, tables, 'kv4', record=True)
    o = K.OracleModel.load(ex, tables)
    for ids in K.nll_programs(val)[:4]:
        rec = K.lockstep(m, o, ids)
        assert rec['agree'] is True and rec['n_steps'] == len(ids) - 1
        assert rec['n_writes'] == 2 * 2 * (len(ids) - 1)


def test_the_oracle_catches_a_wrong_encode(tiny, tables, monkeypatch):
    ex, _, val = tiny
    monkeypatch.setattr(K, 'encode', lambda x, m: np.zeros_like(np.asarray(x)))
    m = K.Kv4Model.load(ex, tables, 'kv4', record=True)
    o = K.OracleModel.load(ex, tables)
    rec = K.lockstep(m, o, K.nll_programs(val)[0])
    assert rec['agree'] is False and rec['first_mismatch'] is not None


# ================================================================ NLL and the clip shares

def test_program_nll_is_attn_probes_float64_log_softmax_over_positions_1_to_t_minus_1(tiny):
    ex, _, val = tiny
    ids = K.nll_programs(val)[0]
    m = IntModel(load_export(ex), horizons=True)
    lg = sequence_logits(m, ids)[:-1].astype(np.float64) / 256
    lg -= lg.max(axis=1, keepdims=True)
    want = (np.log(np.exp(lg).sum(axis=1)) - lg[np.arange(len(ids) - 1), ids[1:]]).sum()
    total, n = K.program_nll(m, ids)
    assert n == len(ids) - 1 and total == pytest.approx(want, rel=1e-12)


def test_the_nll_record_has_four_ways_the_oracle_and_the_clip_shares(tiny, tables):
    ex, _, val = tiny
    rec = K.nll_record(ex, tables, val, workers=2)
    assert set(rec['nll']) == {'int8', 'kv4', 'k4', 'v4', 'oracle'}
    assert rec['oracle']['agree'] is True and rec['nll']['oracle'] == rec['nll']['kv4']
    assert rec['n_programs'] == min(128, len(val_programs(np.fromfile(val, dtype='<u2'))))
    assert rec['n_tokens'] == sum(len(p) - 1 for p in K.nll_programs(val))
    assert set(rec['clip']) == {'k', 'v'} and len(rec['clip']['k']) == 2
    one = K.nll_record(ex, tables, val, workers=1)
    assert json.dumps(one, sort_keys=True) == json.dumps(rec, sort_keys=True)


def test_the_dev_replay_feeds_every_generated_token_but_a_final_eos(tiny, tables):
    from model.tokens import EOS
    ex, _, val = tiny
    a = K.nll_programs(val)[0][:10]
    rec = K.replay_clip(ex, tables, [a, a + [EOS]], workers=1)
    total = rec['clip']['k'][0]['total']
    assert total == 2 * len(a) * 8          # values: head_dim 8 per row


def test_the_tables_hash_is_the_files(tables):
    assert K.tables_sha256(tables) == hashlib.sha256(Path(tables).read_bytes()).hexdigest()


# ================================================================ slow: the real W8-s10 export

@slow
@have_real
def test_slow_exhaustive_encode_on_real_calibrated_tables(tmp_path):
    ex = REAL / 'runs/e16/export/W8-s10'
    rec = K.calibrate(ex, REAL / 'runs/e4/data/SEED/train.bin', workers=24)
    path = tmp_path / 'W8-s10.json'
    K.write_tables(path, rec)
    t = K.read_tables(path, ex)
    chk = K.exhaustive_check(t.M, t.T)
    assert chk['pass'] is True and chk['n_checked'] == 8 * 2 * 32 * 256
    assert t.M.min() >= 3612 and t.M.max() <= 65535
    assert 20_000 < rec['n_tokens'] <= 64 * 512


@slow
@have_real
def test_slow_oracle_agrees_on_real_nll_programs(tmp_path):
    ex = REAL / 'runs/e16/export/W8-s10'
    rec = K.calibrate(ex, REAL / 'runs/e4/data/SEED/train.bin', workers=24)
    path = tmp_path / 'W8-s10.json'
    K.write_tables(path, rec)
    m = K.Kv4Model.load(ex, path, 'kv4', record=True)
    o = K.OracleModel.load(ex, path)
    for ids in K.nll_programs(REAL / 'data/tok/val.bin')[:3]:
        out = K.lockstep(m, o, ids[:400])
        assert out['agree'] is True, out['first_mismatch']
