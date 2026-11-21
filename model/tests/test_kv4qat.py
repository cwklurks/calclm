"""E21's 4-bit K/V fake quant (model/kv4qat.py) and its training wrapper (model/kv4qat_train.py),
PLAN-E21 "The training-side design", "The code change", Controls 6b, 6d, 7 and 11.

Fast tests run on the CPU with synthetic data. CALCLM_E21_SLOW=1 adds Control 7 at function level
on E19's 17 stored W8 tables (runs/e19/kv4, read only); with CUDA as well, Control 7 at module
level (torch.compile, bf16 autocast) and the Control 6d identity-wrapper smoke run against E16's
W8-s10 log. Nothing is written under runs/.
"""
import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from model import kv4 as K
from model import kv4qat as Q
from model import kv4qat_train as W
from model import model as M
from model import train as tr
from model.model import Attention, ModelConfig, Transformer
from model.tests.test_distill import _data, _log, _same_states

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get('CALCLM_RUNS', ROOT / 'runs'))
SLOW = os.environ.get('CALCLM_E21_SLOW') == '1'
slow = pytest.mark.skipif(not SLOW, reason='CALCLM_E21_SLOW=1 runs the E21 slow checks')
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
E19_TABLES = [RUNS / 'e19' / 'kv4' / f'W8-s{s}.json' for s in range(10, 27)]
SMALL = ModelConfig(vocab_size=30, dim=24, n_layers=2, n_heads=3, n_kv_heads=1, hidden_dim=40,
                    max_seq_len=32, pos='alibi')


def _tables(seed=0, c=32):
    """Random (M, T) for 2 kinds x c channels as kv4 derives them from a step s in [1, 18.2]."""
    s = np.random.default_rng(seed).uniform(1., 18.2, size=(2, c))
    return K.reciprocal(s), K.table(s)


# ================================================================ the scheme's integers

def test_constants_are_e19s():
    assert Q.PERCENTILE == K.PERCENTILE == 99.9 and Q.LEVELS == K.LEVELS == 7
    assert Q.MOMENTUM == 0.99 and Q.BINS == 128
    assert Q.MODES == ('kv4', 'fixed', 'identity')


def test_m_and_t_from_s_equal_kv4s_for_10000_steps():
    """Control 7: the training-side M and T derived from s equal kv4.reciprocal and kv4.table."""
    s = np.linspace(1., 18.2, 10_000)
    m, t = Q.derive_from_s(torch.tensor(s, dtype=torch.float64))
    assert np.array_equal(m.numpy().astype(np.int64), K.reciprocal(s))
    assert np.array_equal(t.numpy().astype(np.int64), K.table(s))


def test_s_from_a_is_kv4s_step_with_its_floor():
    a = torch.tensor([0., 3.5, 7., 7.0000001, 100.], dtype=torch.float64)
    assert np.array_equal(Q.step_from_a(a).numpy(), K.step(a.numpy()))


@slow
def test_m_and_t_from_e19s_17_w8_tables_s_equal_the_stored_integers():
    for p in E19_TABLES:
        rec = json.loads(p.read_text())
        m, t = Q.derive_from_s(torch.tensor(rec['s'], dtype=torch.float64))
        assert np.array_equal(m.numpy().astype(np.int64), np.asarray(rec['M'])), p
        assert np.array_equal(t.numpy().astype(np.int64), np.asarray(rec['T'])), p


# ================================================================ the histogram percentile

@pytest.mark.parametrize('n', [1, 2, 999, 1000, 1001, 32_768, 21_982, 4_321])
def test_the_histogram_percentile_is_numpys_linear_percentile(n):
    rng = np.random.default_rng(n)
    for spread in (3, 40, 128):
        x = np.minimum(np.abs(rng.normal(0, spread, size=(n, 5))).round(), 127)
        x[:, 0] = 7                                    # all ties
        got = Q.hist_percentile(torch.tensor(x, dtype=torch.float32)).numpy()
        want = np.percentile(x.astype(np.float64), 99.9, axis=0, method='linear')
        assert np.array_equal(got, want), (n, spread, got, want)


def test_the_histogram_percentile_handles_ties_at_the_interpolation_points():
    x = np.zeros((2000, 3))
    x[-3:, 0] = 127
    x[-2:, 1] = 5
    x[-1:, 2] = 1
    got = Q.hist_percentile(torch.tensor(x)).numpy()
    assert np.array_equal(got, np.percentile(x, 99.9, axis=0, method='linear'))


# ================================================================ Control 7: function level

def test_the_fixed_table_fake_quant_equals_kv4_on_every_int8_input():
    """All 256 int8 inputs on random channels: encode/decode in float32 equal kv4's integers."""
    m, t = _tables(1)
    x = np.arange(-128, 128)
    xs = np.broadcast_to(x[:, None, None], (256, 2, 32))
    got = Q.fake_quant(torch.tensor(xs, dtype=torch.float32), torch.tensor(m, dtype=torch.float32),
                       torch.tensor(t, dtype=torch.float32)).numpy()
    for k in range(2):
        for c in range(32):
            want = K.decode(K.encode(x, m[k, c]), np.broadcast_to(t[k, c], (256, 15)))
            assert np.array_equal(got[:, k, c].astype(np.int64), want), (k, c)


def test_the_check_covers_ties_at_the_half_step():
    """M = 32768 (s = 2): |x| odd lands exactly on r = 32768, the tie goes to the even code."""
    m = np.full((2, 1), 32768)
    t = K.table(np.full((2, 1), 2.))
    x = torch.tensor([[1.], [3.], [5.], [-1.], [-3.]]).expand(5, 2).reshape(5, 2, 1)
    got = Q.fake_quant(x, torch.tensor(m, dtype=torch.float32),
                       torch.tensor(t, dtype=torch.float32))[:, 0, 0]
    want = K.decode(K.encode(np.array([1, 3, 5, -1, -3]), m[0, 0]),
                    np.broadcast_to(t[0, 0], (5, 15)))
    assert got.tolist() == want.tolist()


def _function_check(m, t, device='cpu') -> int:
    x = np.arange(-128, 128)
    n = 0
    for layer in range(m.shape[0]):
        xs = torch.tensor(np.broadcast_to(x[:, None, None], (256, 2, m.shape[-1])),
                          dtype=torch.float32, device=device)
        got = Q.fake_quant(xs, torch.tensor(m[layer], dtype=torch.float32, device=device),
                           torch.tensor(t[layer], dtype=torch.float32, device=device)).cpu()
        got = got.numpy().astype(np.int64)
        for k in range(2):
            for c in range(m.shape[-1]):
                want = K.decode(K.encode(x, m[layer, k, c]),
                                np.broadcast_to(t[layer, k, c], (256, 15)))
                assert np.array_equal(got[:, k, c], want), (layer, k, c)
                n += 256
    return n


@slow
def test_control7_function_level_on_e19s_17_w8_tables_cpu():
    n = 0
    for p in E19_TABLES:
        tb = K.read_tables(p)
        n += _function_check(tb.m, tb.t)
    assert n == 17 * 8 * 2 * 32 * 256 == 2_228_224


@slow
@cuda
def test_control7_function_level_on_e19s_17_w8_tables_cuda():
    n = sum(_function_check(K.read_tables(p).m, K.read_tables(p).t, 'cuda') for p in E19_TABLES)
    assert n == 2_228_224


def test_the_straight_through_gradient_is_clipped_at_the_largest_level():
    m, t = _tables(2, c=1)
    tt = torch.tensor(t[0], dtype=torch.float32)
    top = float(tt[0, -1])
    x = torch.arange(-127., 128.).reshape(-1, 1).requires_grad_()
    out = Q.ste(x, torch.tensor(m[0], dtype=torch.float32), tt)
    out.sum().backward()
    want = (x.detach().abs() <= top).float()
    assert torch.equal(x.grad, want)
    assert torch.equal(out.detach(), Q.fake_quant(x.detach(), torch.tensor(m[0]).float(), tt))


# ================================================================ Kv4QatAttention

def _forward_body(cls) -> list:
    src = textwrap.dedent(inspect.getsource(cls.forward))
    return ast.parse(src).body[0].body


def test_the_copied_forward_is_attentions_plus_exactly_one_inserted_call():
    """The code change: one copied method; the AST equals Attention.forward with exactly the
    inserted statement, right after q8, k8, v8 land on their int8 grids (the injection point)."""
    a, b = _forward_body(Attention), _forward_body(Q.Kv4QatAttention)
    assert len(b) == len(a) + 1
    i = next(j for j, st in enumerate(a) if 'static_round' in ast.unparse(st))
    assert ast.unparse(a[i]) == 'q8, k8, v8 = (static_round(y) for y in ys)'
    assert ast.unparse(b[i + 1]) == Q.INJECTION == 'k8, v8 = self.kv4qat(k8, v8)'
    rest = b[:i + 1] + b[i + 2:]
    assert [ast.dump(x) for x in rest] == [ast.dump(x) for x in a]


def test_the_copied_forward_runs_in_model_models_namespace():
    """QUANT_ENABLED and every helper resolve as in the original (set_quant_enabled reaches it)."""
    assert Q.Kv4QatAttention.forward.__globals__ is vars(M)


def test_convert_swaps_the_class_and_adds_no_state_dict_keys():
    torch.manual_seed(0)
    plain = Transformer(SMALL)
    keys = list(plain.state_dict())
    torch.manual_seed(0)
    m = Transformer(SMALL)
    Q.convert(m, 'kv4')
    assert all(type(b.attn) is Q.Kv4QatAttention for b in m.layers)
    assert list(m.state_dict()) == keys
    assert _same_states(m.state_dict(), plain.state_dict())
    names = {n for n, _ in m.layers[0].attn.named_buffers()}
    assert set(Q.BUFFERS) <= names
    for n in Q.BUFFERS:
        assert n in m.layers[0].attn._non_persistent_buffers_set


def test_convert_refuses_an_unknown_mode_and_fixed_needs_tables():
    with pytest.raises(ValueError):
        Q.convert(Transformer(SMALL), 'k4')
    with pytest.raises(ValueError):
        Q.convert(Transformer(SMALL), 'fixed')


def _batch(seed=0, b=4, t=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 30, (b, t), generator=g)


def _calibrated(seed=0):
    torch.manual_seed(seed)
    m = Transformer(SMALL)
    m.calibrate(_batch(seed))
    return m


def test_identity_mode_leaves_every_value_and_gradient_unchanged():
    a, b = _calibrated(), _calibrated()
    Q.convert(b, 'identity')
    x, y = _batch(1), _batch(2)
    la = a(x, y)[1]
    lb = b(x, y)[1]
    assert torch.equal(la, lb)
    la.backward()
    lb.backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p.grad, q.grad), n
    assert int(b.layers[0].attn.kvq_calls) == 1 and int(b.layers[0].attn.kvq_changed.sum()) == 0


def test_tracking_mode_initialises_at_the_first_training_forward_and_counts():
    m = _calibrated()
    Q.convert(m, 'kv4')
    attn = m.layers[0].attn
    m.eval()
    with torch.no_grad():
        m(_batch(3))                                   # before initialisation: unchanged
    assert not bool(attn.kvq_init) and int(attn.kvq_pre_init_evals) == 1
    m.train()
    m(_batch(4), _batch(5))
    assert bool(attn.kvq_init) and int(attn.kvq_calls) == 1
    assert (attn.kvq_changed > 0).all()
    m(_batch(6), _batch(7))
    assert int(attn.kvq_calls) == 2


def test_the_running_percentile_is_the_ema_of_batch_percentiles():
    """a <- 0.99 a + 0.01 a_batch per layer, kind and channel; the first batch sets a."""
    m = _calibrated()
    Q.convert(m, 'kv4')
    attn = m.layers[0].attn
    seen = []
    orig = Q.hist_percentile

    def spy(x):
        out = orig(x)
        seen.append(out.clone())
        return out
    Q.hist_percentile = spy
    try:
        with torch.no_grad():
            m(_batch(8))
            m(_batch(9))
    finally:
        Q.hist_percentile = orig
    k1, v1, k2, v2 = seen[0], seen[1], seen[2 * SMALL.n_layers], seen[2 * SMALL.n_layers + 1]
    want = torch.stack([k1, v1]).mul(0.99).add((1 - 0.99) * torch.stack([k2, v2]))
    assert torch.equal(attn.kvq_a, want)
    mm, tt = Q.derive_from_s(Q.step_from_a(want))
    assert torch.equal(attn.kvq_M, mm.float()) and torch.equal(attn.kvq_T, tt.float())


def test_evaluation_uses_the_current_scales_and_does_not_update_them():
    m = _calibrated()
    Q.convert(m, 'kv4')
    m(_batch(1), _batch(2))
    a = m.layers[1].attn.kvq_a.clone()
    m.eval()
    with torch.no_grad():
        m(_batch(3))
    assert torch.equal(m.layers[1].attn.kvq_a, a) and int(m.layers[1].attn.kvq_calls) == 1


def test_the_training_forwards_k_and_v_equal_kv4_decode_of_encode():
    """For fixed tables the training forward's K and V equal kv4.decode(kv4.encode(k8))."""
    m = _calibrated()
    mm, tt = _tables(3, c=SMALL.head_dim)
    Q.convert(m, 'fixed', tables=(np.stack([mm] * SMALL.n_layers),
                                  np.stack([tt] * SMALL.n_layers)))
    got, k8s = {}, {}
    orig = Q.Kv4QatAttention.kv4qat

    def spy(self, k8, v8):
        out = orig(self, k8, v8)
        k8s[id(self)] = (k8.detach().clone(), v8.detach().clone())
        got[id(self)] = (out[0].detach().clone(), out[1].detach().clone())
        return out
    Q.Kv4QatAttention.kv4qat = spy
    try:
        m(_batch(4), _batch(5))
    finally:
        Q.Kv4QatAttention.kv4qat = orig
    for key, (k8, v8) in k8s.items():
        for kind, x in enumerate((k8, v8)):
            x = x.numpy().astype(np.int64)
            want = K.decode(K.encode(x, mm[kind]), np.broadcast_to(tt[kind], (*x.shape, 15)))
            assert np.array_equal(got[key][kind].numpy().astype(np.int64), want)


def test_set_quant_enabled_false_skips_the_fake_quant_on_the_float_path():
    m = _calibrated()
    Q.convert(m, 'kv4')
    M.set_quant_enabled(False)
    try:
        m(_batch(1), _batch(2))
    finally:
        M.set_quant_enabled(True)
    assert int(m.layers[0].attn.kvq_calls) == 0


def test_final_state_reports_integers_and_scales():
    m = _calibrated()
    Q.convert(m, 'kv4')
    m(_batch(1), _batch(2))
    st = Q.final_state(m)
    assert len(st['layers']) == SMALL.n_layers
    lay = st['layers'][0]
    assert np.asarray(lay['M']).shape == (2, SMALL.head_dim)
    assert np.asarray(lay['T']).shape == (2, SMALL.head_dim, 15)
    assert np.array_equal(np.asarray(lay['M']), K.reciprocal(np.asarray(lay['s'])))
    assert np.array_equal(np.asarray(lay['T']), K.table(np.asarray(lay['s'])))
    assert lay['calls'] == 1 and lay['changed']['k'] > 0 and lay['changed']['v'] > 0
    tables = Q.tables_of(st)
    assert tables[0].shape == (SMALL.n_layers, 2, SMALL.head_dim)


# ================================================================ the wrapper (The code change)

W8_SHAPE = ['--dim', '256', '--n-layers', '8', '--n-heads', '8', '--hidden-dim', '576',
            '--ctx', '256', '--pos', 'alibi']
SMALL_ARGS = ['--batch', '2', '--eval-every', '10', '--device', 'cpu', '--amp', 'none',
              '--avg-steps', '10', '--avg-every', '2', '--seed', '11']


def _wrap(tmp_path, name, *extra, mode='kv4', steps=12, shape=None):
    out = tmp_path / name
    argv = [*_data(tmp_path), '--out', str(out), '--steps', str(steps), *SMALL_ARGS,
            *(shape or ['--dim', '24', '--n-layers', '2', '--n-heads', '3', '--hidden-dim',
                        '40', '--ctx', '32', '--pos', 'alibi']), *extra]
    return out, W.main(argv, _mode=mode)


def _plain(tmp_path, name, *extra, steps=12, shape=None):
    out = tmp_path / name
    argv = [*_data(tmp_path), '--out', str(out), '--steps', str(steps), *SMALL_ARGS,
            *(shape or ['--dim', '24', '--n-layers', '2', '--n-heads', '3', '--hidden-dim',
                        '40', '--ctx', '32', '--pos', 'alibi']), *extra]
    return out, tr.main(argv)


def _rec(out):
    return json.loads((out / 'kv4qat.json').read_text())


def test_the_wrapper_writes_kv4qat_json_with_the_plans_fields(tmp_path):
    out, res = _wrap(tmp_path, 'q')
    rec = _rec(out)
    for k in ('mode', 'layers', 'percentile', 'percentile_method', 'histogram_bins', 'momentum',
              'init_step', 'pre_init_eval_rows', 'factory_calls', 'qat_code_sha256',
              'files_sha256', 'expected_calls', 'steps_per_s', 'injection', 'binding'):
        assert k in rec, k
    assert rec['mode'] == 'kv4' and rec['factory_calls'] == 1 and rec['expected_calls'] == 13
    assert rec['init_step'] == 'startup calibration' and rec['pre_init_eval_rows'] == []
    for f in ('log.jsonl', 'inputs.json', 'ckpt.pt', 'avg.pt'):
        assert rec['files_sha256'][f] == hashlib.sha256((out / f).read_bytes()).hexdigest()
    for lay in rec['layers']:
        assert lay['calls'] == 13                       # 12 steps + the start-up calibration
        assert lay['changed']['k'] > 0 and lay['changed']['v'] > 0
    assert rec['qat_code_sha256'] == W.qat_code_sha256()
    assert res['eval_ckpt'] == 'avg.pt'


def test_a_fine_tune_records_its_pre_initialisation_evaluation_rows(tmp_path):
    src, _ = _plain(tmp_path, 'src', steps=4)
    out, _ = _wrap(tmp_path, 'ft', '--init', str(src / 'ckpt.pt'))
    rec = _rec(out)
    assert rec['init_step'] == 0 and rec['expected_calls'] == 12
    rows = rec['pre_init_eval_rows']
    assert len(rows) == 1 and rows[0]['step'] == 0
    first = next(r for r in _log(out) if 'val_loss' in r)
    assert rows[0] == first
    plain_first = next(r for r in _log(src) if 'val_loss' in r and r['step'] == 4)
    assert first['val_loss'] == plain_first['val_loss']      # unchanged K and V before init
    assert all(lay['calls'] == 12 and lay['pre_init_evals'] > 0 for lay in rec['layers'])


def test_the_wrapper_restores_model_trains_transformer(tmp_path):
    _wrap(tmp_path, 'q', steps=2)
    assert tr.Transformer is Transformer


def test_the_identity_mode_is_reachable_from_tests_only(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        W.main(['--help'], _mode='fixed')
    monkeypatch.delitem(sys.modules, 'pytest')
    with pytest.raises(RuntimeError, match='tests only'):
        _wrap(tmp_path, 'i', mode='identity', steps=2)


def test_the_command_line_has_no_mode_option():
    src = Path(W.__file__).read_text()
    assert "'--mode'" not in src and '"--mode"' not in src
    tail = src[src.index("if __name__ == '__main__':"):]
    assert 'from model import kv4qat_train as _w' in tail and '_w.main()' in tail


@pytest.mark.parametrize('extra', [('--float',), ('--qat-start', '2'), ('--avg-steps', '0')])
def test_the_wrapper_refuses_options_outside_e21s_recipes(tmp_path, extra):
    with pytest.raises(SystemExit):
        _wrap(tmp_path, 'bad', *extra, steps=4)


def test_the_post_main_asserts_fail_on_a_second_factory_call(tmp_path, monkeypatch):
    orig = tr._train

    def twice(args):
        tr.Transformer(tr.build_config(args, 30))
        return orig(args)
    monkeypatch.setattr(tr, '_train', twice)
    with pytest.raises(RuntimeError, match='factory'):
        _wrap(tmp_path, 'q', steps=2)


def test_the_post_main_asserts_fail_when_an_attention_is_not_the_qat_class(tmp_path, monkeypatch):
    orig = Q.convert

    def half(model, mode, tables=None):
        orig(model, mode, tables)
        model.layers[0].attn.__class__ = Attention
    monkeypatch.setattr(Q, 'convert', half)
    with pytest.raises(RuntimeError, match='Kv4QatAttention'):
        _wrap(tmp_path, 'q', steps=2)


def test_the_post_main_asserts_fail_on_a_wrong_call_count():
    rec = {'layers': [{'calls': 5, 'changed': {'k': 1, 'v': 1}}]}
    with pytest.raises(RuntimeError, match='calls'):
        W.check_counters(rec['layers'], expected=6, mode='kv4')
    with pytest.raises(RuntimeError, match='changed'):
        W.check_counters([{'calls': 6, 'changed': {'k': 0, 'v': 1}}], expected=6, mode='kv4')
    W.check_counters([{'calls': 6, 'changed': {'k': 0, 'v': 0}}], expected=6, mode='identity')


# ================================================================ Control 6b (CPU, a test)

def _outputs(out):
    log = [{k: v for k, v in r.items() if k != 'elapsed_s'} for r in _log(out)]
    inputs = json.loads((out / 'inputs.json').read_text())
    return log, inputs


def _mapped_inputs(inputs, out, init=None):
    args = dict(inputs['args'], out='<out>')
    rec = dict(inputs, args=args)
    if init is not None:
        args['init'] = '<init>'
        rec['init'] = dict(inputs['init'], path='<init>')
    return rec


def _same_ckpts(a, b):
    for n in ('ckpt.pt', 'avg.pt', 'best.pt'):
        ca, cb = (torch.load(d / n, weights_only=False) for d in (a, b))
        assert _same_states(ca['model'], cb['model']), n
        assert {k: v for k, v in ca.items() if k not in ('model', 'args')} == \
            {k: v for k, v in cb.items() if k not in ('model', 'args')}, n


@pytest.mark.parametrize('fine_tune', [False, True])
def test_control6b_the_identity_wrapper_reproduces_a_plain_run_bit_for_bit(tmp_path, fine_tune):
    """W8-shaped, 30 steps, fp32, no compile, CPU: log.jsonl without elapsed_s, inputs.json after
    mapping out (and init), and every tensor of ckpt.pt and avg.pt."""
    extra, init = (), None
    if fine_tune:
        src, _ = _plain(tmp_path, 'src', steps=3, shape=W8_SHAPE)
        init = src / 'avg.pt'
        extra = ('--init', str(init))
    a, _ = _plain(tmp_path, 'plain', *extra, steps=30, shape=W8_SHAPE)
    b, _ = _wrap(tmp_path, 'wrapped', *extra, mode='identity', steps=30, shape=W8_SHAPE)
    la, ia = _outputs(a)
    lb, ib = _outputs(b)
    assert la == lb
    assert _mapped_inputs(ia, a, init) == _mapped_inputs(ib, b, init)
    _same_ckpts(a, b)
    rec = _rec(b)
    assert rec['mode'] == 'identity' and all(x['calls'] == 30 + (0 if fine_tune else 1)
                                             for x in rec['layers'])


# ================================================================ Control 11 (the positive check)

def test_control11_recomputes_the_logged_final_val_loss_in_fixed_table_mode(tmp_path):
    out, _ = _wrap(tmp_path, 'q', steps=12)
    rec = Q.active_check(out, device='cpu', compile=False, amp='none')
    final = [r for r in _log(out) if r.get('step') == 12 and 'val_loss' in r][0]
    assert rec['logged_val_loss'] == final['val_loss']
    assert abs(rec['recomputed_val_loss'] - final['val_loss']) <= 1e-5
    assert rec['int8_val_loss'] != rec['recomputed_val_loss']
    assert rec['match'] is True and rec['separated'] == (rec['separation'] >= 1e-4)
    assert rec['pass'] == (rec['match'] and rec['separated'])
    assert Q.ACTIVE_MATCH_TOL == 1e-5 and Q.ACTIVE_SEPARATION == 1e-4


def test_control11_fails_when_the_tables_are_not_the_trained_ones(tmp_path):
    out, _ = _wrap(tmp_path, 'q', steps=12)
    rec = _rec(out)
    for lay in rec['layers']:
        lay['M'] = [[65535] * len(r) for r in lay['M']]
    (out / 'kv4qat.json').write_text(json.dumps(rec))
    got = Q.active_check(out, device='cpu', compile=False, amp='none')
    assert got['match'] is False and got['pass'] is False


# ================================================================ CUDA (slow): Control 7, 6d

@slow
@cuda
def test_control7_module_level_through_kv4qatattention_under_compile_and_bf16():
    """E19's stored W8 tables in fixed-table mode; all 256 int8 inputs per channel fed as k8 and
    v8 through the real Kv4QatAttention's compiled call under bf16 autocast; the decoded values
    equal kv4's and the counters (non-persistent buffers) move."""
    cfg = ModelConfig(vocab_size=812, dim=256, n_layers=8, n_heads=8, hidden_dim=576,
                      max_seq_len=256, pos='alibi')
    x = np.arange(-128, 128)
    n = 0
    for p in E19_TABLES:
        tb = K.read_tables(p)
        model = Transformer(cfg)
        Q.convert(model, 'fixed', tables=(tb.m, tb.t))
        model = model.cuda().train()
        for layer, blk in enumerate(model.layers):
            attn = blk.attn
            assert {'kvq_a', 'kvq_M', 'kvq_T', 'kvq_calls', 'kvq_changed'} <= \
                attn._non_persistent_buffers_set
            fn = torch.compile(attn.kv4qat)
            k8 = torch.tensor(np.broadcast_to(x[None, :, None], (1, 256, 32)).copy(),
                              dtype=torch.bfloat16, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                k, v = fn(k8, k8.clone())
            for kind, got in enumerate((k, v)):
                got = got[0].float().cpu().numpy().astype(np.int64)
                for c in range(32):
                    want = K.decode(K.encode(x, tb.m[layer, kind, c]),
                                    np.broadcast_to(tb.t[layer, kind, c], (256, 15)))
                    assert np.array_equal(got[:, c], want), (p.name, layer, kind, c)
                    n += 256
            assert int(attn.kvq_calls) == 1 and (attn.kvq_changed > 0).all()
    assert n == 2_228_224


E16_W8_S10 = RUNS / 'e16' / 'train' / 'W8-s10' / 'train'


@slow
@cuda
def test_control6d_the_identity_wrapper_on_the_gpu_matches_e16s_step0_rows(tmp_path):
    """W8-s10's E16 command (60,000-step args) with --steps 250 through the wrapper with the
    identity quantizer: step-0 val_loss, val_bos_loss and train_loss within 1e-6 of E16's."""
    args = json.loads((E16_W8_S10 / 'inputs.json').read_text())['args']
    argv = W.argv_from_args(args, out=tmp_path / 'id', steps=250,
                            data=RUNS / 'e4' / 'data' / 'SEED', vocab=ROOT / 'corpus' / 'vocab.json')
    W.main(argv, _mode='identity')
    ref = {k: v for r in _log(E16_W8_S10) if r.get('step') == 0 for k, v in r.items()
           if k in ('val_loss', 'val_bos_loss', 'train_loss')}
    got = {k: v for r in _log(tmp_path / 'id') if r.get('step') == 0 for k, v in r.items()
           if k in ('val_loss', 'val_bos_loss', 'train_loss')}
    assert set(ref) == set(got) == {'val_loss', 'val_bos_loss', 'train_loss'}
    assert all(abs(got[k] - ref[k]) <= 1e-6 for k in ref), (ref, got)


# ================================================================ review fixes

def test_the_tracking_mode_compiles_without_graph_breaks_and_equals_eager():
    """The training-mode fake quant (histogram, EMA, M and T) traces as one graph (fullgraph: a
    numpy scalar in the percentile index would branch on data), with buffers equal to eager."""
    import copy
    torch._dynamo.reset()
    a = _calibrated()
    Q.convert(a, 'kv4')
    b = copy.deepcopy(a)
    run = torch.compile(b, fullgraph=True, backend='aot_eager')
    for i in range(3):
        la = a(_batch(10 + i), _batch(20 + i))[1]
        lb = run(_batch(10 + i), _batch(20 + i))[1]
        la.backward()
        lb.backward()
        assert torch.equal(la, lb)
    for x, y in zip(a.layers, b.layers):
        for name in Q.BUFFERS:
            assert torch.equal(getattr(x.attn, name), getattr(y.attn, name)), name
    assert type(Q.QUANTILE) is float and Q.QUANTILE == np.true_divide(99.9, 100)


def test_a_nan_in_k8_does_not_crash_the_histogram():
    x = torch.tensor([[1.], [float('nan')], [3.]])
    assert torch.isfinite(Q.hist_percentile(x)).all()


def test_the_wrapper_asserts_the_device_tables_match_the_recorded_integers():
    lay = {'calls': 6, 'changed': {'k': 1, 'v': 1}, 'buffer_M_equal': True,
           'buffer_T_equal': False, 'initialised': True}
    with pytest.raises(RuntimeError, match='buffer'):
        W.check_counters([lay], expected=6, mode='kv4')
    with pytest.raises(RuntimeError, match='initialised'):
        W.check_counters([lay | {'buffer_T_equal': True, 'initialised': False}], expected=6,
                         mode='kv4')
    W.check_counters([lay | {'buffer_T_equal': True}], expected=6, mode='kv4')


def test_control11_recomputes_with_tf32_as_training_sets_it(tmp_path, monkeypatch):
    out, _ = _wrap(tmp_path, 'q', steps=4)
    torch.backends.cuda.matmul.allow_tf32 = False
    Q.active_check(out, device='cpu', compile=False, amp='none')
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_a_nan_in_k8_does_not_crash_the_fake_quant():
    m, t = _tables(4, c=2)
    x = torch.tensor([[1., float('nan')]])
    mm, tt = torch.tensor(m[0]).float(), torch.tensor(t[0]).float()
    Q.fake_quant(x, mm, tt)
    Q.ste(x, mm, tt)
