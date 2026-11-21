"""CalcLM 4-bit K/V integer reference (PLAN-E19.md "The kv4 scheme", revision 2.1).

The K and V caches hold 15 levels per channel. Per layer l, kind (K or V) and channel c the
scheme stores two integer arrays, computed once from a calibration on SEED training programs:

- a reciprocal M[l][kind][c] = min(round(65536 / s), 65535), a uint16 with 16 fractional bits;
- a table T[l][kind][c][j] = sign(j) * min(127, rint(|j| * s)) for j = -7..7 (rint: half to even),

where s = max(a / 7, 1) and a is the 99.9th percentile (numpy, linear, float64) of |x| over every
k8 / v8 the unchanged int8 intref (horizons on) writes on 64 SEED programs
(default_rng(1901).permutation of model.evaluate.val_programs, each cut to 512 tokens).

Encode, integer only, for an int8 x from scale_out: p = |x| M, q = p >> 16, r = p & 0xFFFF,
j = q + (r > 32768 or (r == 32768 and q odd)), clamped to 7, x's sign; decode is T[j]. p <= 127 *
65535 < 2^23. Quantization applies at the cache write, to K and V, in every layer (the current
token's K and V too, since intref reads every key back from the cache); everything else is intref.

The hook is model.new_cache(): Kv4Model is IntModel with a cache whose K and V rings decode(encode)
on write. OracleModel is the independent check (Control 7): a probe-style IntModel subclass that
copies `attention` and applies a float64 rint encode and the stored tables inline at the write.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.data import load_split
from model.export import load_export
from model.intops import clamp, norm_quant, quant_vec, rsr, scale_out
from model.intref import INT24, IntModel, KVCache, alibi_bias, check_bits
from model.pack import lut_matvec
from model.tokens import EOS

LEVELS = 7                     # codes -7..7: 15 levels
PERCENTILE = 99.9
PERCENTILE_METHOD = 'linear'
STEP_FLOOR = 1.0
FRAC_BITS = 16
ONE = 1 << FRAC_BITS           # 65536
HALF = ONE >> 1                # 32768
MASK = ONE - 1
M_MAX = ONE - 1                # 65535: the reciprocal is a uint16
KINDS = ('k', 'v')
MODE_KINDS = {'identity': (), 'kv4': ('k', 'v'), 'k4': ('k',), 'v4': ('v',)}
CAL_RNG, CAL_N, CAL_LEN = 1901, 64, 512
NLL_RNG, NLL_N, NLL_CAP = 1902, 128, 1024
_CTX = mp.get_context('spawn')


# ---------------------------------------------------------------- the scheme's integers

def step(a) -> np.ndarray:
    """s = max(a / 7, 1), float64."""
    return np.maximum(np.asarray(a, dtype=np.float64) / LEVELS, STEP_FLOOR)


def reciprocal(s) -> np.ndarray:
    """M = min(round(65536 / s), 65535) (round: numpy rint), int64 holding a uint16."""
    return np.minimum(np.rint(ONE / np.asarray(s, dtype=np.float64)), M_MAX).astype(np.int64)


def table(s) -> np.ndarray:
    """T[..., j + 7] = sign(j) min(127, rint(|j| s)) for j = -7..7, int64 (..., 15)."""
    j = np.arange(-LEVELS, LEVELS + 1)
    s = np.asarray(s, dtype=np.float64)[..., None]
    return (np.sign(j) * np.minimum(127, np.rint(np.abs(j) * s))).astype(np.int64)


def tables_from_values(values) -> dict:
    """a, s, M, T from every collected int8 value, values (positions, ..., channels)."""
    a = np.percentile(np.abs(np.asarray(values, dtype=np.float64)), PERCENTILE, axis=0,
                      method=PERCENTILE_METHOD)
    s = step(a)
    return {'a': a, 's': s, 'M': reciprocal(s), 'T': table(s)}


def encode(x, m) -> np.ndarray:
    """The 4-bit code of int8 x with reciprocal m (broadcast per channel); integers only."""
    x, m = np.asarray(x), np.asarray(m)
    if x.dtype.kind not in 'iu' or m.dtype.kind not in 'iu':
        raise TypeError('encode takes integers only')
    x, m = x.astype(np.int64), m.astype(np.int64)
    p = np.abs(x) * m
    q, r = p >> FRAC_BITS, p & MASK
    j = q + ((r > HALF) | ((r == HALF) & ((q & 1) == 1)))
    return np.sign(x) * np.minimum(j, LEVELS)


def decode(j, t) -> np.ndarray:
    """T[j] per channel: j (..., channels), t (..., channels, 15)."""
    j = np.asarray(j, dtype=np.int64)
    return np.take_along_axis(np.asarray(t), (j + LEVELS)[..., None], axis=-1)[..., 0]


def ref_encode(x, m) -> np.ndarray:
    """Control 7's independent reference: clip(sign(x) rint(|x| M / 65536), -7, 7) in float64
    (exact: |x| M / 65536 is a dyadic fraction far below 2^53)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    return np.clip(np.sign(x) * np.rint(np.abs(x) * m / ONE), -LEVELS, LEVELS).astype(np.int64)


def exhaustive_check(m, t) -> dict:
    """Every stored (layer, kind, channel): all 256 int8 inputs through E19's encode and decode
    against ref_encode and the same table (Control 7)."""
    m, t = np.asarray(m, dtype=np.int64), np.asarray(t, dtype=np.int64)
    x = np.arange(-128, 128, dtype=np.int64)
    bad, n = [], 0
    for idx in np.ndindex(m.shape):
        j = encode(x, m[idx])
        want = ref_encode(x, m[idx])
        row = np.broadcast_to(t[idx], (len(x), 2 * LEVELS + 1))     # one channel, 256 inputs
        got, ref = decode(j, row), t[idx][want + LEVELS]
        n += len(x)
        if not (np.array_equal(j, want) and np.array_equal(got, ref)):
            k = int(np.flatnonzero((j != want) | (got != ref))[0])
            bad.append({'index': [int(i) for i in idx], 'x': int(x[k]), 'code': int(j[k]),
                        'ref_code': int(want[k])})
    return {'pass': not bad, 'n_checked': n, 'mismatches': bad[:20], 'n_mismatched': len(bad)}


# ---------------------------------------------------------------- programs

def program_bounds(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """model.evaluate.val_programs's boundaries: <bos> to before the next <bos>."""
    starts = np.flatnonzero(stream == 1)
    return starts, np.append(starts[1:], len(stream))


def select_programs(stream, seed: int, n: int, cut: int | None = None) -> list[list[int]]:
    """val_programs(stream) permuted by default_rng(seed), the first n, each cut to `cut` tokens
    (equal to val_programs without building every program as a list)."""
    stream = np.asarray(stream)
    starts, ends = program_bounds(stream)
    order = np.random.default_rng(seed).permutation(len(starts))[:n]
    out = []
    for i in order:
        a, b = int(starts[i]), int(ends[i])
        if cut is not None:
            b = min(b, a + cut)
        out.append(stream[a:b].tolist())
    return out


def calibration_programs(train_bin) -> list[list[int]]:
    return select_programs(load_split(train_bin), CAL_RNG, CAL_N, CAL_LEN)


def nll_programs(val_bin) -> list[list[int]]:
    return select_programs(load_split(val_bin), NLL_RNG, NLL_N, NLL_CAP)


def programs_sha256(progs) -> str:
    return hashlib.sha256(json.dumps([list(map(int, p)) for p in progs]).encode()).hexdigest()


# ---------------------------------------------------------------- the cache (the hook)

class Quantizer:
    """decode(encode(x)) for the kinds it quantizes, x unchanged for the others. count: per layer
    and kind, [codes saturated at +-7 with |x| beyond T[+-7], values written]. record: a list
    that receives (layer, kind, written value) for every write."""

    def __init__(self, m, t, kinds, count: bool = False, record: list | None = None,
                 n_layers: int | None = None):
        self.m = None if m is None else np.asarray(m, dtype=np.int64)
        self.t = None if t is None else np.asarray(t, dtype=np.int64)
        self.kinds = tuple(kinds)
        if self.kinds and self.m is None:
            raise ValueError('a 4-bit quantizer needs the reciprocals and tables')
        n = n_layers if n_layers is not None else (0 if self.m is None else self.m.shape[0])
        self.count = count
        self.counts = {k: np.zeros((n, 2), dtype=np.int64) for k in KINDS}
        self.record = record

    def __call__(self, layer: int, kind: str, x) -> np.ndarray:
        x = np.asarray(x)
        if kind in self.kinds:
            ki = KINDS.index(kind)
            m, t = self.m[layer, ki], self.t[layer, ki]
            j = encode(x, m)
            out = decode(j, t)
            if self.count:
                clip = ((j == LEVELS) & (x > t[:, -1])) | ((j == -LEVELS) & (x < t[:, 0]))
                self.counts[kind][layer] += (int(clip.sum()), x.size)
        else:
            out = x.astype(np.int64)
            if self.count:
                self.counts[kind][layer, 1] += x.size
        if self.record is not None:
            self.record.append((layer, kind, out.copy()))
        return out


class QuantRing:
    """One of K or V, (n_layers, ctx, head_dim) int64. intref writes it only as ring[i, slot] =
    row (quantized here) and reads ring[i, :n]; any other write is refused."""

    def __init__(self, shape, quant: Quantizer, kind: str):
        self.a = np.zeros(shape, dtype=np.int64)
        self.quant, self.kind = quant, kind

    def __setitem__(self, key, value) -> None:
        if not (isinstance(key, tuple) and len(key) == 2 and
                all(isinstance(k, (int, np.integer)) for k in key)):
            raise TypeError(f'the 4-bit cache takes one (layer, slot) row per write, not {key!r}')
        self.a[key] = self.quant(int(key[0]), self.kind, value)

    def __getitem__(self, key) -> np.ndarray:
        return self.a[key]


# ---------------------------------------------------------------- the tables file

@dataclass(frozen=True)
class Tables:
    m: np.ndarray
    t: np.ndarray
    export_sha256: str
    export_meta_sha256: str
    sha256: str
    path: str

    @property
    def M(self) -> np.ndarray:            # noqa: N802 - the plan's names
        return self.m

    @property
    def T(self) -> np.ndarray:            # noqa: N802
        return self.t


def _sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tables_sha256(path) -> str:
    return _sha(path)


def write_tables(path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)


def read_tables(path, export=None) -> Tables:
    """The stored integers; with export, refused unless the tables' recorded export sha256
    equals the export's model.bin and model.json."""
    path = Path(path)
    raw = path.read_bytes()
    rec = json.loads(raw)
    if export is not None:
        ex = Path(export)
        got = (_sha(ex / 'model.bin'), _sha(ex / 'model.json'))
        if (rec.get('export_sha256'), rec.get('export_meta_sha256')) != got:
            raise ValueError(f'{path}: calibrated on export {rec.get("export_sha256")}, not '
                             f'{export} ({got[0]}): the 4-bit model refuses to build')
    m, t = np.asarray(rec['M'], dtype=np.int64), np.asarray(rec['T'], dtype=np.int64)
    if m.ndim != 3 or t.shape != (*m.shape, 2 * LEVELS + 1) or m.shape[1] != 2:
        raise ValueError(f'{path}: tables of shape {m.shape} / {t.shape}')
    if m.min() < 1 or m.max() > M_MAX or np.abs(t).max() > 127:
        raise ValueError(f'{path}: a reciprocal or table entry out of range')
    return Tables(m, t, rec['export_sha256'], rec['export_meta_sha256'],
                  hashlib.sha256(raw).hexdigest(), str(path))


# ---------------------------------------------------------------- the 4-bit model

class Kv4Model(IntModel):
    """IntModel with E19's cache: mode 'kv4' (K and V 4-bit), 'k4', 'v4' or 'identity' (the
    cache class with encode and decode replaced by the identity: Control 3)."""

    def __init__(self, ex, tables: Tables | None, mode: str = 'kv4', horizons: bool = True,
                 count: bool = False, record: bool = False):
        if mode not in MODE_KINDS:
            raise ValueError(f'kv4 modes are {sorted(MODE_KINDS)}, not {mode!r}')
        if MODE_KINDS[mode] and tables is None:
            raise ValueError(f'mode {mode} needs the checkpoint\'s tables')
        super().__init__(ex, horizons=horizons)
        if tables is not None and tables.m.shape != (self.cfg.n_layers, 2, self.cfg.head_dim):
            raise ValueError(f'tables of shape {tables.m.shape} for a model with '
                             f'{self.cfg.n_layers} layers and head_dim {self.cfg.head_dim}')
        self.kv, self.tables = mode, tables
        self.records: list | None = [] if record else None
        self.quant = Quantizer(None if tables is None else tables.m,
                               None if tables is None else tables.t, MODE_KINDS[mode], count,
                               self.records, self.cfg.n_layers)

    @classmethod
    def load(cls, export, tables_path, mode: str = 'kv4', **kw) -> 'Kv4Model':
        tables = None if tables_path is None else read_tables(tables_path, export)
        if mode not in MODE_KINDS:
            raise ValueError(f'kv4 modes are {sorted(MODE_KINDS)}, not {mode!r}')
        return cls(load_export(export), tables, mode, **kw)

    def new_cache(self) -> KVCache:
        shape = (self.cfg.n_layers, self.cfg.max_seq_len, self.cfg.head_dim)
        return KVCache(QuantRing(shape, self.quant, 'k'), QuantRing(shape, self.quant, 'v'))


class _Recorder(IntModel):
    """The unchanged int8 IntModel; its new_cache records every K and V write (calibration)."""

    def __init__(self, ex, horizons: bool = True):
        super().__init__(ex, horizons=horizons)
        self.records: list = []
        self.quant = Quantizer(None, None, (), record=self.records)

    def new_cache(self) -> KVCache:
        shape = (self.cfg.n_layers, self.cfg.max_seq_len, self.cfg.head_dim)
        return KVCache(QuantRing(shape, self.quant, 'k'), QuantRing(shape, self.quant, 'v'))


# ---------------------------------------------------------------- the float oracle (Control 7)

class OracleModel(IntModel):
    """A separate probe-style IntModel subclass (as experiments/v3-angles/cost-frontier/
    attn_probe.py): `attention` copied from intref, with the float64 rint encode of the stored
    reciprocals and the stored tables applied inline at the cache write. Its cache is
    IntModel's plain KVCache, not E19's cache class."""

    def __init__(self, ex, tables: Tables, horizons: bool = True):
        super().__init__(ex, horizons=horizons)
        self.m = tables.m.astype(np.float64)
        self.t = tables.t
        self.records: list = []

    @classmethod
    def load(cls, export, tables_path, **kw) -> 'OracleModel':
        return cls(load_export(export), read_tables(tables_path, export), **kw)

    def _q(self, i: int, kind: int, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        j = np.clip(np.sign(x) * np.rint(np.abs(x) * self.m[i, kind] / ONE), -LEVELS, LEVELS)
        out = self.t[i, kind][np.arange(len(x)), j.astype(np.int64) + LEVELS]
        self.records.append((i, KINDS[kind], out.copy()))
        return out

    def attention(self, i: int, lp, r, pos: int, cache: KVCache, counter) -> np.ndarray:
        cfg = self.cfg
        win = cfg.max_seq_len
        x8, sx = norm_quant(r, lp.attn_g, lp.attn_eg, counter, self.sqrtd)
        q8, k8, v8 = (scale_out(lut_matvec(w.w, x8), w.m, w.e, sx, 0, -127, 127, counter)
                      for w in (lp.wq, lp.wk, lp.wv))
        slot, n = pos % win, min(pos + 1, win)
        cache.k[i, slot], cache.v[i, slot] = self._q(i, 0, k8), self._q(i, 1, v8)
        keys, vals = cache.k[i, :n], cache.v[i, :n]
        q = q8.reshape(cfg.n_heads, cfg.head_dim)
        scores = q @ keys.T
        use = None
        if self.alibi_exponents is not None:
            dist = (slot - np.arange(n)) % win
            scores = scores - alibi_bias(self.alibi_exponents, dist, lp.sh, cfg.lut_max)
            if self.horizons is not None:
                use = dist[None, :] < self.horizons[:, None]
                scores = np.where(use, scores, 1 - (1 << 23))
            scores = check_bits(scores, 23, "ALiBi scores")
        mults = q.size * n if use is None else cfg.head_dim * int(use.sum())
        counter.add("attn_score", mults)
        idx = np.minimum(rsr(scores.max(axis=1, keepdims=True) - scores, lp.sh), len(lp.lut) - 1)
        e = lp.lut[idx]
        if use is not None:
            e = np.where(use, e, 0)
        a = e @ vals
        counter.add("attn_pv", mults)
        o = np.concatenate([self._head_avg(a[h], int(e[h].sum()), counter)
                            for h in range(cfg.n_heads)])
        o8, so = quant_vec(o, (lp.sv[0], lp.sv[1] - 8), counter)
        delta = scale_out(lut_matvec(lp.wo.w, o8), lp.wo.m, lp.wo.e, so, self.F, *INT24, counter)
        return clamp(r + delta, *INT24)


def _nll_of(lg: np.ndarray, target: int) -> float:
    """Nats of target under Q8 logits: float64 log-softmax of logits / 256, as attn_probe."""
    lg = np.asarray(lg, dtype=np.float64) / 256.0
    lg -= lg.max()
    return float(np.log(np.exp(lg).sum()) - lg[target])


def lockstep(model: Kv4Model, oracle: OracleModel, ids) -> dict:
    """Feed ids[:-1] to both; at every position the int64 logits and every decoded K and V write
    must be equal, bit for bit. Also each model's summed NLL over positions 1..len(ids) - 1."""
    if model.records is None:
        raise ValueError('lockstep needs Kv4Model(record=True)')
    a, b = model.new_cache(), oracle.new_cache()
    first, n_writes, ta, tb = None, 0, 0., 0.
    for p, tok in enumerate(ids[:-1]):
        model.records.clear()
        oracle.records.clear()
        la, lb = model.step(int(tok), p, a), oracle.step(int(tok), p, b)
        ra, rb = model.records, oracle.records
        n_writes += len(ra)
        same = np.array_equal(la, lb) and len(ra) == len(rb) and all(
            x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) for x, y in zip(ra, rb))
        if not same and first is None:
            first = {'position': p, 'logits_equal': bool(np.array_equal(la, lb))}
        ta += _nll_of(la, ids[p + 1])
        tb += _nll_of(lb, ids[p + 1])
    model.records.clear()
    oracle.records.clear()
    n = max(len(ids) - 1, 0)
    return {'agree': first is None, 'first_mismatch': first, 'n_steps': n,
            'n_writes': n_writes, 'nll': {'kv4': (ta, n), 'oracle': (tb, n)}}


# ---------------------------------------------------------------- calibration

def _collect_one(ex, prog) -> np.ndarray:
    m = _Recorder(ex, horizons=True)
    cache = m.new_cache()
    out = np.zeros((len(prog), m.cfg.n_layers, 2, m.cfg.head_dim), dtype=np.int8)
    for p, tok in enumerate(prog):
        m.records.clear()
        m.step(int(tok), p, cache)
        for layer, kind, row in m.records:
            out[p, layer, KINDS.index(kind)] = row
    return out


_EX = None


def _collect_init(export: str) -> None:
    global _EX
    _EX = load_export(export)


def _collect_job(prog) -> np.ndarray:
    return _collect_one(_EX, prog)


def collect(export, programs, workers: int = 1) -> np.ndarray:
    """Every layer's k8 and v8 at every position of every program, teacher-forced through the
    unchanged int8 intref with horizons: (tokens, layers, 2, head_dim) int8, program order."""
    if workers <= 1:
        ex = load_export(export)
        parts = [_collect_one(ex, p) for p in programs]
    else:
        with ProcessPoolExecutor(workers, mp_context=_CTX, initializer=_collect_init,
                                 initargs=(str(export),)) as pool:
            parts = list(pool.map(_collect_job, programs, chunksize=1))
    return np.concatenate(parts)


def calibrate(export, train_bin, workers: int = 1) -> dict:
    """The tables record of one checkpoint (JSON-ready): the scheme, the export's identity, the
    calibration programs' hash and token count, and per layer, kind and channel a, s, M, T."""
    ex = Path(export)
    progs = calibration_programs(train_bin)
    vals = collect(ex, progs, workers)
    t = tables_from_values(vals)
    return {'scheme': {'levels': LEVELS, 'percentile': PERCENTILE, 'step_floor': STEP_FLOOR,
                       'frac_bits': FRAC_BITS, 'm_max': M_MAX, 'rng': CAL_RNG,
                       'n_programs': CAL_N, 'cut': CAL_LEN, 'horizons': True,
                       'percentile_method': PERCENTILE_METHOD},
            'export': str(export), 'export_sha256': _sha(ex / 'model.bin'),
            'export_meta_sha256': _sha(ex / 'model.json'), 'train_bin': str(train_bin),
            'programs_sha256': programs_sha256(progs), 'n_programs': len(progs),
            'n_tokens': int(vals.shape[0]), 'n_layers': int(vals.shape[1]),
            'head_dim': int(vals.shape[3]),
            'a': t['a'].tolist(), 's': t['s'].tolist(), 'M': t['M'].tolist(),
            'T': t['T'].tolist()}


# ---------------------------------------------------------------- NLL, the oracle, clip shares

def program_nll(model: IntModel, ids) -> tuple[float, int]:
    """Summed nats over positions 1..len(ids) - 1 (ids already cut to NLL_CAP): float64
    log-softmax of the Q8 logits / 256, as attn_probe."""
    cache = model.new_cache()
    total = 0.
    for p, tok in enumerate(ids[:-1]):
        total += _nll_of(model.step(int(tok), p, cache), ids[p + 1])
    return total, max(len(ids) - 1, 0)


_M: dict = {}


def _nll_init(export: str, tables_path: str) -> None:
    ex = load_export(export)
    tables = read_tables(tables_path, export)
    _M.clear()
    _M['int8'] = IntModel(ex, horizons=True)
    _M['kv4'] = Kv4Model(ex, tables, 'kv4', count=True, record=True)
    _M['k4'] = Kv4Model(ex, tables, 'k4')
    _M['v4'] = Kv4Model(ex, tables, 'v4')
    _M['oracle'] = OracleModel(ex, tables)


def _nll_job(ids) -> dict:
    out = {name: program_nll(_M[name], ids) for name in ('int8', 'k4', 'v4')}
    kv, oracle = _M['kv4'], _M['oracle']
    kv.quant.counts = {k: np.zeros_like(v) for k, v in kv.quant.counts.items()}
    lock = lockstep(kv, oracle, ids)
    out |= lock['nll']
    return {'nll': out, 'agree': lock['agree'], 'first_mismatch': lock['first_mismatch'],
            'n_writes': lock['n_writes'],
            'clip': {k: v.tolist() for k, v in kv.quant.counts.items()}}


def _shares(counts) -> dict:
    """{kind: [{'layer', 'clip', 'total', 'share'}]} from summed [clip, total] per layer."""
    return {k: [{'layer': i, 'clip': int(c), 'total': int(n), 'share': c / n if n else None}
                for i, (c, n) in enumerate(np.asarray(v).tolist())] for k, v in counts.items()}


def _pool_map(init, initargs, job, items, workers: int) -> list:
    if workers <= 1:
        init(*initargs)
        return [job(x) for x in items]
    with ProcessPoolExecutor(workers, mp_context=_CTX, initializer=init,
                             initargs=initargs) as pool:
        return list(pool.map(job, items, chunksize=1))


def nll_record(export, tables_path, val_bin, workers: int = 1) -> dict:
    """Measurement 6, 7 and Control 7 on one checkpoint's NLL set: mean nats per token int8,
    4-bit K and V, 4-bit K only, 4-bit V only and the oracle; the oracle's agreement (decoded K
    and V at every write and int64 logits at every position); the clip shares per layer and
    kind under 4-bit K and V. Program order fixes the float sums: the worker count does not
    change the record."""
    progs = nll_programs(val_bin)
    res = _pool_map(_nll_init, (str(export), str(tables_path)), _nll_job, progs, workers)
    names = ('int8', 'kv4', 'k4', 'v4', 'oracle')
    sums = {n: sum(r['nll'][n][0] for r in res) for n in names}
    ntok = sum(r['nll']['int8'][1] for r in res)
    counts = {k: np.sum([r['clip'][k] for r in res], axis=0) for k in KINDS}
    bad = [{'program': i, **r['first_mismatch']} for i, r in enumerate(res) if not r['agree']]
    return {'nll': {n: sums[n] / ntok for n in names}, 'n_tokens': int(ntok),
            'n_programs': len(progs), 'programs_sha256': programs_sha256(progs),
            'rng': NLL_RNG, 'cap': NLL_CAP, 'horizons': True,
            'oracle': {'agree': not bad, 'mismatches': bad[:20],
                       'n_writes': int(sum(r['n_writes'] for r in res))},
            'clip': _shares(counts), 'tables_sha256': tables_sha256(tables_path)}


def _replay_init(export: str, tables_path: str) -> None:
    _M.clear()
    _M['kv4'] = Kv4Model.load(export, tables_path, 'kv4', count=True)


def _replay_job(ids) -> dict:
    m = _M['kv4']
    m.quant.counts = {k: np.zeros_like(v) for k, v in m.quant.counts.items()}
    feed = ids[:-1] if ids and ids[-1] == EOS else ids
    cache = m.new_cache()
    for p, tok in enumerate(feed):
        m.step(int(tok), p, cache)
    return {k: v.tolist() for k, v in m.quant.counts.items()}


def replay_clip(export, tables_path, id_lists, workers: int = 1) -> dict:
    """Measurement 7's dev clip share: each generated sample's ids teacher-forced through the
    4-bit model with a counting cache, fed as generation fed them (every token but a final
    <eos>); the samplers stay untouched."""
    res = _pool_map(_replay_init, (str(export), str(tables_path)), _replay_job,
                    [list(map(int, x)) for x in id_lists], workers)
    counts = {k: np.sum([r[k] for r in res], axis=0) for k in KINDS}
    return {'clip': _shares(counts), 'n_samples': len(res),
            'tables_sha256': tables_sha256(tables_path)}
