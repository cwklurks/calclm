"""E21's 4-bit K/V fake quant for training (PLAN-E21.md "The training-side design", revision 2.1).

In the training forward, after static_round and before attention_core, k8 and v8 are replaced by
their 4-bit decoded values per layer, kind and channel, exactly as model/kv4.py encodes and
decodes the cache writes:

- Exact encode for given tables. From the integers M and T (derived from s as kv4.reciprocal and
  kv4.table derive them), in float32 with autocast off: p = |x| M, q = floor(p / 65536),
  r = p - 65536 q, j = q + (r > 32768 or (r = 32768 and q odd)), clamped to 7, x's sign, then
  T[j]. x is integer-valued after static_round and p <= 128 x 65,535 < 2^24, so float32 holds
  every intermediate exactly: for fixed tables the result equals kv4.decode(kv4.encode(x)) bit
  for bit (Control 7).
- Gradients: a clipped straight-through estimator, as static_round: forward T[j], gradient 1 for
  |x| <= |T[+-7]| and 0 beyond.
- Scales in training ('kv4', the tracking mode): per layer, kind and channel, a_batch = the 99.9th
  percentile of |x| over the batch's positions by numpy's linear definition, computed exactly from
  a 128-bin histogram (x is integer-valued); a <- 0.99 a + (1 - 0.99) a_batch (the q/k/v absmax
  EMA's form and momentum), in training mode only; the first training-mode forward sets a =
  a_batch (the start-up calibration batch from scratch, the first training batch for a fine-tune).
  Resolution: the update comes first and the forward quantizes with the updated tables, so the
  initialising forward is defined. s = max(a / 7, 1) and M, T are re-derived from s in float64 on
  every update. Before initialisation (only a fine-tune's step-0 evaluation) K and V pass
  unchanged. Evaluation uses the current tables and updates nothing.
- 'fixed': given tables (Control 7, Control 11, the 4-bit agreement), never updated.
- 'identity': K and V returned unchanged (the same tensors), calls still counted; reachable from
  tests only (model.kv4qat_train refuses it outside pytest).
- State: a, M, T, the initialised flag and the counters (training-mode calls; values changed,
  per kind; pre-initialisation evaluation forwards) are non-persistent buffers of each
  Kv4QatAttention: they live on the device, update inside the compiled graph (branch-free) and
  never enter the state dict.

Kv4QatAttention is model.model.Attention with `forward` copied unchanged and exactly one inserted
statement (INJECTION) right after q8, k8, v8 land on their int8 grids. The copied method runs in
model.model's global namespace, so QUANT_ENABLED and every helper resolve as in the original.
Binding (resolution): convert() swaps each block's attention instance's class (attribute class
swap) and registers the buffers; no state-dict key is added.
"""
from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import torch

from model import kv4 as K
from model import model as M
from model.model import Attention

PERCENTILE = K.PERCENTILE                     # 99.9
PERCENTILE_METHOD = K.PERCENTILE_METHOD       # numpy 'linear'
# numpy's q, 0.9990000000000001; a Python float, so the percentile index is a trace-time
# constant under torch.compile (a numpy scalar would be traced as data and branch)
QUANTILE = float(np.true_divide(PERCENTILE, 100))
MOMENTUM = 0.99
LEVELS = K.LEVELS                             # codes -7..7
BINS = 128                                    # |x| in 0..127 after static_round
ONE, HALF = float(K.ONE), float(K.HALF)
M_MAX = float(K.M_MAX)
MODES = ('kv4', 'fixed', 'identity')
INJECTION = 'k8, v8 = self.kv4qat(k8, v8)'
BUFFERS = ('kvq_a', 'kvq_M', 'kvq_T', 'kvq_init', 'kvq_calls', 'kvq_changed',
           'kvq_pre_init_evals')
ACTIVE_MATCH_TOL, ACTIVE_SEPARATION = 1e-5, 1e-4      # Control 11


# ---------------------------------------------------------------- the scheme's integers

def step_from_a(a: torch.Tensor) -> torch.Tensor:
    """s = max(a / 7, 1), float64 (kv4.step)."""
    return torch.clamp(a.to(torch.float64) / LEVELS, min=K.STEP_FLOOR)


def derive_from_s(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M = min(rint(65536 / s), 65535) and T[..., j + 7] = sign(j) min(127, rint(|j| s)), float64
    holding integers (kv4.reciprocal, kv4.table; torch.round is half to even, as numpy's rint)."""
    s = s.to(torch.float64)
    m = torch.clamp(torch.round(ONE / s), max=M_MAX)
    j = torch.arange(-LEVELS, LEVELS + 1, dtype=torch.float64, device=s.device)
    t = torch.sign(j) * torch.clamp(torch.round(j.abs() * s[..., None]), max=127.)
    return m, t


def _virtual_index(n: int) -> tuple[int, int, float]:
    """numpy's linear percentile: the neighbouring order statistics of (n - 1) q and the weight."""
    vi = (n - 1) * QUANTILE
    if vi >= n - 1:
        return n - 1, n - 1, 0.
    lo = int(math.floor(vi))
    return lo, lo + 1, float(vi - lo)


def hist_percentile(absx: torch.Tensor) -> torch.Tensor:
    """The 99.9th percentile (numpy, linear, float64) of each column of absx (positions,
    channels), whose entries are integers in 0..127: a 128-bin histogram per channel, the two
    order statistics around (n - 1) q from the cumulative counts, numpy's lerp."""
    n, c = absx.shape
    idx = absx.nan_to_num(0.).clamp(0, BINS - 1).to(torch.int64).t()   # a NaN never indexes
    hist = torch.zeros(c, BINS, dtype=torch.int64, device=absx.device)
    hist.scatter_add_(1, idx, torch.ones_like(idx))
    cum = hist.cumsum(1)
    lo, hi, g = _virtual_index(n)
    a = (cum <= lo).sum(1).to(torch.float64)
    b = (cum <= hi).sum(1).to(torch.float64)
    d = b - a
    return b - d * (1. - g) if g >= .5 else a + d * g


def fake_quant(x: torch.Tensor, m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """kv4.decode(kv4.encode(x, m), t) in float32: x (..., *m.shape) integer-valued, m the
    reciprocals per channel ((channels,) or (kinds, channels)), t (*m.shape, 15); autocast off."""
    with torch.autocast(x.device.type, enabled=False):
        xf = x.float()
        p = xf.abs() * m.float()
        q = torch.floor(p / ONE)
        r = p - q * ONE
        odd = (q - 2. * torch.floor(q / 2.)) == 1.
        j = q + ((r > HALF) | ((r == HALF) & odd)).float()
        j = torch.clamp(j, max=float(LEVELS)) * torch.sign(xf)
        offs = torch.arange(m.numel(), device=x.device).reshape(m.shape) * (2 * LEVELS + 1)
        idx = (torch.nan_to_num(j) + LEVELS).long().clamp(0, 2 * LEVELS) + offs   # NaN-safe
        return t.float().reshape(-1)[idx]


def ste(x: torch.Tensor, m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward fake_quant(x); gradient 1 where |x| <= |T[+-7]|, 0 beyond (clipped STE)."""
    deq = fake_quant(x.detach(), m, t)
    with torch.autocast(x.device.type, enabled=False):
        keep = (x.detach().float().abs() <= t[:, -1].float()).float()
        return deq + (x - x.detach()).float() * keep


# ---------------------------------------------------------------- the attention class

class Kv4QatAttention(Attention):
    """Attention with the 4-bit K/V fake quant at INJECTION (model.kv4qat's docstring)."""

    def forward(self, x8, sx):
        lins = (self.wq, self.wk, self.wv)
        if self.training and not QUANT_ENABLED:  # float q/k/v and real softmax
            q, k, v = (lin(x8, sx) for lin in lins)
            o, e = _float_attention(q, k, v, self.cfg.n_heads, self.alibi, self.window,
                                    self.cfg.lut_max)
            if self.capture:
                self.last_weights = e.detach()
            return self.wo(*act_quant(o, self.training))
        if self.calibrating:
            with torch.no_grad():
                self.amax.copy_(torch.stack([lin(x8, sx).abs().max() for lin in lins]))
        s = self.scales()
        ys = [lin(x8, sx, post=1.0 / s[i]) for i, lin in enumerate(lins)]
        if self.training and not self.calibrating:
            with torch.no_grad():
                batch = torch.stack([y.detach().abs().max() for y in ys]).float() * s
                m = self.cfg.ema_momentum
                self.amax.mul_(m).add_((1 - m) * batch)
        q8, k8, v8 = (static_round(y) for y in ys)
        k8, v8 = self.kv4qat(k8, v8)
        u = s[0] * s[1] / math.sqrt(self.cfg.head_dim)
        o, e = attention_core(q8, k8, v8, u, s[2], self.cfg.n_heads, self.cfg.lut_max,
                              self.alibi, self.window)
        if self.capture:
            self.last_weights = e.detach()
        return self.wo(*act_quant(o, self.training))

    def _kv4qat_setup(self, mode: str, tables=None) -> None:
        if mode not in MODES:
            raise ValueError(f'kv4qat modes are {MODES}, not {mode!r}')
        hd = self.cfg.head_dim
        dev = self.amax.device
        self.kvq_mode = mode
        z = dict(device=dev)
        self.register_buffer('kvq_a', torch.zeros(2, hd, dtype=torch.float64, **z),
                             persistent=False)
        self.register_buffer('kvq_M', torch.zeros(2, hd, dtype=torch.float32, **z),
                             persistent=False)
        self.register_buffer('kvq_T', torch.zeros(2, hd, 2 * LEVELS + 1, dtype=torch.float32,
                                                  **z), persistent=False)
        self.register_buffer('kvq_init', torch.zeros((), dtype=torch.bool, **z),
                             persistent=False)
        self.register_buffer('kvq_calls', torch.zeros((), dtype=torch.int64, **z),
                             persistent=False)
        self.register_buffer('kvq_changed', torch.zeros(2, dtype=torch.int64, **z),
                             persistent=False)
        self.register_buffer('kvq_pre_init_evals', torch.zeros((), dtype=torch.int64, **z),
                             persistent=False)
        if mode == 'fixed':
            m, t = tables
            m, t = np.asarray(m), np.asarray(t)
            if m.shape != (2, hd) or t.shape != (2, hd, 2 * LEVELS + 1):
                raise ValueError(f'fixed tables of shape {m.shape} / {t.shape} for head_dim {hd}')
            self.kvq_M.copy_(torch.tensor(m, dtype=torch.float32))
            self.kvq_T.copy_(torch.tensor(t, dtype=torch.float32))
            self.kvq_init.fill_(True)

    def _track(self, k8, v8) -> None:
        """Training mode: the batch percentiles, the EMA (the first call sets a) and M, T."""
        hd = self.cfg.head_dim
        ab = torch.stack([hist_percentile(x.detach().abs().reshape(-1, hd)) for x in (k8, v8)])
        ema = self.kvq_a * MOMENTUM + (1 - MOMENTUM) * ab
        self.kvq_a.copy_(torch.where(self.kvq_init, ema, ab))
        m, t = derive_from_s(step_from_a(self.kvq_a))
        self.kvq_M.copy_(m.float())
        self.kvq_T.copy_(t.float())
        self.kvq_init.fill_(True)

    def kv4qat(self, k8, v8):
        """INJECTION: k8 and v8 replaced by their 4-bit decoded values (mode-dependent)."""
        mode = self.kvq_mode
        if mode == 'identity':
            if self.training:
                with torch.no_grad():
                    self.kvq_calls.add_(1)
            return k8, v8
        if self.training:
            with torch.no_grad():
                if mode == 'kv4':
                    self._track(k8, v8)
                self.kvq_calls.add_(1)
        init = self.kvq_init
        outs = []
        for kind, x in enumerate((k8, v8)):
            y = ste(x, self.kvq_M[kind], self.kvq_T[kind])
            y = torch.where(init, y, x.float())
            if self.training:
                with torch.no_grad():
                    self.kvq_changed[kind].add_((y.detach() != x.detach().float()).sum())
            outs.append(y)
        if not self.training:
            with torch.no_grad():
                self.kvq_pre_init_evals.add_((~init).long())
        return outs[0], outs[1]


# the copied forward resolves its names in model.model's namespace, exactly as the original
Kv4QatAttention.forward = types.FunctionType(
    Kv4QatAttention.forward.__code__, vars(M), 'forward', Kv4QatAttention.forward.__defaults__,
    Kv4QatAttention.forward.__closure__)
Kv4QatAttention.forward.__qualname__ = 'Kv4QatAttention.forward'


def convert(model, mode: str, tables=None):
    """Swap every block's attention to Kv4QatAttention in `mode` (attribute class swap); for
    'fixed', tables = (M (layers, 2, head_dim), T (layers, 2, head_dim, 15)) integers."""
    if mode not in MODES:
        raise ValueError(f'kv4qat modes are {MODES}, not {mode!r}')
    if mode == 'fixed' and tables is None:
        raise ValueError('the fixed-table mode needs the tables (M, T)')
    for i, blk in enumerate(model.layers):
        attn = blk.attn
        if type(attn) is not Attention:
            raise TypeError(f'block {i}: {type(attn).__name__}, not a plain Attention')
        attn.__class__ = Kv4QatAttention
        attn._kv4qat_setup(mode, None if tables is None else
                           (np.asarray(tables[0])[i], np.asarray(tables[1])[i]))
    return model


# ---------------------------------------------------------------- records

def final_state(model) -> dict:
    """Per layer: a, s (float64), the integers M and T derived from s, the counters."""
    layers = []
    for blk in model.layers:
        at = blk.attn
        a = at.kvq_a.detach().cpu()
        s = step_from_a(a)
        m, t = derive_from_s(s)
        ch = at.kvq_changed.detach().cpu().tolist()
        layers.append({'a': a.tolist(), 's': s.tolist(),
                       'M': m.to(torch.int64).tolist(), 'T': t.to(torch.int64).tolist(),
                       'buffer_M_equal': bool(torch.equal(at.kvq_M.detach().cpu(), m.float())),
                       'buffer_T_equal': bool(torch.equal(at.kvq_T.detach().cpu(), t.float())),
                       'initialised': bool(at.kvq_init), 'calls': int(at.kvq_calls),
                       'changed': {'k': int(ch[0]), 'v': int(ch[1])},
                       'pre_init_evals': int(at.kvq_pre_init_evals), 'mode': at.kvq_mode})
    return {'layers': layers}


def tables_of(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(M, T) int64 arrays (layers, 2, head_dim[, 15]) from a kv4qat.json-style record."""
    return (np.asarray([lay['M'] for lay in rec['layers']], dtype=np.int64),
            np.asarray([lay['T'] for lay in rec['layers']], dtype=np.int64))


def fixed_model(ckpt, tables, device: str = 'cpu'):
    """A checkpoint of model.train as a Kv4QatAttention model in fixed-table mode (eval mode)."""
    from model.model import load_checkpoint
    model = load_checkpoint(ckpt)
    convert(model, 'fixed', tables)
    return model.to(device).eval()


# ---------------------------------------------------------------- Control 11 (the positive check)

def _val_loss(model, args: dict, device: str, compile_: bool, amp: str) -> float:
    """model.train's logged val loss of `model`: its val windows (eval_windows cap), its eval
    batching (64) and evaluate(), on the given path."""
    from contextlib import nullcontext

    from model import train as tr
    from model.data import load_split, val_windows
    torch.backends.cuda.matmul.allow_tf32 = True          # as model.train's _train sets it
    val = load_split(Path(args['data']) / 'val.bin')
    xs, ys = val_windows(val, int(args['ctx']))
    n = int(args.get('eval_windows', '0'))
    if n:
        xs, ys = xs[:n], ys[:n]
    model = model.to(device)
    run = torch.compile(model) if compile_ else model

    def amp_ctx():
        return (torch.autocast(device.split(':')[0], dtype=torch.bfloat16) if amp == 'bf16'
                else nullcontext())
    return tr.evaluate(run, xs, ys, device, amp_ctx)


def active_check(run_dir, device: str | None = None, compile: bool | None = None,  # noqa: A002
                 amp: str | None = None) -> dict:
    """Control 11: the logged final val_loss (step --steps, the final weights) recomputed from
    ckpt.pt with kv4qat.json's final M and T in fixed-table mode, on the training's path (device,
    torch.compile and autocast from inputs.json unless given; model.train's eval batching), must
    match within 1e-5; the plain int8 val loss of the same ckpt.pt on the same path must differ
    from it by at least 1e-4."""
    from model.model import load_checkpoint
    d = Path(run_dir)
    args = json.loads((d / 'inputs.json').read_text())['args']
    rec = json.loads((d / 'kv4qat.json').read_text())
    device = device or args['device']
    compile_ = (args.get('compile') == 'True') if compile is None else compile
    amp = amp or args['amp']
    steps = int(args['steps'])
    rows = [json.loads(x) for x in (d / 'log.jsonl').read_text().splitlines() if x.strip()]
    logged = [r['val_loss'] for r in rows if r.get('step') == steps and 'val_loss' in r]
    if len(logged) != 1:
        raise ValueError(f'{d}/log.jsonl: {len(logged)} val_loss rows at step {steps}')
    fixed = fixed_model(d / 'ckpt.pt', tables_of(rec))
    got = _val_loss(fixed, args, device, compile_, amp)
    plain = _val_loss(load_checkpoint(d / 'ckpt.pt'), args, device, compile_, amp)
    match = abs(got - logged[0]) <= ACTIVE_MATCH_TOL
    sep = abs(got - plain)
    return {'logged_val_loss': logged[0], 'recomputed_val_loss': got, 'int8_val_loss': plain,
            'difference': got - logged[0], 'separation': sep, 'match': match,
            'separated': sep >= ACTIVE_SEPARATION, 'pass': bool(match and sep >= ACTIVE_SEPARATION),
            'path': {'device': device, 'compile': compile_, 'amp': amp, 'eval_batch': 64},
            'tolerance': ACTIVE_MATCH_TOL, 'min_separation': ACTIVE_SEPARATION}
