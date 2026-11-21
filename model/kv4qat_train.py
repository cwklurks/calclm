"""E21's QAT training entry point (PLAN-E21.md "The code change"): model.train with the 4-bit K/V
fake quant, without editing model.train's files.

    python -m model.kv4qat_train <model.train's arguments, unchanged>

It rebinds `model.train.Transformer`, in its own process, to a factory that builds the plain
Transformer, swaps each block's attention class to model.kv4qat.Kv4QatAttention in 'kv4' (the
tracking mode; no new state-dict keys) and keeps a reference to the model it built; then it calls
model.train.main, restores the name, and checks:

- the factory was called exactly once;
- every block's attention is a Kv4QatAttention in the run's mode;
- each attention's training-mode call counter equals the run's training forwards (--steps, plus
  the start-up calibration forward from scratch), and in 'kv4' its values-changed counters for K
  and for V are both above 0;
- log.jsonl, inputs.json, ckpt.pt and avg.pt exist.

Any failure exits non-zero (the runner deletes the run whole as a crash). Then it writes
kv4qat.json into --out: mode, the final a, s and the integers M and T per layer, kind and channel,
the percentile definition, momentum, the initialisation step, the pre-initialisation evaluation
rows (a wrapper cannot mark the log), the counters, the factory's call count, the QAT code's
sha256, the sha256 of log.jsonl, inputs.json, ckpt.pt and avg.pt, and the training steps per
second (the overhead against W8f is the runner's).

Refused (outside E21's recipes, PLAN-E21 Arms 3-4): --float, --teacher, --qat-start other than 0,
--feat-dir and --avg-steps 0. The identity mode (`main(argv, _mode='identity')`) is reachable from
tests only: there is no command-line option for it, and it is refused unless pytest is loaded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from model import kv4qat as Q
from model import train as T
from model.model import Transformer

QAT_FILES = ('model/kv4qat.py', 'model/kv4qat_train.py')
STORE_TRUE = ('compile', 'feat_shuffle', 'feat_fold', 'float')
RUN_FILES = ('log.jsonl', 'inputs.json', 'ckpt.pt', 'avg.pt')
BINDING = 'attribute class swap (model.kv4qat.convert) inside a rebound model.train.Transformer'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def qat_code_sha256() -> str:
    """sha256 over the two QAT files' sha256 lines (as lab's code manifests)."""
    root = Path(__file__).resolve().parents[1]
    lines = ''.join(f'{_sha(root / f)}  {f}\n' for f in QAT_FILES)
    return hashlib.sha256(lines.encode()).hexdigest()


def argv_from_args(args: dict, **over) -> list[str]:
    """model.train's argv from an inputs.json "args" record (strings, parser order), with
    overrides (name: value; None drops the option)."""
    rec = {**args, **{k: (None if v is None else str(v)) for k, v in over.items()}}
    out = []
    for k, v in rec.items():
        if v is None or v == 'None':
            continue
        flag = '--' + k.replace('_', '-')
        if k in STORE_TRUE:
            if v == 'True':
                out.append(flag)
            elif v != 'False':
                raise ValueError(f'{k}: {v!r} is not a boolean flag value')
            continue
        out += [flag, v]
    return out


def _refuse(args) -> None:
    bad = []
    if args.float:
        bad.append('--float')
    if args.teacher is not None:
        bad.append('--teacher')
    if args.qat_start:
        bad.append('--qat-start')
    if args.feat_dir is not None:
        bad.append('--feat-dir')
    if args.avg_steps <= 0:
        bad.append('--avg-steps 0 (E21 evaluates avg.pt)')
    if bad:
        raise SystemExit(f'model.kv4qat_train: {bad} are outside E21\'s recipes')


def check_counters(layers: list[dict], expected: int, mode: str) -> None:
    for i, lay in enumerate(layers):
        if mode == 'kv4' and 'initialised' in lay:
            if not lay['initialised']:
                raise RuntimeError(f'layer {i}: the running percentile was never initialised')
            if not (lay['buffer_M_equal'] and lay['buffer_T_equal']):
                raise RuntimeError(f'layer {i}: the device buffers M and T differ from the '
                                   'integers derived from the final scales (kv4qat.json)')
        if lay['calls'] != expected:
            raise RuntimeError(f'layer {i}: {lay["calls"]} training-mode fake-quant calls, not the '
                               f'run\'s {expected} training forwards')
        if mode == 'kv4' and not (lay['changed']['k'] > 0 and lay['changed']['v'] > 0):
            raise RuntimeError(f'layer {i}: values changed {lay["changed"]}: the fake quant changed '
                               'no K or no V value; the QAT path was not active (Control 11)')


def _rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / 'log.jsonl').read_text().splitlines() if x.strip()]


def _steps_per_s(rows: list[dict]) -> float | None:
    """Training steps per second between the first and the last train_loss rows."""
    tr = [r for r in rows if 'train_loss' in r]
    if len(tr) < 2 or tr[-1]['elapsed_s'] <= tr[0]['elapsed_s']:
        return None
    return (tr[-1]['step'] - tr[0]['step']) / (tr[-1]['elapsed_s'] - tr[0]['elapsed_s'])


def record(args, built: list, mode: str) -> dict:
    """The post-main checks and kv4qat.json's content (raises RuntimeError on any failure)."""
    if len(built) != 1:
        raise RuntimeError(f'the factory was called {len(built)} times, not exactly once')
    model = built[0]
    for i, blk in enumerate(model.layers):
        if type(blk.attn) is not Q.Kv4QatAttention or blk.attn.kvq_mode != mode:
            raise RuntimeError(f'block {i}\'s attention is {type(blk.attn).__name__} '
                               f'({getattr(blk.attn, "kvq_mode", None)}), not a Kv4QatAttention '
                               f'in {mode} mode')
    out = Path(args.out)
    missing = [f for f in RUN_FILES if not (out / f).is_file()]
    if missing:
        raise RuntimeError(f'{out}: {missing} missing after training')
    init = args.init is not None
    expected = args.steps + (0 if init else 1)
    st = Q.final_state(model)
    check_counters(st['layers'], expected, mode)
    rows = _rows(out)
    pre = sum(lay['pre_init_evals'] for lay in st['layers'])
    if init and mode == 'kv4' and pre == 0:
        raise RuntimeError('a fine-tune with no pre-initialisation evaluation: the step-0 '
                           'evaluation did not pass K and V unchanged')
    pre_rows = [r for r in rows if r.get('step') == 0 and 'val_loss' in r] if pre else []
    return {'mode': mode, **st, 'percentile': Q.PERCENTILE,
            'percentile_method': Q.PERCENTILE_METHOD, 'histogram_bins': Q.BINS,
            'momentum': Q.MOMENTUM, 'init_step': 0 if init else 'startup calibration',
            'pre_init_eval_rows': pre_rows, 'expected_calls': expected,
            'factory_calls': len(built), 'qat_code_sha256': qat_code_sha256(),
            'qat_files': list(QAT_FILES), 'injection': Q.INJECTION, 'binding': BINDING,
            'files_sha256': {f: _sha(out / f) for f in RUN_FILES},
            'steps_per_s': _steps_per_s(rows)}


def main(argv=None, *, _mode: str = 'kv4') -> dict:
    if _mode not in ('kv4', 'identity'):
        raise ValueError(f'model.kv4qat_train trains in kv4 (or identity, tests only), not '
                         f'{_mode!r}')
    if _mode == 'identity' and 'pytest' not in sys.modules:
        raise RuntimeError('the identity mode is reachable from tests only')
    args = T.parse_args(argv)
    _refuse(args)
    built: list = []

    def factory(cfg):
        model = Transformer(cfg)
        Q.convert(model, _mode)
        built.append(model)
        return model

    saved = T.Transformer
    T.Transformer = factory
    try:
        result = T.main(argv)
    finally:
        T.Transformer = saved
    rec = record(args, built, _mode)
    out = Path(args.out) / 'kv4qat.json'
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    return result


if __name__ == '__main__':
    # Run main() in the imported model.kv4qat_train module (E15's __main__ lesson), so the
    # rebinding and the module state are the imported module's.
    from model import kv4qat_train as _w
    _w.main()
