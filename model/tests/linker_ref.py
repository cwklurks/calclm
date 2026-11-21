"""Independent reference for model/linker.py's tests (adapted from an independent review's
reference implementations): odec2's loop over the original scratch modules
(experiments/v3-angles odec.py and bind.py, commit 6f93203) with rule 7 written from
PLAN-v2.1's text. It never imports
model.linker; only intref, intmask, intops, tokens and the scratch code."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from model.intmask import IntMask
from model.intops import rsr, udiv, xorshift32
from model.intref import SamplerConfig, choose, ranked_candidates
from model.tokens import BOS, EOS

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "experiments" / "v3-angles"
EXPORT = ROOT / "runs" / "v2" / "export"
ARMS = {"HZ": dict(old=False, hazard=True, r1=False, r2=False),
        "LD": dict(old=True, hazard=True, r1=True, r2=True),
        "LDs": dict(old=True, hazard=True, r1=True, r2=True, soft=True)}


def scratch():
    """The original odec and bind modules (appended to sys.path, so nothing is shadowed).
    Importing them inserts the main checkout at the front of sys.path, which spawned worker
    processes inherit; that insertion is undone here. Skips the calling test without them
    (the scratch folder is not in the public export)."""
    if not (SCRATCH / "obligations-ending" / "odec.py").exists():
        import pytest
        pytest.skip("needs the original scratch modules (experiments/v3-angles, not in this "
                    "checkout)")
    for d in ("obligations-ending", "linker-wildcard"):
        if str(SCRATCH / d) not in sys.path:
            sys.path.append(str(SCRATCH / d))
    before = list(sys.path)
    try:
        import bind
        import odec
    finally:
        sys.path[:] = before
    return odec, bind


def init(export=EXPORT):
    """Load the export into odec's globals (its vocab tables); returns the model."""
    odec, _ = scratch()
    if odec._m is None:
        odec._load(str(export))
    return odec._m


def _weights(m, lg, banned, top_k=20, tq8=205):
    cand = ranked_candidates(lg, top_k, banned)
    inv_t, top = udiv(32 << 16, tq8), int(lg[cand[0]])
    ws = [int(m.expw[min(rsr(min(top - int(lg[c]), 32767) * inv_t, 16), len(m.expw) - 1)])
          for c in cand]
    return sum(ws), ws[cand.index(EOS)] if EOS in cand else 0


def _cell(pending, depth, prev, pos):
    s = 2 if pending else (3 if depth > 0 else (0 if prev in (138, 150) else 1))
    return s * 4 + int(np.digitize(pos, [256, 512, 1024]))


def gen_ref(seed, E, old, hazard, r1, r2, soft=False, bosfix=True, cap=1024, body=24):
    odec, bind = scratch()
    init()
    XT, prefix_mask, no_dup_mask, LBL, NL = odec.XT, odec.prefix_mask, odec.no_dup_mask, odec.LBL, odec.NL
    m, v = odec._m, len(odec._vocab)
    cfg = SamplerConfig(205, 20, False, seed)
    mask, sp, row = IntMask(odec._vocab), XT(tuple(odec._vocab)), bind.Row()
    tokens, cache, state = [BOS], m.new_cache(), seed if seed else 2463534242
    mask.reset(); logits = m.step(BOS, 0, cache); mask.step(BOS); sp.step(BOS)  # noqa: E702
    if not bosfix:
        row.step(BOS)
    queued = []
    st = dict(forced_lbl=0, hazard_stop=0, r1=0, r2=0, draws=0)
    while len(tokens) < cap:
        b = mask.banned(); nm = sp.name; R = cap - len(tokens); P = sorted(sp.unres)  # noqa: E702
        D = sum(3 + len(x) + body for x in P) + 1
        lg = logits
        if old and nm is not None and nm.kind == "lbl":
            if P:
                pm = prefix_mask(v, nm.chars, set(P))
                if pm is not None:
                    if soft:
                        lg = logits.copy()
                        lg[~pm] += 512
                    else:
                        b |= pm
                if soft:
                    b |= no_dup_mask(v, nm.chars, sp.defined)
            else:
                b |= no_dup_mask(v, nm.chars, sp.defined)
        if old and nm is not None and nm.kind in ("goto", "menu") and R <= D + 3 + body:
            pm = prefix_mask(v, nm.chars, set(P) | set(sp.defined))
            if pm is not None:
                b |= pm
        if old and nm is not None and nm.kind == "goto" and sp.goto_ctx == (0, 0):
            bad = {k for k, x in sp.exit_since.items() if not x}
            if bad:
                xb = no_dup_mask(v, nm.chars, bad)
                if xb.any() and not (b | xb)[4:].all():
                    b |= xb
        at_start = (sp.at_start and not sp.in_str and sp.name is None and sp.pending is None
                    and sp.if_state == 0)
        closure = (not row.in_str and not row.toks and row.after_transfer and not row.if_pending
                   and not row.done)
        if queued:
            tok = queued.pop(0)
        elif r1 and closure and not P and not row.pending:
            tok = EOS; st["r1"] += 1  # noqa: E702
        elif old and P and at_start and R <= D:
            tok = LBL; st["forced_lbl"] += 1  # noqa: E702
        elif r2 and closure and (P or row.pending) and at_start:
            tok = LBL; st["r2"] += 1  # noqa: E702
        else:
            stop = False
            if hazard and not (old and P) and not b[EOS] and len(tokens) > 1:
                W, w_e = _weights(m, lg, b)
                Ec = E[_cell(bool(P), len(sp.blocks), tokens[-1], len(tokens) - 1)]
                h16 = (65536 * w_e * Ec) // (w_e * Ec + (W - w_e) * 65536)
                state = xorshift32(state); st["draws"] += 1  # noqa: E702
                stop = (state >> 16) < h16
            if stop:
                tok = EOS; st["hazard_stop"] += 1  # noqa: E702
            else:
                if hazard or (old and P):
                    b[EOS] = True
                allowed = ~b
                allowed[[0, 1, 3]] = False
                if not allowed.any():
                    b[EOS] = False
                tok, state = choose(m, lg, cfg, state, b)
                if old and tok == EOS and P and R > 4:
                    tok, queued = (LBL, []) if at_start else (NL, [LBL])
        tokens.append(tok)
        if tok == EOS:
            break
        mask.step(tok); sp.step(tok); row.step(tok)  # noqa: E702
        logits = m.step(tok, len(tokens) - 1, cache)
    return {"seed": seed, "ids": [int(t) for t in tokens], "stats": st}


def fit_rows_ref(prog) -> np.ndarray:
    """Teacher-forced rows (cell, w_e, W, y) under M's bans; cell -1 where M leaves only <eos>."""
    odec, _ = scratch()
    m = init()
    mask, sp, cache = IntMask(m.vocab), odec.XT(tuple(m.vocab)), m.new_cache()
    mask.reset()
    out = []
    for t in range(min(len(prog) - 1, 1024)):
        tok = int(prog[t])
        lg = m.step(tok, t, cache); mask.step(tok); sp.step(tok)  # noqa: E702
        b = mask.banned()
        W, w_e = _weights(m, lg, b)
        allowed = ~b
        allowed[[0, 1, 3]] = False
        only = bool(allowed[EOS] and allowed.sum() == 1)
        cell = -1 if only else _cell(bool(sp.unres), len(sp.blocks), tok, t)
        out.append((cell, w_e, W, int(prog[t + 1]) == EOS))
    return np.array(out, dtype=np.int64)


def hot_E(hot) -> list[int]:
    """Offsets +8 in the `hot` cells and -8 elsewhere: a wrong cell flips stops."""
    return [round(65536 * np.exp(8.0 if c in hot else -8.0)) for c in range(16)]

