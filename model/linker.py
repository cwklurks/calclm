"""Linker decoder (LD, PLAN-v2.1.md E2) on the integer reference, on top of the shipped masks.

Rules 1-6 are experiments/v3-angles/rt/odec2.py and its imports (obligations-ending/odec.py,
linker-wildcard/bind.py, commit 6f93203), ported without change: odd behaviour of the scratch
code is kept and marked "(kept)". Rule 7 replaces odec2's float hazard by an integer one drawn
from the sampler's own xorshift32 state, so the whole decoder is bit-exact and portable to C.

    uv run python -m model.linker fit --export runs/v2/export --data data/tok --out offs.json
    uv run python -m model.linker sample --export runs/v2/export --arm LD --offsets offs.json \
        --n 1000 --out ld.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

from model.intmask import IntMask
from model.intops import rsr, udiv, xorshift32
from model.intref import IntModel, SamplerConfig, choose, ranked_candidates
from model.tokens import BOS, EOS, PAD, UNK
from tracker import spec

TEMP, TOP_K, CAP, BODY = 0.8, 20, 1024, 24      # PLAN-v2.1 sampler; BODY: odec's deadline slack
TEMP_Q8 = round(TEMP * 256)                    # 205
BOOST_Q8 = 512                                  # soft bind: +2.0 logits
N_CELLS, POS_BINS, FIT_POSITIONS = 16, (256, 512, 1024), 1024
B_CLIP = 8.0
# If While Repeat For IS> DS< Stop Return Menu( getKey Input Prompt: odec's exit tokens (rule 4)
EXIT_HEX = ("CE", "D1", "D2", "D3", "DA", "DB", "D9", "D5", "E6", "AD", "DC", "DD")
ROW_OPEN = {"lparen", "func", "statopen", "dim", "equstr", "listmatr", "matrlist", "for", "isds",
            "menu"}


@dataclass(frozen=True)
class Rules:
    old: bool = False        # rules 1-4 (bind, no early end, deadline, loop exit): odec2's `old`
    soft_bind: bool = False  # rule 1's pending-name restriction becomes +BOOST_Q8 on the allowed
                             # continuations; the no-redefinition ban applies always
    row_bos_fix: bool = True # Row classifies the first statement too (False: odec2's Row bug)
    hazard: bool = False     # rule 7, the integer stop hazard
    r1: bool = False         # closure point, nothing pending: the program ends
    r2: bool = False         # closure point, references pending: Lbl is forced
    hazard_full: bool = False  # E2b: rule 7's weights are T 1.0 over every allowed id


ARMS: dict[str, Rules] = {
    "M": Rules(), "OLD": Rules(old=True), "HZ": Rules(hazard=True),
    "LD": Rules(old=True, hazard=True, r1=True, r2=True),
    "LDs": Rules(old=True, soft_bind=True, hazard=True, r1=True, r2=True),
    "LD2": Rules(old=True, hazard=True, r1=True, r2=True, hazard_full=True)}


@dataclass(frozen=True)
class Ids:
    """Token ids and roles the rules use, derived from the vocab (the scratch code hard-codes
    the shipped vocab's ids: Lbl 55, Goto 48, Menu( 112, newline 4, : 15, Stop 138, Return 150)."""
    vocab: tuple[str, ...]
    roles: tuple[tuple[str, object], ...]
    lbl: int
    goto: int
    menu: int
    nl: int
    colon: int
    stop: int
    ret: int
    char: dict[str, int]      # label-name character -> id
    name: dict[int, str]      # id -> label-name character
    sep: frozenset[int]       # newline and colon
    exit: frozenset[int]


@lru_cache(maxsize=4)
def _vocab_ids(vocab: tuple[str, ...]) -> Ids:
    roles = spec.roles(vocab)
    idx = {h: i for i, h in enumerate(vocab)}
    name = {i: a for i, (r, a) in enumerate(roles) if r == "name"}
    return Ids(vocab, roles, idx["D6"], idx["D7"], idx["E6"], idx["3F"], idx["3E"], idx["D9"],
               idx["D5"], {a: i for i, a in name.items()}, name,
               frozenset(i for i, (r, _) in enumerate(roles) if r in ("newline", "colon")),
               frozenset(idx[h] for h in EXIT_HEX))


def vocab_ids(vocab) -> Ids:
    return _vocab_ids(tuple(vocab))


# ---- state: the spec tracker + odec's exit flags, and bind.py's statement rows ----------------

class XT(spec.Tracker):
    """tracker.spec's Tracker plus, per label, whether an exit token occurred since its Lbl
    (`exit_since`), and the (If state, block depth) at the latest Goto statement (`goto_ctx`)."""

    def __init__(self, ids: Ids):
        self.exit_ids = ids.exit
        super().__init__(ids.vocab)

    def reset(self) -> None:
        super().reset()
        self.exit_since: dict[str, bool] = {}
        self.goto_ctx: tuple[int, int] | None = None

    def _statement(self, role: str) -> None:
        if role == "goto":
            self.goto_ctx = (self.if_state, len(self.blocks))
        super()._statement(role)

    def _register_label(self, kind: str, name: str) -> None:
        if kind == "lbl" and name not in self.defined:
            self.exit_since[name] = False
        super()._register_label(kind, name)
        if kind == "goto":                  # (kept) a finished Goto counts as every label's exit
            for k in self.exit_since:
                self.exit_since[k] = True

    def step(self, tok: int):
        if int(tok) in self.exit_ids:       # (kept) seen before parsing, so inside strings too
            for k in self.exit_since:
                self.exit_since[k] = True
        return super().step(tok)


class Row:
    """Statement-level binding state (linker-wildcard/bind.py's Row, the part odec2 reads):
    references and definitions per finished statement, block depth, single-line If bodies, and
    whether the last statement was a top-level Goto, Menu(, Stop or Return (`after_transfer`).

    Row has no <bos> case: fed <bos> (odec2, Rules.row_bos_fix False), <bos> joins the first
    statement's tokens and that statement is never classified (a first-line Lbl, Goto, If, loop
    or Stop goes unseen). With the fix, <bos> is not fed. (kept) Names are cut to 2 characters; a stray top-level End right after a transfer keeps
    the state. bind.py's import-time patches (lbl_after_transfer, the R5 output flag) and its
    allowed_override rules change none of the fields odec2 reads, so they are not ported."""

    def __init__(self, ids: Ids):
        self.ids = ids
        self.in_str = self.if_pending = self.after_transfer = self.done = False
        self.toks: list[int] = []
        self.depth = 0
        self.pending: list[str] = []
        self.defined: set[str] = set()

    def _role(self, t: int) -> str:
        return self.ids.roles[t][0]

    def _name(self, toks) -> str:
        s = ""
        for t in toks:
            if t not in self.ids.name:
                break
            s += self.ids.name[t]
        return s[:2]

    def _menu_labels(self, toks) -> list[str]:
        args: list[list[int]] = [[]]
        d, ins = 0, False
        for t in toks:
            r = self._role(t)
            if ins:
                if r == "quote":
                    ins = False
                args[-1].append(t)
                continue
            if r == "quote":
                ins = True
                args[-1].append(t)
                continue
            if r in ROW_OPEN:
                d += 1
            if r == "rparen":
                if d == 0:
                    break
                d -= 1
            if r == "comma" and d == 0:
                args.append([])
                continue
            args[-1].append(t)
        return [self._name(a) for k, a in enumerate(args) if k >= 2 and k % 2 == 0 and self._name(a)]

    def _end_statement(self) -> None:
        toks, self.toks = self.toks, []
        if not toks:
            return
        r0, body = self._role(toks[0]), False
        if self.if_pending:
            self.if_pending = False
            if r0 == "then":
                self.depth += 1
                self.after_transfer = False
                return
            body = True
        d, refs = self.depth, []
        if r0 == "end" and self.depth == 0 and self.after_transfer and not body:
            return
        if r0 == "if":
            self.if_pending = True
        elif r0 in ("for", "while", "repeat"):
            self.depth += 1
        elif r0 == "end":
            self.depth = max(0, self.depth - 1)
        elif r0 == "lbl":
            x = self._name(toks[1:])
            if x:
                self.defined.add(x)
                if x in self.pending:
                    self.pending.remove(x)
        elif r0 == "goto":
            refs = [self._name(toks[1:])]
        elif r0 == "menu":
            refs = self._menu_labels(toks[1:])
        for x in refs:
            if x and x not in self.defined and x not in self.pending:
                self.pending.append(x)
        self.after_transfer = ((r0 in ("goto", "menu") or toks[0] in (self.ids.stop, self.ids.ret))
                               and not body and d == 0)

    def step(self, t: int) -> None:
        if self.done:
            return
        if t == EOS:
            self._end_statement()
            self.done = True
            return
        r = self._role(t) if t < len(self.ids.roles) else "other"
        if self.in_str:
            if r in ("quote", "store"):
                self.in_str = False
            elif r == "newline":
                self.in_str = False
                self._end_statement()
                return
            self.toks.append(t)
            return
        if t in self.ids.sep:
            self._end_statement()
            return
        if r == "quote":
            self.in_str = True
        self.toks.append(t)

    def closure(self) -> bool:
        """A statement boundary right after a top-level transfer, outside strings and If."""
        return (not self.in_str and not self.toks and self.after_transfer and not self.if_pending
                and not self.done)


# ---- rules 1, 3 and 4: bans inside label names --------------------------------------------------

def prefix_mask(ids: Ids, v: int, chars, cands) -> np.ndarray | None:
    """Ban everything at a label-name position except continuations toward a candidate name
    (and a separator when the name so far is a candidate); None when no candidate fits."""
    b = np.ones(v, dtype=bool)
    so_far = "".join(chars)
    ok = False
    for c in cands:
        if c.startswith(so_far) and len(c) > len(so_far):
            b[ids.char[c[len(so_far)]]] = False
            ok = True
    if so_far in cands:
        b[ids.nl] = b[ids.colon] = False
        ok = True
    return b if ok else None


def no_dup_mask(ids: Ids, v: int, chars, names) -> np.ndarray:
    """Forbid finishing a label name on one of `names` (a 1-character prefix: the second
    character that would complete one; a full name: the separators and <eos>)."""
    b = np.zeros(v, dtype=bool)
    so_far = "".join(chars)
    if so_far and so_far in names:
        b[ids.nl] = b[ids.colon] = b[EOS] = True
    if len(so_far) == 1:
        for ch, i in ids.char.items():
            if so_far + ch in names:
                b[i] = True
    return b


def rule_bans(rules: Rules, ids: Ids, sp: XT, P: list[str], R: int, D: int, b: np.ndarray,
              logits: np.ndarray, body: int = BODY) -> np.ndarray:
    """odec2's `old` bans (rules 1, 3, 4) added to b in place. Returns the logits to sample from:
    unchanged, or boosted on the pending continuations under soft bind (which keeps the
    no-redefinition ban while references are pending too)."""
    nm, v = sp.name, len(b)
    if not rules.old or nm is None:
        return logits
    if nm.kind == "lbl":                                            # rule 1: bind
        pm = prefix_mask(ids, v, nm.chars, set(P)) if P else None
        if pm is not None and rules.soft_bind:
            logits = logits + np.where(pm, 0, BOOST_Q8)
        elif pm is not None:
            b |= pm
        if not P or rules.soft_bind:
            b |= no_dup_mask(ids, v, nm.chars, sp.defined)
    if nm.kind in ("goto", "menu") and R <= D + 3 + body:           # rule 3: no new debt late
        pm = prefix_mask(ids, v, nm.chars, set(P) | set(sp.defined))
        if pm is not None:
            b |= pm
    if nm.kind == "goto" and sp.goto_ctx == (0, 0):                 # rule 4: loop exit
        bad = {k for k, x in sp.exit_since.items() if not x}
        if bad:
            xb = no_dup_mask(ids, v, nm.chars, bad)
            if xb.any() and not (b | xb)[4:].all():                 # (kept) ids from 4 up
                b |= xb
    return logits


# ---- rule 7: the integer stop hazard ----------------------------------------------------------

def topk_weights(model: IntModel, logits: np.ndarray, cfg: SamplerConfig, banned=None):
    """(candidates, expw weights) exactly as intref.choose computes them."""
    cand = ranked_candidates(logits, cfg.top_k, banned)
    inv_t = udiv(32 << 16, cfg.temperature_q8)
    top, last = int(logits[cand[0]]), len(model.expw) - 1
    return cand, [int(model.expw[min(rsr(min(top - int(logits[v]), (1 << 15) - 1) * inv_t, 16), last)])
                  for v in cand]


FULL_TEMP_Q8 = 256   # E2b: rule 7 at T 1.0


def hazard_weights(model: IntModel, logits: np.ndarray, cfg: SamplerConfig, banned, full: bool):
    """Rule 7's (candidates, weights): the sampler's top-k at its temperature (E2), or with
    `full` (E2b, PLAN-v2.2-E2b.md) intref's exp-table weights at T 1.0 over every allowed id."""
    if full:
        cfg = SamplerConfig(FULL_TEMP_Q8, len(logits), False, cfg.seed)
    return topk_weights(model, logits, cfg, banned)


def hazard16(w_e: int, W: int, E_c: int) -> int:
    """floor(65536 * w_e*E_c / (w_e*E_c + (W - w_e)*65536)); 65536 when <eos> is all of W."""
    den = w_e * E_c + (W - w_e) * 65536
    return 65536 * w_e * E_c // den if den else 0


def hazard_draw(state: int, h16: int) -> tuple[bool, int]:
    """One xorshift32 step of the sampler's state: stop iff its high 16 bits are below h16."""
    state = xorshift32(state)
    return (state >> 16) < h16, state


def hazard_cell(ids: Ids, pending: bool, depth: int, prev: int, pos: int) -> int:
    """state * 4 + position bin. States: 2 pending, 3 in a block, 0 after Stop/Return, else 1."""
    s = 2 if pending else (3 if depth > 0 else (0 if prev in (ids.stop, ids.ret) else 1))
    return 4 * s + sum(pos >= x for x in POS_BINS)


def offsets_to_E(b) -> list[int]:
    return [round(65536 * math.exp(x)) for x in b]


# ---- the decoder ------------------------------------------------------------------------------

def generate(model: IntModel, rules: Rules, seed: int, E: list[int] | None = None, cap: int = CAP,
             temperature_q8: int = TEMP_Q8, top_k: int = TOP_K, body: int = BODY,
             mask_level: int = 1) -> dict:
    """<bos>, then sample under the masks M and `rules` until <eos> or `cap` tokens in all.
    E: the 16 integer hazard offsets (offsets_to_E), required with rules.hazard.
    mask_level: IntMask's level (1 v2.3's masks, 2 v2.4's safety masks).
    Returns {"ids": [BOS, ..., (EOS)], "stats": counts of forced and hazard events}."""
    if rules.hazard and (E is None or len(E) != N_CELLS):
        raise ValueError(f"rule 7 needs {N_CELLS} integer offsets E (offsets_to_E)")
    ids, cfg = vocab_ids(model.vocab), SamplerConfig(temperature_q8, top_k, False, seed)
    mask, sp, row = IntMask(model.vocab, mask_level), XT(ids), Row(ids)
    tokens, cache, state = [BOS], model.new_cache(), seed if seed else 2463534242
    mask.reset()
    logits = model.step(BOS, 0, cache)
    mask.step(BOS)
    sp.step(BOS)
    if not rules.row_bos_fix:
        row.step(BOS)                               # odec2: <bos> joins statement 1 (see Row)
    queued: list[int] = []
    st = dict(forced_lbl=0, hazard_stop=0, r1=0, r2=0, eos_sub=0, hazard_draws=0)
    if rules.hazard_full:            # E2b's stop rate by cell (LD's stats stay as E2 wrote them)
        st |= dict(draws_c=[0] * N_CELLS, stops_c=[0] * N_CELLS)
    while len(tokens) < cap:
        b = mask.banned()
        R, P = cap - len(tokens), sorted(sp.unres)
        D = sum(3 + len(x) + body for x in P) + 1
        lg = rule_bans(rules, ids, sp, P, R, D, b, logits, body)
        at_start = (sp.at_start and not sp.in_str and sp.name is None and sp.pending is None
                    and sp.if_state == 0)
        closure = row.closure()
        if queued:
            tok = queued.pop(0)
        elif rules.r1 and closure and not P and not row.pending:
            tok = EOS
            st["r1"] += 1
        elif rules.old and P and at_start and R <= D:
            tok = ids.lbl
            st["forced_lbl"] += 1
        elif rules.r2 and closure and (P or row.pending) and at_start:
            tok = ids.lbl                           # rule 1 then picks a pending name
            st["r2"] += 1
        else:
            stop = False
            if rules.hazard and not (rules.old and P) and not b[EOS] and len(tokens) > 1:
                cand, w = hazard_weights(model, lg, cfg, b, rules.hazard_full)
                w_e = w[cand.index(EOS)] if EOS in cand else 0
                c = hazard_cell(ids, bool(P), len(sp.blocks), tokens[-1], len(tokens) - 1)
                stop, state = hazard_draw(state, hazard16(w_e, sum(w), E[c]))
                st["hazard_draws"] += 1
                if rules.hazard_full:
                    st["draws_c"][c] += 1
                    st["stops_c"][c] += stop
            if stop:
                tok = EOS
                st["hazard_stop"] += 1
            else:
                if rules.hazard or (rules.old and P):   # (kept) hazard on: <eos> only by rule 7
                    b[EOS] = True
                allowed = ~b
                allowed[[PAD, BOS, UNK]] = False
                if not allowed.any():
                    b[EOS] = False
                tok, state = choose(model, lg, cfg, state, b)
                if rules.old and tok == EOS and P and R > 4:   # rule 2: <eos> becomes Lbl
                    st["eos_sub"] += 1
                    tok, queued = (ids.lbl, []) if at_start else (ids.nl, [ids.lbl])
        tokens.append(tok)
        if tok == EOS:
            break
        mask.step(tok)
        sp.step(tok)
        row.step(tok)
        logits = model.step(tok, len(tokens) - 1, cache)
    return {"ids": [int(t) for t in tokens], "stats": st}


# ---- the offset fit ---------------------------------------------------------------------------

def solve_offset(w_e, W, y_sum: float, lo: float = -B_CLIP, hi: float = B_CLIP) -> float:
    """b with sum_i h_i(b) = y_sum, h_i(b) = w_e e^b / (w_e e^b + W - w_e), in float64. Sum h is
    increasing in b: Newton from 0, kept inside a shrinking bracket by bisection; b is clipped
    to [lo, hi] when the root lies outside."""
    w_e, W = np.asarray(w_e, dtype=np.float64), np.asarray(W, dtype=np.float64)
    ones = float(((w_e > 0) & (W == w_e)).sum())        # h = 1 at any b
    live = (w_e > 0) & (W > w_e)
    z = np.log(w_e[live]) - np.log(W[live] - w_e[live])

    def f(b: float) -> tuple[float, float]:
        h = 1.0 / (1.0 + np.exp(-(z + b)))
        return ones + float(h.sum()) - y_sum, float((h * (1.0 - h)).sum())

    if f(lo)[0] >= 0:
        return lo
    if f(hi)[0] <= 0:
        return hi
    b = 0.0
    for _ in range(200):
        g, d = f(b)
        if abs(g) <= 1e-12 * max(1.0, y_sum):
            break
        lo, hi = (b, hi) if g < 0 else (lo, b)
        nb = b - g / d if d > 0 else (lo + hi) / 2
        b = nb if lo < nb < hi else (lo + hi) / 2
    return b


def solve_cells(rows: np.ndarray) -> tuple[list[float], dict]:
    """rows (N, 4): cell, w_e, W, y (cell -1 is left out). Per cell: the offset, or 0 for a cell
    with no positions ("empty") or with positions but no <eos> ("no_eos"; its root is at -inf).
    PLAN-v2.1 says only "a cell with no data gets 0"; no_eos extends that (on s10 every live cell
    has 15 or more <eos>, so it does not arise)."""
    b, info = [], {"n": [], "eos": [], "status": []}
    for c in range(N_CELLS):
        r = rows[rows[:, 0] == c]
        n, y = len(r), int(r[:, 3].sum()) if len(r) else 0
        if n == 0 or y == 0:
            bc, status = 0.0, "empty" if n == 0 else "no_eos"
        else:
            bc = solve_offset(r[:, 1], r[:, 2], float(y))
            status = "clip_lo" if bc == -B_CLIP else "clip_hi" if bc == B_CLIP else "fit"
        b.append(float(bc))
        info["n"].append(n)
        info["eos"].append(y)
        info["status"].append(status)
    return b, info


def train_programs(data_dir, n: int, seed: int = 0) -> list[np.ndarray]:
    """The fit sample: `n` train programs that end with <eos>, numpy rng `seed`, in stream order."""
    from model.data import load_split, program_starts
    stream = load_split(Path(data_dir) / "train.bin")
    st = program_starts(stream)
    progs = [stream[a:e] for a, e in zip(st, np.append(st[1:], len(stream)))]
    progs = [p for p in progs if len(p) and p[-1] == EOS]
    idx = np.random.default_rng(seed).choice(len(progs), n, replace=False)
    return [progs[i] for i in sorted(idx)]


def programs_sha256(progs) -> str:
    return hashlib.sha256(b"".join(np.asarray(p, dtype="<u2").tobytes() for p in progs)).hexdigest()


def _fit_one(prog, temperature_q8: int = TEMP_Q8, top_k: int = TOP_K,
             full: bool = False) -> np.ndarray:
    """Teacher-force one program through intref with masks M: per position t < FIT_POSITIONS that
    predicts token t+1, the row (cell, w_e, W, y = token t+1 is <eos>).

    Cell -1 (left out of the fit): M leaves no id but <eos> (its fallback). Teacher forcing gets
    there when a human program reads a custom list the tracker sees as unassigned; the sampler
    never does (M bans the ʟ), h = 1 there whatever b, and such positions pin the fit at -8."""
    m = _model
    ids, cfg = vocab_ids(m.vocab), SamplerConfig(temperature_q8, top_k, False, 1)
    mask, sp, cache = IntMask(m.vocab), XT(ids), m.new_cache()
    mask.reset()
    n = min(len(prog) - 1, FIT_POSITIONS)
    rows = np.zeros((n, 4), dtype=np.int64)
    for t in range(n):
        tok = int(prog[t])
        logits = m.step(tok, t, cache)
        mask.step(tok)
        sp.step(tok)
        b = mask.banned()
        cand, w = hazard_weights(m, logits, cfg, b, full)
        w_e = w[cand.index(EOS)] if EOS in cand else 0
        allowed = ~b
        allowed[[PAD, BOS, UNK]] = False
        only_eos = allowed[EOS] and allowed.sum() == 1
        rows[t] = (-1 if only_eos else hazard_cell(ids, bool(sp.unres), len(sp.blocks), tok, t),
                   w_e, sum(w), int(prog[t + 1]) == EOS)
    return rows


def fit_rows(model: IntModel, programs, workers: int = 16, full: bool = False) -> np.ndarray:
    """(N, 4) rows (cell, w_e, W, y) over all programs, in program order. full: rule 7's E2b
    weights (hazard_weights)."""
    one = partial(_fit_one, full=full)
    if workers <= 1:
        _set_model(model)
        parts = [one(p) for p in programs]
    else:
        with ProcessPoolExecutor(workers, mp_context=_CTX, initializer=_set_model,
                                 initargs=(model,)) as pool:
            parts = list(pool.map(one, programs, chunksize=1))
    return np.concatenate(parts) if parts else np.zeros((0, 4), dtype=np.int64)


def fit_offsets(model: IntModel, programs, workers: int = 16) -> list[float]:
    """The 16 hazard offsets b_c fitted on teacher-forced programs (PLAN-v2.1 rule 7)."""
    return solve_cells(fit_rows(model, programs, workers))[0]


# ---- workers and CLI --------------------------------------------------------------------------

_model: IntModel | None = None
_CTX = mp.get_context("spawn")   # forkserver (3.14 default) needs a socket; fork warns


def _set_model(model: IntModel) -> None:
    global _model
    _model = model


def load_model(export, horizons: bool = False) -> IntModel:
    if not horizons:
        return IntModel.load(export)
    if "horizons" not in inspect.signature(IntModel.load).parameters:
        raise RuntimeError("--horizons: this intref has no ALiBi horizons (IntModel.load(dir, "
                           "horizons=True) is missing)")
    return IntModel.load(export, horizons=True)


def _load_worker(export: str, horizons: bool) -> None:
    _set_model(load_model(export, horizons))


def _sample_one(seed: int, rules: Rules, arm: str | None, E, cap: int, horizons: bool,
                offsets_sha256: str | None, mask_level: int = 1) -> dict:
    out = generate(_model, rules, seed, E=E, cap=cap, mask_level=mask_level)
    level = {"mask_level": mask_level} if mask_level != 1 else {}   # level 1: records as before
    return {"seed": seed, "temp": TEMP, "top_k": TOP_K, "cap": cap, "masks": True, "arm": arm,
            "horizons": horizons, "offsets_sha256": offsets_sha256, **level, **out}


def sample_records(export, rules: Rules, seeds, E=None, cap: int = CAP, horizons: bool = False,
                   workers: int = 16, arm: str | None = None,
                   offsets_sha256: str | None = None, mask_level: int = 1) -> list[dict]:
    """One record per seed, in seed order (the export is loaded once per worker)."""
    job = partial(_sample_one, rules=rules, arm=arm, E=E, cap=cap, horizons=horizons,
                  offsets_sha256=offsets_sha256, mask_level=mask_level)
    with ProcessPoolExecutor(workers, mp_context=_CTX, initializer=_load_worker,
                             initargs=(str(export), horizons)) as pool:
        return list(pool.map(job, seeds, chunksize=1))


def _cmd_sample(a) -> None:
    rules = ARMS[a.arm]
    E = sha = None
    if a.offsets:
        raw = Path(a.offsets).read_bytes()
        off, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        # the export path only: offsets fitted without horizons serve --horizons runs too
        if "export" not in off or Path(off["export"]).resolve() != Path(a.export).resolve():
            raise SystemExit(f"--offsets {a.offsets} was fitted on export {off.get('export')!r}, "
                             f"not {a.export}")
        E = [int(x) for x in off["E"]]
        if bool(off.get("hazard_full", False)) != rules.hazard_full:
            raise SystemExit(f"--offsets {a.offsets} were fitted with hazard_full="
                             f"{off.get('hazard_full', False)}; arm {a.arm} needs "
                             f"{rules.hazard_full}")
        if rules.hazard_full and bool(off.get("horizons", False)) != a.horizons:
            raise SystemExit(f"--offsets {a.offsets} were fitted with horizons="
                             f"{off.get('horizons', False)}, this run has {a.horizons}")
    elif rules.hazard:
        raise SystemExit(f"arm {a.arm} uses the stop hazard: pass --offsets (from `fit`)")
    t0 = time.time()
    recs = sample_records(a.export, rules, list(range(a.seed0, a.seed0 + a.n)), E, a.cap,
                          a.horizons, a.workers, a.arm, sha, a.mask_level)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tot = {k: sum(r["stats"][k] for r in recs) for k, v in recs[0]["stats"].items()
           if isinstance(v, int)}
    print(json.dumps({"arm": a.arm, "n": len(recs),
                      "complete": sum(r["ids"][-1] == EOS for r in recs),
                      "mean_len": round(sum(len(r["ids"]) for r in recs) / len(recs), 1), **tot,
                      "wall_s": round(time.time() - t0, 1)}))


def _cmd_fit(a) -> None:
    t0 = time.time()
    progs = train_programs(a.data, a.n_programs)
    rows = fit_rows(load_model(a.export, a.horizons), progs, a.workers, full=a.hazard_full)
    b, cells = solve_cells(rows)
    out = {"b": b, "E": offsets_to_E(b), "n_programs": len(progs), "cells": cells,
           "skipped_only_eos": {"n": int((rows[:, 0] == -1).sum()),
                                "eos": int(rows[rows[:, 0] == -1, 3].sum())},
           "programs_sha256": programs_sha256(progs), "export": str(Path(a.export).resolve()),
           "hazard_full": a.hazard_full, "horizons": a.horizons,
           "temperature_q8": FULL_TEMP_Q8 if a.hazard_full else TEMP_Q8,
           "top_k": "all" if a.hazard_full else TOP_K, "positions": FIT_POSITIONS,
           "wall_s": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"b": [round(x, 3) for x in b], "wall_s": out["wall_s"]}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="sample programs with one arm")
    s.add_argument("--export", required=True, type=Path)
    s.add_argument("--arm", required=True, choices=list(ARMS))
    s.add_argument("--offsets", type=Path, help="fit output (its integer E); needed by HZ/LD/LDs")
    s.add_argument("--horizons", action="store_true")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed0", type=int, default=1)
    s.add_argument("--cap", type=int, default=CAP)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--mask-level", type=int, choices=(1, 2), default=1,
                   help="1: v2.3's masks; 2: v2.4's safety masks (model/safety.py)")
    f = sub.add_parser("fit", help="fit the 16 stop-hazard offsets on train programs")
    f.add_argument("--export", required=True, type=Path)
    f.add_argument("--data", default=Path("data/tok"), type=Path)
    f.add_argument("--n-programs", type=int, default=2000)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--hazard-full", action="store_true",
                   help="E2b: fit rule 7 on T 1.0 weights over every allowed id")
    f.add_argument("--horizons", action="store_true", help="fit with ALiBi horizons on")
    a = ap.parse_args(argv)
    _cmd_sample(a) if a.cmd == "sample" else _cmd_fit(a)


if __name__ == "__main__":
    main()
