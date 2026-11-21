"""The v2.1 analysis metrics (PLAN-v2.1.md, "Common setup", Metrics) over a lab.score run.

    run_metrics(samples, results_dir, cap) -> dict    one run's metrics, unrounded (percent;
                                                      distinct_4 and overlap_32 as score.json's
                                                      ratios; None when undefined)
    per_sample(samples, results_dir, cap) -> dict     per sample index: strict, computed2,
                                                      strict2, lus booleans (paired sign tests)

`samples` are the raw sampled id lists in seed order; `results_dir` is lab.score's --out for
them (results.jsonl, score.json, config.json), scored with --yields at --max-len `cap`; both
functions refuse a run that is not.

Moved from the v3-angles scratch code (experiments/v3-angles, commit 6f93203) without semantic
change, except for the plan's amendment (before any decision sample), which makes two rules
follow TI-OS (checked in the emulator):
- string literals: a string opens at a quote and closes at the next quote, →, newline or <eos>;
  a colon is part of it, and a string still open where the (cut) program ends is a literal.
  guards.literals ended strings at a colon; tail.literals ran on past → and dropped an open one.
- TRIV: TI-OS shows a negative number with ⁻ (U+207B), which now counts like the ASCII minus.
Amendment 3: axes_only ignores the TI-OS home screen's "Done" and cursor (HOME_DONE_CELLS).
Each function names its source:
- rt/fullscore.py: computed2, metrics (-> run_metrics), pair
- rt/rtcommon.py: aligned, cls (-> outcome_cls), axes_only, strict2, signp
- rt/tail.py: literals (-> positioned_literals), profile's tail-only share (-> tail_only_share)
- behaviour-tags/guards.py: literals (-> string_literals), vis_lines, computed
- behaviour-tags/metrics.py: cls (-> shown_cls), real
- failure-anatomy/common.py: names
- linker-wildcard/cfg.py: statements, name_of, menu_labels, analyze (dead-code share)
string_literals (computed2) and positioned_literals (tail-only) share that string rule and keep
their own filters.
"""
from __future__ import annotations

import functools
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from eval.truncate import trim_to_line
from lab.score import ECHO, ERROR_CLASSES, VOCAB, is_strict_ok, newline_id, visible_lines
from model.tokens import EOS
from tracker.spec import default_vocab, roles

NL = newline_id(VOCAB)
QUOTE, STORE = (json.loads(VOCAB.read_text()).index(h) for h in ("2A", "04"))   # " and →
PROMPT = re.compile(r"^.{0,16}\?\s*█?$")   # a bare Input/Prompt prompt line, cursor shown
TRIV = re.compile(r"^[-⁻]?[01](\.0*)?$|^0+$")  # a line showing only 0 or ±1 (- or TI's ⁻)


@functools.cache
def names() -> tuple[str, ...]:
    """Display name of every vocab id (tivars, English); specials keep their key."""
    from tivars.models import TI_84PCE

    d = {b.hex().upper(): t.langs["en"].display for b, t in TI_84PCE.tokens.bytes.items()}
    return tuple(d.get(h, h) for h in default_vocab())


# ---- string literals -------------------------------------------------------------------------

def _strings(ids) -> list[tuple[int, list[int]]]:
    """(position of the opening quote, token ids inside) of every string literal: it closes at
    the next quote, →, newline or <eos>, a colon is part of it, and one still open at the end of
    the ids (a cut program) counts."""
    out, cur, start = [], None, 0
    for i, t in enumerate(ids):
        if cur is None:
            if t == QUOTE:
                cur, start = [], i
        elif t in (QUOTE, STORE, NL, EOS):
            out.append((start, cur))
            cur = None
        else:
            cur.append(t)
    if cur is not None:
        out.append((start, cur))
    return out


def string_literals(ids) -> list[str]:
    """guards.literals: the text of every string literal after <bos>, stripped, blank ones
    dropped."""
    nm = names()
    texts = ("".join(nm[t] for t in toks).strip() for _, toks in _strings(ids[1:]))
    return [x for x in texts if x]


def positioned_literals(ids) -> list[tuple[int, str]]:
    """tail.literals: (position of the opening quote, stripped text without special tokens) of
    every string literal of 3+ characters."""
    nm = names()
    out = []
    for start, toks in _strings(ids):
        s = "".join(nm[t] for t in toks if t > 3).strip()
        if len(s) >= 3:
            out.append((start, s))
    return out


# ---- screens -----------------------------------------------------------------------------------

def real(lines) -> list[str]:
    """metrics.real: stripped non-blank lines that are neither the harness's input echo nor a
    bare prompt."""
    return [x.strip() for x in lines
            if x.strip() and not ECHO.match(x.strip()) and not PROMPT.match(x.strip())]


def vis_lines(r: dict) -> set[str]:
    """guards.vis_lines: real lines of the final home screen and of every yield screen, with the
    cursor block removed."""
    lines = real(r.get("home_text") or [])
    for y in r.get("yields", []):
        lines += real(y[3])
    return {x.strip().replace("█", "") for x in lines if x.strip().replace("█", "")}


# The TI-OS home screen after a program returns (emulator screenshots of s10 G2b programs that
# showed a graph and then quit): "Done" right-aligned on row 0, which the screen reader reads as
# four U+FFFD, and the entry cursor on row 1, read as "_". Not program output; axes_only drops
# these cells, only when all five are there with exactly these glyphs.
HOME_DONE_CELLS = {(0, 22): "�", (0, 23): "�", (0, 24): "�", (0, 25): "�", (1, 0): "_"}


def axes_only(r: dict) -> bool:
    """rtcommon.axes_only: the screen is blank, or every lit cell is on its busiest row or its
    busiest column (bare axes); HOME_DONE_CELLS removed first."""
    lcd = r.get("lcd_text") or []
    lit = {(i, j): ch for i, row in enumerate(lcd) for j, ch in enumerate(row) if ch != " "}
    if all(lit.get(k) == ch for k, ch in HOME_DONE_CELLS.items()):
        lit = {k: ch for k, ch in lit.items() if k not in HOME_DONE_CELLS}
    pts = list(lit)
    if not pts:
        return True
    rows = Counter(i for i, _ in pts)
    cols = Counter(j for _, j in pts)
    r0 = rows.most_common(1)[0][0]
    c0 = cols.most_common(1)[0][0]
    return all(i == r0 or j == c0 for i, j in pts)


# ---- per-program verdicts ----------------------------------------------------------------------

def strict2(r: dict) -> bool:
    """rtcommon.strict2: strict OK, but a graph screen counts only if it shows more than axes."""
    if r["outcome"] != "OK":
        return False
    lines = visible_lines(r)
    if lines and not all(ECHO.match(x) for x in lines):
        return True
    return bool(r.get("graph")) and not axes_only(r)


def computed2(ids, r: dict) -> bool:
    """fullscore.computed2: OK or TIMEOUT, and a graph that is more than axes, or a visible line
    (final or yield screen) that is no sub- or superstring of a string literal and not only 0/1."""
    if r["outcome"] not in ("OK", "TIMEOUT"):
        return False
    if r.get("graph") and not axes_only(r):
        return True
    lits = string_literals(ids)
    comp = [x for x in vis_lines(r) if not any(x in L or L in x for L in lits)]
    return any(not TRIV.match(x.strip()) for x in comp)


def shown_cls(r: dict) -> dict:
    """metrics.cls: what a run showed (strict, shown_ok, shown_any, flash, noerr)."""
    o = r["outcome"]
    end = real(r.get("home_text") or [])
    yt = [x for y in r.get("yields", []) for x in real(y[3])]
    seen = real(r.get("seen") or [])
    graph = bool(r.get("graph"))
    shown_any = o in ("OK", "TIMEOUT") and (graph or bool(end) or bool(yt))
    return dict(o=o, strict=o == "OK" and (graph or bool(end)),
                shown_ok=o == "OK" and (graph or bool(end) or bool(yt)), shown_any=shown_any,
                flash=o in ("OK", "TIMEOUT") and not shown_any and bool(seen),
                noerr=o in ("OK", "TIMEOUT"), ts=r.get("timeout_state"))


def computed(ids, r: dict) -> tuple[bool, bool]:
    """guards.computed (the first 'computed', superseded by computed2): (shown, computed)."""
    c = shown_cls(r)
    if not (c["shown_ok"] or c["shown_any"]):
        return False, False
    lits = string_literals(ids)
    comp = [x for x in vis_lines(r) if not any(x in L or L in x for L in lits)]
    return True, bool(comp) or bool(r.get("graph"))


def outcome_cls(r: dict) -> str:
    """rtcommon.cls: strict_ok, blank_ok (OK but not strict), ERR:<NAME>, or the outcome."""
    if is_strict_ok(r):
        return "strict_ok"
    if r["outcome"] == "OK":
        return "blank_ok"
    if r["outcome"] == "ERR":
        return "ERR:" + str(r["error"])
    return r["outcome"]


def is_lus(r: dict) -> bool:
    """LABEL, UNDEFINED or SYNTAX: lab.score's err_label + err_undefined + err_syntax."""
    return r["outcome"] == "ERR" and r["error"] in ERROR_CLASSES


# ---- control flow (linker-wildcard/cfg.py) -----------------------------------------------------

_R = roles(default_vocab())
_ID = {h: i for i, h in enumerate(default_vocab())}
STOP, RET = _ID["D9"], _ID["D5"]
BOS = 1


def statements(ids) -> list[tuple[int, int, list[int]]]:
    """(start, end exclusive, tokens) per statement; newline or colon ends one outside strings."""
    out, cur, start, in_str = [], [], 0, False
    for i, t in enumerate(ids):
        if t == BOS:
            start = i + 1
            continue
        role = _R[t][0] if t < len(_R) else "other"
        if t == EOS:
            break
        if in_str:
            if role in ("quote", "store"):
                in_str = False
            elif role == "newline":
                out.append((start, i, cur))
                cur, start, in_str = [], i + 1, False
                continue
            cur.append(t)
            continue
        if role in ("newline", "colon"):
            out.append((start, i, cur))
            cur, start = [], i + 1
            continue
        if role == "quote":
            in_str = True
        cur.append(t)
    if cur:
        out.append((start, len(ids), cur))
    return out


def name_of(toks) -> str:
    s = ""
    for t in toks:
        r, a = _R[t]
        if r != "name":
            break
        s += a
    return s[:2]


def menu_labels(toks) -> list[str]:
    """Label names of a Menu( statement's arguments (tokens after Menu()."""
    args, depth, in_str = [[]], 0, False
    for t in toks:
        r = _R[t][0]
        if in_str:
            if r == "quote":
                in_str = False
            args[-1].append(t)
            continue
        if r == "quote":
            in_str = True
            args[-1].append(t)
            continue
        if r in ("lparen", "func", "statopen", "dim", "equstr", "listmatr", "matrlist", "for",
                 "isds", "menu"):
            depth += 1
        if r == "rparen":
            if depth == 0:
                break
            depth -= 1
        if r == "comma" and depth == 0:
            args.append([])
            continue
        args[-1].append(t)
    return [name_of(a) for k, a in enumerate(args) if k >= 2 and k % 2 == 0 and name_of(a)]


def _statement_info(sts) -> list[dict]:
    info, depth, if_pending = [], 0, False
    for s, e, toks in sts:
        if not toks:
            info.append(dict(s=s, e=e, kind="empty", depth=depth, body=False))
            continue
        r0 = _R[toks[0]][0]
        body = False
        if if_pending:
            if_pending = False
            if r0 == "then":
                depth += 1
                info.append(dict(s=s, e=e, kind="then", depth=depth - 1, body=False))
                continue
            body = True
        d, lbl, refs = depth, None, []
        if r0 == "if":
            if_pending = True
        elif r0 in ("for", "while", "repeat"):
            depth += 1
        elif r0 == "end":
            depth = max(0, depth - 1)
        elif r0 == "lbl":
            lbl = name_of(toks[1:])
        elif r0 == "goto":
            refs = [name_of(toks[1:])]
        elif r0 == "menu":
            refs = menu_labels(toks[1:])
        transfer = (r0 in ("goto", "menu") or toks[0] in (STOP, RET)) and not body and d == 0
        info.append(dict(s=s, e=e, kind=r0, depth=d, body=body, lbl=lbl,
                         refs=[x for x in refs if x], transfer=transfer, n=e - s + 1))
    return info


def analyze(ids):
    """cfg.analyze: static top-level control flow. Returns (statement info, regions split at
    every Lbl, region reachable, statement dead, label -> region)."""
    info = _statement_info(statements(ids))
    regions = [[]]
    for k, st in enumerate(info):
        if st.get("lbl"):
            regions.append([])
        regions[-1].append(k)
    lbl_region = {}
    for ri, reg in enumerate(regions):
        for k in reg:
            if info[k].get("lbl") and info[k]["lbl"] not in lbl_region:
                lbl_region[info[k]["lbl"]] = ri
    live_end, falls = [], []      # statements after live_end are dead; falls into the next
    for reg in regions:
        cut, f = len(reg), True
        for j, k in enumerate(reg):
            if info[k].get("transfer"):
                cut, f = j + 1, False
                break
        live_end.append(cut)
        falls.append(f)
    reach = [False] * len(regions)
    reach[0] = True
    stack = [0]
    while stack:
        ri = stack.pop()
        for k in regions[ri][:live_end[ri]]:
            for nm in info[k].get("refs", []):
                t = lbl_region.get(nm)
                if t is not None and not reach[t]:
                    reach[t] = True
                    stack.append(t)
        if falls[ri] and ri + 1 < len(regions) and not reach[ri + 1]:
            reach[ri + 1] = True
            stack.append(ri + 1)
    dead = [False] * len(info)
    for ri, reg in enumerate(regions):
        for j, k in enumerate(reg):
            if not reach[ri] or j >= live_end[ri]:
                dead[k] = True
    return info, regions, reach, dead, lbl_region


def dead_share(ids) -> float:
    """fullscore's dead code: tokens in dead statements over tokens in all statements, spans
    counted as end - start + 1."""
    info, _, _, dead, _ = analyze(ids)
    m = sum(s["e"] - s["s"] + 1 for s in info) or 1
    return sum(s["e"] - s["s"] + 1 for s, x in zip(info, dead) if x) / m


# ---- a whole run -------------------------------------------------------------------------------

def aligned(samples, results_dir: Path, cap: int = 1024) -> dict[int, tuple[list, list, dict]]:
    """rtcommon.aligned: sample index -> (raw ids, trimmed ids, result record), for the samples
    lab.score ran: trim_to_line at cap, programs left as <bos> alone are not run."""
    path = Path(results_dir) / "results.jsonl"
    recs = sorted((json.loads(x) for x in path.read_text().splitlines() if x.strip()),
                  key=lambda r: r["index"])
    trimmed = [trim_to_line(list(map(int, s)), cap, NL, EOS) for s in samples]
    keep = [i for i, t in enumerate(trimmed) if len(t) > 1]
    if len(keep) != len(recs):
        raise ValueError(f"{path}: {len(recs)} records for {len(keep)} programs kept at cap {cap}")
    return {i: (list(samples[i]), trimmed[i], r) for i, r in zip(keep, recs)}


def tail_only_share(A: dict, tail: int = 64) -> float | None:
    """tail.profile's tail_only (unrounded): among strict-OK programs, the share whose earliest
    on-screen string literal (3+ characters, final home screen) starts in the last `tail`
    tokens of the trimmed program. None without strict-OK programs."""
    keys = [i for i in A if is_strict_ok(A[i][2])]
    tail_only = 0
    for i in keys:
        _, t, r = A[i]
        scr = " ".join(visible_lines(r))
        m = [p for p, s in positioned_literals(t) if s in scr]
        if m and min(m) >= len(t) - tail:
            tail_only += 1
    return tail_only / len(keys) if keys else None


def _checked_run(results_dir: Path, cap: int) -> tuple[dict, dict]:
    """(score.json, config.json) of a lab.score run cut at `cap` with yield capture on."""
    d = Path(results_dir)
    score = json.loads((d / "score.json").read_text())
    if score.get("max_len") != cap:
        raise ValueError(f"{d}: score.json max_len is {score.get('max_len')}, not {cap}")
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    if config.get("yields") is not True:
        raise ValueError(f"{d}: config.json does not say yields: true (score with --yields)")
    return score, config


def _pct(flags) -> float | None:
    return 100 * np.mean(flags) if len(flags) else None


def run_metrics(samples: list[list[int]], results_dir: Path, cap: int = 1024) -> dict:
    """fullscore.metrics plus tail-only, L+U+S, lab.score's distinct-4 and overlap-32, and the
    harness's own failures. Unrounded.

    Percentages over the programs run (n_run), except complete (over n_samples), fin_* (over
    finished samples: <eos> within cap; fin_len_med in tokens) and tail_only (over strict-OK
    programs); None when that denominator is 0. distinct_4 and overlap_32 are score.json's
    distinct_4 and overlap_32.mean_seen (ratios). harness_errors counts HARNESS_ERROR records (a
    worker that died, raised or gave no result in time); wall_limit_hits counts records whose
    wall_s exceeds the driver's wall_limit_s (every wall-limit stop is one; the driver stops
    answering then, so those outcomes depend on host load)."""
    sj, config = _checked_run(results_dir, cap)
    A = aligned(samples, results_dir, cap)
    v = list(A.values())
    R = {"strict": _pct([is_strict_ok(r) for _, _, r in v]),
         "strict2": _pct([strict2(r) for _, _, r in v]),
         "computed2": _pct([computed2(t, r) for _, t, r in v])}
    c = [outcome_cls(r) for _, _, r in v]
    for key, k in (("label", "ERR:LABEL"), ("timeout", "TIMEOUT"), ("syntax", "ERR:SYNTAX"),
                   ("undefined", "ERR:UNDEFINED")):
        R[key] = _pct([x == k for x in c])
    R["lus"] = R["label"] + R["undefined"] + R["syntax"] if v else None
    R["noerr"] = _pct([r["outcome"] in ("OK", "TIMEOUT") for _, _, r in v])
    R["complete"] = 100 * sum(EOS in s[:cap] for s, _, _ in v) / len(samples) if samples else None
    f = np.array([len(s) - 2 for s, _, _ in v if EOS in s[:cap]])
    R["fin_len_med"] = float(np.median(f)) if len(f) else None
    R["fin_le40"] = 100 * float(np.mean(f <= 40)) if len(f) else None
    share = tail_only_share(A)
    R["tail_only"] = 100 * share if share is not None else None
    R["dead"] = _pct([dead_share(t) for _, t, _ in v])
    out = {k: None if x is None else float(x) for k, x in R.items()}
    limit = config.get("config", {}).get("wall_limit_s", float("inf"))
    out.update(distinct_4=sj["distinct_4"], overlap_32=sj["overlap_32"]["mean_seen"],
               n_samples=len(samples), n_run=len(A),
               harness_errors=sum(r["outcome"] == "HARNESS_ERROR" for _, _, r in v),
               wall_limit_hits=sum((r.get("wall_s") or 0) > limit for _, _, r in v))
    return out


def per_sample(samples: list[list[int]], results_dir: Path, cap: int = 1024) -> dict[int, dict]:
    """Sample index (seed order, 0-based) -> {strict, computed2, strict2, lus}, for every sample
    lab.score ran."""
    _checked_run(results_dir, cap)
    return {i: {"strict": is_strict_ok(r), "computed2": computed2(t, r), "strict2": strict2(r),
                "lus": is_lus(r)}
            for i, (_, t, r) in aligned(samples, results_dir, cap).items()}


def signp(up: int, dn: int) -> float:
    """rtcommon.signp: two-sided exact sign-test p for up vs dn discordant pairs."""
    n = up + dn
    if not n:
        return 1.0
    k = min(up, dn)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def pair(B: dict, A: dict, f) -> tuple[int, int, str]:
    """fullscore.pair: over the indices both runs share, (A gains, A losses, sign-test p)."""
    c = sorted(set(A) & set(B))
    up = sum(f(A[i]) and not f(B[i]) for i in c)
    dn = sum(f(B[i]) and not f(A[i]) for i in c)
    return up, dn, "%.2g" % signp(up, dn)
