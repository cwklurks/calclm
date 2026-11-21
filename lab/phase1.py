"""Phase 1 acceptance checks for the v2.4 safety masks (docs/agents/phase1-safety-masks.md
section 5). CPU only; nothing here runs the emulator.

- `scan`: blocked patterns in sampled programs, counted on generated tokens only (a prompt's
  tokens are skipped). Written apart from the mask code: it lexes the program itself (strings end
  at ", -> or a newline; statements at a newline or a colon outside strings) and matches regexes
  on the strings' display text, where every token shown as one character is that character and
  every longer token is one non-letter placeholder. Strings joined by + ("B"+"Y") are matched as
  one. Only the token lists come from model.safety (they are the specification). The same
  patterns on the full display text (longer tokens spelled out) are counted apart, report-only
  (display_*): they are outside what the masks block, e.g. sin(C) shows "(C)".
- `compare`: the pass rates of two judged runs of the same (task, seed) jobs, paired, against
  the pre-stated tolerance TAU (2 SE of the difference of two independent rates at rc2's prompted
  dev rate, 139/2000).

    python -m lab.phase1 scan --samples S.jsonl [--prompted] --out OUT.json
    python -m lab.phase1 compare --a A/judged.jsonl --b B/judged.jsonl --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from model import safety as S

STORE, QUOTE, NEWLINE, COLON = "04", "2A", "3F", "3E"
IO, LOOPVAR = ("DC", "DD"), ("D3", "DA", "DB")       # Input Prompt; For( IS>( DS<(
PATTERNS = {"str_by": re.compile(r"(?<![a-z])by", re.I),   # a word starting with BY
            "str_copyr": re.compile(r"copyr", re.I),
            "str_c": re.compile(r"\(c\)", re.I)}
KEYS = ("always", "window_store", *PATTERNS)
DISPLAY_KEYS = tuple("display_" + k[4:] for k in PATTERNS)
PLACEHOLDER = "\x01"
PLUS = "70"


def tolerance(passes: int, n: int) -> float:
    """2 SE of the difference of two independent rates at rate passes / n, n jobs each."""
    p = passes / n
    return 2 * math.sqrt(2 * p * (1 - p) / n)


TAU = tolerance(139, 2000)          # rc2's prompted dev rate (release review): 1.61 points


@lru_cache(maxsize=4)
def _full_display(vocab: tuple[str, ...]) -> tuple[str, ...]:
    from tivars.models import TI_84PCE
    sheet = TI_84PCE.tokens.bytes
    return tuple("" if h.startswith("<") else sheet[bytes.fromhex(h)].langs["en"].display
                 for h in vocab)


@lru_cache(maxsize=4)
def _display(vocab: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(d if len(d) <= 1 else PLACEHOLDER for d in _full_display(vocab))


def _string_hits(toks: list[tuple[int, bool]], vocab, counts: dict) -> None:
    """Pattern matches in one string (or +-joined strings) whose last character is generated:
    on the token-class text (counted) and on the full display text (display_*, report-only)."""
    for disp, keys in ((_display(tuple(vocab)), PATTERNS), (_full_display(tuple(vocab)),
                                                             DISPLAY_KEYS)):
        chars, gen = [], []
        for t, g in toks:
            chars.append(disp[t])
            gen += [g] * len(disp[t])
        text = "".join(chars)
        for key, pat in zip(keys, PATTERNS.values()):
            counts[key] += sum(gen[m.end() - 1] for m in pat.finditer(text))


def scan(ids, vocab, start: int) -> dict:
    """Counts of blocked patterns among ids[start:] (ids[:start] is context only)."""
    always = {i for i, h in enumerate(vocab) if h in set(S.ALWAYS_HEX)}
    window = {i for i, h in enumerate(vocab) if h.startswith(S.WINDOW_PAGE)}
    counts = dict.fromkeys(KEYS + DISPLAY_KEYS, 0)
    in_str, at_start, first, prev = False, True, None, None
    toks: list[tuple[int, bool]] = []      # the current string's tokens (joined across "+")
    join = 0                               # 1: a string closed by a quote, 2: then a +
    for pos, t in enumerate(ids):
        t = int(t)
        h, g = vocab[t], pos >= start
        if g and t in always:
            counts["always"] += 1
        if join and not in_str and not (join == 1 and h == PLUS) and not (join == 2 and h == QUOTE):
            _string_hits(toks, vocab, counts)
            join, toks = 0, []
        if h.startswith("<"):                      # <bos>, <eos>: a string ends with the program
            if in_str:
                _string_hits(toks, vocab, counts)
            in_str, at_start, first, prev, toks = False, True, None, None, []
            continue
        if in_str:
            if h in (QUOTE, STORE, NEWLINE):
                in_str = False
                if h == QUOTE:
                    join = 1                       # a + and a quote may continue it
                else:
                    _string_hits(toks, vocab, counts)
                    toks = []
                if h == NEWLINE:
                    at_start, first = True, None
            else:
                toks.append((t, g))
            prev = h
            continue
        if join == 1 and h == PLUS:
            join = 2
        elif join == 2 and h == QUOTE:
            join, in_str = 0, True
            prev = h
            continue
        if g and t in window and (prev == STORE or first in IO or prev in LOOPVAR):
            counts["window_store"] += 1
        if h == QUOTE:
            in_str, toks = True, []
        if h in (NEWLINE, COLON):
            at_start, first = True, None
        elif at_start:
            at_start, first = False, h
        prev = h
    if in_str or join:
        _string_hits(toks, vocab, counts)
    counts["total"] = sum(counts[k] for k in KEYS)
    return counts


def compare(a: list[dict], b: list[dict], tau: float = TAU) -> dict:
    """b's pass rate minus a's over the same (task, seed) jobs: d, the paired SE and 95% CI, a
    task-clustered SE, passes lost and gained; pass when d >= -tau."""
    pa = {(r["task"], r["seed"]): bool(r["pass"]) for r in a}
    pb = {(r["task"], r["seed"]): bool(r["pass"]) for r in b}
    if pa.keys() != pb.keys() or len(pa) != len(a) or len(pb) != len(b):
        raise ValueError("the two runs must hold the same (task, seed) jobs, once each")
    keys = sorted(pa)
    n = len(keys)
    diff = [int(pb[k]) - int(pa[k]) for k in keys]
    d = sum(diff) / n
    se = math.sqrt(sum((x - d) ** 2 for x in diff) / (n - 1) / n)
    by_task = defaultdict(float)
    for k, x in zip(keys, diff):
        by_task[k[0]] += x - d
    se_task = math.sqrt(sum(v * v for v in by_task.values())) / n
    return {"n": n, "pass_a": sum(pa.values()), "pass_b": sum(pb.values()),
            "rate_a": sum(pa.values()) / n, "rate_b": sum(pb.values()) / n, "d": d,
            "se_paired": se, "ci95": [d - 1.96 * se, d + 1.96 * se], "se_task": se_task,
            "ci95_task": [d - 1.96 * se_task, d + 1.96 * se_task],
            "lost": sum(x < 0 for x in diff), "gained": sum(x > 0 for x in diff),
            "pairs_same": sum(pa[k] == pb[k] for k in keys), "tau": tau, "pass": d >= -tau}


# ---- CLI ----------------------------------------------------------------------------------------

def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def scan_file(path: Path, vocab: list[str], prompted: bool) -> dict:
    from lab.tasks_v2 import TASKS_V2
    rows = _rows(path)
    tot = dict.fromkeys((*KEYS, "total", *DISPLAY_KEYS), 0)
    hits = []
    for r in rows:
        start = 1 + len(TASKS_V2[r["task"]].opening_ids()) if prompted else 1
        c = scan(r["ids"], vocab, start)
        for k in tot:
            tot[k] += c[k]
        if c["total"] or any(c[k] for k in DISPLAY_KEYS):
            hits.append({k: r[k] for k in ("task", "seed") if k in r} | {
                k: v for k, v in c.items() if v and k not in DISPLAY_KEYS})
    return {"file": str(path), "n": len(rows), "programs_with_hits": len(hits), "counts": tot,
            "hits": hits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="blocked patterns in a samples .jsonl")
    s.add_argument("--samples", type=Path, required=True)
    s.add_argument("--prompted", action="store_true", help="rows have task openings to skip")
    s.add_argument("--vocab", type=Path, default=Path(__file__).resolve().parents[1] / "corpus"
                   / "vocab.json")
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="paired pass rates of two judged runs")
    c.add_argument("--a", type=Path, required=True, help="the reference arm (level 1)")
    c.add_argument("--b", type=Path, required=True, help="the arm under test (level 2)")
    c.add_argument("--tau", type=float, default=TAU)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        out = scan_file(a.samples, json.loads(a.vocab.read_text()), a.prompted)
    else:
        out = compare(_rows(a.a), _rows(a.b), a.tau)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "hits"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
