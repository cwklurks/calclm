"""The math tier: is a generated program a working, input-driven math program?

Judged from two `lab.score --yields` runs of the same samples, one typed "2" and one typed "7"
(the harness types that value at every Input/Prompt). Per program (trimmed ids, record A from
the "2" run, record B from the "7" run):
- takes_input: the program asks for a value: it contains an Input or Prompt token (static, from
  the trimmed tokens).
- answered: both records show the harness answered at least one input wait.
- no_error: both outcomes are OK or TIMEOUT.
- computes: lab.function.computed2 is True for both (a visible, non-trivial result).
- responds: the input changes what is shown: the response lines of A differ from B's, where a
  response line is a visible line that is not an echo of the typed value (just the value, or
  "?" then the value: "A=?2") and not a sub- or superstring of one of the program's string
  literals (computed2's rule; it drops "A=2" after Input "A=",A).
- math: takes_input and answered and no_error and computes and responds.
The typed pair is a parameter (default "2" and "7"); E4's held-out secondary uses "3" and "5".

    lab.score --yields --input-text 2 --samples S --out DIR_A
    lab.score --yields --input-text 7 --samples S --out DIR_B   # the same samples
    python -m lab.mathtier --samples S --a DIR_A --b DIR_B      # metrics as JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.function import _ID, _checked_run, aligned, computed2, string_literals, vis_lines
from lab.score import read_samples

INPUT, PROMPT = _ID["DC"], _ID["DD"]
TYPED_A, TYPED_B = "2", "7"
FLAGS = ("takes_input", "answered", "no_error", "computes", "responds", "math")


def takes_input(ids) -> bool:
    """The program asks for a value: an Input or Prompt token is in its trimmed tokens."""
    return INPUT in ids or PROMPT in ids


def response_lines(r: dict, typed: str, ids=()) -> set[str]:
    """vis_lines without the lines that merely echo `typed`, the harness's answer: a line that,
    with spaces removed, is the value itself or ends with "?" followed by it, and without lines
    that are sub- or superstrings of the program's string literals (computed2's rule; it drops
    the echo after an Input prompt string, "A=2")."""
    lits = string_literals(ids)
    out = set()
    for line in vis_lines(r):
        s = line.replace(" ", "")
        if s == typed or s.endswith("?" + typed) or any(line in L or L in line for L in lits):
            continue
        out.add(line)
    return out


def classify(ids, rec_a: dict, rec_b: dict, typed=(TYPED_A, TYPED_B)) -> dict:
    """ids: the trimmed program; rec_a/rec_b: its run records, typed typed[0] and typed[1]
    ("2" and "7" unless E4's held-out pair, "3" and "5", is given)."""
    c = {"takes_input": takes_input(ids),
         "answered": all((r.get("actions") or {}).get("input", 0) >= 1 for r in (rec_a, rec_b)),
         "no_error": all(r["outcome"] in ("OK", "TIMEOUT") for r in (rec_a, rec_b)),
         "computes": computed2(ids, rec_a) and computed2(ids, rec_b),
         "responds": response_lines(rec_a, typed[0], ids) != response_lines(rec_b, typed[1], ids)}
    c["math"] = all(c[k] for k in ("takes_input", "answered", "no_error", "computes", "responds"))
    return c


def math_metrics(samples: list[list[int]], dir_a: Path, dir_b: Path, cap: int = 1024,
                 typed=(TYPED_A, TYPED_B)) -> dict:
    """Percentages of the programs run (n_run) for every flag, and the count of math programs.
    Both dirs must be lab.score --yields runs of the same samples, typed typed[0] and typed[1]
    ("2" and "7" by default)."""
    _, ca = _checked_run(dir_a, cap)
    _, cb = _checked_run(dir_b, cap)
    got = (ca.get("config", {}).get("input_text"), cb.get("config", {}).get("input_text"))
    if got != tuple(typed):
        raise ValueError(f"{dir_a}, {dir_b}: input_text must be {typed[0]!r} and {typed[1]!r}, "
                         f"not {got!r} (lab.score --yields --input-text)")
    A, B = aligned(samples, dir_a, cap), aligned(samples, dir_b, cap)
    flags = [classify(A[i][1], A[i][2], B[i][2], typed) for i in sorted(set(A) & set(B))]
    n = len(flags)
    out = {"n_run": n, "n_math": sum(f["math"] for f in flags)}
    out |= {k: (100 * sum(f[k] for f in flags) / n if n else None) for k in FLAGS}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=True, type=Path, help="jsonl of {ids: [...]} or lists")
    ap.add_argument("--a", required=True, type=Path, help='the "2" run: lab.score --out DIR')
    ap.add_argument("--b", required=True, type=Path, help='the "7" run: lab.score --out DIR')
    ap.add_argument("--cap", type=int, default=1024)
    a = ap.parse_args(argv)
    print(json.dumps(math_metrics(read_samples(a.samples), a.a, a.b, a.cap), indent=1))


if __name__ == "__main__":
    main()
