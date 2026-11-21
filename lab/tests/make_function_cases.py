"""Write function_cases.json beside this script: the public lab/tests/function_cases.json.

The private fixture holds model-generated programs and their emulator screens, which CalcLM
never publishes. This one has the same shape (samples, records, cases, metrics) and the same
case names that lab/tests/test_function.py asks for, but every program and every screen below
was written by hand for it: short, made-up programs, not sampled from the model and not taken
from the corpus. The screens are what TI-OS would show for them, written out by hand; nothing
here was run in the emulator.

Each case states the verdicts it was written to show (`want`). The script computes every
verdict with lab.function, checks that it agrees with `want`, and freezes the result. So the
public test_function.py is a regression test of lab.function on these programs, plus the
hand-stated verdicts; it is not the private fixture's check against the original scratch code.

    python -m lab.tests.make_function_cases          rewrite the JSON
    python -m lab.tests.make_function_cases --check  fail if the JSON is stale
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("function_cases.json")
CAP = 1024
MAX_IDS = 50          # tools.release_guard counts longer id arrays as programs; stay below it

# (name, program text, ends with <eos>, final screen, wanted verdicts, record extras)
# Screens: home_text lines as TI-OS shows them; "lcd" is a graph screen (rows of 26 cells) when
# given, else the home lines. An Input/Prompt line answered by the harness reads "X=?1".
AXES = [" " * 13 + "." + " " * 12] * 5 + ["." * 26] + [" " * 13 + "." + " " * 12] * 4
DONE = {(0, 22): "�", (0, 23): "�", (0, 24): "�", (0, 25): "�", (1, 0): "_"}


def grid(points=(), cells=None, base=None):
    g = [list(r) for r in (base or [" " * 26] * 10)]
    for i, j in points:
        g[i][j] = "."
    for (i, j), ch in (cells or {}).items():
        g[i][j] = ch
    return ["".join(r) for r in g]


CASES = [
    ("computed_home", "Prompt R\npiR*R→A\nDisp A", True, ["R=?1", "3.141592654"],
     dict(strict=True, strict2=True, computed2=True, tail_only=False), {}),
    ("literal_substring", 'Disp "GOOD JOB"', True, ["GOOD JOB"],
     dict(strict=True, computed2=False, tail_only=True), {}),
    ("literal_superstring", '5→X\nOutput(1,1,"X=")\nOutput(1,3,X)', True, ["X=5"],
     dict(strict=True, computed2=False), {}),
    # sample 3 is not a case: it has no newline and no <eos>, so trim_to_line leaves <bos>
    ("trivial_only", "0→A\nDisp A\nDisp 1", True, ["0", "1"],
     dict(strict=True, computed2=False), {}),
    ("yield_only", "Prompt N\nDisp 2N\nPause \nClrHome", True, [],
     dict(strict=False, strict2=False, computed2=True, cls="blank_ok"),
     dict(yields=[["input", 40, False, ["N=?█"]], ["pause", 80, False, ["N=?1", "2"]]])),
    ("yield_only_timeout", "Lbl A\nClrHome\nPrompt X\nDisp X+48\nPause \nGoto A", True, ["X=?"],
     dict(strict=False, computed2=True, cls="TIMEOUT"),
     dict(outcome="TIMEOUT", timeout_state="input",
          yields=[["input", 30, False, ["X=?█"]], ["pause", 70, False, ["X=?1", "49"]]])),
    ("axes_only_graph", "ClrDraw\nLine(⁻5,0,5,0)", True, [],
     dict(strict=True, strict2=False, computed2=False, axes_only=True),
     dict(graph=True, lcd=AXES)),
    ("drawn_graph", "ClrDraw\nPt-On(2,3)\nPt-On(⁻4,1)", True, [],
     dict(strict=True, strict2=True, computed2=True, axes_only=False),
     dict(graph=True, lcd=grid([(2, 18), (4, 5)], base=AXES))),
    ("minus_one", "⁻1→A\nWhile 1\nDisp A\nPause \nEnd", True, ["⁻1"],
     dict(strict=False, computed2=False, cls="TIMEOUT"),
     dict(outcome="TIMEOUT", timeout_state="pause")),
    ("blank_ok", "ClrHome", True, [],
     dict(strict=False, computed2=False, cls="blank_ok", axes_only=True), {}),
    ("echo_only", "Prompt A", True, ["A=?1"],
     dict(strict=False, computed2=False, cls="blank_ok"), {}),
    ("err_label", "Goto Z", True, [],
     dict(strict=False, lus=True, cls="ERR:LABEL"), dict(outcome="ERR", error="LABEL")),
    ("err_undefined", "Disp [A]", True, [],
     dict(strict=False, lus=True, cls="ERR:UNDEFINED"), dict(outcome="ERR", error="UNDEFINED")),
    ("err_syntax", "Disp 2+", True, [],
     dict(strict=False, lus=True, cls="ERR:SYNTAX"), dict(outcome="ERR", error="SYNTAX")),
    ("err_other", 'Disp "LOG ZERO"\nDisp log(0)', True, ["LOG ZERO"],
     dict(strict=False, lus=False, cls="ERR:DOMAIN"), dict(outcome="ERR", error="DOMAIN")),
    ("dead_code", 'Goto B\nDisp "SKIPPED"\nLbl B\nDisp 7', True, ["7"],
     dict(strict=True, computed2=True), {}),
    ("tail_only", '5→A\nDisp A\nDisp "DONE"', True, ["5", "DONE"],
     dict(strict=True, computed2=True, tail_only=True), {}),
    ("number_and_label", 'Disp "SUM"\nDisp 3+4', True, ["SUM", "7"],
     dict(strict=True, computed2=True, tail_only=True), {}),
    ("unfinished", 'Disp 3\nDisp 4\nDisp "PAR', False, ["3", "4"],
     dict(strict=True, computed2=True), {}),
    ("short_finished", "Stop", True, [],
     dict(strict=False, computed2=False, cls="blank_ok"), {}),
    ("colon_in_string", 'Disp "TIME:0"\nDisp 1', True, ["TIME:0", "1"],
     dict(strict=True, computed2=False), {}),
    ("store_ends_string", '"HI→Str1\nDisp ln(0)', True, [],
     dict(strict=False, computed2=False, cls="ERR:DOMAIN"), dict(outcome="ERR", error="DOMAIN")),
    ("unclosed_at_cut", 'Disp 5\nDisp "HI THERE\nDisp 6', False, ["5", "HI THERE"],
     dict(strict=True, tail_only=True), {}),
    ("done_screen", "ClrDraw\nPt-On(0,0)\nClrDraw", True, [],
     dict(strict=True, strict2=False, computed2=False, axes_only=True),
     dict(graph=True, lcd=grid(cells=DONE))),
    ("done_other_row", "ClrDraw\nPt-On(1,1)", True, [],
     dict(strict=True, strict2=True, computed2=True, axes_only=False),
     dict(graph=True, lcd=grid(cells={(2, 22): "�", (2, 23): "�", (2, 24): "�",
                                      (2, 25): "�", (3, 0): "_"}))),
]
NOT_RUN = (3, "Disp 12")      # (sample index, text): no newline, no <eos>


def encode(text: str, eos: bool) -> list[int]:
    from tivars.models import TI_84PCE
    from tivars.types import TIProgram

    from corpus.tokenize import Vocab

    ids = Vocab.load(ROOT / "corpus" / "vocab.json").encode(
        TIProgram.encode(text, model=TI_84PCE), add_special=True)
    ids = ids if eos else ids[:-1]
    if 3 in ids or len(ids) > MAX_IDS:
        raise ValueError(f"{text!r}: unknown token or more than {MAX_IDS} ids")
    return ids


def record(index: int, home: list[str], extra: dict) -> dict:
    lcd = extra.get("lcd") or [(home[i] if i < len(home) else "").ljust(26) for i in range(10)]
    return {"index": index, "outcome": extra.get("outcome", "OK"), "error": extra.get("error"),
            "graph": extra.get("graph", False), "home_text": home, "lcd_text": lcd,
            "timeout_state": extra.get("timeout_state"), "yields": extra.get("yields", [])}


def build() -> dict:
    sys.path.insert(0, str(ROOT))
    from eval.truncate import trim_to_line
    from lab import function as Fn
    from lab.score import is_strict_ok

    samples, records, cases = [], [], []
    for name, text, eos, home, want, extra in CASES:
        if len(samples) == NOT_RUN[0]:
            samples.append(encode(NOT_RUN[1], eos=False))
        sample = len(samples)
        samples.append(encode(text, eos))
        r = record(len(records), home, extra)
        records.append(r)
        t = trim_to_line(samples[sample], CAP, Fn.NL, 2)
        e = {"strict": is_strict_ok(r), "strict2": Fn.strict2(r), "computed2": Fn.computed2(t, r),
             "lus": Fn.is_lus(r), "computed": list(Fn.computed(t, r)),
             "axes_only": Fn.axes_only(r), "class": Fn.outcome_cls(r),
             "literals": Fn.string_literals(t),
             "tail_literals": [list(x) for x in Fn.positioned_literals(t)],
             "vis_lines": sorted(Fn.vis_lines(r)), "dead": Fn.dead_share(t),
             "tail_only": (Fn.tail_only_share({0: (samples[sample], t, r)}) == 1.0
                           if is_strict_ok(r) else None)}
        for k, v in want.items():
            got = e["class" if k == "cls" else k]
            if got != v:
                raise AssertionError(f"{name}: {k} is {got!r}, written to be {v!r}")
        cases.append({"name": name, "sample": sample, "expect": e})
    trimmed = trim_to_line(samples[NOT_RUN[0]], CAP, Fn.NL, 2)
    assert trimmed == [1], "sample 3 must be left as <bos> alone"

    fix = {"about": "Hand-written stand-in for the private fixture, made by "
                    "make_function_cases.py. Every program and screen was written by hand; "
                    "none is model output or corpus text. The programs as typed are in that "
                    "script.",
           "cap": CAP, "samples": samples, "records": records, "cases": cases}
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (run / "score.json").write_text(json.dumps(
            {"distinct_4": 0.5, "overlap_32": {"mean_seen": 0.0}, "max_len": CAP}))
        (run / "config.json").write_text(json.dumps({"yields": True, "config": {}}))
        m = Fn.run_metrics(samples, run, CAP)
        A = Fn.aligned(samples, run, CAP)
    fix["metrics"] = {k: round(m[k], 1) for k in ("strict", "strict2", "computed2", "noerr",
                                                   "complete", "fin_len_med", "fin_le40", "dead")}
    fix["metrics"].update({"ERR:LABEL": round(m["label"], 1),
                           "ERR:UNDEFINED": round(m["undefined"], 1),
                           "ERR:SYNTAX": round(m["syntax"], 1),
                           "TIMEOUT": round(m["timeout"], 1)})
    fix["tail_only_share"] = Fn.tail_only_share(A)
    return fix


def dumps(fix: dict) -> str:
    return json.dumps(fix, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the JSON is out of date")
    a = ap.parse_args(argv)
    text = dumps(build())
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is out of date; run python -m {__spec__.name}")
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
