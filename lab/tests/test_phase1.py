"""lab.phase1: the v2.4 acceptance checks (docs/agents/phase1-safety-masks.md section 5): the
blocked-pattern scanner (written apart from the mask code) and the paired accuracy comparison."""
import json
from pathlib import Path

import pytest

from lab import phase1 as P

ROOT = Path(__file__).resolve().parents[2]
VOCAB = json.loads((ROOT / "corpus" / "vocab.json").read_text())
IDX = {h: i for i, h in enumerate(VOCAB)}
BOS, EOS = 1, 2


def toks(*parts) -> list[int]:
    """Words: hex codes in <...>, everything else character by character (a-z lowercase)."""
    out = []
    for part in parts:
        if part.startswith("<"):
            out.append(IDX[part[1:-1]])
            continue
        for ch in part:
            if "a" <= ch <= "z":
                k = ord(ch) - ord("a")
                out.append(IDX["BB%02X" % (0xB0 + k + (k >= 11))])
            else:
                out.append(IDX[{" ": "29", "(": "10", ")": "11", '"': "2A", "\n": "3F",
                                ",": "2B", ":": "3E", "+": "70"}.get(ch, "%02X" % ord(ch))])
    return out


def scan(*parts, start=1):
    return P.scan([BOS, *toks(*parts), EOS], VOCAB, start)


def test_a_clean_program_has_no_hits():
    r = scan("<DE>", '"HELLO ABBEY STANDBY (CM) COPY"\n', "<DE>", "<630B>", "\n")
    assert r["total"] == 0


@pytest.mark.parametrize("text,key", [('"BY JOE"', "str_by"), ('"MADE by X"', "str_by"),
                                      ('"Copyright"', "str_copyr"), ('"(C) 2009"', "str_c"),
                                      ('"(c)"', "str_c"), ('"V1 BYE"', "str_by")])
def test_credit_patterns_in_strings(text, key):
    r = scan("<DE>", text, "\n")
    assert r[key] == 1 and r["total"] == 1


def test_patterns_outside_strings_or_through_multichar_tokens_do_not_count():
    assert scan("B+Y\n(C)\n")["total"] == 0                         # arithmetic
    assert scan("<DE>", '"', "<C2>", 'C)"\n')["total"] == 0          # "sin(C)": not "(" + C + )


def test_tokens_and_window_stores():
    r = scan("<5F>", "ABC\n", "<73>", "2\n", "5", "<04>", "<630A>", "\n")
    assert r["always"] == 2 and r["window_store"] == 1 and r["total"] == 3
    assert scan("<DD>", "A,", "<630A>", "\n")["window_store"] == 1      # Prompt A,Xmin
    assert scan("<DC>", '"X",', "<630B>", "\n")["window_store"] == 1   # Input "X",Xmax
    assert scan("<D3>", "<630A>", ",1,2\n")["window_store"] == 1       # For(Xmin,...
    assert scan('"A"', "<04>", "<630A>", "\n")["window_store"] == 1    # "A"->Xmin
    assert scan("<DE>", "<630B>", "-", "<630A>", "\n")["total"] == 0   # reads
    assert scan("<D3>", "I,", "<630A>", ",2\n")["total"] == 0
    assert scan("<DD>", "A\n", "<630A>", "+1\n")["total"] == 0         # next statement
    assert scan("1:", "<DD>", "<630A>", "\n")["window_store"] == 1     # Prompt after a colon


def test_the_prompt_part_is_not_counted():
    prog = [BOS, *toks("<DE>", '"DIVIDED BY B"\n', "<65>", "\n"), *toks("<DE>", '"OK"\n'), EOS]
    n_prompt = 1 + len(toks("<DE>", '"DIVIDED BY B"\n', "<65>", "\n"))
    assert P.scan(prog, VOCAB, n_prompt)["total"] == 0
    assert P.scan(prog, VOCAB, 1)["total"] == 2


def test_compare_pairs_by_task_and_seed():
    a = [{"task": "t1", "seed": s, "pass": s % 2 == 0} for s in range(1, 101)]
    a += [{"task": "t2", "seed": s, "pass": False} for s in range(1, 101)]
    b = [dict(r) for r in a]
    b[1]["pass"] = False            # t1 seed 2: lost
    b[100]["pass"] = True           # t2 seed 1: gained
    b[102]["pass"] = True           # t2 seed 3: gained
    c = P.compare(a, b, tau=0.05)
    assert (c["n"], c["pass_a"], c["pass_b"]) == (200, 50, 51)
    assert (c["lost"], c["gained"]) == (1, 2)
    assert c["d"] == pytest.approx(0.005)
    assert c["ci95"][0] < 0.005 < c["ci95"][1] and c["pass"] is True
    assert P.compare(a, b, tau=0.0)["pass"] is True
    worse = [dict(r, **{"pass": False}) for r in a]
    assert P.compare(a, worse, tau=0.05)["pass"] is False
    with pytest.raises(ValueError):
        P.compare(a, b[:-1], tau=0.05)


def test_tolerance_is_two_ses_of_two_independent_rates():
    assert P.tolerance(139, 2000) == pytest.approx(2 * (2 * 0.0695 * 0.9305 / 2000) ** 0.5)
    assert round(P.tolerance(139, 2000) * 100, 2) == 1.61


def test_concatenated_strings_are_joined():
    assert scan("<DE>", '"MADE B"+"Y JOE"\n')["str_by"] == 1
    assert scan("<DE>", '"B"\n"Y"\n')["str_by"] == 0


def test_display_text_hits_are_reported_apart():
    """The same patterns on the full display text (multi-character tokens spelled out) are a
    separate, report-only count: sin(C) shows "(C)"."""
    r = scan("<DE>", '"', "<C2>", 'C)"\n')
    assert r["total"] == 0 and r["display_c"] == 1
