"""lab.mathtier: the math tier, from a program's tokens and its two input-typed run records.

A program is judged twice, with the harness typing "2" (run A) and "7" (run B). It is math when
it asks for input, was answered, did not error, computed a non-trivial result, and showed a
different result for the two values."""
import json

import pytest

from lab import mathtier as M
from lab.tests.test_function import rec, tok, write_run
from model.tokens import EOS

INPUT = "DC"


def _run(d, records, typed, cap=1024):
    """A lab.score --yields run whose config.json records the typed input value."""
    d = write_run(d, records, cap=cap)
    cfg = json.loads((d / "config.json").read_text())
    cfg["config"]["input_text"] = typed
    (d / "config.json").write_text(json.dumps(cfg))
    return d


# ---- takes_input (static) -------------------------------------------------------------------

def test_input_and_prompt_tokens_ask_for_a_value():
    assert M.takes_input(tok(INPUT, "A")) and M.takes_input(tok("DD", "A"))
    assert not M.takes_input(tok("DE", "6"))


# ---- one program, two runs ------------------------------------------------------------------

def test_a_working_input_driven_math_program():
    ids = tok(INPUT, "A")
    a = rec("OK", home=["6"], actions={"input": 1})
    b = rec("OK", home=["21"], actions={"input": 1})
    assert M.classify(ids, a, b) == {"takes_input": True, "answered": True, "no_error": True,
                                     "computes": True, "responds": True, "math": True}


def test_a_constant_output_does_not_respond():
    ids = tok(INPUT, "A")
    a = rec("OK", home=["42"], actions={"input": 1})
    b = rec("OK", home=["42"], actions={"input": 1})
    c = M.classify(ids, a, b)
    assert not c["responds"] and not c["math"] and c["computes"]


def test_an_echo_only_screen_does_not_respond():
    ids = tok(INPUT, "A")
    a = rec("OK", home=["A=?2"], actions={"input": 1})
    b = rec("OK", home=["A=?7"], actions={"input": 1})
    assert M.response_lines(a, "2") == set() and M.response_lines(b, "7") == set()
    c = M.classify(ids, a, b)
    assert not c["responds"] and not c["math"]


def test_an_error_in_one_run_fails_no_error():
    ids = tok(INPUT, "A")
    a = rec("OK", home=["6"], actions={"input": 1})
    b = rec("ERR", error="SYNTAX", home=["6"], actions={"input": 1})
    c = M.classify(ids, a, b)
    assert not c["no_error"] and not c["math"]


def test_a_program_without_input_fails_takes_input():
    ids = tok("DE", "6")
    a = rec("OK", home=["6"], actions={"input": 0})
    b = rec("OK", home=["6"], actions={"input": 0})
    c = M.classify(ids, a, b)
    assert not c["takes_input"] and not c["math"]


# ---- a whole pair of runs --------------------------------------------------------------------

def test_math_metrics_requires_the_two_typed_runs_and_counts(tmp_path):
    samples = [tok(INPUT, "A") + [EOS], tok("DE", "6") + [EOS]]
    recs_a = [rec("OK", home=["6"], actions={"input": 1}, index=0), rec("OK", home=["6"], index=1)]
    recs_b = [rec("OK", home=["21"], actions={"input": 1}, index=0), rec("OK", home=["6"], index=1)]
    da = _run(tmp_path / "a", recs_a, "2")
    db = _run(tmp_path / "b", recs_b, "7")
    assert M.math_metrics(samples, da, db) == {
        "n_run": 2, "n_math": 1, "takes_input": 50.0, "answered": 50.0, "no_error": 100.0,
        "computes": 100.0, "responds": 50.0, "math": 50.0}
    with pytest.raises(ValueError, match="input_text"):
        M.math_metrics(samples, da, _run(tmp_path / "c", recs_b, "1"))
    with pytest.raises(ValueError, match="input_text"):
        M.math_metrics(samples, db, da)                     # "7" then "2": the wrong order


def test_an_input_prompt_string_echo_does_not_respond():
    """Input "A=",A shows A=2 and A=7: the typed value after the program's own prompt string
    is an echo, not a result."""
    from lab.tests.test_function import EQ, Q
    ids = tok(INPUT, Q, "A", EQ, Q, "2B", "A")                  # Input "A=",A (TI = 6A, , 2B)
    a = rec("OK", home=["A=2"], actions={"input": 1})
    b = rec("OK", home=["A=7"], actions={"input": 1})
    c = M.classify(ids, a, b)
    assert c["responds"] is False and c["math"] is False


def test_a_result_after_the_echo_still_responds():
    from lab.tests.test_function import EQ, NL, Q
    ids = tok(INPUT, Q, "A", EQ, Q, "2B", "A", NL, "DE", "A", "82", "3")   # Input "A=",A / Disp A*3
    a = rec("OK", home=["A=2", "6"], actions={"input": 1})
    b = rec("OK", home=["A=7", "21"], actions={"input": 1})
    assert M.classify(ids, a, b)["math"] is True


# ---- the typed pair (E4's held-out inputs 3 and 5) --------------------------------------------

def test_the_typed_pair_defaults_to_2_and_7():
    assert M.TYPED_A == "2" and M.TYPED_B == "7"
    ids = tok(INPUT, "A")
    a = rec("OK", home=["A=?2"], actions={"input": 1})
    b = rec("OK", home=["A=?7"], actions={"input": 1})
    assert M.classify(ids, a, b) == M.classify(ids, a, b, typed=("2", "7"))


def test_echoes_of_the_held_out_pair_are_not_responses():
    """Typed 3 and 5, "A=?3" and "A=?5" are echoes; under the default 2/7 they are results."""
    ids = tok(INPUT, "A")
    a = rec("OK", home=["A=?3"], actions={"input": 1})
    b = rec("OK", home=["A=?5"], actions={"input": 1})
    assert M.classify(ids, a, b, typed=("3", "5"))["responds"] is False
    assert M.classify(ids, a, b)["responds"] is True


def test_math_metrics_with_the_held_out_pair_requires_runs_typed_3_and_5(tmp_path):
    samples = [tok(INPUT, "A") + [EOS]]
    recs_a = [rec("OK", home=["9"], actions={"input": 1}, index=0)]
    recs_b = [rec("OK", home=["15"], actions={"input": 1}, index=0)]
    d3, d5 = _run(tmp_path / "3", recs_a, "3"), _run(tmp_path / "5", recs_b, "5")
    got = M.math_metrics(samples, d3, d5, typed=("3", "5"))
    assert got["math"] == 100.0 and got["n_math"] == 1
    with pytest.raises(ValueError, match="input_text"):
        M.math_metrics(samples, d3, d5)                      # the default pair is still 2 and 7
    with pytest.raises(ValueError, match="input_text"):
        M.math_metrics(samples, d5, d3, typed=("3", "5"))
