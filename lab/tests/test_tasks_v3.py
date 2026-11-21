"""lab.tasks_v3: task suite v3, v2's judgment plus two offline rules (no TIMEOUT; the reference
is the last number shown).

The checks against saved runs (offline v2 reproduction, the experiment map, the CEmu build
report) read runs/ (git-ignored, read-only) and skip without it; in a worktree point
CALCLM_RUNS at the main checkout's runs/ directory."""
import json
import os
from pathlib import Path

import pytest

from lab import tasks as T
from lab import tasks_v2 as V
from lab import tasks_v3 as V3
from lab.stats import paired_greater

RUNS = Path(os.environ.get("CALCLM_RUNS", T.ROOT / "runs"))
CIRCLE = T.TASKS["geo_circle_area"]          # refs 38.48..., 452.38..., 2.0106...
REFS = CIRCLE.expected()


def num(x: float) -> str:
    return f"{x:.10g}"


def rec(lines=(), outcome="OK", yields=(), j=0, extra=0):
    """A v2 driver record with what the judges read (geo_circle_area's tuple j harness)."""
    h = V.harness(CIRCLE, j)
    return {"outcome": outcome, "home_text": list(lines), "yields": [list(y) for y in yields],
            "extra_inputs": extra, "decoy": h.decoy, "prefill": dict(h.prefill)}


def answer(j, *after, outcome="OK", label="AREA"):
    """Tuple j's run: the prompt echo, a labelled answer, then `after` lines."""
    return rec(["R=?" + CIRCLE.inputs[j][0], label, num(REFS[j]), *after], outcome, j=j)


def sample(*per_tuple):
    return [f(j) for j, f in enumerate(per_tuple)]


GOOD = (lambda j: answer(j),) * 3


# ---- the last number shown -------------------------------------------------------------------

def test_last_number_is_the_bottom_number_of_the_final_home_screen():
    assert V3.last_number(rec(["R=?3.5", "AREA", "38.48451001"])) == pytest.approx(38.48451001)


def test_last_number_reads_a_line_left_to_right():
    assert V3.last_number(rec(["1 2 3"])) == 3


def test_last_number_reads_screens_in_display_order_home_screen_last():
    r = rec(["AREA", "5"], yields=[("input", 10, False, ["R=?"]),
                                   ("pause", 20, False, ["AREA", "7"])])
    assert V3.last_number(r) == 5


def test_a_final_home_screen_without_numbers_falls_back_to_the_last_pause_screen():
    r = rec(["DONE"], yields=[("input", 10, False, ["R=?"]),
                              ("pause", 20, False, ["AREA", "7"]),
                              ("pause", 30, False, ["AREA", "7", "AGAIN", "8"])])
    assert V3.last_number(r) == 8


def test_screens_before_the_last_input_are_not_read():
    """As v2: only the screens after the last input was answered count."""
    r = rec([], yields=[("pause", 5, False, ["9"]), ("input", 10, False, ["R=?"])])
    assert V3.last_number(r) is None


def test_display_order_is_v2s_screens_reordered():
    r = rec(["H"], yields=[("input", 1, False, ["R=?"]), ("pause", 2, False, ["P1"]),
                           ("menu", 3, False, ["P2"])])
    assert V3.screens3(r) == [["P1"], ["P2"], ["H"]]
    assert sorted(map(tuple, V3.screens3(r))) == sorted(map(tuple, V.screens(r)))


def test_last_number_uses_v2s_parser_including_its_conventions():
    assert V3.last_number(rec(["X=", "⁻2"])) == -2
    assert V3.last_number(rec(["1.5ᴇ3"])) == 1500
    assert V3.last_number(rec(["AREA", "38.5", "L1"])) == 38.5     # digits after a letter


# ---- judge3 ----------------------------------------------------------------------------------

def test_judge3_keeps_judge2s_fields_and_passes_a_clean_sample():
    recs = sample(*GOOD)
    j2, j3 = V.judge2(recs, CIRCLE), V3.judge3(recs, CIRCLE)
    assert {k: j3[k] for k in j2} == j2
    assert j3["pass"] and j3["pass_v3"] and j3["pass_v3_labelled"]
    assert j3["timeouts"] == [False] * 3 and j3["last"] == [True] * 3
    assert j3["lost"] is None and j3["not_last_kinds"] == [None] * 3


def test_a_timeout_passes_v2_but_not_v3():
    recs = sample(GOOD[0], lambda j: answer(j, outcome="TIMEOUT"), GOOD[2])
    j = V3.judge3(recs, CIRCLE)
    assert j["pass"] and not j["pass_v3"] and not j["pass_v3_labelled"]
    assert j["timeouts"] == [False, True, False] and j["lost"] == "timeout"


def test_a_number_shown_after_the_answer_fails_v3():
    recs = sample(GOOD[0], GOOD[1], lambda j: answer(j, "VOLUME", "123.25"))
    j = V3.judge3(recs, CIRCLE)
    assert j["pass"] and not j["pass_v3"]
    assert j["last"] == [True, True, False] and j["lost"] == "not_last"
    assert j["not_last_kinds"] == [None, None, "other"]


def test_an_echo_of_a_typed_value_after_the_answer_fails_v3_and_is_classed_typed():
    recs = sample(GOOD[0], GOOD[1], lambda j: answer(j, "RADIUS", CIRCLE.inputs[j][0]))
    j = V3.judge3(recs, CIRCLE)
    assert j["pass"] and not j["pass_v3"] and j["not_last_kinds"][2] == "typed"


def test_a_stale_prefilled_value_after_the_answer_is_classed_prefill():
    q = V.harness(CIRCLE, 2).prefill[16][1]            # Q's prefilled value
    recs = sample(GOOD[0], GOOD[1], lambda j: answer(j, q))
    assert V3.judge3(recs, CIRCLE)["not_last_kinds"][2] == "prefill"


def test_a_wrong_number_before_the_answer_still_passes_v3():
    """The rule is about the last number only: an earlier contradictory value is allowed."""
    recs = sample(GOOD[0], GOOD[1],
                  lambda j: rec(["R=?0.8", "WRONG", "75.39822369", "AREA", num(REFS[2])], j=j))
    assert V3.judge3(recs, CIRCLE)["pass_v3"]


def test_timeout_and_not_last_together_are_lost_to_both():
    recs = sample(GOOD[0], lambda j: answer(j, outcome="TIMEOUT"),
                  lambda j: answer(j, "123.25"))
    assert V3.judge3(recs, CIRCLE)["lost"] == "both"


def test_v3_labelled_needs_v2s_label_check():
    recs = sample(GOOD[0], GOOD[1], lambda j: answer(j, label="RESULT"))
    j = V3.judge3(recs, CIRCLE)
    assert j["pass_v3"] and not j["pass_labelled"] and not j["pass_v3_labelled"]


def test_a_v2_failure_is_a_v3_failure_and_not_counted_as_lost():
    recs = sample(GOOD[0], GOOD[1], lambda j: rec(["R=?0.8", "AREA", "99"], j=j))
    j = V3.judge3(recs, CIRCLE)
    assert not j["pass"] and not j["pass_v3"] and j["lost"] is None


def test_judge3_of_an_empty_sample():
    j = V3.judge3(None, CIRCLE)
    assert j["empty"] and not j["pass_v3"] and not j["pass_v3_labelled"]
    assert j["last"] == [False] * 3 and j["lost"] is None


def test_judge3_refuses_records_run_under_another_harness_like_judge2():
    recs = sample(*GOOD)
    recs[1]["decoy"] = "9.99"
    with pytest.raises(ValueError, match="another harness"):
        V3.judge3(recs, CIRCLE)


# ---- metrics ---------------------------------------------------------------------------------

def row(tid="geo_circle_area", seed=1, v2=False, v3=False, lab=False, v3lab=False, lost=None,
        kinds=(None, None, None)):
    return {"task": tid, "seed": seed, "pass": v2, "pass_v3": v3, "pass_labelled": lab,
            "pass_v3_labelled": v3lab, "lost": lost, "not_last_kinds": list(kinds)}


def test_metrics3_counts_rates_and_losses():
    rows = [row(seed=1, v2=True, v3=True, lab=True, v3lab=True),
            row(seed=2, v2=True, lost="timeout"),
            row(seed=3, v2=True, lab=True, lost="not_last", kinds=(None, "typed", "other")),
            row(seed=4, v2=True, lost="both", kinds=("prefill", None, None)),
            row(seed=5)]
    m = V3.metrics3(rows)
    assert m["n_samples"] == 5 and m["n_pass"] == 4 and m["n_pass_v3"] == 1
    assert m["task_correct"] == 80.0 and m["task_correct_v3"] == 20.0
    assert m["task_correct_labelled"] == 40.0 and m["task_correct_v3_labelled"] == 20.0
    assert m["lost"] == {"timeout": 1, "not_last": 1, "both": 1}
    assert m["not_last_kinds"] == {"typed": 1, "other": 1, "prefill": 1}


def test_metrics3_seed_filter_keeps_only_sampling_seeds_up_to_the_bound():
    rows = [row(seed=1, v2=True, v3=True), row(seed=60, v2=True)]
    assert V3.metrics3(rows, max_seed=50)["n_samples"] == 1


def test_compare_is_lab_stats_paired_test_and_its_two_sided_p():
    a, b = [1.0, 2.0, 3.5, 2.5], [1.0, 1.5, 2.0, 2.5]
    one = V3.compare(a, b, two_sided=False)
    ref = paired_greater(a, b)
    assert one["diff"] == ref["diff"] and one["p"] == ref["p"]
    two = V3.compare(a, b, two_sided=True)
    assert two["p"] == pytest.approx(2 * min(ref["p"], 1 - ref["p"]))


# ---- v3 loophole mutants -----------------------------------------------------------------------

def test_v3_loopholes_cover_both_rules_on_v2_tasks():
    kinds = {k for ms in V3.V3_LOOPHOLES.values() for k, _ in ms}
    assert {"timeout", "spray", "prefill"} <= kinds
    assert set(V3.V3_LOOPHOLES) <= set(V.TASKS_V2)


# ---- the experiment map ----------------------------------------------------------------------

def test_source_dirs_follow_each_experiments_arms():
    r = Path("/R")
    assert V3.source(r, "e10", "L4", 11, "dev") == (r / "e10/s11/L4/tasks/v2", None)
    assert V3.source(r, "e10", "W", 11, "heldout") == (r / "e8/s11/W/heldout100", None)
    assert V3.source(r, "e10", "W", 23, "heldout") == (r / "e10/s23/W/heldout100", None)
    # E8's W-s10 is not E10's W-s10 (E7's export): E10 has no W-s10 held-out
    assert V3.source(r, "e10", "W", 10, "heldout") == (r / "e10/s10/W/heldout100", None)
    assert V3.source(r, "e11", "L8", 12, "heldout") == (r / "e11/s12/L8/heldout50", None)
    assert V3.source(r, "e13", "L4", 12, "dev") == (r / "e11/s12/L4/tasks/v2", None)
    assert V3.source(r, "e13", "BR", 12, "heldout") == (r / "e13/s12/BR/heldout50", None)
    assert V3.source(r, "e14", "V21", 12, "heldout") == (r / "e10/s12/L4/heldout100", 50)
    assert V3.source(r, "e14", "LD2", 12, "dev") == (r / "e14/s12/LD2/tasks/v2", None)


def test_rescore_refuses_to_write_inside_a_source_experiment_dir(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        V3.check_out(tmp_path / "e10" / "x", tmp_path)
    with pytest.raises(ValueError, match="inside"):
        V3.check_out(tmp_path, tmp_path)               # OUT = RUNS writes OUT/e10/...
    V3.check_out(tmp_path / "v3", tmp_path)


def test_report_runs_the_same_output_guard(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        V3.report(tmp_path, tmp_path)


def test_a_seed_bound_also_filters_the_written_rows(tmp_path, monkeypatch):
    rows = [row(seed=1, v2=True, v3=True), row(seed=60, v2=True)]
    src = tmp_path / "src"
    unit = ("e14", "V21", 11, "heldout", src, 50)
    monkeypatch.setattr(V3, "_units", lambda runs: iter([unit]))
    monkeypatch.setattr(V3, "_rescore_one", lambda a: (str(a[0]), rows))

    class Pool:                                        # in-process stand-in for the pool
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, f, xs): return map(f, xs)
    monkeypatch.setattr(V3, "ProcessPoolExecutor", Pool)
    V3.rescore(tmp_path / "runs", tmp_path / "out")
    d = tmp_path / "out" / "e14" / "s11" / "V21" / "heldout"
    assert [json.loads(x)["seed"] for x in (d / "judged3.jsonl").read_text().splitlines()] == [1]
    assert json.loads((d / "metrics3.json").read_text())["n_samples"] == 1


# ---- against saved runs (read-only) ----------------------------------------------------------

REPRO = [RUNS / p for p in ("e10/s11/L4/tasks/v2", "e10/s11/W/tasks/v2",
                            "e13/s11/BR/tasks/v2", "e14/s11/LD2/tasks/v2",
                            "e10/s11/L4/heldout100", "e14/s11/LD2/heldout50")]


@pytest.mark.skipif(not all((d / "judged.jsonl").exists() for d in REPRO),
                    reason="no saved E10/E13/E14 runs (set CALCLM_RUNS)")
@pytest.mark.parametrize("d", REPRO, ids=lambda d: "/".join(d.parts[-4:]))
def test_offline_v2_reproduces_every_saved_judged_row(d):
    saved = [json.loads(x) for x in (d / "judged.jsonl").read_text().splitlines() if x.strip()]
    assert V3.rejudge_v2(d, RUNS.parent) == saved


BUILD = RUNS / "v3" / "build" / "build3.json"


@pytest.mark.skipif(not BUILD.exists(), reason="no v3 CEmu build (python -m lab.tasks_v3 build3)")
def test_cemu_build_every_gold_passes_v3_and_every_mutant_fails():
    rep = json.loads(BUILD.read_text())
    assert rep["n_tasks"] == 60 and rep["ok"]
    for t in rep["tasks"]:
        assert t["gold"]["pass_v3"] and all(t["gold"]["last"]), t["task"]
        assert not any(m["pass"] for m in t["mutants"] + t["loopholes"]), t["task"]
        assert all(m["pass"] and not m["pass_v3"] for m in t["v3_loopholes"]), t["task"]
