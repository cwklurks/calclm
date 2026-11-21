"""lab.tasks: the correctness-aware task suite (docs/agents/tasks/F-task-suite.md)."""
import contextlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from lab import tasks as T
from model.tokens import BOS, EOS


def rec(lines=(), outcome="OK", yields=(), extra=0):
    """A lab.score --yields result record with just what the judge reads."""
    return {"outcome": outcome, "home_text": list(lines), "yields": [list(y) for y in yields],
            "extra_inputs": extra}


CIRCLE = T.TASKS["geo_circle_area"]


# ---- reading numbers off the screen ----------------------------------------------------------

@pytest.mark.parametrize("line,want", [
    ("AREA=78.53981634", [78.53981634]),
    ("            ⁻2.5", [-2.5]),
    ("1.5ᴇ12", [1.5e12]),
    ("4.2ᴇ⁻5", [4.2e-5]),
    (".5", [0.5]),
    ("R=?3.5", [3.5]),
    ("-4 AND 6", [-4, 6]),
    ("5-3", [5, 3]),
    ("CIRCLE AREA", []),
    ("Y=AX²+BX+C", []),
    ("  12 34.5  ", [12, 34.5]),
    ("⁻1.5ᴇ⁻3", [-1.5e-3]),
    ("12.", [12]),
    ("L1 Y1 STR1", []),                 # digits inside names are not numbers
    ("X1=?3.5", [3.5]),
])
def test_numbers_reads_ti_display_forms(line, want):
    assert T.numbers(line) == pytest.approx(want)


def test_matches_uses_relative_1e3_and_absolute_1e6_at_zero():
    assert T.matches(100.09, 100) and T.matches(99.91, 100)
    assert not T.matches(100.2, 100) and not T.matches(99.8, 100)
    assert T.matches(-2.5024, -2.5) and not T.matches(2.5, -2.5)
    assert T.matches(5e-7, 0) and T.matches(-1e-6, 0) and not T.matches(1e-5, 0)


def test_value_reads_typed_text():
    assert T.value("3.5") == 3.5 and T.value("⁻4") == -4 and T.value("12") == 12


# ---- the echo guard and the suite's shape ----------------------------------------------------

def test_echo_guard_rejects_a_reference_equal_to_an_input_of_its_tuple():
    t = replace(CIRCLE, ref=lambda r: r)
    with pytest.raises(ValueError, match="echo"):
        T.check_echo(t)


def test_echo_guard_rejects_a_reference_equal_to_another_tuples_value():
    t = replace(CIRCLE, inputs=(("3.5",), ("12",), ("38.48451",)))   # 3.5 -> 38.4845...
    with pytest.raises(ValueError, match="echo"):
        T.check_echo(t)
    t = replace(CIRCLE, ref=lambda r: 5.0 if r < 10 else 6.0)      # same answer twice
    with pytest.raises(ValueError, match="echo"):
        T.check_echo(t)


def test_echo_guard_rejects_a_reference_shown_by_the_opening():
    t = replace(CIRCLE, title=("CIRCLE AREA", "38.4845"))
    with pytest.raises(ValueError, match="opening"):
        T.check_echo(t)


def test_echo_guard_passes_the_suite():
    for t in T.SUITE:
        T.check_echo(t)


def test_suite_has_five_tasks_per_topic_in_eight_topics():
    assert len(T.SUITE) == 40 and len(T.TASKS) == 40
    assert Counter(t.topic for t in T.SUITE) == {tp: 5 for tp in T.TOPICS}
    assert len(T.TOPICS) == 8


def test_every_task_has_three_full_tuples_and_two_mutants():
    for t in T.SUITE:
        assert len(t.inputs) == 3 and all(len(v) == len(t.vars) for v in t.inputs), t.id
        assert len(t.mutants) >= 2 and t.gold not in t.mutants, t.id


def test_inputs_avoid_the_math_tier_values_and_include_big_and_fractional_ones():
    for t in T.SUITE:
        vals = [T.value(v) for tup in t.inputs for v in tup]
        assert not {2.0, 7.0} & set(vals), t.id
        assert any(v > 10 for v in vals), t.id
        assert t.integer_inputs or any(v != int(v) for v in vals), t.id
        if t.integer_inputs:
            assert all(v == int(v) for v in vals), t.id


def test_trig_tasks_set_degree_in_the_opening():
    trig = [t for t in T.SUITE if any(f in t.gold for f in ("sin(", "cos(", "tan("))]
    assert trig
    for t in trig:
        assert t.degree and "\nDegree\n" in t.opening(), t.id


def test_openings_tokenize_without_unk_and_end_with_a_newline():
    nl = T.NL
    for t in T.SUITE:
        ids = t.opening_ids()
        assert T.UNK not in ids and ids[-1] == nl and BOS not in ids, t.id
        for text in (t.gold, *t.mutants):
            assert T.UNK not in T.text_ids(text), (t.id, text)


def test_check_static_rejects_an_unk_opening():
    t = replace(CIRCLE, title=("CIRCLE AREA",), vars=("R",))
    T.check_static(t)
    bad = replace(CIRCLE, title=("CIRCLE AREA \x01",))
    with pytest.raises(ValueError):
        T.check_static(bad)


def test_program_is_bos_opening_completion_eos():
    ids = T.program_ids(CIRCLE, CIRCLE.gold)
    op = CIRCLE.opening_ids()
    assert ids[0] == BOS and ids[1:1 + len(op)] == op and ids[-1] == EOS
    assert ids[1 + len(op):-1] == T.text_ids(CIRCLE.gold)


def test_suite_sha256_is_stable_and_covers_answers():
    a = T.suite_sha256(T.SUITE)
    assert a == T.suite_sha256(T.SUITE) and len(a) == 64
    changed = [replace(CIRCLE, ref=lambda r: 2 * r), *T.SUITE[1:]]
    assert T.SUITE[0] is CIRCLE and T.suite_sha256(changed) != a


# ---- judging one run and one sample ----------------------------------------------------------

def test_tuple_passes_on_the_final_home_screen():
    assert T.tuple_passes(rec(["CIRCLE AREA", "R=?3.5", "     38.48451001"]), 38.48451)
    assert not T.tuple_passes(rec(["CIRCLE AREA", "R=?3.5", "     38.6"]), 38.48451)


def test_yields_count_only_after_the_last_input_was_answered():
    before = rec(["R=?3.5"], yields=[("input", 10, False, ["R=?"]),
                                     ("pause", 20, False, ["38.48451"]),
                                     ("input", 30, False, ["R=?"])])
    assert not T.tuple_passes(before, 38.48451)
    after = rec(["DONE"], yields=[("input", 10, False, ["R=?"]),
                                  ("pause", 20, False, ["AREA", "38.48451"])])
    assert T.tuple_passes(after, 38.48451)
    no_input = rec([], yields=[("pause", 20, False, ["38.48451"])])
    assert T.tuple_passes(no_input, 38.48451)


def test_sample_passes_only_when_all_three_tuples_pass_without_errors():
    refs = CIRCLE.expected()
    good = [rec([f"{x:.10g}"]) for x in refs]
    assert T.judge(good, refs)["pass"]
    timeout = [rec([f"{x:.10g}"], outcome="TIMEOUT") for x in refs]
    assert T.judge(timeout, refs)["pass"]
    two = [*good[:2], rec(["1"])]
    j = T.judge(two, refs)
    assert not j["pass"] and j["tuples"] == [True, True, False]
    err = [*good[:2], rec([f"{refs[2]:.10g}"], outcome="ERR")]
    assert not T.judge(err, refs)["pass"]
    harness = [*good[:2], rec([f"{refs[2]:.10g}"], outcome="HARNESS_ERROR")]
    j = T.judge(harness, refs)
    assert not j["pass"] and j["harness_error"]


def test_an_empty_sample_is_judged_failed():
    j = T.judge(None, CIRCLE.expected())
    assert j == {"tuples": [False] * 3, "decoys": [False] * 3, "outcomes": [None] * 3,
                 "extra_inputs": [None] * 3, "harness_error": False, "empty": True,
                 "pass": False, "pass_literal": False}


def test_a_screen_that_also_shows_another_tuples_answer_fails_that_tuple():
    """Decoy rule (amendment): a program listing many numbers (a counting loop) shows every
    tuple's answer; a tuple whose screens also show another tuple's reference fails."""
    refs = (12.0, 15.0, 13.0)
    counting = rec([], outcome="TIMEOUT",
                   yields=[("input", 1, False, ["A=?"]), ("input", 2, False, ["B=?"]),
                           *[("pause", 3 + i, False, [str(i)]) for i in range(1, 30)]])
    j = T.judge([counting] * 3, refs)
    assert j["decoys"] == [True] * 3 and j["tuples"] == [False] * 3 and not j["pass"]
    assert j["pass_literal"]                       # the spec's rule alone would pass it
    honest = [rec([f"{x:g}"]) for x in refs]
    j = T.judge(honest, refs)
    assert j["pass"] and j["decoys"] == [False] * 3


def test_a_run_stopped_by_the_wall_clock_limit_is_a_harness_error():
    from eval.driver import Config
    refs = CIRCLE.expected()
    good = [rec([f"{x:.10g}"]) for x in refs]
    good[1] = {**good[1], "outcome": "TIMEOUT", "wall_s": Config().wall_limit_s + 1}
    j = T.judge(good, refs)
    assert j["harness_error"] and not j["pass"]


# ---- metrics ---------------------------------------------------------------------------------

def judged(task, seed, tuples, ok=True):
    return {"task": task, "seed": seed, "tuples": tuples, "harness_error": False,
            "empty": False, "pass": ok and all(tuples), "pass_literal": ok and all(tuples)}


def test_task_metrics_counts_every_sample_in_the_denominator():
    ids = [t.id for t in T.SUITE]
    rows = [judged(i, s, [False] * 3) for i in ids for s in (1, 2)]
    rows[0] = judged(ids[0], 1, [True] * 3)                       # task 0: 1 of 2
    rows[2] = judged(ids[1], 1, [True, False, False])             # partial only
    rows[3] = {**judged(ids[1], 2, [False] * 3), "empty": True}   # empty still counts
    m = T.task_metrics(rows, T.SUITE)
    assert m["n_samples"] == 80 and m["n_pass"] == 1
    assert m["task_correct"] == pytest.approx(100 / 80)
    assert m["tasks_solved"] == 1 and m["n_tasks"] == 40
    assert m["partial"] == pytest.approx(200 / 80)
    assert m["per_task"][ids[0]] == pytest.approx(50.0) and m["per_task"][ids[1]] == 0.0
    assert m["per_topic"][T.SUITE[0].topic] == pytest.approx(100 / 10)
    assert m["n_empty"] == 1 and m["suite_sha256"] == T.suite_sha256(T.SUITE)
    assert m["task_correct_literal"] == pytest.approx(100 / 80)


def test_task_metrics_refuses_missing_or_unknown_tasks():
    rows = [judged(t.id, 1, [False] * 3) for t in T.SUITE]
    with pytest.raises(ValueError):
        T.task_metrics(rows[1:], T.SUITE)
    with pytest.raises(ValueError):
        T.task_metrics([*rows, judged("nope", 1, [False] * 3)], T.SUITE)


# ---- sample and score without the model or the emulator --------------------------------------

def test_sample_draws_k_seeds_per_task_from_the_opening(tmp_path, monkeypatch):
    calls = []

    def fake_draw(export, jobs, workers):
        calls.append((export, workers))
        return [[BOS, *T.TASKS[tid].opening_ids(), 5, EOS] for tid, _ in jobs]
    monkeypatch.setattr(T, "draw", fake_draw)
    small = T.SUITE[:2]
    T.sample("EXPORT", k=3, seed0=4, out=tmp_path, workers=2, tasks=small)
    rows = [json.loads(x) for x in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert [(r["task"], r["seed"]) for r in rows] == [(t.id, s) for t in small for s in (4, 5, 6)]
    assert all(r["ids"][1:1 + len(T.TASKS[r["task"]].opening_ids())] ==
               T.TASKS[r["task"]].opening_ids() for r in rows)
    meta = json.loads((tmp_path / "sample.json").read_text())
    assert meta["k"] == 3 and meta["seed0"] == 4 and meta["suite_sha256"] == T.suite_sha256(small)
    assert meta["temp"] == 0.8 and meta["top_k"] == 20 and meta["cap"] == 1024
    assert meta["masks"] is True and meta["horizons"] is True
    assert calls == [("EXPORT", 2)]


def test_draw_one_is_v21_decoding_from_the_opening(monkeypatch):
    """The real sampling call: intref.generate with the opening as prompt, cap 1024, T 0.8 /
    top-k 20, the tracker's masks, an export loaded with ALiBi horizons."""
    import model.intref as IR
    from model.intmask import IntMask
    loaded, calls = [], []

    class FakeModel:
        vocab = T._vocab().entries

    monkeypatch.setattr(IR.IntModel, "load",
                        classmethod(lambda cls, d, horizons=False: loaded.append((d, horizons))
                                    or FakeModel()))

    def fake_generate(model, prompt, max_new, cfg, cap=None, mask=None):
        calls.append((prompt, max_new, cfg, cap, mask))
        return [BOS, *prompt, EOS]
    monkeypatch.setattr(IR, "generate", fake_generate)
    task = T.TASKS["phy_force_x"]
    out = T.draw("EX", [(task.id, 9)], workers=1)
    assert loaded == [("EX", True)]
    [(prompt, max_new, cfg, cap, mask)] = calls
    assert prompt == task.opening_ids() and max_new == 1024 and cap == 1024
    assert cfg == IR.SamplerConfig(205, 20, False, 9) and isinstance(mask, IntMask)
    assert out == [[BOS, *task.opening_ids(), EOS]]


def test_draw_refuses_an_export_with_another_vocab(monkeypatch):
    import model.intref as IR

    class FakeModel:
        vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "3F"]
    monkeypatch.setattr(IR.IntModel, "load", classmethod(lambda cls, d, horizons=False:
                                                         FakeModel()))
    with pytest.raises(ValueError, match="vocab"):
        T.draw("EX", [("geo_circle_area", 1)], workers=1)


def write_samples(d, rows, tasks):
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "sample.json").write_text(json.dumps({"k": 2, "seed0": 1,
                                               "suite_sha256": T.suite_sha256(tasks)}))


def pooled_stand_in(make_runner, log):
    """A stand-in for T.pooled_runner / V.pooled_runner2: yields make_runner(workers,
    budget_ms); log gets ("enter", workers, budget_ms) and "exit"."""
    @contextlib.contextmanager
    def pooled(workers, budget_ms=30_000):
        log.append(("enter", workers, budget_ms))
        try:
            yield make_runner(workers, budget_ms)
        finally:
            log.append("exit")
    return pooled


class Queued:
    """A pooled runner's handle: the records come with result()."""

    def __init__(self, log, recs):
        self.log, self.recs = log, recs

    def result(self):
        self.log.append("result")
        return self.recs


def queued(runner, log):
    """`runner` made a pooled one: each call is logged and returns a Queued handle."""
    def run(*a):
        log.append("submit")
        return Queued(log, runner(*a))
    return run


def gold_and_mutant_rows(tasks):
    return ([{"task": t.id, "seed": s, "ids": T.program_ids(t, t.gold if s == 1 else
                                                           t.mutants[0])}
             for t in tasks for s in (1, 2)] + [{"task": tasks[0].id, "seed": 3, "ids": [BOS]}])


def shows_gold(tasks):
    """Stand-in emulator: the reference for gold programs, 0 for anything else."""
    def run(programs, values, out):
        task = next(t for t in tasks if values in t.inputs)
        ref = task.expected()[task.inputs.index(values)]
        return [rec([f"{ref:.10g}"]) if p == T.program_ids(task, task.gold) else rec(["0"])
                for p in programs]
    return run


def test_score_queues_every_batch_before_it_awaits_a_record(tmp_path):
    """A pooled runner gets all tasks' batches first, then the records are awaited; the
    judgment is byte for byte the one a plain runner gives."""
    small = T.SUITE[:3]
    rows = gold_and_mutant_rows(small)
    write_samples(tmp_path / "plain", rows, small)
    write_samples(tmp_path / "pooled", rows, small)
    T.score(tmp_path / "plain", runner=shows_gold(small), tasks=small)
    log = []
    T.score(tmp_path / "pooled", runner=queued(shows_gold(small), log), tasks=small)
    assert log == ["submit"] * 9 + ["result"] * 9
    for name in ("judged.jsonl", "metrics.json"):
        assert (tmp_path / "pooled" / name).read_bytes() == (tmp_path / "plain" / name
                                                            ).read_bytes()


def test_score_without_a_runner_runs_every_batch_on_one_pool(tmp_path, monkeypatch):
    small = T.SUITE[:2]
    write_samples(tmp_path, gold_and_mutant_rows(small), small)
    log, calls = [], []
    monkeypatch.setattr(T, "pooled_runner", pooled_stand_in(
        lambda w, b: queued(lambda *a: calls.append(a) or shows_gold(small)(*a), log), log))
    m = T.score(tmp_path, tasks=small, workers=5, budget_ms=7000)
    assert log[0] == ("enter", 5, 7000) and log[-1] == "exit" and len(calls) == 6
    assert log.index("result") > max(i for i, x in enumerate(log) if x == "submit")
    assert m["n_pass"] == 2 and m["budget_ms"] == 7000


def test_run_programs_takes_a_handle_and_checks_its_record_count(tmp_path):
    prog = T.program_ids(CIRCLE, CIRCLE.gold)
    log = []
    [recs] = T.run_programs(CIRCLE, [prog], queued(lambda p, v, o: [rec([v[0]])], log),
                            tmp_path)
    assert [r["home_text"] for r in recs] == [[v[0]] for v in CIRCLE.inputs]
    assert log == ["submit"] * 3 + ["result"] * 3
    collect = T.submit_programs(CIRCLE, [prog], queued(lambda p, v, o: [], []), tmp_path)
    with pytest.raises(ValueError, match="0 records for 1 programs"):
        collect()


def test_score_runs_each_sample_once_per_tuple_and_writes_metrics(tmp_path):
    small = T.SUITE[:2]
    gold = [T.program_ids(t, t.gold) for t in small]
    rows = [{"task": small[0].id, "seed": 1, "ids": gold[0]},
            {"task": small[0].id, "seed": 2, "ids": [BOS]},              # empty: not run
            {"task": small[1].id, "seed": 1, "ids": gold[1]},
            {"task": small[1].id, "seed": 2, "ids": T.program_ids(small[1], small[1].mutants[0])}]
    write_samples(tmp_path, rows, small)
    seen = []

    def runner(programs, values, out):
        seen.append((values, len(programs), out))
        task = next(t for t in small if values in t.inputs)
        ref = task.expected()[task.inputs.index(values)]
        return [rec([f"{ref:.10g}"]) if p in gold else rec(["0"]) for p in programs]

    m = T.score(tmp_path, runner=runner, tasks=small)
    assert sorted(v for v, _, _ in seen) == sorted(v for t in small for v in t.inputs)
    assert [n for _, n, _ in seen] == [1, 1, 1, 2, 2, 2]
    assert m["n_samples"] == 4 and m["n_pass"] == 2 and m["n_empty"] == 1
    assert json.loads((tmp_path / "metrics.json").read_text()) == m
    judged_rows = [json.loads(x) for x in (tmp_path / "judged.jsonl").read_text().splitlines()]
    assert [(r["task"], r["seed"], r["pass"]) for r in judged_rows] == [
        (small[0].id, 1, True), (small[0].id, 2, False), (small[1].id, 1, True),
        (small[1].id, 2, False)]


def test_cli_gold_then_score_reaches_full_marks(tmp_path, monkeypatch):
    """`gold` writes the gold completions as samples; `score` judges them (stand-in emulator
    that shows the reference whenever the program is a gold one)."""
    golds = {tuple(T.program_ids(t, t.gold)): t for t in T.SUITE}

    def fake_runner(workers, budget_ms=30_000):
        def run(programs, values, out):
            out_recs = []
            for p in programs:
                t = golds[tuple(p)]
                out_recs.append(rec([f"{t.expected()[t.inputs.index(values)]:.10g}"]))
            return out_recs
        return run
    monkeypatch.setattr(T, "pooled_runner", pooled_stand_in(fake_runner, []))
    assert T.main(["gold", "--out", str(tmp_path)]) == 0
    assert T.main(["score", str(tmp_path), "--workers", "1"]) == 0
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["task_correct"] == 100.0 and m["tasks_solved"] == 40 and m["n_samples"] == 40


def test_score_checks_the_samples_before_any_emulator_run(tmp_path):
    small = T.SUITE[:2]
    ran = []

    def runner(*a):
        ran.append(a)
        return []
    write_samples(tmp_path, [{"task": "nope", "seed": 1, "ids": [BOS, EOS]}], small)
    with pytest.raises(ValueError, match="unknown"):
        T.score(tmp_path, runner=runner, tasks=small)
    dup = {"task": small[0].id, "seed": 1, "ids": [BOS, EOS]}
    write_samples(tmp_path, [dup, dup], small)
    with pytest.raises(ValueError, match="twice"):
        T.score(tmp_path, runner=runner, tasks=small)
    assert ran == []


def test_score_refuses_samples_drawn_for_another_suite(tmp_path):
    small = T.SUITE[:2]
    write_samples(tmp_path, [], small)
    with pytest.raises(ValueError, match="suite"):
        T.score(tmp_path, runner=lambda *a: [], tasks=T.SUITE[:3])


# ---- the metric's own tests in CEmu: gold passes, mutants fail -------------------------------

@pytest.fixture(scope="module")
def emu_runner():
    """Runs token-id programs in one in-process emulator with yield capture (skips without the
    ROM)."""
    from eval import driver, inputs
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    env = E.prepare()

    def run(programs, values, out):
        cfg = driver.Config(input_values=tuple(values))
        specs = inputs.from_token_ids(programs, T.VOCAB, out)
        return [driver.run_one(env.emu, s, env.clean_state, cfg, yields=True) for s in specs]
    return run


@pytest.mark.parametrize("task", T.SUITE, ids=lambda t: t.id)
def test_gold_passes_and_mutants_fail_in_cemu(emu_runner, tmp_path, task):
    v = T.validate(task, emu_runner, tmp_path)
    assert v["gold"]["pass"], v
    assert not any(m["pass"] for m in v["mutants"]), v


@pytest.mark.parametrize("tid", ["nt_gcd", "nt_lcm", "phy_ohm_voltage", "chem_percent_yield"])
def test_a_counting_loop_solves_no_task_in_cemu(emu_runner, tmp_path, tid):
    """The reviewer's counterexample: Disp 1, 2, ... 120 with a Pause each matched every
    integer answer under the spec's rule alone; the decoy rule fails it."""
    task = T.TASKS[tid]
    prog = T.program_ids(task, "For(I,1,120)\nDisp I\nPause \nEnd")
    [recs] = T.run_programs(task, [prog], emu_runner, tmp_path)
    j = T.judge(recs, task.expected())
    assert not j["pass"] and not any(j["tuples"]), j


@pytest.mark.parametrize("tid,body", [
    ("nt_gcd", "For(I,1,A)\nIf not(fPart(A/I))\nThen\nDisp I\nPause \nEnd\nEnd"),
    ("nt_gcd", "For(I,1,B)\nIf not(fPart(B/I))\nThen\nDisp I\nPause \nEnd\nEnd"),
    ("nt_lcm", "For(I,1,40)\nDisp AI\nPause \nEnd"),
    ("nt_lcm", "For(I,1,40)\nDisp BI\nPause \nEnd"),
], ids=["divisors-of-A", "divisors-of-B", "multiples-of-A", "multiples-of-B"])
def test_listing_divisors_or_multiples_solves_neither_gcd_nor_lcm_in_cemu(emu_runner, tmp_path,
                                                                         tid, body):
    """The reviewer's second counterexample: a program that lists the divisors (multiples) of
    an input shows its gcd (lcm); the tuples are chosen so that it also shows another
    tuple's answer on at least one tuple."""
    task = T.TASKS[tid]
    [recs] = T.run_programs(task, [T.program_ids(task, body)], emu_runner, tmp_path)
    j = T.judge(recs, task.expected())
    assert not j["pass"], j


def test_numbers_skips_a_power_of_ten_too_large_for_a_float():
    """E6 s14 S0: a program printed a number with a huge exponent; float() overflowed and the
    scorer crashed. Such a number cannot match any task reference, so it is skipped."""
    from lab.tasks import numbers
    assert numbers("1ᴇ999") == []
    assert numbers("X=2.5 1ᴇ999 3") == [2.5, 3.0]
    assert numbers("1ᴇ99") == [1e99]
