"""lab.tasks_v2: task suite v2, hardened judging and held-out tasks
(docs/agents/tasks/F2-task-suite-v2.md).

The checks against saved E6/E7 runs (v1 byte-identity, the audited coincidental passes) read
runs/ (git-ignored, read-only) and skip without it; in a worktree point CALCLM_RUNS at the
main checkout's runs/ directory."""
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from lab import tasks as T
from lab import tasks_v2 as V
from model.tokens import BOS, EOS

V1_SHA = "fc41a22a"          # the prefix every v1 result records; the full value is pinned below
RUNS = Path(os.environ.get("CALCLM_RUNS", T.ROOT / "runs"))


def rec(lines=(), outcome="OK", yields=(), task=None, j=0, extra=0):
    """A v2 driver record with just what the judge reads (the harness of `task`'s tuple j)."""
    r = {"outcome": outcome, "home_text": list(lines), "yields": [list(y) for y in yields],
         "extra_inputs": extra}
    if task is not None:
        h = V.harness(task, j)
        r |= {"decoy": h.decoy, "prefill": dict(h.prefill)}
    return r


CIRCLE = T.TASKS["geo_circle_area"]


# ---- v1 stays exactly as it is ---------------------------------------------------------------

def test_v1_suite_sha_is_pinned():
    full = "fc41a22a7767d44e0780b9cd1d3d994349a91d7834c02e77a97bca91f1839bed"
    assert T.suite_sha256(T.SUITE) == full == V.V1_SUITE_SHA256 and full.startswith(V1_SHA)


def test_v2_dev_tasks_are_v1s_tasks_unchanged():
    assert V.SUITE_V2[:40] == T.SUITE
    assert all(a is b for a, b in zip(V.SUITE_V2, T.SUITE))
    assert V.DEV_IDS == tuple(t.id for t in T.SUITE)


# ---- the held-out tasks ----------------------------------------------------------------------

def test_heldout_ids_are_a_module_constant_of_twenty_ho_tasks():
    assert len(V.HELDOUT_IDS) == 20 and len(set(V.HELDOUT_IDS)) == 20
    assert V.HELDOUT_IDS == tuple(t.id for t in V.SUITE_V2[40:])
    assert all(i.startswith("ho_") for i in V.HELDOUT_IDS)
    assert not set(V.HELDOUT_IDS) & set(T.TASKS)
    assert "never" in V.__doc__ and "held-out" in V.__doc__.lower()


def test_heldout_topics_follow_the_spec():
    got = Counter(V.TASKS_V2[i].topic for i in V.HELDOUT_IDS)
    assert got == {"statistics": 5, "number theory": 5, "chemistry": 4, "finance": 2,
                   "algebra": 2, "physics": 2}


def test_heldout_tasks_have_v1s_shape():
    for t in V.HELDOUT:
        T.check_static(t)
        T.check_echo(t)
        assert len(t.inputs) == 3 and len(t.mutants) >= 2 and t.gold not in t.mutants, t.id
        vals = [T.value(v) for tup in t.inputs for v in tup]
        assert not {2.0, 7.0} & set(vals) and any(v > 10 for v in vals), t.id
        assert t.integer_inputs == (t.topic == "number theory"), t.id
        assert t.opening().startswith("ClrHome\n") and "\nPrompt " in t.opening(), t.id
        assert all(len(x) <= 26 for x in t.title), t.id         # one home-screen row each


def test_heldout_formulas_differ_from_every_dev_formula():
    """Different formulas, not renamed ones: the gold completions' computation keys
    (lab.synth.computation_key: canonical variables) never match a dev task's."""
    from lab.synth import computation_key
    dev = {computation_key(t.gold) for t in T.SUITE}
    for t in V.HELDOUT:
        assert computation_key(t.gold) not in dev, t.id
    assert len({computation_key(t.gold) for t in V.HELDOUT}) == 20


def test_synthetic_overlap_is_recorded_for_every_heldout_task():
    assert set(V.SYNTH_OVERLAP) == set(V.HELDOUT_IDS)
    assert set(V.SYNTH_OVERLAP.values()) <= {"none", "title-similar", "same key"}


def test_trig_heldout_tasks_set_degree():
    for t in V.HELDOUT:
        if any(f in t.gold for f in ("sin(", "cos(", "tan(")):
            assert t.degree, t.id


# ---- decoy and prefill -----------------------------------------------------------------------

def apart(a, b):
    return not T.matches(a, b) and not T.matches(b, a)


@pytest.mark.parametrize("task", V.SUITE_V2, ids=lambda t: t.id)
def test_decoy_and_prefill_follow_the_rules(task):
    refs = task.expected()
    for j, tup in enumerate(task.inputs):
        h = V.harness(task, j)
        assert h.values == tuple(tup)
        d = T.value(h.decoy)
        ins = [T.value(v) for v in tup]
        assert all(apart(d, x) and apart(d, 2 * x) and apart(d, x / 2) for x in ins)
        assert all(apart(d, r) for r in refs)
        names = [n for n, _ in h.prefill]
        assert names == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["θ"]
        vals = [T.value(x) for _, x in h.prefill]
        assert all(re.fullmatch(r"\d\.\d\d", x) for _, x in h.prefill)
        assert all(1.5 <= v <= 9.5 and v != 0 for v in vals)
        assert all(apart(a, b) for i, a in enumerate(vals) for b in vals[i + 1:])
        assert all(apart(v, x) for v in vals for x in [*ins, d, *refs])


def test_harness_is_deterministic_and_seeded_per_task_and_tuple():
    a, b = V.harness(CIRCLE, 0), V.harness(CIRCLE, 0)
    assert a == b
    assert V.harness(CIRCLE, 1).prefill != a.prefill
    assert V.harness(T.TASKS["geo_hypotenuse"], 0).prefill != a.prefill


@pytest.mark.parametrize("bad", ["3.5", "3.5005", "7", "1.75", "24", "6", "38.48"],
                         ids=["input", "near-input", "double", "half", "double-t1", "half-t1",
                              "reference"])
def test_check_harness_rejects_a_bad_decoy(monkeypatch, bad):
    """CIRCLE's tuples are 3.5, 12, 0.8; tuple 0's reference is 38.4845..."""
    V.check_harness(CIRCLE)
    monkeypatch.setattr(V, "decoy", lambda task, j: bad if j in (0, 1) else "5.55")
    with pytest.raises(ValueError, match="decoy"):
        V.check_harness(CIRCLE)


@pytest.mark.parametrize("fill", [
    (("A", "3.85"), ("B", "3.85")),          # not distinct
    (("A", "3.50"), ("B", "4.85")),          # tuple 0's input (3.5)
    (("A", "12.25"), ("B", "4.85")),         # outside 1.5-9.5
    (("A", "4.85"), ("AB", "5.15")),         # not a real variable
], ids=["repeat", "input", "range", "name"])
def test_check_harness_rejects_a_bad_prefill(monkeypatch, fill):
    monkeypatch.setattr(V, "prefill", lambda task, j: fill)
    with pytest.raises(ValueError, match="prefill"):
        V.check_harness(CIRCLE)


def test_suite_v2_sha_is_its_own_and_stable():
    assert V.SUITE_V2_SHA256 == V.suite_v2_sha256() and len(V.SUITE_V2_SHA256) == 64
    assert V.SUITE_V2_SHA256 != V.V1_SUITE_SHA256
    assert V.SUITE_V2_SHA256 == V.PINNED_SUITE_V2_SHA256


# ---- judging under v2 ------------------------------------------------------------------------

def test_judge2_adds_labels_to_v1s_judgment():
    refs = CIRCLE.expected()
    good = [rec(["AREA", f"{x:.10g}"], task=CIRCLE, j=i) for i, x in enumerate(refs)]
    j = V.judge2(good, CIRCLE)
    v1 = T.judge(good, refs)
    assert {k: j[k] for k in v1} == v1
    assert j["pass"] and j["labelled"] == [True] * 3 and j["pass_labelled"]
    bare = [rec([f"{x:.10g}"], task=CIRCLE, j=i) for i, x in enumerate(refs)]
    j = V.judge2(bare, CIRCLE)
    assert j["pass"] and j["labelled"] == [False] * 3 and not j["pass_labelled"]


def test_label_on_the_same_line_or_the_line_above_counts():
    ref = CIRCLE.expected()[0]
    kw = V.LABELS[CIRCLE.id]
    assert V.labelled(rec([f"AREA={ref:.8g}"]), ref, kw)
    assert V.labelled(rec(["area is", f"{ref:.8g}"]), ref, kw)          # case does not matter
    assert not V.labelled(rec(["AREA", "R IS", f"{ref:.8g}"]), ref, kw)
    assert not V.labelled(rec(["AREA=?3.5", f"{ref:.8g}"]), ref, kw)    # a prompt's echo
    y = rec(["DONE"], yields=[("input", 1, False, ["R=?"]), ("pause", 2, False,
                                                            ["AREA", f"{ref:.8g}"])])
    assert V.labelled(y, ref, kw)


def test_every_task_has_a_short_keyword_list():
    assert set(V.LABELS) == set(V.TASKS_V2)
    assert all(1 <= len(k) <= 6 and all(x == x.upper() and x for x in k)
               for k in V.LABELS.values())


def test_judge2_refuses_records_run_under_another_harness():
    refs = CIRCLE.expected()
    v1_style = [rec([f"{x:.10g}"]) for x in refs]
    with pytest.raises(ValueError, match="harness"):
        V.judge2(v1_style, CIRCLE)
    wrong = [rec([f"{x:.10g}"], task=CIRCLE, j=(i + 1) % 3) for i, x in enumerate(refs)]
    with pytest.raises(ValueError, match="harness"):
        V.judge2(wrong, CIRCLE)
    harness_err = [*[rec([f"{x:.10g}"], task=CIRCLE, j=i) for i, x in enumerate(refs[:2])],
                   {"outcome": "HARNESS_ERROR", "home_text": [], "yields": []}]
    j = V.judge2(harness_err, CIRCLE)
    assert j["harness_error"] and not j["pass"]
    rejected = [*harness_err[:2], {"outcome": "REJECTED", "home_text": [], "yields": []}]
    j = V.judge2(rejected, CIRCLE)                 # the driver returns before the harness ran
    assert not j["pass"] and not j["pass_labelled"]


def test_judge2_of_an_empty_sample():
    j = V.judge2(None, CIRCLE)
    assert j["empty"] and not j["pass"] and j["labelled"] == [False] * 3
    assert not j["pass_labelled"]


# ---- metrics ---------------------------------------------------------------------------------

def judged(task, seed, ok=False, labelled=False):
    return {"task": task, "seed": seed, "tuples": [ok] * 3, "harness_error": False,
            "empty": False, "pass": ok, "pass_literal": ok, "labelled": [labelled] * 3,
            "pass_labelled": ok and labelled}


def test_task_metrics2_keeps_v1s_metrics_and_adds_the_splits():
    rows = [judged(t.id, s) for t in V.SUITE_V2 for s in (1, 2)]
    rows[0] = judged(V.DEV_IDS[0], 1, ok=True, labelled=True)
    rows[2] = judged(V.DEV_IDS[1], 1, ok=True)
    rows[80] = judged(V.HELDOUT_IDS[0], 1, ok=True)
    m = V.task_metrics2(rows, V.SUITE_V2)
    v1 = T.task_metrics(rows, V.SUITE_V2)
    assert {k: m[k] for k in v1 if k != "suite_sha256"} == {k: v for k, v in v1.items()
                                                           if k != "suite_sha256"}
    assert m["suite_sha256"] == V.SUITE_V2_SHA256
    assert m["task_correct"] == pytest.approx(300 / 120)
    assert m["task_correct_dev"] == pytest.approx(200 / 80)
    assert m["task_correct_heldout"] == pytest.approx(100 / 40)
    assert m["task_correct_labelled"] == pytest.approx(100 / 120)
    assert m["tasks_solved"] == 3 and m["tasks_solved_dev"] == 2
    assert m["tasks_solved_heldout"] == 1 and m["tasks_solved_labelled"] == 1


def test_task_metrics2_on_one_split_leaves_the_other_empty():
    rows = [judged(i, 1) for i in V.HELDOUT_IDS]
    m = V.task_metrics2(rows, V.HELDOUT)
    assert m["task_correct_heldout"] == 0.0 and m["task_correct_dev"] is None
    assert m["tasks_solved_dev"] is None and m["n_tasks"] == 20


# ---- score2 and sample2 without the emulator or the model ------------------------------------

def write_samples(d, rows, sha, **meta):
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "sample.json").write_text(json.dumps({"k": 1, "seed0": 1, "export": "EX",
                                               "suite_sha256": sha, **meta}))


def fake_runner(seen):
    """Shows the reference (labelled) for gold programs, 0 otherwise; checks the harness."""
    golds = {tuple(T.program_ids(t, t.gold)): t for t in V.SUITE_V2}

    def run(programs, h, out):
        seen.append((h, len(programs), out))
        recs = []
        for p in programs:
            t = golds.get(tuple(p))
            j = t.inputs.index(h.values) if t else 0
            assert t is None or V.harness(t, j) == h
            line = [f"{V.LABELS[t.id][0]}", f"{t.expected()[j]:.10g}"] if t else ["0"]
            recs.append({"outcome": "OK", "home_text": line, "yields": [], "extra_inputs": 0,
                         "decoy": h.decoy, "prefill": dict(h.prefill)})
        return recs
    return run


def test_score2_rejudges_v1_samples_on_the_dev_tasks_under_v2(tmp_path):
    rows = [{"task": t.id, "seed": 1, "ids": T.program_ids(t, t.gold)} for t in T.SUITE]
    rows += [{"task": T.SUITE[0].id, "seed": 2, "ids": [BOS]}]
    write_samples(tmp_path, rows, V.V1_SUITE_SHA256)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    seen = []
    m = V.score2(tmp_path, runner=fake_runner(seen))
    assert len(seen) == 120 and {n for _, n, _ in seen} <= {1, 2}
    out = tmp_path / "v2"
    assert all(Path(o).is_relative_to(out / "runs") for _, _, o in seen)
    assert json.loads((out / "metrics.json").read_text()) == m
    assert m["suite_sha256"] == V.SUITE_V2_SHA256 and m["source"] == "v1"
    assert m["n_samples"] == 41 and m["n_pass"] == 40 and m["n_empty"] == 1
    assert m["task_correct_dev"] == pytest.approx(4000 / 41) and m["task_correct_heldout"] is None
    assert m["task_correct_labelled"] == pytest.approx(4000 / 41)
    assert m["samples_sha256"] == hashlib.sha256(before["samples.jsonl"]).hexdigest()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    assert len((out / "judged.jsonl").read_text().splitlines()) == 41


@pytest.mark.parametrize("sha,tasks", [(V.V1_SUITE_SHA256, T.SUITE),
                                       (V.SUITE_V2_SHA256, V.HELDOUT[:3])])
def test_score2_queues_every_batch_before_it_awaits_a_record(tmp_path, sha, tasks):
    """A pooled runner gets all tasks' batches (each under its tuple's harness) first, then
    the records are awaited; the judgment is byte for byte the one a plain runner gives."""
    from lab.tests.test_tasks import queued
    rows = [{"task": t.id, "seed": s, "ids": T.program_ids(t, t.gold if s == 1 else
                                                           t.mutants[0])}
            for t in tasks for s in (1, 2)] + [{"task": tasks[0].id, "seed": 3, "ids": [BOS]}]
    meta = {} if sha == V.V1_SUITE_SHA256 else {"tasks": [t.id for t in tasks]}
    outs = []
    for name in ("plain", "pooled"):
        write_samples(tmp_path / name, rows, sha, **meta)
        outs.append(tmp_path / name / ("v2" if sha == V.V1_SUITE_SHA256 else ""))
    seen_plain, seen_pooled, log = [], [], []
    V.score2(tmp_path / "plain", runner=fake_runner(seen_plain), out=outs[0])
    V.score2(tmp_path / "pooled", runner=queued(fake_runner(seen_pooled), log), out=outs[1])
    assert log == ["submit"] * 3 * len(tasks) + ["result"] * 3 * len(tasks)
    assert [(h, n) for h, n, _ in seen_pooled] == [(h, n) for h, n, _ in seen_plain]
    assert (outs[1] / "judged.jsonl").read_bytes() == (outs[0] / "judged.jsonl").read_bytes()
    plain, pooled = (json.loads((o / "metrics.json").read_text()) for o in outs)
    assert pooled.pop("samples_dir") != plain.pop("samples_dir") and pooled == plain


def test_score2_without_a_runner_runs_every_batch_on_one_pool(tmp_path, monkeypatch):
    from lab.tests.test_tasks import pooled_stand_in, queued
    rows = [{"task": t.id, "seed": 1, "ids": T.program_ids(t, t.gold)} for t in T.SUITE]
    write_samples(tmp_path, rows, V.V1_SUITE_SHA256)
    log, seen = [], []
    monkeypatch.setattr(V, "pooled_runner2", pooled_stand_in(
        lambda w, b: queued(fake_runner(seen), log), log))
    m = V.score2(tmp_path, workers=6, budget_ms=9000)
    assert log[0] == ("enter", 6, 9000) and log[-1] == "exit" and len(seen) == 120
    assert log.index("result") > max(i for i, x in enumerate(log) if x == "submit")
    assert m["n_pass"] == 40 and m["budget_ms"] == 9000


def test_score2_never_writes_into_a_v1_samples_dir(tmp_path):
    write_samples(tmp_path, [], V.V1_SUITE_SHA256)
    with pytest.raises(ValueError, match="v1"):
        V.score2(tmp_path, out=tmp_path, runner=lambda *a: [])


def test_score2_refuses_samples_of_an_unknown_suite(tmp_path):
    write_samples(tmp_path, [], "0" * 64)
    with pytest.raises(ValueError, match="suite"):
        V.score2(tmp_path, runner=lambda *a: [])


def test_sample2_draws_the_heldout_tasks_and_scores_them(tmp_path, monkeypatch):
    calls = []

    def fake_draw(export, jobs, workers):
        calls.append((export, jobs, workers))
        return [T.program_ids(V.TASKS_V2[tid], V.TASKS_V2[tid].gold) for tid, _ in jobs]
    monkeypatch.setattr(V, "draw2", fake_draw)
    seen = []
    m = V.sample2("EX", k=2, seed0=4, out=tmp_path, workers=3, split="heldout",
                  runner=fake_runner(seen))
    [(export, jobs, workers)] = calls
    assert export == "EX" and workers == 3
    assert jobs == [(i, s) for i in V.HELDOUT_IDS for s in (4, 5)]
    meta = json.loads((tmp_path / "sample.json").read_text())
    assert meta["suite_sha256"] == V.SUITE_V2_SHA256 and meta["split"] == "heldout"
    assert meta["tasks"] == list(V.HELDOUT_IDS) and meta["k"] == 2 and meta["seed0"] == 4
    assert m["task_correct_heldout"] == 100.0 and m["task_correct_dev"] is None
    assert m["tasks_solved_heldout"] == 20 and m["source"] == "v2"
    assert json.loads((tmp_path / "metrics.json").read_text()) == m


def test_draw2_is_v21_decoding_from_a_heldout_opening(monkeypatch):
    import model.intref as IR
    from model.intmask import IntMask
    calls = []

    class FakeModel:
        vocab = T._vocab().entries

    monkeypatch.setattr(IR.IntModel, "load",
                        classmethod(lambda cls, d, horizons=False: FakeModel()))

    def fake_generate(model, prompt, max_new, cfg, cap=None, mask=None):
        calls.append((prompt, max_new, cfg, cap, mask))
        return [BOS, *prompt, EOS]
    monkeypatch.setattr(IR, "generate", fake_generate)
    task = V.TASKS_V2[V.HELDOUT_IDS[0]]
    out = V.draw2("EX", [(task.id, 9)], workers=1)
    [(prompt, max_new, cfg, cap, mask)] = calls
    assert prompt == task.opening_ids() and max_new == 1024 and cap == 1024
    assert cfg == IR.SamplerConfig(205, 20, False, 9) and isinstance(mask, IntMask)
    assert out == [[BOS, *task.opening_ids(), EOS]]


def test_cli_score2_and_sample2_dispatch(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(V, "score2", lambda d, **kw: got.append(("score2", d, kw)) or {})
    monkeypatch.setattr(V, "sample2", lambda *a, **kw: got.append(("sample2", a, kw)) or {})
    monkeypatch.setattr(V, "build2", lambda out, runner: got.append(("build2", out)) or
                        {"ok": True, "n_tasks": 60, "tasks": [], "suite_sha256": "x"})
    assert T.main(["score2", str(tmp_path), "--workers", "2"]) == 0
    assert T.main(["sample2", "--export", "E", "--k", "25", "--seed0", "1", "--tasks",
                   "heldout", "--out", str(tmp_path / "s")]) == 0
    assert T.main(["build2", "--out", str(tmp_path / "b")]) == 0
    assert got[0][0] == "score2" and got[0][1] == tmp_path and got[0][2]["workers"] == 2
    assert got[1][0] == "sample2" and got[1][2]["split"] == "heldout" and got[1][2]["k"] == 25
    assert got[2][0] == "build2"


# ---- v1 byte-identity on saved E6 and E7 records ---------------------------------------------

def v1_dirs():
    """Every saved v1 score dir: E6 (all arms), E7 (W) and the TASKS.md baselines."""
    found = [*RUNS.glob("e6/s*/*/tasks"), *RUNS.glob("e7/s*/W/tasks"), *RUNS.glob("tasks/*")]
    return sorted(d for d in found if (d / "judged.jsonl").exists())


def test_rejudge_v1_reproduces_a_judged_file(tmp_path):
    """Without saved runs: score a tiny suite with a stand-in runner that writes records the
    way run_specs does, then rebuild judged.jsonl and metrics.json from those records."""
    small = T.SUITE[:2]
    rows = [{"task": t.id, "seed": s, "ids": T.program_ids(t, t.gold if s == 1 else t.mutants[0])}
            for t in small for s in (1, 2)] + [{"task": small[0].id, "seed": 3, "ids": [BOS]}]
    d = tmp_path / "tasks"
    write_samples(d, rows, T.suite_sha256(small))

    def runner(programs, values, out):
        task = next(t for t in small if values in t.inputs)
        ref = task.expected()[task.inputs.index(values)]
        recs = [{"index": i, "outcome": "OK", "home_text": [f"{ref:.10g}" if p == T.program_ids(
            task, task.gold) else "0"], "yields": [], "extra_inputs": 0}
            for i, p in enumerate(programs)]
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        return recs
    T.score(d, runner=runner, tasks=small)
    text, metrics = V.rejudge_v1(d, tasks=small)
    assert text == (d / "judged.jsonl").read_text()
    assert metrics == (d / "metrics.json").read_text()


@pytest.mark.skipif(not v1_dirs(), reason="no saved E6/E7 task-suite runs (set CALCLM_RUNS)")
@pytest.mark.parametrize("d", v1_dirs(), ids=lambda d: "/".join(d.parts[-4:-1]))
def test_saved_v1_records_judge_byte_identically(d):
    """E6 Amendment 1's neutrality check, again: every saved v1 score dir re-judged from its
    emulator records gives its judged.jsonl and metrics.json byte for byte."""
    text, metrics = V.rejudge_v1(d)
    assert text == (d / "judged.jsonl").read_text()
    assert metrics == (d / "metrics.json").read_text()


# ---- the metric's own tests in CEmu ----------------------------------------------------------

@pytest.fixture(scope="module")
def emu():
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    return E.prepare()


def runner_for(env, hardened):
    """In-process runner; hardened: v2's harness (decoy and prefill), else v1's."""
    from eval import driver, inputs

    def run(programs, h, out):
        cfg = driver.Config(input_values=tuple(h.values),
                            **({"decoy": h.decoy, "prefill": h.prefill} if hardened else {}))
        specs = inputs.from_token_ids(programs, T.VOCAB, out)
        return [driver.run_one(env.emu, s, env.clean_state, cfg, yields=True) for s in specs]
    return run


@pytest.mark.parametrize("task", V.SUITE_V2, ids=lambda t: t.id)
def test_gold_passes_and_mutants_and_loopholes_fail_under_v2_in_cemu(emu, tmp_path, task):
    v = V.validate2(task, runner_for(emu, True), tmp_path)
    assert v["gold"]["pass"], v
    assert not any(m["pass"] for m in v["mutants"]), v
    assert not any(m["pass"] for m in v["loopholes"]), v
    assert v["ok"]


def test_loophole_mutants_cover_ten_tasks_and_both_kinds():
    assert len(V.LOOPHOLES) >= 10
    kinds = Counter(k for ms in V.LOOPHOLES.values() for k, _ in ms)
    assert kinds["extra"] >= 5 and kinds["unset"] >= 5 and set(kinds) == {"extra", "unset"}
    assert set(V.LOOPHOLES) <= set(V.TASKS_V2)


@pytest.mark.parametrize("tid", sorted(V.LOOPHOLES))
def test_loophole_mutants_pass_under_v1s_harness_in_cemu(emu, tmp_path, tid):
    """Each loophole mutant is a real exploit: v1's harness (last value at extra prompts,
    variables 0) passes it; v2's (test above) fails it."""
    task = V.TASKS_V2[tid]
    progs = [T.program_ids(task, body) for _, body in V.LOOPHOLES[tid]]
    for recs in V.run_programs2(task, progs, runner_for(emu, False), tmp_path):
        j = T.judge(recs, task.expected())
        assert j["pass"], (tid, j)


def coincidental():
    if not (RUNS / "e6" / "audit" / "labels.json").exists():
        return []
    return V.coincidental_items(RUNS)


@pytest.mark.skipif(not coincidental(), reason="no E6/E7 audits (set CALCLM_RUNS)")
def test_e6_and_e7_coincidental_passes_fail_under_v2_in_cemu(emu, tmp_path):
    items = coincidental()
    assert len(items) == 9                              # 6 in E6, 3 in E7
    res = V.regression(items, runner_for(emu, True), tmp_path)
    assert all(r["v1_pass"] for r in res), res          # they did pass under v1
    assert [r["item"] for r in res if r["v2_pass"]] == [], res
