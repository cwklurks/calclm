"""End to end: CLI over a directory of .8xp files with worker processes."""
import json

import pytest
from tivars import TIProgram

from eval import driver, handmade, inputs, run_programs


def make_dir(tmp_path, names):
    d = tmp_path / "progs"
    d.mkdir()
    for case in handmade.CASES:
        if case.name in names:
            p = TIProgram(name=case.name)
            p.load_string(case.src)
            p.save(str(d / f"{case.name}.8xp"))
    (d / "JUNK.8xp").write_bytes(b"garbage")
    return d


def test_cli_dir_parallel(emu_env, tmp_path):
    d = make_dir(tmp_path, {"OKDISP", "ESYNTAX", "INPUT", "TLOOP"})
    out = tmp_path / "run"
    rc = run_programs.main([str(d), "--out", str(out), "--workers", "2", "--budget-ms", "4000",
                            "--shots", "2"])
    assert rc == 0
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == list(range(5))
    labels = {r["name"]: r["label"] for r in rows}
    assert labels == {"ESYNTAX": "ERR:SYNTAX", "INPUT": "OK", "OKDISP": "OK",
                      "TLOOP": "TIMEOUT", "JUNK": "REJECTED"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n"] == 5 and summary["programs_per_min"] > 0
    assert (out / "summary.md").exists()
    assert len(list((out / "shots").glob("*.png"))) == 2


def test_api_token_ids(emu_env, tmp_path):
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE", "34", "32", "3F", "71"]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    # Disp 42 / Disp - (syntax error) / <unk>
    seqs = [[1, 4, 5, 6, 2], [1, 4, 8, 2], [1, 4, 3, 2]]
    res = run_programs.run_token_ids(seqs, tmp_path / "vocab.json", tmp_path / "run", workers=1,
                                     budget_ms=3000)
    assert [r["label"] for r in res] == ["OK", "ERR:SYNTAX", "REJECTED"]


def crash_on_tloop(job):
    """Worker function that kills its process for one program (stands in for a CEmu crash)."""
    import os
    if job[0].name == "TLOOP":
        os._exit(3)
    return run_programs._work(job)


def test_lost_worker_is_recorded_and_run_completes(emu_env, tmp_path):
    specs = [inputs.from_bodies([(c.name, c.body())], tmp_path / "w", start=i)[0]
             for i, c in enumerate(map(handmade.case, ("OKDISP", "OKLOOP", "OKOUT", "TLOOP")))]
    assert specs[3].name == "TLOOP"
    res = run_programs.run_specs(specs, tmp_path / "run", workers=2,
                                 cfg=driver.Config(budget_ms=3000), shots=0,
                                 work=crash_on_tloop, job_timeout_s=6)
    assert [r["label"] for r in res[:3]] == ["OK", "OK", "OK"]
    assert res[3]["outcome"] == "HARNESS_ERROR"


# ---- EmuPool: one pool of workers across many batches ---------------------------------------

FILL = tuple((v, f"{1.11 + 0.31 * i:.2f}") for i, v in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZθ"))
HARNESSED = driver.Config(budget_ms=4000, input_values=("4",), decoy="3.17", prefill=FILL)


def batch_specs(tmp_path, tag, bodies):
    return inputs.from_bodies([(f"{tag}{i}", handmade.Case(f"{tag}{i}", b, "OK").body())
                               for i, b in enumerate(bodies)], tmp_path / f"in-{tag}")


def two_batches(tmp_path):
    """A harnessed batch (input values, decoy, prefill) and a plain one with a loop."""
    a = batch_specs(tmp_path, "PA", ["Prompt A\nDisp 100A+B", "Prompt A,C\nDisp A+C",
                                     "Disp 42", "Disp 1+"])
    b = batch_specs(tmp_path, "PB", ["Input A\nDisp A*2", "While 1\nEnd", "Disp 7"])
    return [(a, HARNESSED), (b, driver.Config(budget_ms=3000))]


def no_wall(rows):
    return [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]


def untimed(path):
    s = json.loads(path.read_text())
    return {k: v for k, v in s.items() if k not in ("wall_s", "programs_per_min")}


def test_pool_batches_give_run_specs_records_and_files_in_cemu(emu_env, tmp_path):
    ref = [run_programs.run_specs(specs, tmp_path / f"ref{k}", workers=2, cfg=cfg, shots=0,
                                  title="T", yields=True)
           for k, (specs, cfg) in enumerate(two_batches(tmp_path))]
    with run_programs.EmuPool(3) as pool:
        handles = [pool.submit(specs, tmp_path / f"pool{k}", workers=2, cfg=cfg, shots=0,
                               title="T", yields=True)
                   for k, (specs, cfg) in enumerate(two_batches(tmp_path))]
        got = [h.result() for h in reversed(handles)][::-1]   # either batch may ask first
        assert pool._got == [None] * 7      # the pool keeps no record once its batch has it
    assert [no_wall(g) for g in got] == [no_wall(r) for r in ref]
    assert got[0][0]["prefill"] == dict(FILL) and got[0][1]["extra_inputs"] == 1
    assert [r["outcome"] for r in got[1]] == ["OK", "TIMEOUT", "OK"]
    for k in range(2):
        a, b = tmp_path / f"ref{k}", tmp_path / f"pool{k}"
        assert (b / "config.json").read_bytes() == (a / "config.json").read_bytes()
        assert untimed(b / "summary.json") == untimed(a / "summary.json")
        assert no_wall(map(json.loads, (b / "results.jsonl").read_text().splitlines())) == \
            no_wall(map(json.loads, (a / "results.jsonl").read_text().splitlines()))


def test_pool_runs_in_this_process_with_one_worker_in_cemu(emu_env, tmp_path):
    [(specs, cfg), _] = two_batches(tmp_path)
    ref = run_programs.run_specs(specs, tmp_path / "ref", workers=1, cfg=cfg, shots=0)
    with run_programs.EmuPool(1) as pool:
        h = pool.submit(specs, tmp_path / "pool", workers=1, cfg=cfg, shots=0)
        assert len(pool._got) == len(specs)                   # ran at submit
        assert no_wall(h.result()) == no_wall(ref)


def crash_on_tloop_job(job):
    """EmuPool worker function that kills its process for one program."""
    import os
    if job[0].name == "TLOOP":
        os._exit(3)
    return run_programs._work_job(job)


def test_pool_survives_a_lost_worker_in_cemu(emu_env, tmp_path):
    specs = [inputs.from_bodies([(c.name, c.body())], tmp_path / "w", start=i)[0]
             for i, c in enumerate(map(handmade.case, ("OKDISP", "TLOOP", "OKOUT")))]
    later = batch_specs(tmp_path, "LT", ["Disp 42", "Disp 5"])
    with run_programs.EmuPool(2, work=crash_on_tloop_job) as pool:
        first = pool.submit(specs, tmp_path / "a", cfg=driver.Config(budget_ms=3000), shots=0,
                            job_timeout_s=6)
        second = pool.submit(later, tmp_path / "b", cfg=driver.Config(budget_ms=3000), shots=0)
        assert [r["outcome"] for r in first.result()] == ["OK", "HARNESS_ERROR", "OK"]
        assert [r["label"] for r in second.result()] == ["OK", "OK"]


def hang_on_hang_job(job):
    """EmuPool worker function that never returns for programs HANG0 and HANG1."""
    import time
    if job[0].name in ("HANG0", "HANG1"):
        time.sleep(3600)
    return run_programs._work_job(job)


def test_pool_replaces_workers_stuck_past_the_timeout_in_cemu(emu_env, tmp_path):
    """As run_specs' pool ended with its batch, a stuck worker must not hold its slot: two
    stuck jobs on a 2-worker pool cost only themselves, later batches still run."""
    stuck = batch_specs(tmp_path, "HANG", ["Disp 1", "Disp 2", "Disp 3"])
    with run_programs.EmuPool(2, work=hang_on_hang_job) as pool:
        first = pool.submit(stuck, tmp_path / "a", cfg=driver.Config(budget_ms=3000), shots=0,
                            job_timeout_s=4)
        later = [pool.submit(batch_specs(tmp_path, f"L{k}x", ["Disp 42", "Disp 5"]),
                             tmp_path / f"l{k}", cfg=HARNESSED, shots=0, job_timeout_s=60)
                 for k in range(2)]
        assert [r["outcome"] for r in first.result()] == ["HARNESS_ERROR", "HARNESS_ERROR",
                                                          "OK"]
        assert [r["label"] for h in later for r in h.result()] == ["OK"] * 4


def test_a_closed_pool_refuses_to_collect(emu_env, tmp_path):
    with run_programs.EmuPool(2) as pool:
        h = pool.submit(batch_specs(tmp_path, "CL", ["Disp 1"]), tmp_path / "c", shots=0)
    with pytest.raises(RuntimeError, match="closed"):
        h.result()


def test_pool_removes_its_prefill_files_in_cemu(emu_env, tmp_path, monkeypatch):
    from eval import emu as E
    monkeypatch.setattr(driver, "_PREFILLED", {})
    before = set(E.CACHE.iterdir())
    [(specs, cfg), _] = two_batches(tmp_path)
    with run_programs.EmuPool(2) as pool:
        pool.submit(specs, tmp_path / "run", cfg=cfg, shots=0).result()
    assert set(E.CACHE.iterdir()) == before and driver.PREFILL_ROOT is None


def test_pool_keeps_the_wall_clock_limit_per_run_in_cemu(emu_env, tmp_path):
    """driver's wall_limit_s still stops each run; the pool's age does not count."""
    cfg = driver.Config(budget_ms=600_000, wall_limit_s=1.0)
    with run_programs.EmuPool(2) as pool:
        hs = [pool.submit(batch_specs(tmp_path, f"W{k}", ["While 1\nEnd", "Disp 3"]),
                          tmp_path / f"w{k}", cfg=cfg, shots=0) for k in range(2)]
        recs = [r for h in hs for r in h.result()]
    assert [r["outcome"] for r in recs] == ["TIMEOUT", "OK"] * 2
    assert all(r["emu_ms"] < 600_000 and r["wall_s"] < 30 for r in recs)
