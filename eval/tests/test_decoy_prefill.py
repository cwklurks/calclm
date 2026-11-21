"""Task suite v2's hardened harness (docs/agents/tasks/F2-task-suite-v2.md): driver.Config.decoy
answers Input/Prompt waits past input_values with a decoy value instead of the last value, and
driver.Config.prefill stores values in real variables over the link before the program runs.
Both are off by default, and then nothing changes."""
import json

import pytest

from eval import driver, inputs, run_programs
from eval.handmade import Case

FOUR = Case("FOURP", "Prompt A,B,C,D\nDisp 1000A+100B+10C+D", "OK")
VARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("θ",)
FILL = tuple((v, f"{1.11 + 0.31 * i:.2f}") for i, v in enumerate(VARS))   # 1.11 ... 9.17


# ---- without the emulator ------------------------------------------------------------------

class FakeEmu:
    def __init__(self):
        self.keys = []

    def run(self, ms):
        pass

    def inject_key(self, key):
        self.keys.append(key)


def fake_run(cfg):
    spec = inputs.ProgramSpec(0, "P", "P", "test", Case("P", "Prompt A", "OK").body(), None)
    return driver._Run(FakeEmu(), spec, cfg)


def cemu_keys():
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("CEmu harness not available")
    E.cemu()
    from tools.cemu import keys
    return keys


def typed(run, n):
    for _ in range(n):
        run.answer("input")
    out, cur = [], []
    for k in run.emu.keys:
        if k == "Enter":
            out.append(cur)
            cur = []
        else:
            cur.append(k)
    return out


def test_defaults_are_off():
    c = driver.Config()
    assert c.decoy is None and c.prefill is None
    assert c == driver.Config(decoy=None, prefill=None)


def test_waits_past_the_tuple_get_the_decoy_and_are_counted():
    text_keys = cemu_keys().text_keys
    run = fake_run(driver.Config(input_values=("1", "3.5", "12"), decoy="4.37"))
    got = typed(run, 5)
    assert got == [text_keys("1"), text_keys("3.5"), text_keys("12"), text_keys("4.37"),
                   text_keys("4.37")]
    assert run.extra_inputs == 2


def test_decoy_needs_input_values_and_a_value():
    with pytest.raises(ValueError):
        driver.Config(decoy="4.37")
    with pytest.raises(ValueError):
        driver.Config(input_values=("1",), decoy="")
    with pytest.raises(ValueError):
        driver.Config(input_values=("1",), decoy=4.37)


def test_prefill_must_name_distinct_real_variables_with_string_values():
    driver.Config(prefill=FILL)
    for bad in ((), (("A", 3.5),), (("A", "3.5"), ("A", "4.5")), (("AB", "3.5"),),
                (("Str1", "3.5"),), (("a", "3.5"),)):
        with pytest.raises(ValueError):
            driver.Config(prefill=bad)


def test_config_json_records_decoy_and_prefill_only_when_set():
    off = run_programs._config_meta(driver.Config())
    assert "decoy" not in off and "prefill" not in off
    on = run_programs._config_meta(driver.Config(input_values=("3",), decoy="4.37",
                                                 prefill=(("A", "5.25"),)))
    on = json.loads(json.dumps(on))
    assert on["decoy"] == "4.37" and on["prefill"] == [["A", "5.25"]]


def test_run_token_ids_threads_decoy_and_prefill_only_when_given(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(run_programs.inputs, "from_token_ids", lambda *a, **k: [])
    monkeypatch.setattr(run_programs, "run_specs", lambda specs, out, cfg, **kw: seen.setdefault(
        "cfg", cfg) and [])
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000,
                               input_values=("3",), decoy="4.37", prefill=(("A", "5.25"),))
    assert seen["cfg"] == driver.Config(budget_ms=5000, input_values=("3",), decoy="4.37",
                                        prefill=(("A", "5.25"),))
    seen.clear()
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000)
    assert seen["cfg"] == driver.Config(budget_ms=5000)


def test_lab_score_run_emulator_threads_decoy_and_prefill(monkeypatch, tmp_path):
    from lab import score
    seen = {}

    def fake(programs, vocab, out, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr("eval.run_programs.run_token_ids", fake)
    score.run_emulator([[1, 5, 2]], tmp_path, input_values=("3",), decoy="4.37",
                       prefill=(("A", "5.25"),))
    assert seen["decoy"] == "4.37" and seen["prefill"] == (("A", "5.25"),)
    seen.clear()
    score.run_emulator([[1, 5, 2]], tmp_path, input_values=("3",))
    assert "decoy" not in seen and "prefill" not in seen


# ---- in the emulator -----------------------------------------------------------------------

def run(emu_env, tmp_path, body, cfg, name="HARNV2"):
    [spec] = inputs.from_bodies([(name, Case(name, body, "OK").body())],
                                tmp_path / str(len(list(tmp_path.iterdir()))))
    return driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg)


def shown(r):
    return [x.strip() for x in r["home_text"] if x.strip()]


def test_an_extra_prompt_gets_the_decoy_in_cemu(emu_env, tmp_path):
    cfg = driver.Config(budget_ms=6000, input_values=("1", "3", "5"), decoy="8.25")
    r = run(emu_env, tmp_path, FOUR.src, cfg)
    assert r["label"] == "OK" and r["actions"]["input"] == 4, r
    assert "1358.25" in shown(r) and r["extra_inputs"] == 1 and r["decoy"] == "8.25", r


def test_prefilled_variables_hold_their_values_in_cemu(emu_env, tmp_path):
    cfg = driver.Config(budget_ms=6000, prefill=FILL)
    r = run(emu_env, tmp_path, "ClrHome\nDisp 100A+B", cfg)
    vals = dict(FILL)
    want = 100 * float(vals["A"]) + float(vals["B"])
    assert r["label"] == "OK" and f"{want:g}" in shown(r), r
    assert r["prefill"] == vals and "decoy" not in r
    for v, x in FILL:                                   # every variable, read back after the run
        assert emu_env.emu.get_real(v) == pytest.approx(float(x), abs=1e-12), v


def test_a_prompt_overwrites_a_prefilled_variable_in_cemu(emu_env, tmp_path):
    cfg = driver.Config(budget_ms=6000, input_values=("4",), prefill=FILL)
    r = run(emu_env, tmp_path, "Prompt A\nDisp 100A+B", cfg)
    want = 400 + float(dict(FILL)["B"])
    assert r["label"] == "OK" and f"{want:g}" in shown(r), r


def test_prefill_leaves_ans_and_the_home_screen_alone_in_cemu(emu_env, tmp_path):
    base = run(emu_env, tmp_path, "Disp Ans", driver.Config(budget_ms=6000))
    filled = run(emu_env, tmp_path, "Disp Ans", driver.Config(budget_ms=6000, prefill=FILL))
    assert filled.pop("prefill") == dict(FILL)
    for r in (base, filled):
        r.pop("wall_s")
    assert base == filled                         # Ans still 0; nothing else on the screen


def test_the_prefilled_state_is_reused_for_the_same_values_in_cemu(emu_env, tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(driver, "_PREFILLED", {})
    calls = []
    real = driver.prefill
    monkeypatch.setattr(driver, "prefill", lambda emu, pairs: calls.append(pairs) or
                        real(emu, pairs))
    other = (("A", "6.25"), *FILL[1:])
    body = "ClrHome\nDisp 100A+B"
    a = run(emu_env, tmp_path, body, driver.Config(budget_ms=6000, prefill=FILL))
    b = run(emu_env, tmp_path, body, driver.Config(budget_ms=6000, prefill=FILL))
    c = run(emu_env, tmp_path, body, driver.Config(budget_ms=6000, prefill=other))
    d = run(emu_env, tmp_path, body, driver.Config(budget_ms=6000, prefill=FILL))
    assert calls == [FILL, other, FILL]
    for r in (a, b, c, d):
        r.pop("wall_s")
    assert a == b == d and "626.42" in shown(c), c          # 100 x 6.25 + B (1.42)


def test_each_process_cache_gets_its_own_file_in_cemu(emu_env, tmp_path, monkeypatch):
    """Sandboxed runs reuse tiny PIDs: two processes must never share a cache file."""
    monkeypatch.setattr(driver, "PREFILL_ROOT", tmp_path / "root")
    monkeypatch.setattr(driver, "_PREFILLED", {})
    a = driver.prefilled_state(emu_env.emu, emu_env.clean_state, FILL)
    monkeypatch.setattr(driver, "_PREFILLED", {})               # as if another process
    b = driver.prefilled_state(emu_env.emu, emu_env.clean_state, FILL)
    assert a != b and a.exists() and b.exists()
    assert a.parent.parent == b.parent.parent == tmp_path / "root"


def test_a_removed_cache_file_is_rebuilt_not_reused_in_cemu(emu_env, tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(driver, "PREFILL_ROOT", tmp_path / "root")
    monkeypatch.setattr(driver, "_PREFILLED", {})
    a = driver.prefilled_state(emu_env.emu, emu_env.clean_state, FILL)
    shutil.rmtree(tmp_path / "root")
    b = driver.prefilled_state(emu_env.emu, emu_env.clean_state, FILL)
    assert b.exists(), b
    r = run(emu_env, tmp_path, "ClrHome\nDisp 100A+B", driver.Config(budget_ms=6000,
                                                                      prefill=FILL))
    assert r["label"] == "OK" and r["prefill"] == dict(FILL) and a


@pytest.mark.parametrize("workers", [1, 2])
def test_run_specs_removes_its_prefill_files_in_cemu(emu_env, tmp_path, workers, monkeypatch):
    """Pool workers are terminated (no atexit), so run_specs owns the cleanup."""
    from eval import emu as E
    monkeypatch.setattr(driver, "_PREFILLED", {})
    before = set(E.CACHE.iterdir())
    specs = inputs.from_bodies([(f"PF{i}", Case(f"PF{i}", "Disp 100A+B", "OK").body())
                                for i in range(2)], tmp_path / "in")
    recs = run_programs.run_specs(specs, tmp_path / "run", workers=workers, shots=0,
                                  cfg=driver.Config(budget_ms=6000, prefill=FILL))
    want = 100 * float(dict(FILL)["A"]) + float(dict(FILL)["B"])
    assert all(r["label"] == "OK" and f"{want:g}" in shown(r) for r in recs), recs
    assert set(E.CACHE.iterdir()) == before


def test_a_shared_prefilled_state_is_loaded_instead_of_built_in_cemu(emu_env, tmp_path,
                                                                     monkeypatch):
    """run_one(prefilled=...): a state save_prefilled wrote for cfg.prefill (EmuPool builds one
    per batch for all its workers); when that file is missing, the run builds its own."""
    shared = driver.save_prefilled(emu_env.emu, emu_env.clean_state, FILL, tmp_path / "s.img")
    monkeypatch.setattr(driver, "_PREFILLED", {})
    calls = []
    real = driver.prefill
    monkeypatch.setattr(driver, "prefill", lambda emu, pairs: calls.append(pairs) or
                        real(emu, pairs))
    cfg = driver.Config(budget_ms=6000, prefill=FILL)
    [spec] = inputs.from_bodies([("PS", Case("PS", "Disp 100A+B", "OK").body())], tmp_path)
    a = driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg, prefilled=shared)
    assert calls == []
    b = driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg,
                       prefilled=tmp_path / "missing.img")
    assert calls == [FILL]
    for r in (a, b):
        r.pop("wall_s")
    want = 100 * float(dict(FILL)["A"]) + float(dict(FILL)["B"])
    assert a == b and a["label"] == "OK" and f"{want:g}" in shown(a), a


def test_the_pool_builds_each_prefill_once_for_all_its_workers_in_cemu(emu_env, tmp_path):
    other = (("A", "6.25"), *FILL[1:])
    specs = inputs.from_bodies([(f"PP{i}", Case(f"PP{i}", "Disp 100A+B", "OK").body())
                                for i in range(4)], tmp_path / "in")
    with run_programs.EmuPool(2) as pool:
        hs = [pool.submit(specs, tmp_path / f"r{k}", cfg=driver.Config(budget_ms=6000,
                                                                       prefill=fill), shots=0)
              for k, fill in enumerate((FILL, other, FILL))]
        assert list(pool._shared) == [FILL, other]
        paths = list(pool._shared.values())
        recs = [h.result() for h in hs]
        assert not any(p.exists() for p in paths) and not pool._shared   # removed once used
    shows = [f"{100 * float(dict(f)['A']) + float(dict(f)['B']):g}" for f in (FILL, other, FILL)]
    for rs, want in zip(recs, shows):
        assert all(r["label"] == "OK" and want in shown(r) for r in rs), rs


def test_without_prefill_variables_are_zero_in_cemu(emu_env, tmp_path):
    r = run(emu_env, tmp_path, "ClrHome\nDisp 100A+B+5", driver.Config(budget_ms=6000))
    assert r["label"] == "OK" and "5" in shown(r) and "prefill" not in r, r
