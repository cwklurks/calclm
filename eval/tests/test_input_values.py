"""Typed input sequences (driver.Config.input_values, docs/agents/tasks/F-task-suite.md): the
k-th Input/Prompt wait is answered with input_values[k]; waits beyond the tuple get the last
value and are counted in the record's `extra_inputs`. None (the default) changes nothing."""
import hashlib
import json
from dataclasses import asdict

import pytest

from eval import driver, handmade, inputs, run_programs
from eval.handmade import Case

CFG = driver.Config(budget_ms=6000)
THREE = Case("THREEP", "ClrHome\nPrompt A\nPrompt B\nPrompt C\nDisp 100A+10B+C", "OK")
FOUR = Case("FOURP", "Prompt A,B,C,D\nDisp 1000A+100B+10C+D", "OK")


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


def test_default_config_has_no_input_values():
    assert driver.Config().input_values is None
    assert driver.Config() == driver.Config(input_values=None)


def cemu_keys():
    """tools.cemu.keys (the CEmu harness's key names; the driver types with text_keys)."""
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("CEmu harness not available")
    E.cemu()
    from tools.cemu import keys
    return keys


def test_each_wait_gets_its_own_value_then_the_last_one_repeats():
    text_keys = cemu_keys().text_keys
    run = fake_run(driver.Config(input_values=("1", "3.5", "12")))
    got = typed(run, 5)
    assert got == [text_keys("1"), text_keys("3.5"), text_keys("12"), text_keys("12"),
                   text_keys("12")]
    assert run.extra_inputs == 2 and run.actions["input"] == 5


def test_without_input_values_every_wait_gets_input_text():
    text_keys = cemu_keys().text_keys
    run = fake_run(driver.Config(input_text="4"))
    assert typed(run, 3) == [text_keys("4")] * 3
    assert run.extra_inputs == 0


def test_run_token_ids_puts_input_values_on_the_config_only_when_given(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(run_programs.inputs, "from_token_ids", lambda *a, **k: [])
    monkeypatch.setattr(run_programs, "run_specs", lambda specs, out, cfg, **kw: seen.setdefault(
        "cfg", cfg) and [])
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000,
                               input_values=("3", "4.5"))
    assert seen["cfg"].input_values == ("3", "4.5") and seen["cfg"].budget_ms == 5000
    seen.clear()
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000)
    assert seen["cfg"] == driver.Config(budget_ms=5000)


def test_lab_score_run_emulator_passes_input_values_through(monkeypatch, tmp_path):
    from lab import score
    seen = {}

    def fake(programs, vocab, out, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr("eval.run_programs.run_token_ids", fake)
    score.run_emulator([[1, 5, 2]], tmp_path, input_values=("3", "5"), yields=True)
    assert seen["input_values"] == ("3", "5")
    seen.clear()
    score.run_emulator([[1, 5, 2]], tmp_path)
    assert "input_values" not in seen


def test_config_json_records_input_values_only_when_set():
    on = run_programs._config_meta(driver.Config(input_values=("3", "4")))
    off = run_programs._config_meta(driver.Config())
    assert json.loads(json.dumps(on))["input_values"] == ["3", "4"]
    assert "input_values" not in off
    # the keys, in order, of the config.json a default run wrote before input_values existed
    assert list(off) == ["budget_ms", "step_ms", "quiet_ms", "end_quiet_ms", "getkey_every_ms",
                         "grace_ms", "input_text", "feed_keys", "max_actions", "key_probe",
                         "probe_ms", "locate_errors", "wall_limit_s", "preloaded"]
    assert off == {k: (sorted(v) if isinstance(v, frozenset) else v)
                   for k, v in asdict(driver.Config()).items()
                   if k not in ("input_values", "decoy", "prefill")}   # F2's options, unset


def test_input_values_must_be_a_non_empty_tuple_of_strings():
    with pytest.raises(ValueError):
        driver.Config(input_values=())
    with pytest.raises(ValueError):
        driver.Config(input_values=(3.5, "4"))


# ---- in the emulator -----------------------------------------------------------------------

def run(emu_env, tmp_path, case, cfg):
    [spec] = inputs.from_bodies([(case.name, case.body())], tmp_path / str(id(cfg)))
    return driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg)


def shown(r):
    return [x.strip() for x in r["home_text"] if x.strip()]


def test_three_prompts_get_three_values_in_order(emu_env, tmp_path):
    r = run(emu_env, tmp_path, THREE, driver.Config(budget_ms=6000, input_values=("1", "3", "5")))
    assert r["label"] == "OK" and r["actions"]["input"] == 3, r
    assert "135" in shown(r) and r["extra_inputs"] == 0, r


def test_decimal_values_are_typed(emu_env, tmp_path):
    r = run(emu_env, tmp_path, THREE, driver.Config(budget_ms=6000,
                                                    input_values=("0.5", "1.5", "12")))
    assert r["label"] == "OK" and "77" in shown(r), r          # 50 + 15 + 12


def test_waits_past_the_tuple_get_the_last_value_and_are_counted(emu_env, tmp_path):
    r = run(emu_env, tmp_path, FOUR, driver.Config(budget_ms=6000, input_values=("1", "3", "5")))
    assert r["label"] == "OK" and r["actions"]["input"] == 4, r
    assert "1355" in shown(r) and r["extra_inputs"] == 1, r


def test_input_waits_are_sequenced_like_prompt_waits(emu_env, tmp_path):
    case = Case("THREEI", "Input A\nInput \"B?\",B\nInput C\nDisp 100A+10B+C", "OK")
    r = run(emu_env, tmp_path, case, driver.Config(budget_ms=6000, input_values=("4", "3", "5")))
    assert r["label"] == "OK" and r["actions"]["input"] == 3, r
    assert "435" in shown(r) and r["extra_inputs"] == 0, r


# Records of the driver before input_values existed (commit b15a725), Config(budget_ms=6000)
# with yield capture, wall_s dropped: sha256 of the sorted-key JSON, first 16 hex digits.
# ROM digest 65c6b7d3d144aa5f (the author's OS 5.8 ROM); the emulator is deterministic.
PINNED_ROM = "65c6b7d3d144aa5f"
PINNED = {"INPUT": "a61ff8d3d40bcab6", "PROMPT": "dbc1469673f4c938", "INSTR": "5a2fb1b32009d826",
          "TINPUT": "a7b077d2eec151e8", "OKDISP": "a02763a596a09287", "MENU": "3402f2ab0f35b2fa",
          "PAUSE": "a991f260bcc93274", "GETKEY": "b03a31eb48bb7a42",
          "ESYNTAX": "077395ba805bc43b"}


@pytest.mark.parametrize("name", PINNED)
def test_an_existing_config_judges_identically(emu_env, tmp_path, name):
    """input_values None: the record is byte for byte the one the old driver produced."""
    if emu_env.emu.digest != PINNED_ROM:
        pytest.skip("records were pinned with another ROM")
    case = handmade.case(name)
    for cfg in (CFG, driver.Config(budget_ms=6000, input_values=None)):
        [spec] = inputs.from_bodies([(case.name, case.body())], tmp_path / str(id(cfg)))
        r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg, yields=True)
        r.pop("wall_s")
        blob = json.dumps(r, sort_keys=True, ensure_ascii=False).encode()
        assert hashlib.sha256(blob).hexdigest()[:16] == PINNED[name], r


def test_a_one_value_tuple_judges_like_input_text(emu_env, tmp_path):
    a = run(emu_env, tmp_path, handmade.case("PROMPT"), driver.Config(budget_ms=6000,
                                                                      input_text="4"))
    b = run(emu_env, tmp_path, handmade.case("PROMPT"), driver.Config(budget_ms=6000,
                                                                      input_values=("4",)))
    assert b.pop("extra_inputs") == 1
    a.pop("wall_s"), b.pop("wall_s")
    assert a == b


def test_run_specs_writes_input_values_into_config_json(emu_env, tmp_path):
    [spec] = inputs.from_bodies([(THREE.name, THREE.body())], tmp_path / "in")
    cfg = driver.Config(budget_ms=6000, input_values=("1", "3", "5"))
    [r] = run_programs.run_specs([spec], tmp_path / "run", workers=1, cfg=cfg, shots=0)
    assert r["extra_inputs"] == 0 and r["label"] == "OK"
    meta = json.loads((tmp_path / "run" / "config.json").read_text())
    assert meta["config"]["input_values"] == ["1", "3", "5"]
