"""Run the hand-made programs on the emulated TI-84 Plus CE and check every outcome class.
Skipped when the ROM is not on this box."""
import pytest

from eval import driver, handmade, inputs

CFG = driver.Config(budget_ms=6000)


@pytest.mark.parametrize("case", handmade.CASES, ids=lambda c: c.name)
def test_handmade(emu_env, case, tmp_path):
    [spec] = inputs.from_bodies([(case.name, case.body())], tmp_path)
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, CFG)
    assert r["label"] == case.expect, r
    if case.timeout_state:
        assert r["timeout_state"] == case.timeout_state, r
    if case.screen_has:
        assert any(case.screen_has in line for line in r["home_text"]), r
    if case.graph is not None:
        assert r["graph"] is case.graph, r
    assert 0 < r["emu_ms"] <= CFG.budget_ms + CFG.grace_ms + 500


def test_clean_state_every_program(emu_env, tmp_path):
    first, second = inputs.from_bodies(
        [("SETA", handmade.Case("SETA", "5→A\nArchive A", "OK").body()),
         ("SHOWA", handmade.Case("SHOWA", "Disp A", "OK").body())], tmp_path)
    assert driver.run_one(emu_env.emu, first, emu_env.clean_state, CFG)["label"] == "OK"
    r = driver.run_one(emu_env.emu, second, emu_env.clean_state, CFG)
    shown = [line.strip() for line in r["home_text"] if line.strip()]
    assert r["label"] == "OK" and shown == ["0"], r


def test_rejected_spec_never_touches_emulator(tmp_path):
    [spec] = inputs.from_bodies([("EMPTY", b"")], tmp_path)
    r = driver.run_one(None, spec, None, CFG)
    assert r["outcome"] == "REJECTED" and r["label"] == "REJECTED"


def test_screenshot_written(emu_env, tmp_path):
    [spec] = inputs.from_bodies([("OKDISP", handmade.case("OKDISP").body())], tmp_path)
    shot = tmp_path / "shot.png"
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, CFG, shot=shot)
    assert shot.exists() and r["screenshot"] == str(shot)


def test_action_counts(emu_env, tmp_path):
    [spec] = inputs.from_bodies([("PROMPT", handmade.case("PROMPT").body())], tmp_path)
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, CFG)
    assert r["actions"]["input"] == 2 and r["last_action"] == "input"


@pytest.mark.parametrize("budget", range(3000, 3600, 50))
@pytest.mark.parametrize("name,state", [("TINPUT", "input"), ("TMENU", "menu"),
                                        ("TLOOP", "computing"), ("TGETKEY", "getkey")])
def test_timeout_state_stable_whatever_the_budget(emu_env, tmp_path, name, state, budget):
    """The budget can expire while the program is between two prompts; the reported state
    must still be what it is waiting for."""
    case = handmade.case(name)
    [spec] = inputs.from_bodies([(name, case.body())], tmp_path)
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, driver.Config(budget_ms=budget))
    assert (r["label"], r["timeout_state"]) == ("TIMEOUT", state)


def test_theta_name_is_sent_and_run(emu_env, tmp_path):
    [spec] = inputs.from_bodies([("θX1", handmade.Case("θX1", "Disp 3", "OK").body())], tmp_path)
    assert spec.name == "θX1"
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, CFG)
    assert r["label"] == "OK" and any("3" in line for line in r["home_text"]), r


_SHORT_LINES = "".join(f"{i}→A\n" for i in range(1, 200))                  # lines 1-199
_WRAPPED_LINES = "".join(f"\"LINE {i} IS LONG ENOUGH TO WRAP IN THE EDITOR\"→Str1\n"
                         for i in range(1, 160))                               # lines 1-159


@pytest.mark.parametrize("src,line_no,line", [
    ("Disp 1\nDisp 1/0\nDisp 3", 2, "Disp 1/0"),
    ("Disp 1\nDisp 2+\nDisp 3", 2, "Disp 2+"),
    ("1→A\nprgmNOPE\nDisp 3", 2, "prgmNOPE"),
    ("For(I,1,3)\nDisp I\nEnd\nDisp ln(0)", 4, "Disp ln(0)"),
    # long programs: 2:Goto takes the editor a while to reach the error line
    pytest.param(_SHORT_LINES + "Disp 1/0\nDisp 3", 200, "Disp 1/0", id="line200"),
    pytest.param(_WRAPPED_LINES + "Disp ln(0)\n" + _SHORT_LINES, 160, "Disp ln(0)",
                 id="line160-wrapped"),
])
def test_error_location(emu_env, tmp_path, src, line_no, line):
    [spec] = inputs.from_bodies([("TERR", handmade.Case("TERR", src, "").body())], tmp_path)
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state, CFG)
    assert r["outcome"] == "ERR"
    assert (r["error_line_no"], r["error_line"]) == (line_no, line), r
    assert r["lcd_text"][0].strip().startswith("ERROR:")   # screen captured before Goto


def test_input_text_is_configurable(emu_env, tmp_path):
    [spec] = inputs.from_bodies([("INPUT", handmade.case("INPUT").body())], tmp_path)
    r = driver.run_one(emu_env.emu, spec, emu_env.clean_state,
                       driver.Config(budget_ms=6000, input_text="⁻2.5"))
    assert r["label"] == "OK" and any(line.strip() == "⁻5" for line in r["home_text"]), r
