"""Yield capture (PLAN-v2.1.md, Judging): opt-in, records the home screen at every Input, Pause,
Menu and getKey wait, and changes nothing else in the result record."""
import json

import pytest

from eval import driver, handmade, inputs, run_programs
from eval.handmade import Case

CFG = driver.Config(budget_ms=6000)

YIELD_CASES = (
    Case("YPAUSE", "ClrHome\nDisp \"FIRST\"\nPause \nClrHome\nDisp \"LAST\"", "OK",
         screen_has="LAST"),
    Case("YINPUT", "ClrHome\nDisp \"ASK\"\nInput \"N=\",A\nClrHome\nDisp A+1", "OK",
         screen_has="2"),
    Case("YGETKEY", "ClrHome\nDisp \"KEYS\"\nRepeat K\ngetKey→K\nEnd\nClrHome\nDisp 5", "OK",
         screen_has="5"),
    Case("YMENU", "Menu(\"PICK\",\"GO\",A,\"QUIT\",Q)\nLbl A\nClrHome\nDisp \"DONE\"\nLbl Q", "OK",
         screen_has="DONE"),
)
# outcome classes and input paths from the hand-made set, for the invariance check
INVARIANT = (*YIELD_CASES, *map(handmade.case, (
    "OKDISP", "ESYNTAX", "ELABEL", "TLOOP", "INPUT", "PAUSE", "MENU", "GETKEY", "TINPUT", "TMENU",
    "GRPAUSE", "GREND")))


def run(emu_env, tmp_path, case, yields, cfg=CFG):
    [spec] = inputs.from_bodies([(case.name, case.body())], tmp_path / ("on" if yields else "off"))
    if yields:
        return driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg, yields=True)
    return driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg)


def shown(y, text):
    return any(text in line for line in y[3])


# ---- without the emulator ------------------------------------------------------------------

class FakeEmu:
    """Just enough of tools.cemu.Emu for _Run.answer: a clock, a key queue and a home screen."""

    def __init__(self, screen=None):
        self.calls, self.screen = [], screen

    def run(self, ms):
        self.calls.append(("run", ms))

    def inject_key(self, key):
        self.calls.append(("key", key))

    def home_text(self):
        self.calls.append(("home_text",))
        if self.screen is None:
            raise RuntimeError("no screen")
        return self.screen


def fake_run(emu, yields):
    spec = inputs.ProgramSpec(0, "P", "P", "test", Case("P", "Pause ", "OK").body(), None)
    return driver._Run(emu, spec, CFG, yields=yields) if yields else driver._Run(emu, spec, CFG)


def test_snapshot_is_taken_before_the_answer():
    emu = FakeEmu(["HELLO           ", "                ", "  X=?           "])
    run = fake_run(emu, yields=True)
    run.graph = True
    run.answer("pause")
    assert emu.calls == [("home_text",), ("key", "Enter")]
    assert run.yields == [["pause", 0, True, ["HELLO           ", "  X=?           "]]]


def test_failed_screen_read_records_no_lines_and_still_answers():
    emu = FakeEmu(screen=None)
    run = fake_run(emu, yields=True)
    run.answer("pause")
    assert run.yields == [["pause", 0, False, []]]
    assert ("key", "Enter") in emu.calls


def test_off_never_reads_the_screen():
    emu = FakeEmu(["HELLO"])
    run = fake_run(emu, yields=False)
    run.answer("pause")
    assert emu.calls == [("key", "Enter")]


def test_rejected_spec_gets_empty_yields_only_when_on(tmp_path):
    [spec] = inputs.from_bodies([("EMPTY", b"")], tmp_path)
    assert "yields" not in driver.run_one(None, spec, None, CFG)
    r = driver.run_one(None, spec, None, CFG, yields=True)
    assert r["outcome"] == "REJECTED" and r["yields"] == []


# ---- in the emulator -----------------------------------------------------------------------

def test_pause_yield_shows_the_screen_the_end_screen_lost(emu_env, tmp_path):
    r = run(emu_env, tmp_path, YIELD_CASES[0], yields=True)
    assert r["label"] == "OK" and not any("FIRST" in x for x in r["home_text"]), r
    [y] = r["yields"]
    assert y[0] == "pause" and shown(y, "FIRST") and not shown(y, "LAST") and y[2] is False
    assert 0 < y[1] < r["emu_ms"]


def test_input_yield_shows_the_prompt(emu_env, tmp_path):
    r = run(emu_env, tmp_path, YIELD_CASES[1], yields=True)
    assert r["label"] == "OK" and any("2" in x for x in r["home_text"]), r
    [y] = r["yields"]
    # while Input edits, the screen reader returns only the line being edited ("N=" + cursor),
    # not the "ASK" above it
    assert y[0] == "input" and shown(y, "N=") and not shown(y, "ASK")


def test_getkey_and_menu_waits_are_recorded(emu_env, tmp_path):
    r = run(emu_env, tmp_path, YIELD_CASES[2], yields=True)
    assert r["label"] == "OK" and r["yields"], r
    assert {y[0] for y in r["yields"]} == {"getkey"} and shown(r["yields"][0], "KEYS")
    assert len(r["yields"]) == r["actions"]["getkey"]
    r = run(emu_env, tmp_path, YIELD_CASES[3], yields=True)
    assert r["label"] == "OK" and [y[0] for y in r["yields"]] == ["menu"], r


def test_graph_pause_yield_carries_the_graph_flag(emu_env, tmp_path):
    r = run(emu_env, tmp_path, handmade.case("GRPAUSE"), yields=True)
    assert [y[0] for y in r["yields"]] == ["graph_wait"] and r["yields"][0][2] is True, r


def test_no_wait_no_yields(emu_env, tmp_path):
    r = run(emu_env, tmp_path, handmade.case("OKDISP"), yields=True)
    assert r["label"] == "OK" and r["yields"] == []


@pytest.mark.parametrize("case", INVARIANT, ids=lambda c: c.name)
def test_yields_add_a_field_and_change_nothing_else(emu_env, tmp_path, case):
    off = run(emu_env, tmp_path, case, yields=False)
    on = run(emu_env, tmp_path, case, yields=True)
    assert off["label"] == case.expect, off
    assert "yields" not in off and isinstance(on.pop("yields"), list)
    off.pop("wall_s"), on.pop("wall_s")
    assert on == off


def test_run_specs_threads_yields_to_worker_processes(emu_env, tmp_path):
    def specs(d):
        return inputs.from_bodies([(c.name, c.body()) for c in YIELD_CASES[:2]], tmp_path / d)

    on = run_programs.run_specs(specs("a"), tmp_path / "on", workers=2, cfg=CFG, shots=0,
                                yields=True)
    off = run_programs.run_specs(specs("b"), tmp_path / "off", workers=2, cfg=CFG, shots=0)
    assert [y[0] for y in on[0]["yields"]] == ["pause"]
    assert [y[0] for y in on[1]["yields"]] == ["input"]
    assert all("yields" not in r for r in off)
    rows = [json.loads(x) for x in (tmp_path / "on" / "results.jsonl").read_text().splitlines()]
    assert [r["yields"] for r in rows] == [r["yields"] for r in on]
    assert json.loads((tmp_path / "on" / "config.json").read_text())["yields"] is True
    assert "yields" not in json.loads((tmp_path / "off" / "config.json").read_text())


def test_run_token_ids_passes_yields_through(emu_env, tmp_path):
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE", "34", "32", "3F", "71"]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    res = run_programs.run_token_ids([[1, 4, 5, 6, 2]], tmp_path / "vocab.json", tmp_path / "run",
                                     workers=1, budget_ms=3000, yields=True)
    assert res[0]["label"] == "OK" and res[0]["yields"] == []
