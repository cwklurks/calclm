"""Key probes for the game tier (Config.key_probe, opt-in): the harness feeds a scripted key
sequence (GAME_KEYS: the four arrows, then Clear) to getKey loops and records, for every fed key,
whether the screen changed between just before the press and probe_ms after it. Off by default;
off, results are what they were."""
import pytest

from eval import driver, inputs
from eval.handmade import Case

GAME = driver.Config(budget_ms=8000, feed_keys=driver.GAME_KEYS, key_probe=True)
PLAIN = driver.Config(budget_ms=8000)

MOVER = Case("GMOVE", "ClrHome\n0→X\nRepeat K=45\ngetKey→K\nIf K=26:X+1→X\nOutput(1,1,X)\nEnd",
             "OK")
ANYKEY = Case("GANY", "ClrHome\nDisp \"HI\"\nRepeat getKey\nEnd\nDisp 1", "OK")
STILL = Case("GSTILL", "ClrHome\nDisp \"WAIT\"\nRepeat K=45\ngetKey→K\nEnd", "OK")


def run(emu_env, tmp_path, case, cfg):
    [spec] = inputs.from_bodies([(case.name, case.body())], tmp_path / f"{case.name}{id(cfg)}")
    return driver.run_one(emu_env.emu, spec, emu_env.clean_state, cfg)


def test_the_game_keys_are_the_arrows_then_clear():
    assert driver.GAME_KEYS == ("Right", "Down", "Left", "Up", "Clear")
    assert driver.Config().key_probe is False
    assert driver.Config().feed_keys == ("Enter", "Right", "Down", "2nd", "Left", "Up", "Clear")


def test_a_program_that_moves_on_an_arrow_and_quits_on_clear(emu_env, tmp_path):
    r = run(emu_env, tmp_path, MOVER, GAME)
    assert r["outcome"] == "OK", r
    probes = r["key_probes"]
    assert [p[0] for p in probes] == list(driver.GAME_KEYS)
    assert probes[0][1] is True                      # Right: X went 0 -> 1 on screen
    assert [p[1] for p in probes[1:4]] == [False, False, False]
    assert r["last_action"] == "getkey"


def test_a_program_that_ends_on_any_key(emu_env, tmp_path):
    r = run(emu_env, tmp_path, ANYKEY, GAME)
    assert r["outcome"] == "OK" and [p[0] for p in r["key_probes"]] == ["Right"], r


def test_a_program_that_quits_on_clear_without_reacting(emu_env, tmp_path):
    r = run(emu_env, tmp_path, STILL, GAME)
    assert r["outcome"] == "OK", r
    assert [p[0] for p in r["key_probes"]] == list(driver.GAME_KEYS)
    assert not any(p[1] for p in r["key_probes"][:4])


@pytest.mark.parametrize("case", [MOVER, ANYKEY, STILL], ids=lambda c: c.name)
def test_probes_off_record_nothing_new(emu_env, tmp_path, case):
    r = run(emu_env, tmp_path, case, PLAIN)
    assert "key_probes" not in r


def test_run_token_ids_turns_key_probes_on_for_games(monkeypatch, tmp_path):
    from eval import run_programs
    seen = {}
    monkeypatch.setattr(run_programs.inputs, "from_token_ids", lambda *a, **k: [])
    monkeypatch.setattr(run_programs, "run_specs", lambda specs, out, cfg, **kw: seen.setdefault(
        "cfg", cfg) and [])
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000, game=True)
    assert seen["cfg"].key_probe and seen["cfg"].feed_keys == driver.GAME_KEYS
    assert seen["cfg"].budget_ms == 5000
    seen.clear()
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000)
    assert seen["cfg"] == driver.Config(budget_ms=5000)


def test_lab_score_passes_game_through(monkeypatch, tmp_path):
    from lab import score
    seen = {}

    def fake(programs, vocab, out, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr("eval.run_programs.run_token_ids", fake)
    score.run_emulator([[1, 5, 2]], tmp_path, game=True, yields=True)
    assert seen["game"] is True and seen["yields"] is True
