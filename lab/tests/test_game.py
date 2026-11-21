"""lab.game: the game tier, from a program's tokens and its key-probe run record."""
import pytest

from lab import game as G
from lab.tests.test_function import COLON, NL, rec, tok, write_run
from model.tokens import EOS

GK, REPEAT, WHILE, FOR, END, IF, THEN, LBL, GOTO, STORE = (
    "AD", "D2", "D1", "D3", "D4", "CE", "CF", "D6", "D7", "04")


# ---- getKey in a loop (static) --------------------------------------------------------------

@pytest.mark.parametrize("ids, want", [
    (tok(REPEAT, "K", NL, GK, STORE, "K", NL, END), True),                 # in a Repeat body
    (tok(WHILE, "1", NL, GK, STORE, "K", NL, END), True),                  # in a While body
    (tok(REPEAT, GK, NL, END), True),                                      # in the loop's condition
    (tok(LBL, "A", NL, GK, STORE, "K", NL, GOTO, "A"), True),              # a Goto loop
    (tok(LBL, "A", NL, IF, "K", NL, GOTO, "A", NL, GK, STORE, "K"), False),  # getKey after the Goto
    (tok(GK, STORE, "K", NL, REPEAT, "K", NL, END), False),                # before the loop
    (tok(IF, "K", NL, THEN, NL, GK, STORE, "K", NL, END), False),          # If-Then is not a loop
    (tok(GK, STORE, "K"), False),
])
def test_getkey_in_a_loop(ids, want):
    assert G.getkey_in_loop(ids) is want


# ---- one run --------------------------------------------------------------------------------

LOOP = tok(REPEAT, "K", "=", "4", "5", NL, GK, STORE, "K", NL, END)


def probes(*keys_changed):
    return [[k, c, 100 * i] for i, (k, c) in enumerate(keys_changed)]


def test_a_game_reacts_to_an_arrow_and_quits_on_clear():
    r = rec("OK", key_probes=probes(("Right", True), ("Down", False), ("Left", False),
                                     ("Up", False), ("Clear", True)), last_action="getkey")
    c = G.classify(LOOP, r)
    assert c == {"getkey_loop": True, "reacts": True, "quits_on_clear": True, "any_key": False,
                 "interactive": True, "game": True}


def test_no_screen_change_on_any_arrow_is_not_a_game():
    r = rec("OK", key_probes=probes(("Right", False), ("Down", False), ("Left", False),
                                     ("Up", False), ("Clear", True)), last_action="getkey")
    assert G.classify(LOOP, r)["game"] is False and G.classify(LOOP, r)["quits_on_clear"]


def test_ending_on_the_first_key_is_any_key_not_a_quit():
    r = rec("OK", key_probes=probes(("Right", True)), last_action="getkey")
    c = G.classify(LOOP, r)
    assert c["any_key"] and not c["quits_on_clear"] and not c["game"]


@pytest.mark.parametrize("outcome", ["ERR", "TIMEOUT"])
def test_an_error_or_a_timeout_never_quits(outcome):
    r = rec(outcome, key_probes=probes(("Right", True), ("Down", False), ("Left", False),
                                        ("Up", False), ("Clear", True)), last_action="getkey")
    c = G.classify(LOOP, r)
    assert c["reacts"] and not c["quits_on_clear"] and not c["game"]


def test_a_static_loop_is_required():
    r = rec("OK", key_probes=probes(("Right", True), ("Down", False), ("Left", False),
                                     ("Up", False), ("Clear", True)), last_action="getkey")
    assert G.classify(tok(GK, STORE, "K"), r)["game"] is False


# ---- a whole run ----------------------------------------------------------------------------

def test_game_metrics_need_a_key_probe_run_and_count_over_programs_run(tmp_path):
    good = rec("OK", index=0, key_probes=probes(("Right", True), ("Down", False), ("Left", False),
                                                 ("Up", False), ("Clear", True)),
               last_action="getkey")
    plain = rec("OK", index=1, home=["5"])
    samples = [LOOP + [EOS], tok("DE", "5") + [EOS]]
    d = write_run(tmp_path / "a", [good, plain])
    with pytest.raises(ValueError):
        G.game_metrics(samples, d)                       # config.json has no key_probe
    import json
    cfg = json.loads((d / "config.json").read_text())
    cfg["config"] |= {"key_probe": True, "feed_keys": list(G.GAME_KEYS)}
    (d / "config.json").write_text(json.dumps(cfg))
    m = G.game_metrics(samples, d)
    assert m["n_run"] == 2 and m["game"] == 50.0 and m["getkey_loop"] == 50.0
    assert m["n_game"] == 1


@pytest.mark.parametrize("outcome, state, want", [("TIMEOUT", "getkey", True),
                                                   ("TIMEOUT", "input", False),
                                                   ("ERR", None, False), ("OK", None, True)])
def test_interactive_means_reacting_without_crashing(outcome, state, want):
    """Human val: about as many reacting getKey loops still answer keys at the budget as end."""
    r = rec(outcome, key_probes=probes(("Right", True), ("Down", False)), last_action="getkey",
            timeout_state=state)
    assert G.classify(LOOP, r)["interactive"] is want
