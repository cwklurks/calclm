"""model/topics.py: CalcLM v2.5's topic menu (PHASE3-topic-menu.md): the ten candidates, the
fmix32 seed mix, the prompted reference at the deployed settings, the check lines and the
pre-stated menu rule."""
import json
import math
from pathlib import Path

import pytest

from lab import tasks_v2 as V
from model import topics as TP

ROOT = Path(__file__).resolve().parents[2]
VOCAB = json.loads((ROOT / "corpus" / "vocab.json").read_text())
L4 = ROOT / "runs" / "e10" / "export" / "L4-s23"
L4_FIT = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"

PLAN = {"CYL": "geo_cylinder_volume", "CIRC": "geo_circle_area", "SPEED": "phy_average_speed",
        "OHM": "phy_ohm_voltage", "C2F": "conv_c_to_f", "F2C": "conv_f_to_c",
        "HYP": "geo_hypotenuse", "HEAT": "ho_chem_heat", "QUAD": "alg_quadratic_root",
        "SPH": "geo_sphere_volume"}


# ---- the candidates --------------------------------------------------------------------------

def test_the_ten_candidates_are_the_plans():
    assert [t.key for t in TP.CANDIDATES] == list(PLAN)
    assert {t.key: t.task for t in TP.CANDIDATES} == PLAN
    assert "phy_force_x" not in {t.task for t in TP.CANDIDATES}
    assert len({t.label for t in TP.CANDIDATES}) == 10
    assert all(len(t.label) <= 20 for t in TP.CANDIDATES)


@pytest.mark.parametrize("key", list(PLAN))
def test_each_prompt_is_its_tasks_opening(key):
    assert TP.prompt(key) == V.TASKS_V2[PLAN[key]].opening_ids()
    assert TP.prompt(key)[0] != 1                              # no <bos>: the decoder adds it


@pytest.mark.parametrize("key", list(PLAN))
def test_no_opening_holds_a_token_the_level_2_masks_refuse(key):
    """The prompt is fed unmasked, so every opening token must be one level 2 would allow
    where it stands (no setting command such as Degree, no credit line, no call)."""
    from model.intmask import IntMask
    m = IntMask(VOCAB, level=2)
    m.reset()
    m.step(1)
    for tok in TP.prompt(key):
        assert not m.banned()[tok], (key, VOCAB[tok])
        m.step(tok)


# ---- the seed mix ----------------------------------------------------------------------------

def test_mix_seed_is_murmurhash3s_fmix32():
    """Published MurmurHash3_x86_32 vectors for the empty input are fmix32(seed)."""
    assert TP.mix_seed(0) == 0
    assert TP.mix_seed(1) == 0x514E28B7
    assert TP.mix_seed(0xFFFFFFFF) == 0x81F16F39
    out = [TP.mix_seed(s) for s in range(1, 5001)]
    assert len(set(out)) == 5000 and all(0 < x < 1 << 32 for x in out)


def test_mix_seed_refuses_values_outside_32_bits():
    for bad in (-1, 1 << 32):
        with pytest.raises(ValueError):
            TP.mix_seed(bad)


# ---- the reference ---------------------------------------------------------------------------

def test_generate_topic_is_generate_prompted_at_the_deployed_settings(monkeypatch):
    from lab import e14
    from model.linker import ARMS
    seen = []
    monkeypatch.setattr(e14, "generate_prompted",
                        lambda *a, **kw: seen.append((a, kw)) or {"ids": [1, 2], "stats": {}})
    out = TP.generate_topic("MODEL", [7] * 16, "CYL", 20)
    assert out == {"ids": [1, 2], "stats": {}}
    assert seen == [(("MODEL", ARMS["LD2"], TP.mix_seed(20), [7] * 16, TP.prompt("CYL")),
                     {"cap": 256, "mask_level": 2})]
    assert (TP.CAP, TP.MASK_LEVEL, TP.TEMPERATURE_Q8, TP.TOP_K) == (256, 2, 205, 20)
    TP.generate_topic("MODEL", [7] * 16, "SPH", 3, cap=1024, seed_mix="none")
    assert seen[-1] == (("MODEL", ARMS["LD2"], 3, [7] * 16, TP.prompt("SPH")),
                        {"cap": 1024, "mask_level": 2})
    with pytest.raises(KeyError):
        TP.generate_topic("MODEL", [7] * 16, "NOPE", 3)


@pytest.mark.skipif(not (L4 / "model.bin").exists(), reason="needs the L4-s23 export")
def test_generate_topic_on_l4_s23_matches_generate_prompted():
    from lab.e14 import generate_prompted
    from model.intref import IntModel
    from model.linker import ARMS
    m = IntModel.load(L4, horizons=True)
    E = json.loads(L4_FIT.read_text())["E"]
    got = TP.generate_topic(m, E, "CIRC", 600001)
    want = generate_prompted(m, ARMS["LD2"], TP.mix_seed(600001), E, TP.prompt("CIRC"),
                             cap=256, mask_level=2)
    assert got == want and len(got["ids"]) <= 256
    assert got["ids"][:1 + len(TP.prompt("CIRC"))] == [1, *TP.prompt("CIRC")]


# ---- check lines -----------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(PLAN))
def test_check_line_value_is_the_tasks_reference(key):
    t = TP.topic(key)
    task = V.TASKS_V2[t.task]
    assert len(t.check) == len(task.vars)
    assert t.check not in task.inputs                           # not one of the suite's tests
    want = task.ref(*map(float, t.check))
    assert TP.check_value(key) == want
    line = TP.check_line(key)
    assert len(line) <= 40 and line.startswith("Check: ")
    shown = line.rsplit(" ", 1)[1]
    assert math.isclose(float(shown), want, abs_tol=0.005 + 1e-12)
    for var, val in zip(task.vars, t.check):
        assert f"{var}={val}" in line


def test_check_line_example_from_the_plan():
    assert TP.check_line("CYL") == "Check: R=2, H=3 should show 37.70"


# ---- the menu rule (section 1b, fixed before the measurement) --------------------------------

def _fixture(passes: dict, k=100, tokens=100.0):
    return {"tok_per_s": 0.674,
            "topics": {key: {"task": PLAN[key], "k": k, "passes": p, "mean_tokens": tokens}
                       for key, p in passes.items()},
            "random": {"mean_tokens": 749.0}}


def test_menu_rule_main_rarely_dropped_and_random_last():
    fx = _fixture({"CYL": 80, "CIRC": 45, "SPEED": 10, "OHM": 31, "C2F": 9, "F2C": 3,
                   "HYP": 7, "HEAT": 2, "QUAD": 0, "SPH": 5})
    menu = TP.menu_entries(fx)
    assert [e["key"] for e in menu] == ["CYL", "CIRC", "OHM", "SPEED", "C2F", "HYP", "RANDOM"]
    assert [e["rate"] for e in menu] == ["8 in 10", "5 in 10", "3 in 10", "1 in 10", "rarely",
                                         "rarely", "rarely"]
    assert menu[-1]["label"] == "Random program"


def test_menu_rule_ties_and_rounding():
    fx = _fixture({"CYL": 14, "CIRC": 14, "SPEED": 11, "OHM": 96})
    menu = TP.menu_entries(fx)
    assert [e["key"] for e in menu] == ["OHM", "CYL", "CIRC", "SPEED", "RANDOM"]   # ties: plan order
    assert [e["rate"] for e in menu][:4] == ["10 in 10", "1 in 10", "1 in 10", "1 in 10"]


def test_menu_stop_rule_needs_three_main_entries():
    fx = _fixture({"CYL": 80, "CIRC": 45, "SPEED": 9, "OHM": 9})
    assert TP.main_entries(fx) == ["CYL", "CIRC"]
    with pytest.raises(TP.StopForAuthor):
        TP.menu_entries(fx)


def test_menu_time_is_minutes_at_the_measured_speed():
    fx = _fixture({"CYL": 80, "CIRC": 45, "OHM": 31}, tokens=80.0)
    fx["topics"]["OHM"]["mean_tokens"] = 190.0
    menu = {e["key"]: e for e in TP.menu_entries(fx)}
    assert menu["CYL"]["time"] == "2 min"          # 80 / 0.674 / 60 = 1.98
    assert menu["OHM"]["time"] == "5 min"          # 4.70
    assert menu["RANDOM"]["time"] == "19 min"      # 749 / 0.674 / 60 = 18.5
    fx["topics"]["CYL"]["mean_tokens"] = 10.0
    assert TP.menu_entries(fx)[0]["time"] == "1 min"


def test_wilson_interval():
    lo, hi = TP.wilson(40, 50)
    assert round(100 * lo) == 67 and round(100 * hi) == 89      # the plan's table, CYL
    assert TP.wilson(0, 100)[0] == 0.0
