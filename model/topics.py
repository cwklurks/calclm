"""CalcLM v2.5's topic menu (PHASE3-topic-menu.md): the candidate topics, the seed mix and the
prompted reference at the settings the calculator uses for a topic.

A topic is a task of the suite (lab/tasks.py dev, lab/tasks_v2.py held-out); its prompt is the
task's opening_ids() (ClrHome, the Disp title, the Prompt of its inputs), fed after <bos> with
E14's rules (lab.e14.feed_prompt). A topic run is LD2 with ALiBi horizons, T 0.8, top-k 20,
mask level 2, a 256-token cap (opening included) and the seed mixed by fmix32 (section 2b: E14
drew its first hazard draw from the raw seed, which for small seeds almost always stopped).

The menu's numbers come from a measurement (section 3b) stored, numbers only, in
ce/tests/fixtures/topics_v2.6.json (v2.6-rc2: E24's L4c-s11 with its LD2 fit FIT; v2.5's, on
E10's L4-s23, is topics_v2.5.json); menu_entries applies section 1b's rule, fixed before the
measurement. ce/calclm/tools/gen_topics.py turns the result into ce/calclm/src/topics.h.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

CAP, MASK_LEVEL, TEMPERATURE_Q8, TOP_K = 256, 2, 205, 20
SEED_MIXES = ("none", "fmix32")
FIXTURE = Path(__file__).resolve().parents[1] / "ce" / "tests" / "fixtures" / "topics_v2.6.json"
# the LD2 fit the menu was measured with; topics.h binds the program to it and to the export
FIT = Path(__file__).resolve().parents[1] / "ce" / "tests" / "fixtures" / "ld2_offsets_L4c-s11.json"
MAIN_MIN, RARELY_MIN, RARELY_MAX = 0.10, 0.03, 2      # section 1b's rule
RANDOM_LABEL = "Random program"


class StopForAuthor(Exception):
    """Fewer than three topics reach 10%: the menu would be mostly long shots (section 1b)."""


@dataclass(frozen=True)
class Topic:
    key: str
    task: str                    # lab.tasks_v2.TASKS_V2 id
    label: str                   # menu text (at most 20 characters)
    check: tuple[str, ...]       # check inputs, in the task's Prompt order


# Candidates: tasks with at least 2 passes in 50 in rc2's prompted release review, except
# phy_force_x (its opening sets Degree), in the plan's order (section 1b).
CANDIDATES: tuple[Topic, ...] = (
    Topic("CYL", "geo_cylinder_volume", "Cylinder volume", ("2", "3")),
    Topic("CIRC", "geo_circle_area", "Circle area", ("2",)),
    Topic("SPEED", "phy_average_speed", "Average speed", ("90", "6")),
    Topic("OHM", "phy_ohm_voltage", "Ohm's law voltage", ("2", "6")),
    Topic("C2F", "conv_c_to_f", "Temperature C to F", ("10",)),
    Topic("F2C", "conv_f_to_c", "Temperature F to C", ("212",)),
    Topic("HYP", "geo_hypotenuse", "Hypotenuse", ("6", "8")),
    Topic("HEAT", "ho_chem_heat", "Heat energy Q=MCT", ("2", "3", "4")),
    Topic("QUAD", "alg_quadratic_root", "Quadratic root", ("1", "5", "6")),
    Topic("SPH", "geo_sphere_volume", "Sphere volume", ("3",)),
)
_BY_KEY = {t.key: t for t in CANDIDATES}


def topic(key: str) -> Topic:
    return _BY_KEY[key]


def task(key: str):
    from lab.tasks_v2 import TASKS_V2
    return TASKS_V2[topic(key).task]


def prompt(key: str) -> list[int]:
    """The opening's ids (no <bos>), exactly what the evaluation fed."""
    return task(key).opening_ids()


# ---- the seed mix ---------------------------------------------------------------------------

def mix_seed(seed: int) -> int:
    """MurmurHash3's 32-bit finalizer (fmix32): a bijection on 32 bits with fmix32(0) = 0.
    The C core's clm_mix_seed is the same function."""
    if not 0 <= seed < 1 << 32:
        raise ValueError("seed must be in [0, 2**32)")
    h = seed
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def seeded(seed: int, seed_mix: str) -> int:
    if seed_mix not in SEED_MIXES:
        raise ValueError(f"seed mix {seed_mix!r}: one of {SEED_MIXES}")
    return mix_seed(seed) if seed_mix == "fmix32" else seed


# ---- the reference --------------------------------------------------------------------------

def generate_topic(model, E, key: str, seed: int, cap: int = CAP,
                   seed_mix: str = "fmix32") -> dict:
    """What the calculator writes for topic `key` and the user's seed: lab.e14's prompted LD2
    at mask level 2 (model: an IntModel with horizons). Returns {"ids", "stats"}; ids are
    [<bos>, *opening, ..., (<eos>)]."""
    from lab import e14
    from model.linker import ARMS
    p = prompt(key)
    return e14.generate_prompted(model, ARMS["LD2"], seeded(seed, seed_mix), E, p, cap=cap,
                                 mask_level=MASK_LEVEL)


# ---- the check line -------------------------------------------------------------------------

def check_value(key: str) -> float:
    return float(task(key).ref(*map(float, topic(key).check)))


def _number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def check_line(key: str) -> str:
    """'Check: R=2, H=3 should show 37.70' (at most 40 columns: the status line)."""
    t = topic(key)
    args = ", ".join(f"{v}={x}" for v, x in zip(task(key).vars, t.check))
    return f"Check: {args} should show {_number(check_value(key))}"


# ---- the menu rule --------------------------------------------------------------------------

def wilson(passes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _rate(fx: dict, key: str) -> float:
    t = fx["topics"][key]
    return t["passes"] / t["k"]


def _minutes(tokens: float, tok_per_s: float) -> str:
    return f"{max(1, round(tokens / tok_per_s / 60))} min"


def _ranked(fx: dict) -> list[str]:
    order = [t.key for t in CANDIDATES if t.key in fx["topics"]]
    return sorted(order, key=lambda k: -_rate(fx, k))          # stable: ties keep plan order


def main_entries(fx: dict) -> list[str]:
    return [k for k in _ranked(fx) if _rate(fx, k) >= MAIN_MIN]


def menu_entries(fx: dict) -> list[dict]:
    """Section 1b's rule on a measurement: at least 10% a main entry ("N in 10", N =
    round(10 p), at least 1) in descending order of rate; 3% to under 10% "rarely", the two
    highest at most; under 3% dropped; "Random program" last. Raises StopForAuthor when fewer
    than three topics reach 10%."""
    main = main_entries(fx)
    if len(main) < 3:
        raise StopForAuthor(f"only {len(main)} topics reach 10%: {main}")
    rarely = [k for k in _ranked(fx) if RARELY_MIN <= _rate(fx, k) < MAIN_MIN][:RARELY_MAX]
    tps = fx["tok_per_s"]
    out = []
    for k in main + rarely:
        t = fx["topics"][k]
        tenths = (20 * t["passes"] + t["k"]) // (2 * t["k"])      # round(10 p), halves up
        rate = f"{max(1, tenths)} in 10" if k in main else "rarely"
        out.append({"key": k, "label": topic(k).label, "rate": rate,
                    "time": _minutes(fx["topics"][k]["mean_tokens"], tps)})
    out.append({"key": "RANDOM", "label": RANDOM_LABEL, "rate": "rarely",
                "time": _minutes(fx["random"]["mean_tokens"], tps)})
    return out


def load_fixture(path: Path = FIXTURE) -> dict:
    fx = json.loads(Path(path).read_text())
    for key in ("tok_per_s", "topics", "random", "export_sha256", "cap", "mask_level",
                "seed_mix"):
        if key not in fx:
            raise ValueError(f"{path}: no {key!r}")
    if (fx["cap"], fx["mask_level"], fx["seed_mix"]) != (CAP, MASK_LEVEL, "fmix32"):
        raise ValueError(f"{path}: not measured at the topic settings")
    for key, t in fx["topics"].items():
        if topic(key).task != t["task"]:
            raise ValueError(f"{path}: topic {key} is task {t['task']}")
    return fx
