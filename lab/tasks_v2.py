"""Task suite v2 (docs/agents/tasks/F2-task-suite-v2.md): v1's 40 tasks (lab/tasks.py, unchanged)
judged with a hardened harness, plus 20 held-out tasks.

HELD-OUT RULE: the held-out tasks (HELDOUT_IDS, ids prefixed ho_) are for measurement only.
They must never be used for training data, synthetic generation prompts or RL rewards, in any
form (their titles, openings, formulas, gold completions, mutants or answers).

The hardened harness (eval.driver.Config.decoy and .prefill), per task and input tuple:
- decoy: an Input/Prompt wait past the task's variables is answered with a decoy value, not
  the tuple's last value again. It differs (beyond the tolerance) from every input of its
  tuple, from twice and half of each, and from every reference of the task.
- prefill: before the run, the real variables A-Z and θ hold distinct values with two
  decimals in 1.5-9.5, apart from the tuple's inputs, its decoy and the task's references; the
  task's own variables are then overwritten by its prompts. A program that relies on a
  variable being 0 in a fresh emulator fails.
Both are deterministic, seeded from (CONTENT_SHA256, task id, tuple index); records carry
them (`decoy`, `prefill`), and judge2 refuses records run under another harness.

Judging: lab.tasks.judge, unchanged, plus `labelled`: a passing tuple whose matching number
has one of the task's LABELS keywords on its line or on the line above (not a prompt's echo).
Metrics (task_metrics2): v1's task_metrics, unchanged, with the v2 suite sha, plus
task_correct_dev / _heldout / _labelled and tasks_solved_dev / _heldout / _labelled.

    python -m lab.tasks build2 [--out DIR] [--workers N]      # validate v2 in CEmu
    python -m lab.tasks score2 DIR [--out OUT] [--workers N]  # v1 samples: dev tasks -> DIR/v2/
    python -m lab.tasks sample2 --export E --k 25 --seed0 1 --tasks heldout --out DIR

Openings, gold completions, mutants and loophole mutants here are hand-written for this suite.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from lab import tasks as T
from lab.tasks import Task, matches, value

VARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("θ",)
FILL_LO, FILL_HI = 1.5, 9.5
V1_SUITE_SHA256 = T.suite_sha256(T.SUITE)

Runner2 = Callable[[list, "Harness", Path], list]   # (programs, harness, out dir) -> records


def _comb(n, k):
    return math.comb(int(n), int(k))


def _digit_product(n):
    return math.prod(int(d) for d in str(int(n)))


HELDOUT: tuple[Task, ...] = (
    # ---- statistics
    Task("ho_stat_geo_mean", "statistics", ("GEOMETRIC MEAN", "OF A AND B"), ("A", "B"),
         (("4", "9"), ("1.5", "13.5"), ("12", "30")), lambda a, b: math.sqrt(a * b),
         "Disp √(AB)", ("Disp (A+B)/2", "Disp √(A+B)", "Disp AB/2")),
    Task("ho_stat_harmonic_mean", "statistics", ("HARMONIC MEAN", "OF A AND B"), ("A", "B"),
         (("3", "6"), ("1.5", "4.5"), ("12", "20")), lambda a, b: 2 * a * b / (a + b),
         "Disp 2AB/(A+B)", ("Disp AB/(A+B)", "Disp (A+B)/2", "Disp 2/(A+B)")),
    Task("ho_stat_midrange3", "statistics", ("MIDRANGE OF 3 NUMBERS", "(LARGEST+SMALLEST)/2"),
         ("A", "B", "C"), (("4", "8", "15"), ("1.5", "6", "12.5"), ("30", "45", "11")),
         lambda a, b, c: (max(a, b, c) + min(a, b, c)) / 2,
         "Disp (max(A,max(B,C))+min(A,min(B,C)))/2",
         ("Disp (A+B+C)/3", "Disp max(A,max(B,C))-min(A,min(B,C))",
          "Disp (max(A,B)+min(A,B))/2")),
    Task("ho_stat_binomial", "statistics", ("BINOMIAL PROBABILITY", "P(X=K) IN N TRIALS",
                                            "SUCCESS PROBABILITY P"), ("N", "K", "P"),
         (("12", "3", "0.25"), ("10", "4", "0.5"), ("15", "5", "0.35")),
         lambda n, k, p: _comb(n, k) * p ** k * (1 - p) ** (n - k),
         "Disp (N nCr K)P^K(1-P)^(N-K)",
         ("Disp P^K(1-P)^(N-K)", "Disp (N nCr K)P^K", "Disp (N nCr K)P^(N-K)(1-P)^K")),
    Task("ho_stat_margin_error", "statistics", ("MARGIN OF ERROR, 95%", "SAMPLE PROPORTION P",
                                                "SAMPLE SIZE N", "USE Z=1.96"), ("P", "N"),
         (("0.5", "400"), ("0.3", "150"), ("0.85", "1000")),
         lambda p, n: 1.96 * math.sqrt(p * (1 - p) / n),
         "Disp 1.96√(P(1-P)/N)", ("Disp √(P(1-P)/N)", "Disp 1.96P(1-P)/N", "Disp 1.96√(P/N)")),
    # ---- number theory
    Task("ho_nt_digital_root", "number theory", ("DIGITAL ROOT OF N", "(REPEATED SUM OF DIGITS)"),
         ("N",), (("4599",), ("98765",), ("1234",)), lambda n: 1 + (n - 1) % 9,
         "Disp 1+remainder(N-1,9)",
         ("Disp remainder(N,9)", "Disp remainder(N-1,9)",
          "0→S\nWhile N>0\nS+remainder(N,10)→S\nint(N/10)→N\nEnd\nDisp S"),
         integer_inputs=True),
    Task("ho_nt_sum_cubes", "number theory", ("SUM OF CUBES", "1³+2³+...+N³"), ("N",),
         (("5",), ("10",), ("24",)), lambda n: (n * (n + 1) / 2) ** 2,
         "Disp (N(N+1)/2)²", ("Disp N(N+1)/2", "Disp N³", "Disp N²(N+1)/2"),
         integer_inputs=True),
    Task("ho_nt_sum_range", "number theory", ("SUM OF THE WHOLE NUMBERS", "FROM A TO B"),
         ("A", "B"), (("3", "10"), ("15", "40"), ("1", "100")),
         lambda a, b: (b - a + 1) * (a + b) / 2,
         "Disp (B-A+1)(A+B)/2", ("Disp (B-A)(A+B)/2", "Disp (A+B)/2", "Disp B(B+1)/2"),
         integer_inputs=True),
    Task("ho_nt_num_digits", "number theory", ("NUMBER OF DIGITS", "OF A WHOLE NUMBER N"), ("N",),
         (("4567",), ("89",), ("123456",)), lambda n: len(str(int(n))),
         "Disp int(log(N))+1", ("Disp log(N)", "Disp int(log(N))", "Disp int(ln(N))+1"),
         integer_inputs=True),
    Task("ho_nt_digit_product", "number theory", ("PRODUCT OF THE DIGITS", "OF A WHOLE NUMBER N"),
         ("N",), (("1234",), ("98765",), ("3456",)), _digit_product,
         "1→P\nWhile N>0\nPremainder(N,10)→P\nint(N/10)→N\nEnd\nDisp P",
         ("0→S\nWhile N>0\nS+remainder(N,10)→S\nint(N/10)→N\nEnd\nDisp S",
          "Disp remainder(N,10)", "Disp N/10"), integer_inputs=True),
    # ---- chemistry
    Task("ho_chem_ph", "chemistry", ("PH OF A STRONG ACID", "MOLES OF H+ IONS N",
                                     "VOLUME V IN ML"), ("N", "V"),
         (("0.0035", "250"), ("0.25", "1500"), ("0.00045", "120")),
         lambda n, v: -math.log10(1000 * n / v),
         "Disp ⁻log(1000N/V)", ("Disp ⁻log(N/V)", "Disp log(1000N/V)", "Disp ⁻ln(1000N/V)")),
    Task("ho_chem_combined_gas", "chemistry", ("COMBINED GAS LAW", "START PRESSURE P, VOLUME V",
                                               "START TEMP T IN K", "END PRESSURE Q, TEMP U (K)",
                                               "END VOLUME?"), ("P", "V", "T", "Q", "U"),
         (("1", "12", "300", "1.5", "350"), ("0.8", "4.5", "273", "1.2", "310"),
          ("3", "25", "400", "1", "250")),
         lambda p, v, t, q, u: p * v * u / (q * t),
         "Disp VPU/(QT)", ("Disp VPT/(QU)", "Disp VP/Q", "Disp QVU/(PT)")),
    Task("ho_chem_heat", "chemistry", ("HEAT ENERGY Q (J)", "MASS M IN G",
                                       "SPECIFIC HEAT C IN J/G°C", "TEMP CHANGE T IN °C"),
         ("M", "C", "T"), (("50", "4.18", "15"), ("12.5", "0.9", "40"), ("200", "0.385", "6")),
         lambda m, c, t: m * c * t,
         "Disp MCT", ("Disp MC", "Disp MT/C", "Disp M+C+T")),
    Task("ho_chem_half_life", "chemistry", ("RADIOACTIVE DECAY", "AMOUNT LEFT AFTER TIME T",
                                            "START AMOUNT A", "HALF-LIFE H (SAME UNITS)"),
         ("A", "H", "T"), (("80", "5", "12"), ("100", "5730", "3000"), ("12.5", "8", "20")),
         lambda a, h, t: a * 0.5 ** (t / h),
         "Disp A(1/2)^(T/H)", ("Disp A(1/2)^(H/T)", "Disp A/2", "Disp A(1/2)T/H")),
    # ---- finance
    Task("ho_fin_continuous", "finance", ("COMPOUND INTEREST", "COMPOUNDED CONTINUOUSLY",
                                          "FINAL AMOUNT FROM P,", "R % PER YEAR, T YEARS"),
         ("P", "R", "T"), (("1500", "3.5", "4"), ("400", "6", "2.5"), ("75", "12", "10")),
         lambda p, r, t: p * math.exp(r * t / 100),
         "Disp Pe^^(RT/100)", ("Disp Pe^^(RT)", "Disp P(1+R/100)^T", "Disp e^^(RT/100)")),
    Task("ho_fin_loan_payment", "finance", ("MONTHLY LOAN PAYMENT", "LOAN AMOUNT P", "R % PER YEAR",
                                            "T YEARS, PAID MONTHLY"), ("P", "R", "T"),
         (("10000", "6", "5"), ("2500", "4.5", "3"), ("180000", "3.5", "30")),
         lambda p, r, t: p * (r / 1200) / (1 - (1 + r / 1200) ** (-12 * t)),
         "Disp P(R/1200)/(1-(1+R/1200)^(⁻12T))",
         ("Disp P(R/100)/(1-(1+R/100)^(⁻T))", "Disp P/(12T)",
          "Disp P(R/1200)/(1-(1+R/1200)^(12T))")),
    # ---- algebra
    Task("ho_alg_y_intercept", "algebra", ("Y-INTERCEPT OF THE LINE", "THROUGH (A,B), (C,D)"),
         ("A", "B", "C", "D"),
         (("1", "4.5", "5", "12.5"), ("0.5", "3", "4.5", "13"), ("12", "8", "16", "11")),
         lambda a, b, c, d: b - a * (d - b) / (c - a),
         "Disp B-A(D-B)/(C-A)", ("Disp B+A(D-B)/(C-A)", "Disp (D-B)/(C-A)", "Disp B-A(C-A)/(D-B)")),
    Task("ho_alg_geo_term", "algebra", ("GEOMETRIC SEQUENCE", "FIRST TERM A", "COMMON RATIO R",
                                        "TERM NUMBER N"), ("A", "R", "N"),
         (("3", "1.5", "12"), ("100", "0.8", "15"), ("1.5", "3", "10")),
         lambda a, r, n: a * r ** (n - 1),
         "Disp AR^(N-1)", ("Disp AR^N", "Disp (AR)^(N-1)", "Disp A+R(N-1)")),
    # ---- physics
    Task("ho_phy_pendulum", "physics", ("PENDULUM PERIOD (S)", "LENGTH L IN M", "USE G=9.8 M/S²"),
         ("L",), (("1.5",), ("0.25",), ("12",)), lambda x: 2 * math.pi * math.sqrt(x / 9.8),
         "Disp 2pi√(L/9.8)", ("Disp pi√(L/9.8)", "Disp 2pi√(9.8/L)", "Disp 2piL/9.8")),
    Task("ho_phy_projectile_range", "physics", ("PROJECTILE RANGE (M)", "SPEED V IN M/S",
                                                "LAUNCH ANGLE A IN DEGREES",
                                                "LEVEL GROUND, G=9.8 M/S²"), ("V", "A"),
         (("20", "45"), ("12.5", "30"), ("35", "15")),
         lambda v, a: v * v * math.sin(math.radians(2 * a)) / 9.8,
         "Disp V²sin(2A)/9.8", ("Disp V²sin(A)/9.8", "Disp Vsin(2A)/9.8", "Disp V²cos(2A)/9.8"),
         degree=True),
)
HELDOUT_IDS: tuple[str, ...] = (
    "ho_stat_geo_mean", "ho_stat_harmonic_mean", "ho_stat_midrange3", "ho_stat_binomial",
    "ho_stat_margin_error", "ho_nt_digital_root", "ho_nt_sum_cubes", "ho_nt_sum_range",
    "ho_nt_num_digits", "ho_nt_digit_product", "ho_chem_ph", "ho_chem_combined_gas",
    "ho_chem_heat", "ho_chem_half_life", "ho_fin_continuous", "ho_fin_loan_payment",
    "ho_alg_y_intercept", "ho_alg_geo_term", "ho_phy_pendulum", "ho_phy_projectile_range")
assert HELDOUT_IDS == tuple(t.id for t in HELDOUT)

# Overlap with the synthetic set (data/synth/scale/kept.jsonl, checked read-only on
# 2026-10-19): "same key" if the gold's computation key (lab.synth.computation_key, variables
# canonical) occurs in a kept program's key; "title-similar" if a kept title names the same
# quantity (a loose keyword match); else "none". No held-out formula occurs there.
SYNTH_OVERLAP: dict[str, str] = {
    "ho_stat_geo_mean": "title-similar", "ho_stat_harmonic_mean": "title-similar",
    "ho_stat_midrange3": "none", "ho_stat_binomial": "title-similar",
    "ho_stat_margin_error": "title-similar", "ho_nt_digital_root": "none",
    "ho_nt_sum_cubes": "title-similar", "ho_nt_sum_range": "title-similar",
    "ho_nt_num_digits": "title-similar", "ho_nt_digit_product": "none",
    "ho_chem_ph": "title-similar", "ho_chem_combined_gas": "none",
    "ho_chem_heat": "title-similar", "ho_chem_half_life": "title-similar",
    "ho_fin_continuous": "title-similar", "ho_fin_loan_payment": "title-similar",
    "ho_alg_y_intercept": "title-similar", "ho_alg_geo_term": "title-similar",
    "ho_phy_pendulum": "none", "ho_phy_projectile_range": "title-similar",
}

SUITE_V2: tuple[Task, ...] = (*T.SUITE, *HELDOUT)
TASKS_V2: dict[str, Task] = {t.id: t for t in SUITE_V2}
DEV_IDS: tuple[str, ...] = tuple(t.id for t in T.SUITE)
SPLITS = {"dev": T.SUITE, "heldout": HELDOUT, "all": SUITE_V2}
CONTENT_SHA256 = T.suite_sha256(SUITE_V2)   # v1's hash over the 60 tasks: seeds the harness

# Words naming each task's quantity (task_correct_labelled); matched case-insensitively.
LABELS: dict[str, tuple[str, ...]] = {
    "geo_circle_area": ("AREA",), "geo_hypotenuse": ("HYP", "SIDE C", "C="),
    "geo_cylinder_volume": ("VOL",), "geo_sphere_volume": ("VOL",),
    "geo_triangle_sas": ("AREA",), "alg_slope": ("SLOPE", "M="),
    "alg_linear_root": ("X=", "ROOT", "SOL"), "alg_quadratic_root": ("ROOT", "X=", "SOL"),
    "alg_distance": ("DIST", "D="), "alg_vertex_x": ("VERT", "X="),
    "phy_kinetic_energy": ("KINETIC", "ENERGY", "KE", "E="),
    "phy_average_speed": ("SPEED", "VEL", "S="), "phy_ohm_voltage": ("VOLT", "V="),
    "phy_potential_energy": ("POTENTIAL", "ENERGY", "PE", "E="),
    "phy_force_x": ("FORCE", "FX", "COMP", "X-"),
    "fin_simple_interest": ("INTEREST", "I="),
    "fin_compound_yearly": ("AMOUNT", "TOTAL", "FINAL", "BAL", "A="),
    "fin_compound_monthly": ("AMOUNT", "TOTAL", "FINAL", "BAL", "A="),
    "fin_sale_price": ("PRICE", "SALE", "COST"), "fin_percent_change": ("CHANGE", "%"),
    "conv_c_to_f": ("FAHR", "°F", "F="), "conv_f_to_c": ("CELS", "°C", "C="),
    "conv_in_to_cm": ("CM", "CENTI"), "conv_ms_to_kmh": ("KM",), "conv_deg_to_rad": ("RAD",),
    "stat_mean3": ("MEAN", "AVERAGE", "AVG"), "stat_range3": ("RANGE",),
    "stat_zscore": ("Z=", "Z-", "SCORE"), "stat_pop_sd3": ("DEV", "SD", "STD"),
    "stat_weighted_mean": ("MEAN", "AVERAGE", "AVG"),
    "nt_gcd": ("GCD", "GCF", "DIVISOR", "FACTOR"), "nt_lcm": ("LCM", "MULTIPLE"),
    "nt_remainder": ("REMAINDER", "REM", "MOD"), "nt_digit_sum": ("SUM",),
    "nt_factorial": ("FACTORIAL", "!"), "chem_moles": ("MOL",),
    "chem_molarity": ("MOLAR", "MOL/L", "CONC", "M="),
    "chem_ideal_gas": ("PRESS", "ATM", "P="), "chem_dilution": ("CONC", "FINAL", "C="),
    "chem_percent_yield": ("YIELD", "%"),
    "ho_stat_geo_mean": ("GEOMETRIC", "MEAN"), "ho_stat_harmonic_mean": ("HARMONIC", "MEAN"),
    "ho_stat_midrange3": ("MIDRANGE", "MID"), "ho_stat_binomial": ("PROB", "P(", "P="),
    "ho_stat_margin_error": ("MARGIN", "ERROR", "E="),
    "ho_nt_digital_root": ("ROOT",), "ho_nt_sum_cubes": ("SUM",),
    "ho_nt_sum_range": ("SUM", "TOTAL"), "ho_nt_num_digits": ("DIGIT",),
    "ho_nt_digit_product": ("PRODUCT",), "ho_chem_ph": ("PH",),
    "ho_chem_combined_gas": ("VOL", "V2", "V="), "ho_chem_heat": ("HEAT", "ENERGY", "Q="),
    "ho_chem_half_life": ("LEFT", "REMAIN", "AMOUNT"),
    "ho_fin_continuous": ("AMOUNT", "TOTAL", "FINAL", "BAL", "A="),
    "ho_fin_loan_payment": ("PAYMENT", "MONTHLY", "PAY"),
    "ho_alg_y_intercept": ("INTERCEPT", "Y-INT", "B="), "ho_alg_geo_term": ("TERM", "AN="),
    "ho_phy_pendulum": ("PERIOD", "T="), "ho_phy_projectile_range": ("RANGE", "DIST", "R="),
}

# Loophole mutants (the metric's own tests, never samples): programs v1's harness passes by
# accident. "extra" reads a spare prompt that v1 answered with the tuple's last value again;
# "unset" relies on a variable it never sets being 0 in a fresh emulator.
LOOPHOLES: dict[str, tuple[tuple[str, str], ...]] = {
    "geo_circle_area": (("extra", "Prompt D\nDisp piRD"), ("unset", "Disp piR^2+Q")),
    "geo_cylinder_volume": (("extra", "Prompt K\nDisp piR^2K"),),
    "geo_hypotenuse": (("extra", "Prompt C\nDisp √(A²+BC)"), ("unset", "Disp √(A²+B²+C)")),
    "alg_linear_root": (("extra", "Prompt C\nDisp (⁻B-C)/(2A)"),
                        ("unset", "Disp (⁻B-√(B²-4AC))/(2A)")),
    "phy_ohm_voltage": (("extra", "Prompt S\nDisp IS"),),
    "phy_kinetic_energy": (("extra", "Prompt W\nDisp MVW/2"),),
    "fin_simple_interest": (("extra", "Prompt Y\nDisp PRY/100"),),
    "stat_mean3": (("extra", "Prompt D\nDisp (A+B+D)/3"), ("unset", "Disp (A+B+C+D)/3")),
    "nt_digit_sum": (("unset", "While N>0\nS+remainder(N,10)→S\nint(N/10)→N\nEnd\nDisp S"),),
    "conv_c_to_f": (("extra", "Prompt F\nDisp 9F/5+32"),),
    "ho_nt_sum_range": (("unset", "For(I,A,B)\nS+I→S\nEnd\nDisp S"),),
    "ho_stat_midrange3": (("extra", "Prompt D\nDisp (max(A,max(B,D))+min(A,min(B,D)))/2"),),
    "ho_fin_continuous": (("extra", "Prompt Y\nDisp Pe^^(RY/100)"),),
    "ho_chem_heat": (("extra", "Prompt D\nDisp MCD"),),
    "ho_phy_pendulum": (("extra", "Prompt G\nDisp 2pi√(G/9.8)"),
                        ("unset", "Disp 2pi√(L/9.8)+Z")),
}


# ---- the hardened harness ----------------------------------------------------------------------

@dataclass(frozen=True)
class Harness:
    values: tuple[str, ...]
    decoy: str
    prefill: tuple[tuple[str, str], ...]


def _apart(a: float, b: float) -> bool:
    return not matches(a, b) and not matches(b, a)


def _candidates(*key) -> Iterator[str]:
    """Deterministic values with two decimals (the second nonzero) in 1.5-9.5."""
    seed = hashlib.sha256(json.dumps([CONTENT_SHA256, *key]).encode()).digest()
    for n in itertools.count():
        block = hashlib.sha256(seed + n.to_bytes(8, "big")).digest()
        for i in range(0, 32, 4):
            c = 150 + int.from_bytes(block[i:i + 4], "big") % 801     # hundredths
            if c % 10:
                yield f"{c // 100}.{c % 100:02d}"


def _decoy_bad(task: Task, j: int) -> list[float]:
    ins = [value(v) for v in task.inputs[j]]
    return [*ins, *(2 * x for x in ins), *(x / 2 for x in ins), *task.expected()]


def decoy(task: Task, j: int) -> str:
    """The answer at tuple j's extra Input/Prompt waits."""
    bad = _decoy_bad(task, j)
    return next(c for c in _candidates("decoy", task.id, j)
                if all(_apart(value(c), b) for b in bad))


def prefill(task: Task, j: int) -> tuple[tuple[str, str], ...]:
    """(variable, value) for A-Z and θ before each run of tuple j."""
    bad = [*(value(v) for v in task.inputs[j]), value(decoy(task, j)), *task.expected()]
    out: list[str] = []
    for c in _candidates("prefill", task.id, j):
        if all(_apart(value(c), b) for b in bad):
            out.append(c)
            bad.append(value(c))
            if len(out) == len(VARS):
                return tuple(zip(VARS, out))
    raise AssertionError("unreachable")


@functools.cache
def harness(task: Task, j: int) -> Harness:
    return Harness(tuple(task.inputs[j]), decoy(task, j), prefill(task, j))


def check_harness(task: Task) -> None:
    """The builder's assertion of the decoy and prefill rules, tuple by tuple."""
    refs = task.expected()
    for j, tup in enumerate(task.inputs):
        ins = [value(v) for v in tup]
        d = value(decoy(task, j))
        if not all(_apart(d, b) for b in _decoy_bad(task, j)):
            raise ValueError(f"{task.id} t{j}: decoy {d} is near an input, twice or half an "
                             "input, or a reference")
        pf = prefill(task, j)
        names = [n for n, _ in pf]
        vals = [value(x) for _, x in pf]
        if not all(n in VARS for n in names) or len(set(names)) != len(names):
            raise ValueError(f"{task.id} t{j}: prefill must name distinct variables A-Z, θ")
        if not all(_apart(a, b) for i, a in enumerate(vals) for b in vals[i + 1:]):
            raise ValueError(f"{task.id} t{j}: prefill values must be distinct")
        if not all(FILL_LO <= v <= FILL_HI and len(x.split(".")[-1]) == 2 for v, (_, x)
                   in zip(vals, pf)):
            raise ValueError(f"{task.id} t{j}: prefill values need 2 decimals in 1.5-9.5")
        if not all(_apart(v, b) for v in vals for b in [*ins, d, *refs]):
            raise ValueError(f"{task.id} t{j}: prefill value near an input, the decoy or a "
                             "reference")
        if names != list(VARS):
            raise ValueError(f"{task.id} t{j}: prefill must cover A-Z and θ in order")


def suite_v2_sha256() -> str:
    """Everything that decides a v2 judgment: the 60 tasks, the split, every tuple's decoy and
    prefill, the label keywords and the loophole mutants."""
    blob = {"version": 2, "tasks": [t.spec() for t in SUITE_V2], "heldout": list(HELDOUT_IDS),
            "harness": {t.id: [[h.decoy, [list(p) for p in h.prefill]]
                               for h in (harness(t, j) for j in range(len(t.inputs)))]
                        for t in SUITE_V2},
            "labels": {k: list(v) for k, v in LABELS.items()},
            "loopholes": {k: [list(x) for x in v] for k, v in LOOPHOLES.items()}}
    raw = json.dumps(blob, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


SUITE_V2_SHA256 = suite_v2_sha256()
# Any change to what decides a v2 judgment changes SUITE_V2_SHA256; tests compare it with this.
PINNED_SUITE_V2_SHA256 = "e9b1dcfd2aac4aa2be61ff50287ea2a6ecddc1fcbbd490673a5bbf7618bb773e"


# ---- judging ------------------------------------------------------------------------------------

def screens(r: dict) -> list[list[str]]:
    """The screens lab.tasks.tuple_lines reads (final home screen, yield screens after the last
    input), kept apart and without blank lines."""
    ys = r.get("yields") or []
    last = max((i for i, y in enumerate(ys) if y[0] == "input"), default=-1)
    out = [list(r.get("home_text") or []), *(list(y[3]) for y in ys[last + 1:])]
    return [[x for x in s if x.strip()] for s in out]


def labelled(r: dict, ref: float, keywords) -> bool:
    """The reference is shown with a keyword on its line or the line above (a prompt's echo,
    "X=?..." , does not count)."""
    for lines in screens(r):
        for i, line in enumerate(lines):
            if not any(matches(x, ref) for x in T.numbers(line)):
                continue
            near = [x.upper() for x in (line, *lines[max(0, i - 1):i]) if "=?" not in x]
            if any(k in x for x in near for k in keywords):
                return True
    return False


_EXEMPT = ("HARNESS_ERROR", "REJECTED")   # records the driver returns before the harness ran


def judge2(records: list[dict] | None, task: Task) -> dict:
    """lab.tasks.judge (unchanged) plus labelled tuples and pass_labelled. Refuses records run
    under another harness than this task's v2 one."""
    refs = task.expected()
    j = T.judge(records, refs)
    if records is None:
        return {**j, "labelled": [False] * len(refs), "pass_labelled": False}
    for i, r in enumerate(records):
        if r["outcome"] in _EXEMPT:
            continue
        h = harness(task, i)
        if r.get("decoy") != h.decoy or r.get("prefill") != dict(h.prefill):
            raise ValueError(f"{task.id} t{i}: record run under another harness than v2's")
    lab = [t and labelled(r, ref, LABELS[task.id])
           for t, r, ref in zip(j["tuples"], records, refs)]
    clean = all(o in T.OK_OUTCOMES for o in j["outcomes"]) and not j["harness_error"]
    return {**j, "labelled": lab, "pass_labelled": all(lab) and clean}


def run_programs2(task: Task, programs: list[list[int]], runner: Runner2, out: Path) -> list:
    """lab.tasks.run_programs (the same line trim and empties) with each tuple's harness."""
    return submit_programs2(task, programs, runner, out)()


def submit_programs2(task: Task, programs: list[list[int]], runner: Runner2,
                     out: Path) -> Callable[[], list]:
    """lab.tasks.submit_programs with each tuple's harness (run_programs2 in two steps)."""
    def run(progs, values, d):
        return runner(progs, harness(task, task.inputs.index(values)), d)
    return T.submit_programs(task, programs, run, out)


def validate2(task: Task, runner: Runner2, out: Path) -> dict:
    """Gold passes; every mutant and loophole mutant fails (under the v2 harness)."""
    loops = [b for _, b in LOOPHOLES.get(task.id, ())]
    progs = [T.program_ids(task, x) for x in (task.gold, *task.mutants, *loops)]
    gold, *rest = (judge2(r, task) for r in run_programs2(task, progs, runner, out))
    n = len(task.mutants)
    return {"task": task.id, "gold": gold, "mutants": rest[:n], "loopholes": rest[n:],
            "ok": gold["pass"] and not any(m["pass"] for m in rest)}


def emulator_runner2(workers: int = 8, budget_ms: int = 30_000) -> Runner2:
    """lab.score.run_emulator with yield capture and the v2 harness."""
    from lab.score import run_emulator

    def run(programs, h: Harness, out):
        return run_emulator(programs, out, T.VOCAB, workers=min(workers, len(programs)),
                            budget_ms=budget_ms, yields=True, input_values=h.values,
                            decoy=h.decoy, prefill=h.prefill)
    return run


@contextlib.contextmanager
def pooled_runner2(workers: int = 8, budget_ms: int = 30_000):
    """emulator_runner2 on one pool of emulator workers kept for the whole block (see
    lab.tasks.pooled_runner): a call queues its batch and returns a handle."""
    from eval.run_programs import EmuPool

    with EmuPool(workers) as pool:
        yield lambda programs, h, out: T.pool_run(
            pool, programs, out, workers, budget_ms, input_values=h.values, decoy=h.decoy,
            prefill=h.prefill)


def build2(out: Path, runner: Runner2, tasks=SUITE_V2) -> dict:
    for t in tasks:
        T.check_static(t)
        T.check_echo(t)
        check_harness(t)
    res = [validate2(t, runner, Path(out) / "runs") for t in tasks]
    report = {"suite_sha256": SUITE_V2_SHA256, "n_tasks": len(tasks),
              "ok": all(r["ok"] for r in res), "tasks": res}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "build.json").write_text(json.dumps(report, indent=1))
    return report


# ---- metrics ------------------------------------------------------------------------------------

def task_metrics2(judged: list[dict], tasks=SUITE_V2) -> dict:
    """v1's task_metrics (unchanged values, the v2 suite sha), plus per-split and labelled
    rates; a split with no task here is None."""
    m = T.task_metrics(judged, tasks)
    m["suite_sha256"] = SUITE_V2_SHA256
    n = len(judged)
    by = {t.id: [r for r in judged if r["task"] == t.id] for t in tasks}

    def split(ids):
        tids = [t for t in by if t in ids]
        rows = [r for t in tids for r in by[t]]
        if not tids:
            return None, None
        return (T._rate(sum(r["pass"] for r in rows), len(rows)),
                sum(any(r["pass"] for r in by[t]) for t in tids))
    (dev, dev_solved), (ho, ho_solved) = split(set(DEV_IDS)), split(set(HELDOUT_IDS))
    m |= {"task_correct_dev": dev, "task_correct_heldout": ho,
          "task_correct_labelled": T._rate(sum(r["pass_labelled"] for r in judged), n),
          "tasks_solved_dev": dev_solved, "tasks_solved_heldout": ho_solved,
          "tasks_solved_labelled": sum(any(r["pass_labelled"] for r in rows)
                                       for rows in by.values()),
          "n_tasks_dev": sum(t in DEV_IDS for t in by),
          "n_tasks_heldout": sum(t in HELDOUT_IDS for t in by)}
    return m


# ---- sampling and scoring -----------------------------------------------------------------------

def _draw_one2(job: tuple[str, int]) -> list[int]:
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate

    task_id, seed = job
    cfg = SamplerConfig(round(T.TEMP * 256), T.TOP_K, False, seed)
    return [int(t) for t in generate(T._model, TASKS_V2[task_id].opening_ids(), T.CAP, cfg,
                                     T.CAP, IntMask(T._model.vocab))]


def draw2(export, jobs: list[tuple[str, int]], workers: int) -> list[list[int]]:
    """lab.tasks.draw for any v2 task (the same v2.1 decoding)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1:
        T._load(str(export))
        return [_draw_one2(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=T._load,
                             initargs=(str(export),)) as pool:
        return list(pool.map(_draw_one2, jobs, chunksize=1))


def _rows(d: Path, tasks) -> list[dict]:
    rows = [json.loads(x) for x in (d / "samples.jsonl").read_text().splitlines() if x.strip()]
    unknown = {r["task"] for r in rows} - {t.id for t in tasks}
    if unknown:
        raise ValueError(f"{d}: samples for unknown tasks {sorted(unknown)}")
    keys = [(r["task"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{d}: a (task, seed) appears twice")
    return rows


def score2(d: Path, out: Path | None = None, runner: Runner2 | None = None, workers: int = 8,
           budget_ms: int = 30_000) -> dict:
    """Judge DIR/samples.jsonl under the v2 harness. v1 samples (suite fc41a22a...) are
    re-judged on the dev tasks, reusing them (the openings are unchanged), into DIR/v2/; v2
    samples (sample2) into DIR. Writes OUT/judged.jsonl, OUT/metrics.json, OUT/runs/. Without
    a runner, every batch runs on one pool of `workers` emulator processes (pooled_runner2)."""
    d = Path(d)
    meta = json.loads((d / "sample.json").read_text())
    sha = meta.get("suite_sha256")
    if sha == V1_SUITE_SHA256:
        source, tasks, out = "v1", T.SUITE, Path(out or d / "v2")
        if out.resolve() == d.resolve():
            raise ValueError(f"{d}: v1 samples are re-judged into another directory (DIR/v2)")
    elif sha == SUITE_V2_SHA256:
        source, tasks, out = "v2", tuple(TASKS_V2[i] for i in meta["tasks"]), Path(out or d)
    else:
        raise ValueError(f"{d}: samples were drawn for another suite (suite_sha256 {sha})")
    rows = _rows(d, tasks)
    judged = []
    session = pooled_runner2(workers, budget_ms) if runner is None else contextlib.nullcontext(
        runner)
    with session as run:
        pending = []      # every task's batches are queued before any record is awaited
        for t in tasks:
            mine = [r for r in rows if r["task"] == t.id]
            pending.append((t, mine, submit_programs2(t, [r["ids"] for r in mine], run,
                                                      out / "runs")))
        for t, mine, collect in pending:
            judged += [{"task": t.id, "seed": r["seed"], **judge2(rec, t)}
                       for r, rec in zip(mine, collect())]
    m = task_metrics2(judged, tasks)
    m |= {"samples_sha256": hashlib.sha256((d / "samples.jsonl").read_bytes()).hexdigest(),
          "k": meta.get("k"), "export": meta.get("export"), "budget_ms": budget_ms,
          "source": source, "samples_dir": str(d)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def sample2(export, k: int, seed0: int, out: Path, workers: int = 8, split: str = "heldout",
            runner: Runner2 | None = None, budget_ms: int = 30_000) -> dict:
    """k samples per task of a split (seeds seed0..seed0+k-1, v2.1 decoding), then score2."""
    tasks = SPLITS[split]
    jobs = [(t.id, s) for t in tasks for s in range(seed0, seed0 + k)]
    ids = draw2(export, jobs, workers)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(
        json.dumps({"task": tid, "seed": s, "ids": x}) + "\n" for (tid, s), x in zip(jobs, ids)))
    ex = Path(export) / "model.bin"
    meta = {"export": str(export), "k": k, "seed0": seed0, "n": len(jobs),
            "suite_sha256": SUITE_V2_SHA256, "split": split, "tasks": [t.id for t in tasks],
            "temp": T.TEMP, "top_k": T.TOP_K, "cap": T.CAP, "masks": True, "horizons": True,
            "export_sha256": hashlib.sha256(ex.read_bytes()).hexdigest() if ex.exists() else None}
    (out / "sample.json").write_text(json.dumps(meta, indent=1))
    return score2(out, runner=runner, workers=workers, budget_ms=budget_ms)


# ---- checks against saved runs (read-only) ------------------------------------------------------

def rejudge_v1(d: Path, tasks=T.SUITE) -> tuple[str, str]:
    """v1's score() run again on a saved v1 score dir, its emulator replaced by the saved
    records (DIR/runs/<task>/t<j>/results.jsonl); returns the judged.jsonl and metrics.json
    text it writes (in a scratch copy: DIR is never written)."""
    d = Path(d)
    saved = json.loads((d / "metrics.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("samples.jsonl", "sample.json"):
            shutil.copyfile(d / name, work / name)

        def saved_runner(programs, values, out):
            rel = Path(out).relative_to(work)
            recs = [json.loads(x) for x in (d / rel / "results.jsonl").read_text().splitlines()
                    if x.strip()]
            if len(recs) != len(programs):
                raise ValueError(f"{d / rel}: {len(recs)} records for {len(programs)} programs")
            return recs
        T.score(work, runner=saved_runner, tasks=tasks, budget_ms=saved["budget_ms"])
        return (work / "judged.jsonl").read_text(), (work / "metrics.json").read_text()


def _audit_dir(runs: Path, exp: str, arm: str, seed: int) -> Path:
    """Where an audited pass's samples live (E7's S is E6's SEED)."""
    if exp == "e7" and arm == "S":
        return runs / "e6" / f"s{seed}" / "SEED" / "tasks"
    return runs / exp / f"s{seed}" / arm / "tasks"


def coincidental_items(runs: Path) -> list[dict]:
    """E6's and E7's blind-audit passes labelled coincidental, with their token ids (read
    from the saved samples, never from the audit's text)."""
    items = []
    for exp in ("e6", "e7"):
        key = json.loads((Path(runs) / exp / "audit" / "key.json").read_text())["items"]
        labels = json.loads((Path(runs) / exp / "audit" / "labels.json").read_text())["labels"]
        for lab in labels:
            if lab["verdict"] != "coincidental":
                continue
            k = key[lab["item"]]
            d = _audit_dir(Path(runs), exp, k["arm"], k["seed"])
            ids = next(r["ids"] for r in map(json.loads, (d / "samples.jsonl").read_text()
                                             .splitlines())
                       if (r["task"], r["seed"]) == (k["task"], k["sample_seed"]))
            items.append({"exp": exp, "item": lab["item"], **k, "dir": str(d), "ids": ids})
    return items


def regression(items: list[dict], runner: Runner2, out: Path) -> list[dict]:
    """Each item judged again under v2 (and its saved v1 judgment)."""
    res = []
    for it in items:
        task = TASKS_V2[it["task"]]
        v1 = next(r for r in map(json.loads, (Path(it["dir"]) / "judged.jsonl").read_text()
                                 .splitlines())
                  if (r["task"], r["seed"]) == (it["task"], it["sample_seed"]))
        [recs] = run_programs2(task, [it["ids"]], runner, Path(out) / it["item"])
        j = judge2(recs, task)
        res.append({k: it[k] for k in ("exp", "item", "arm", "seed", "task", "sample_seed")}
                   | {"v1_pass": v1["pass"], "v2_pass": j["pass"], "v2": j})
    return res
