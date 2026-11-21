"""The task suite: does the model complete a typed math-tool opening with a program that
computes the right answer? (docs/agents/tasks/F-task-suite.md)

A task is a TI-BASIC opening (ClrHome, [Degree,] a Disp title, a Prompt of its input
variables), 3 input tuples and a Python reference. A sample continues the opening with v2.1
decoding (model.intref.generate with the opening as prompt, masks and ALiBi horizons, T 0.8 /
top-k 20, cap 1024). It runs in CEmu once per tuple, the harness typing the tuple's values at
successive Input/Prompt waits (eval.driver.Config.input_values), with yield capture.

- A tuple passes when a number on the final home screen, or on a yield screen captured after
  the last input was answered, is within relative 1e-3 of the reference (absolute 1e-6 when
  the reference is 0), and (decoy rule, an amendment to the spec) those screens show no
  other tuple's reference: a loop that lists numbers (Disp I, Pause, 1 to 120) would
  otherwise hit integer answers. The echo guard keeps the 3 references apart, so a correct
  program never trips it.
- A sample passes when its 3 tuples pass, every run ends OK or TIMEOUT and no harness error
  occurs (a run stopped at the driver's wall-clock limit counts as one). Samples left empty
  by the 1024-token line trim are not run and fail.
- task_correct: % of all samples that pass; task_correct_literal: the same without the decoy
  rule. Also: per-task and per-topic rates, tasks solved by at least one sample, and partial
  (% of samples with at least one passing tuple, whatever the runs' outcomes).

    python -m lab.tasks build [--workers N] [--out DIR]           # validate the suite in CEmu
    python -m lab.tasks sample --export E --k 25 --seed0 1 --out DIR [--workers N]
    python -m lab.tasks score DIR [--workers N]                   # -> DIR/metrics.json
    python -m lab.tasks gold --out DIR                            # the gold completions as
                                                                  # samples (ceiling check)
    python -m lab.tasks build2 | score2 DIR | sample2 ...         # task suite v2: lab.tasks_v2

The suite's sha256 (suite_sha256: every task's text, inputs and reference answers) is in
sample.json, metrics.json and build.json. Task openings, gold completions and mutants are
hand-written for this suite.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from corpus.tokenize import UNK, Vocab
from eval.truncate import trim_to_line
from model.tokens import BOS, EOS

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "corpus" / "vocab.json"
TEMP, TOP_K, CAP = 0.8, 20, 1024          # v2.1 decoding
REL_TOL, ABS_TOL = 1e-3, 1e-6
FORBIDDEN = (2.0, 7.0)                    # the math tier's typed values
OK_OUTCOMES = ("OK", "TIMEOUT")
TOPICS = ("geometry", "algebra", "physics", "finance", "conversion", "statistics",
          "number theory", "chemistry")

Runner = Callable[[list, tuple, Path], list]   # (programs, input values, out dir) -> records


# ---- tokens --------------------------------------------------------------------------------

@functools.cache
def _vocab() -> Vocab:
    return Vocab.load(VOCAB)


def text_ids(text: str) -> list[int]:
    """TI-BASIC text (tivars load_string syntax) -> token ids (corpus/vocab.json)."""
    from tivars import TIProgram

    p = TIProgram(name="T")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.load_string(text)
    return _vocab().encode(p.data)


NL = text_ids("\n")[0]


# ---- tasks ---------------------------------------------------------------------------------

def value(text: str) -> float:
    """A typed input value as a number ("⁻" is TI's negative sign)."""
    return float(text.replace("⁻", "-"))


@dataclass(frozen=True)
class Task:
    id: str
    topic: str
    title: tuple[str, ...]           # Disp lines
    vars: tuple[str, ...]            # Prompt order
    inputs: tuple[tuple[str, ...], ...]
    ref: Callable[..., float]
    gold: str                        # completion after the opening
    mutants: tuple[str, ...]         # wrong completions (must fail)
    degree: bool = False             # trig in degrees: Degree in the opening
    integer_inputs: bool = False

    def opening(self) -> str:
        title = ",".join(f'"{x}"' for x in self.title)
        return ("ClrHome\n" + ("Degree\n" if self.degree else "") + f"Disp {title}\n"
                f"Prompt {','.join(self.vars)}\n")

    def opening_ids(self) -> list[int]:
        return text_ids(self.opening())

    def expected(self) -> tuple[float, ...]:
        return tuple(float(self.ref(*map(value, tup))) for tup in self.inputs)

    def spec(self) -> dict:
        """Everything that decides a judgment (suite_sha256 hashes these)."""
        return {"id": self.id, "topic": self.topic, "opening": self.opening(),
                "vars": list(self.vars), "inputs": [list(t) for t in self.inputs],
                "expected": list(self.expected()), "gold": self.gold,
                "mutants": list(self.mutants)}


def program_ids(task: Task, completion: str) -> list[int]:
    """<bos>, the opening, a completion, <eos>."""
    return [BOS, *task.opening_ids(), *text_ids(completion), EOS]


def suite_sha256(tasks) -> str:
    blob = json.dumps([t.spec() for t in tasks], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _deg(x: float) -> float:
    return math.radians(x)


def _pop_sd(a, b, c):
    m = (a + b + c) / 3
    return math.sqrt(((a - m) ** 2 + (b - m) ** 2 + (c - m) ** 2) / 3)


def _digit_sum(n):
    return sum(int(d) for d in str(int(n)))


SUITE: tuple[Task, ...] = (
    # ---- geometry
    Task("geo_circle_area", "geometry", ("CIRCLE AREA", "RADIUS R"), ("R",),
         (("3.5",), ("12",), ("0.8",)), lambda r: math.pi * r * r,
         "Disp piR^2", ("Disp 2piR", "Disp piR^3", "Disp R")),
    Task("geo_hypotenuse", "geometry", ("RIGHT TRIANGLE", "HYPOTENUSE", "LEGS A AND B"),
         ("A", "B"), (("3", "4"), ("1.5", "8"), ("12", "9")), math.hypot,
         "Disp √(A²+B²)", ("Disp A²+B²", "Disp A+B", "Disp √(A²+B)")),
    Task("geo_cylinder_volume", "geometry", ("CYLINDER VOLUME", "RADIUS R, HEIGHT H"),
         ("R", "H"), (("3", "5"), ("1.5", "10.5"), ("12", "0.5")),
         lambda r, h: math.pi * r * r * h,
         "Disp piR^2H", ("Disp 2piRH", "Disp piRH", "Disp piR^2H/3")),
    Task("geo_sphere_volume", "geometry", ("SPHERE VOLUME", "RADIUS R"), ("R",),
         (("4",), ("1.5",), ("11",)), lambda r: 4 / 3 * math.pi * r ** 3,
         "Disp 4/3piR^3", ("Disp 4piR^2", "Disp 4/3piR^2", "Disp R^3")),
    Task("geo_triangle_sas", "geometry", ("TRIANGLE AREA", "SIDES A AND B", "ANGLE C IN DEGREES"),
         ("A", "B", "C"), (("5", "8", "30"), ("3.5", "4", "120"), ("12", "9", "40")),
         lambda a, b, c: a * b * math.sin(_deg(c)) / 2,
         "Disp ABsin(C)/2", ("Disp ABsin(C)", "Disp ABcos(C)/2", "Disp AB/2"), degree=True),
    # ---- algebra
    Task("alg_slope", "algebra", ("SLOPE OF THE LINE", "THROUGH (A,B), (C,D)"),
         ("A", "B", "C", "D"),
         (("1", "4", "5", "16"), ("0.5", "1", "4.5", "11"), ("10", "4", "15", "1")),
         lambda a, b, c, d: (d - b) / (c - a),
         "Disp (D-B)/(C-A)", ("Disp (C-A)/(D-B)", "Disp (D-B)/(C+A)", "Disp D-B")),
    Task("alg_linear_root", "algebra", ("SOLVE AX+B=0 FOR X",), ("A", "B"),
         (("4", "10"), ("0.5", "3"), ("12", "9")), lambda a, b: -b / a,
         "Disp ⁻B/A", ("Disp B/A", "Disp ⁻A/B")),
    Task("alg_quadratic_root", "algebra", ("LARGER ROOT OF", "AX²+BX+C=0"), ("A", "B", "C"),
         (("1", "9", "14"), ("0.5", "6", "16"), ("1", "13", "30")),
         lambda a, b, c: (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a),
         "Disp (⁻B+√(B²-4AC))/(2A)",
         ("Disp (⁻B-√(B²-4AC))/(2A)", "Disp (⁻B+√(B²-4AC))/A", "Disp (B+√(B²-4AC))/(2A)")),
    Task("alg_distance", "algebra", ("DISTANCE BETWEEN", "(A,B) AND (C,D)"), ("A", "B", "C", "D"),
         (("1", "3", "4", "8"), ("0.5", "1.5", "6", "4"), ("12", "5", "3", "9")),
         lambda a, b, c, d: math.hypot(c - a, d - b),
         "Disp √((C-A)²+(D-B)²)",
         ("Disp (C-A)²+(D-B)²", "Disp √((C+A)²+(D+B)²)", "Disp abs(C-A)+abs(D-B)")),
    Task("alg_vertex_x", "algebra", ("PARABOLA Y=AX²+BX+C", "X OF THE VERTEX"), ("A", "B"),
         (("1", "6"), ("0.5", "5"), ("3", "15")), lambda a, b: -b / (2 * a),
         "Disp ⁻B/(2A)", ("Disp B/(2A)", "Disp ⁻B/A", "Disp ⁻2A/B")),
    # ---- physics
    Task("phy_kinetic_energy", "physics", ("KINETIC ENERGY (J)", "MASS M IN KG",
                                           "SPEED V IN M/S"), ("M", "V"),
         (("3", "4"), ("0.5", "12"), ("80", "1.5")), lambda m, v: m * v * v / 2,
         "Disp MV²/2", ("Disp MV²", "Disp MV/2", "Disp MV^3/2")),
    Task("phy_average_speed", "physics", ("AVERAGE SPEED (M/S)", "DISTANCE D IN M",
                                          "TIME T IN S"), ("D", "T"),
         (("100", "8"), ("45", "3.75"), ("13.5", "4")), lambda d, t: d / t,
         "Disp D/T", ("Disp T/D", "Disp DT")),
    Task("phy_ohm_voltage", "physics", ("OHM'S LAW VOLTAGE (V)", "CURRENT I IN AMPS",
                                        "RESISTANCE R IN OHMS"), ("I", "R"),
         (("0.5", "220"), ("3", "4"), ("1.5", "14")), lambda i, r: i * r,
         "Disp IR", ("Disp I/R", "Disp R/I", "Disp I+R")),
    Task("phy_potential_energy", "physics", ("POTENTIAL ENERGY (J)", "MASS M IN KG",
                                             "HEIGHT H IN M", "USE G=9.8 M/S²"), ("M", "H"),
         (("3", "5"), ("0.5", "12"), ("60", "1.5")), lambda m, h: 9.8 * m * h,
         "Disp 9.8MH", ("Disp MH", "Disp 9.8M/H", "Disp 9.8MH/2")),
    Task("phy_force_x", "physics", ("FORCE X-COMPONENT (N)", "FORCE F IN NEWTONS",
                                    "ANGLE A IN DEGREES", "FROM THE X-AXIS"), ("F", "A"),
         (("10", "30"), ("12.5", "60"), ("40", "50")), lambda f, a: f * math.cos(_deg(a)),
         "Disp Fcos(A)", ("Disp Fsin(A)", "Disp F/cos(A)", "Disp F"), degree=True),
    # ---- finance
    Task("fin_simple_interest", "finance", ("SIMPLE INTEREST EARNED", "PRINCIPAL P",
                                            "RATE R % PER YEAR", "TIME T IN YEARS"),
         ("P", "R", "T"), (("1000", "5", "3"), ("250", "4.5", "6"), ("80", "12", "1.5")),
         lambda p, r, t: p * r * t / 100,
         "Disp PRT/100", ("Disp PRT", "Disp P+PRT/100", "Disp P(1+R/100)^T")),
    Task("fin_compound_yearly", "finance", ("COMPOUND INTEREST", "COMPOUNDED YEARLY",
                                            "FINAL AMOUNT FROM P,", "R % PER YEAR, T YEARS"),
         ("P", "R", "T"), (("1000", "5", "3"), ("250", "4.5", "6"), ("80", "12", "11")),
         lambda p, r, t: p * (1 + r / 100) ** t,
         "Disp P(1+R/100)^T", ("Disp P(1+R)^T", "Disp P(1+R/100)T", "Disp P(1+RT/100)")),
    Task("fin_compound_monthly", "finance", ("COMPOUND INTEREST", "COMPOUNDED MONTHLY",
                                             "FINAL AMOUNT FROM P,", "R % PER YEAR, T YEARS"),
         ("P", "R", "T"), (("1000", "5", "3"), ("250", "4.5", "6"), ("80", "12", "1.5")),
         lambda p, r, t: p * (1 + r / 1200) ** (12 * t),
         "Disp P(1+R/1200)^(12T)",
         ("Disp P(1+R/100)^(12T)", "Disp P(1+R/1200)^T", "Disp P(1+R/12)^(12T)")),
    Task("fin_sale_price", "finance", ("SALE PRICE", "PRICE P", "DISCOUNT D %"), ("P", "D"),
         (("80", "25"), ("19.99", "15"), ("1250", "4.5")), lambda p, d: p * (1 - d / 100),
         "Disp P(1-D/100)", ("Disp PD/100", "Disp P(1+D/100)", "Disp P-D")),
    Task("fin_percent_change", "finance", ("PERCENT CHANGE", "FROM OLD VALUE A", "TO NEW VALUE B"),
         ("A", "B"), (("80", "100"), ("12.5", "9"), ("40", "46")),
         lambda a, b: 100 * (b - a) / a,
         "Disp 100(B-A)/A", ("Disp 100(B-A)/B", "Disp (B-A)/A", "Disp 100(A-B)/A")),
    # ---- conversion
    Task("conv_c_to_f", "conversion", ("CELSIUS TO FAHRENHEIT", "TEMPERATURE C IN °C"), ("C",),
         (("25",), ("37.5",), ("4",)), lambda c: 9 * c / 5 + 32,
         "Disp 9C/5+32", ("Disp 5C/9+32", "Disp 9C/5-32", "Disp 9(C+32)/5")),
    Task("conv_f_to_c", "conversion", ("FAHRENHEIT TO CELSIUS", "TEMPERATURE F IN °F"), ("F",),
         (("100",), ("50",), ("71.5",)), lambda f: 5 * (f - 32) / 9,
         "Disp 5(F-32)/9", ("Disp 5F/9-32", "Disp 9(F-32)/5", "Disp F-32")),
    Task("conv_in_to_cm", "conversion", ("INCHES TO CENTIMETERS", "LENGTH L IN INCHES"), ("L",),
         (("12",), ("3.5",), ("60",)), lambda x: 2.54 * x,
         "Disp 2.54L", ("Disp L/2.54", "Disp 2.5L", "Disp L")),
    Task("conv_ms_to_kmh", "conversion", ("M/S TO KM/H", "SPEED V IN M/S"), ("V",),
         (("10",), ("3.5",), ("25",)), lambda v: 3.6 * v,
         "Disp 3.6V", ("Disp V/3.6", "Disp 60V", "Disp V")),
    Task("conv_deg_to_rad", "conversion", ("DEGREES TO RADIANS", "ANGLE D IN DEGREES"), ("D",),
         (("45",), ("12.5",), ("300",)), lambda d: d * math.pi / 180,
         "Disp Dpi/180", ("Disp 180D/pi", "Disp Dpi/360", "Disp D")),
    # ---- statistics
    Task("stat_mean3", "statistics", ("MEAN OF 3 NUMBERS",), ("A", "B", "C"),
         (("4", "9", "23"), ("1.5", "6", "10"), ("20", "11", "50")),
         lambda a, b, c: (a + b + c) / 3,
         "Disp (A+B+C)/3", ("Disp A+B+C/3", "Disp (A+B+C)/2", "Disp (A+B)/2")),
    Task("stat_range3", "statistics", ("RANGE OF 3 NUMBERS", "(LARGEST MINUS SMALLEST)"),
         ("A", "B", "C"), (("9", "4", "15"), ("1.5", "6", "0.5"), ("30", "45", "12")),
         lambda a, b, c: max(a, b, c) - min(a, b, c),
         "Disp max(A,max(B,C))-min(A,min(B,C))",
         ("Disp max(A,max(B,C))", "Disp C-A", "Disp max(A,B)-min(A,B)")),
    Task("stat_zscore", "statistics", ("Z-SCORE", "VALUE X, MEAN M", "STANDARD DEVIATION S"),
         ("X", "M", "S"), (("85", "70", "6"), ("12.5", "10", "1.25"), ("40", "52", "8")),
         lambda x, m, s: (x - m) / s,
         "Disp (X-M)/S", ("Disp X-M/S", "Disp (M-X)/S", "Disp (X-M)S")),
    Task("stat_pop_sd3", "statistics", ("POPULATION STANDARD", "DEVIATION OF 3 NUMBERS"),
         ("A", "B", "C"), (("4", "9", "23"), ("1.5", "6", "10"), ("20", "11", "50")), _pop_sd,
         "(A+B+C)/3→M\nDisp √(((A-M)²+(B-M)²+(C-M)²)/3)",
         ("Disp stdDev({A,B,C})", "(A+B+C)/3→M\nDisp ((A-M)²+(B-M)²+(C-M)²)/3",
          "(A+B+C)/3→M\nDisp √(((A-M)²+(B-M)²+(C-M)²)/2)")),
    Task("stat_weighted_mean", "statistics", ("WEIGHTED MEAN", "VALUE A WITH WEIGHT W",
                                              "VALUE B WITH WEIGHT V"), ("A", "W", "B", "V"),
         (("80", "3", "90", "1"), ("12.5", "0.4", "6", "0.35"), ("50", "5", "20", "15")),
         lambda a, w, b, v: (a * w + b * v) / (w + v),
         "Disp (AW+BV)/(W+V)", ("Disp (A+B)/2", "Disp (AW+BV)/2", "Disp AW+BV")),
    # ---- number theory
    Task("nt_gcd", "number theory", ("GREATEST COMMON", "DIVISOR OF A AND B"), ("A", "B"),
         (("84", "90"), ("168", "36"), ("252", "70")),
         lambda a, b: math.gcd(int(a), int(b)),
         "Disp gcd(A,B)", ("Disp lcm(A,B)", "Disp A-B", "Disp A/B"), integer_inputs=True),
    Task("nt_lcm", "number theory", ("LEAST COMMON", "MULTIPLE OF A AND B"), ("A", "B"),
         (("4", "6"), ("4", "10"), ("6", "15")),
         lambda a, b: math.lcm(int(a), int(b)),
         "Disp lcm(A,B)", ("Disp gcd(A,B)", "Disp AB"), integer_inputs=True),
    Task("nt_remainder", "number theory", ("REMAINDER WHEN A", "IS DIVIDED BY B"), ("A", "B"),
         (("47", "6"), ("100", "8"), ("89", "13")), lambda a, b: a % b,
         "Disp remainder(A,B)", ("Disp int(A/B)", "Disp fPart(A/B)", "Disp A/B"),
         integer_inputs=True),
    Task("nt_digit_sum", "number theory", ("SUM OF THE DIGITS", "OF A WHOLE NUMBER N"), ("N",),
         (("1234",), ("98765",), ("3005",)), _digit_sum,
         "0→S\nWhile N>0\nS+remainder(N,10)→S\nint(N/10)→N\nEnd\nDisp S",
         ("0→S\nWhile N>0\nS+1→S\nint(N/10)→N\nEnd\nDisp S", "Disp remainder(N,10)",
          "Disp N/10"), integer_inputs=True),
    Task("nt_factorial", "number theory", ("FACTORIAL OF N",), ("N",),
         (("5",), ("9",), ("11",)), lambda n: math.factorial(int(n)),
         "Disp N!", ("Disp N(N-1)", "Disp (N-1)!", "Disp N^N"), integer_inputs=True),
    # ---- chemistry
    Task("chem_moles", "chemistry", ("MOLES FROM MASS", "MASS M IN G", "MOLAR MASS W IN G/MOL"),
         ("M", "W"), (("36", "18"), ("12.5", "44"), ("100", "58.44")), lambda m, w: m / w,
         "Disp M/W", ("Disp MW", "Disp W/M")),
    Task("chem_molarity", "chemistry", ("MOLARITY (MOL/L)", "MOLES N", "VOLUME V IN ML"),
         ("N", "V"), (("0.5", "250"), ("1.5", "600"), ("0.03", "80")),
         lambda n, v: 1000 * n / v,
         "Disp 1000N/V", ("Disp N/V", "Disp NV/1000", "Disp 100N/V")),
    Task("chem_ideal_gas", "chemistry", ("IDEAL GAS PRESSURE ATM", "USE R=0.08206",
                                         "MOLES N, TEMP T IN K", "VOLUME V IN L"),
         ("N", "T", "V"), (("1", "300", "12"), ("0.5", "273.15", "5.6"), ("3", "450", "40")),
         lambda n, t, v: 0.08206 * n * t / v,
         "Disp 0.08206NT/V", ("Disp 8.314NT/V", "Disp NT/V", "Disp 0.08206NV/T")),
    Task("chem_dilution", "chemistry", ("DILUTION", "START CONCENTRATION C", "START VOLUME V",
                                        "FINAL VOLUME W", "FINAL CONCENTRATION?"), ("C", "V", "W"),
         (("6", "50", "240"), ("1.5", "20", "125"), ("12", "15", "45")),
         lambda c, v, w: c * v / w,
         "Disp CV/W", ("Disp CW/V", "Disp C", "Disp CV")),
    Task("chem_percent_yield", "chemistry", ("PERCENT YIELD", "ACTUAL YIELD A IN G",
                                             "THEORETICAL YIELD T IN G"), ("A", "T"),
         (("8", "12.5"), ("45", "60"), ("3.6", "4")), lambda a, t: 100 * a / t,
         "Disp 100A/T", ("Disp 100T/A", "Disp A/T")),
)
TASKS: dict[str, Task] = {t.id: t for t in SUITE}


# ---- numbers on the screen -----------------------------------------------------------------

_NUM = re.compile(r"([⁻-]?)([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ᴇ([⁻-]?[0-9]+))?")


def numbers(line: str) -> list[float]:
    """Every number shown on a home-screen line. TI shows a negative number with ⁻ (an ASCII
    minus counts as a sign only where it cannot be a subtraction) and powers of ten with ᴇ.
    Digits right after a letter belong to a name (L1, Str1, X1=?) and are not numbers."""
    out = []
    for m in _NUM.finditer(line):
        sign, mant, exp = m.groups()
        prev = line[m.start() - 1] if m.start() else " "
        if not sign and prev.isalpha():
            continue
        neg = sign == "⁻" or (sign == "-" and not (prev.isalnum() or prev in ")."))
        try:
            x = float(mant) * (10.0 ** int(exp.replace("⁻", "-")) if exp else 1.0)
        except OverflowError:   # no task reference is this large; it can never match
            continue
        out.append(-x if neg else x)
    return out


def matches(x: float, ref: float) -> bool:
    if ref == 0:
        return abs(x) <= ABS_TOL
    return abs(x - ref) <= REL_TOL * abs(ref)


def tuple_lines(r: dict) -> list[str]:
    """The final home screen plus every yield screen captured after the last input was
    answered (yield screens are read before the harness answers: the screens after the last
    "input" yield; all of them when no input was answered)."""
    ys = r.get("yields") or []
    last = max((i for i, y in enumerate(ys) if y[0] == "input"), default=-1)
    lines = list(r.get("home_text") or [])
    for y in ys[last + 1:]:
        lines += y[3]
    return lines


def shows(r: dict, ref: float) -> bool:
    return any(matches(x, ref) for line in tuple_lines(r) for x in numbers(line))


def tuple_passes(r: dict, ref: float) -> bool:
    """The spec's tuple rule alone (judge adds the decoy rule)."""
    return shows(r, ref)


def _harness(r: dict) -> bool:
    """A harness error, or a run the driver stopped at its wall-clock limit (labelled
    TIMEOUT, but its fate depended on host load)."""
    from eval.driver import Config

    return r["outcome"] == "HARNESS_ERROR" or (r.get("wall_s") or 0) > Config().wall_limit_s


def judge(records: list[dict] | None, refs) -> dict:
    """One sample's 3 runs (None: empty after the trim, never run) against the references.
    tuples[i]: run i shows refs[i] and no other tuple's reference (decoys[i]: it shows one,
    as a loop listing numbers does). pass: all tuples, every run OK or TIMEOUT, no harness
    error. pass_literal: the same without the decoy rule (the spec's rule as written)."""
    n = len(refs)
    if records is None:
        return {"tuples": [False] * n, "decoys": [False] * n, "outcomes": [None] * n,
                "extra_inputs": [None] * n, "harness_error": False, "empty": True,
                "pass": False, "pass_literal": False}
    hits = [shows(r, ref) for r, ref in zip(records, refs, strict=True)]
    decoys = [any(shows(r, o) for j, o in enumerate(refs) if j != i)
              for i, r in enumerate(records)]
    outcomes = [r["outcome"] for r in records]
    harness = any(_harness(r) for r in records)
    clean = all(o in OK_OUTCOMES for o in outcomes) and not harness
    tuples = [h and not d for h, d in zip(hits, decoys)]
    return {"tuples": tuples, "decoys": decoys, "outcomes": outcomes,
            "extra_inputs": [r.get("extra_inputs") for r in records], "harness_error": harness,
            "empty": False, "pass": all(tuples) and clean, "pass_literal": all(hits) and clean}


# ---- validating the suite --------------------------------------------------------------------

def check_static(task: Task) -> None:
    """Shape and tokens: 3 full tuples, 2+ mutants, inputs never 2 or 7, a value above 10, a
    non-integer unless the task takes integers, no <unk> in the opening, gold or mutants."""
    vals = [value(v) for tup in task.inputs for v in tup]
    problems = []
    if task.topic not in TOPICS:
        problems.append(f"unknown topic {task.topic!r}")
    if len(task.inputs) != 3 or any(len(t) != len(task.vars) for t in task.inputs):
        problems.append("needs 3 tuples of one value per variable")
    if len(task.mutants) < 2:
        problems.append("needs at least 2 mutants")
    if set(vals) & set(FORBIDDEN):
        problems.append("inputs must never be 2 or 7")
    if not any(v > 10 for v in vals):
        problems.append("needs an input above 10")
    if task.integer_inputs != all(v == int(v) for v in vals):
        problems.append("integer_inputs must say whether every input is an integer")
    ids = task.opening_ids()
    if UNK in ids or ids[-1] != NL:
        problems.append("the opening must tokenize without <unk> and end with a newline")
    if any(UNK in text_ids(x) for x in (task.gold, *task.mutants)):
        problems.append("gold and mutants must tokenize without <unk>")
    if problems:
        raise ValueError(f"{task.id}: " + "; ".join(problems))


def check_echo(task: Task) -> None:
    """The echo guard: every reference differs (beyond the tolerance) from every input value
    of every tuple, from the other tuples' references, and from any number the opening
    shows."""
    refs = task.expected()
    vals = [value(v) for tup in task.inputs for v in tup]
    for i, ref in enumerate(refs):
        if any(matches(v, ref) for v in vals):
            raise ValueError(f"{task.id}: echo: reference {ref} of tuple {i} equals an input")
        if any(matches(o, ref) for j, o in enumerate(refs) if j != i):
            raise ValueError(f"{task.id}: echo: reference {ref} of tuple {i} repeats")
        shown = [x for line in task.title for x in numbers(line)]
        if any(matches(x, ref) for x in shown):
            raise ValueError(f"{task.id}: reference {ref} is a number the opening shows")


def resolve(recs):
    """A runner's return value as records: a list as is, a queued batch (a pooled runner's
    handle, see pooled_runner) once its result() is in."""
    return recs.result() if hasattr(recs, "result") else recs


def run_programs(task: Task, programs: list[list[int]], runner: Runner, out: Path) -> list:
    """Each program once per tuple -> per program, its 3 records (None when the program is
    empty after the line trim at CAP)."""
    return submit_programs(task, programs, runner, out)()


def submit_programs(task: Task, programs: list[list[int]], runner: Runner,
                    out: Path) -> Callable[[], list]:
    """run_programs in two steps: every tuple's batch goes to the runner now, and the returned
    function gives run_programs' result. A pooled runner queues the batches and returns
    handles, so a caller can queue every task's batches before it waits for any record."""
    trimmed = [trim_to_line(list(p), CAP, NL, EOS) for p in programs]
    keep = [i for i, t in enumerate(trimmed) if len(t) > 1]
    batches = []

    def check(j, recs):
        if len(recs) != len(keep):
            raise ValueError(f"{task.id} t{j}: {len(recs)} records for {len(keep)} programs")
        return recs
    for j, vals in enumerate(task.inputs):
        if not keep:
            break
        recs = runner([trimmed[i] for i in keep], tuple(vals), Path(out) / task.id / f"t{j}")
        batches.append((j, recs if hasattr(recs, "result") else check(j, recs)))

    def collect() -> list:
        per = {i: [] for i in keep}
        for j, recs in batches:
            for i, r in zip(keep, check(j, resolve(recs))):
                per[i].append(r)
        return [per.get(i) for i in range(len(programs))]
    return collect


def validate(task: Task, runner: Runner, out: Path) -> dict:
    """The metric's own test for one task: the gold completion passes, every mutant fails."""
    progs = [program_ids(task, x) for x in (task.gold, *task.mutants)]
    refs = task.expected()
    gold, *mutants = (judge(r, refs) for r in run_programs(task, progs, runner, out))
    return {"task": task.id, "gold": gold, "mutants": mutants,
            "ok": gold["pass"] and not any(m["pass"] for m in mutants)}


def emulator_runner(workers: int = 8, budget_ms: int = 30_000) -> Runner:
    """lab.score.run_emulator with yield capture, one worker per program at most."""
    from lab.score import run_emulator

    def run(programs, values, out):
        return run_emulator(programs, out, VOCAB, workers=min(workers, len(programs)),
                            budget_ms=budget_ms, yields=True, input_values=values)
    return run


def pool_run(pool, programs, out, workers: int, budget_ms: int, **harness):
    """emulator_runner's lab.score.run_emulator call (yield capture, no screenshots, the run
    dir's path as title, `workers` capped at the batch size) queued on an
    eval.run_programs.EmuPool: a Batch whose result() is run_emulator's records."""
    return pool.submit_token_ids(programs, VOCAB, Path(out),
                                 workers=min(workers, len(programs)), budget_ms=budget_ms,
                                 shots=0, title=str(out), yields=True, **harness)


@contextlib.contextmanager
def pooled_runner(workers: int = 8, budget_ms: int = 30_000):
    """emulator_runner on one pool of emulator workers kept for the whole block: a call
    queues its batch and returns a handle (records via result(); see resolve). The same
    records and run directory files as emulator_runner, timings aside."""
    from eval.run_programs import EmuPool

    with EmuPool(workers) as pool:
        yield lambda programs, values, out: pool_run(
            pool, programs, out, workers, budget_ms, input_values=tuple(values))


def build(out: Path, runner: Runner, tasks=SUITE) -> dict:
    for t in tasks:
        check_static(t)
        check_echo(t)
    res = [validate(t, runner, Path(out) / "runs") for t in tasks]
    report = {"suite_sha256": suite_sha256(tasks), "n_tasks": len(tasks),
              "ok": all(r["ok"] for r in res), "tasks": res}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "build.json").write_text(json.dumps(report, indent=1))
    return report


# ---- sampling -------------------------------------------------------------------------------

_model = None


def _load(export: str) -> None:
    global _model
    from model.intref import IntModel

    _model = IntModel.load(export, horizons=True)
    if list(_model.vocab or []) != _vocab().entries:
        raise ValueError(f"{export}: the export's vocab is not corpus/vocab.json")


def _draw_one(job: tuple[str, int]) -> list[int]:
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate

    task_id, seed = job
    cfg = SamplerConfig(round(TEMP * 256), TOP_K, False, seed)
    return [int(t) for t in generate(_model, TASKS[task_id].opening_ids(), CAP, cfg, CAP,
                                     IntMask(_model.vocab))]


def draw(export, jobs: list[tuple[str, int]], workers: int) -> list[list[int]]:
    """v2.1 decoding from each (task id, seed) job's opening, in job order."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1:
        _load(str(export))
        return [_draw_one(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_load,
                             initargs=(str(export),)) as pool:
        return list(pool.map(_draw_one, jobs, chunksize=1))


def sample(export, k: int, seed0: int, out: Path, workers: int = 8, tasks=SUITE) -> None:
    jobs = [(t.id, s) for t in tasks for s in range(seed0, seed0 + k)]
    ids = draw(export, jobs, workers)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(
        json.dumps({"task": tid, "seed": s, "ids": x}) + "\n" for (tid, s), x in zip(jobs, ids)))
    ex = Path(export) / "model.bin"
    meta = {"export": str(export), "k": k, "seed0": seed0, "n": len(jobs),
            "suite_sha256": suite_sha256(tasks), "temp": TEMP, "top_k": TOP_K, "cap": CAP,
            "masks": True, "horizons": True,
            "export_sha256": hashlib.sha256(ex.read_bytes()).hexdigest() if ex.exists() else None}
    (out / "sample.json").write_text(json.dumps(meta, indent=1))


def gold_samples(out: Path, tasks=SUITE) -> None:
    """The gold completions written as samples (seed 0), for `score` as a ceiling check."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(
        json.dumps({"task": t.id, "seed": 0, "ids": program_ids(t, t.gold)}) + "\n"
        for t in tasks))
    (out / "sample.json").write_text(json.dumps({"export": "gold", "k": 1, "seed0": 0,
                                                 "n": len(tasks),
                                                 "suite_sha256": suite_sha256(tasks)}, indent=1))


# ---- scoring --------------------------------------------------------------------------------

def _rate(k: int, n: int) -> float:
    return 100 * k / n if n else 0.0


def task_metrics(judged: list[dict], tasks=SUITE) -> dict:
    """Metrics over judged samples ({task, tuples, empty, harness_error, pass}); every sample
    counts in the denominator, and every suite task must have samples."""
    by = {t.id: [] for t in tasks}
    for r in judged:
        if r["task"] not in by:
            raise ValueError(f"sample for unknown task {r['task']!r}")
        by[r["task"]].append(r)
    missing = [tid for tid, rows in by.items() if not rows]
    if missing:
        raise ValueError(f"no samples for {missing}")
    n = len(judged)
    passed = sum(r["pass"] for r in judged)
    topic = {t.id: t.topic for t in tasks}
    per_topic = {}
    for tp in dict.fromkeys(t.topic for t in tasks):
        rows = [r for r in judged if topic[r["task"]] == tp]
        per_topic[tp] = _rate(sum(r["pass"] for r in rows), len(rows))
    return {"suite_sha256": suite_sha256(tasks), "n_tasks": len(tasks), "n_samples": n,
            "n_pass": passed, "task_correct": _rate(passed, n),
            "task_correct_literal": _rate(sum(r["pass_literal"] for r in judged), n),
            "tasks_solved": sum(any(r["pass"] for r in rows) for rows in by.values()),
            "partial": _rate(sum(any(r["tuples"]) for r in judged), n),
            "n_empty": sum(r["empty"] for r in judged),
            "harness_errors": sum(r["harness_error"] for r in judged),
            "per_task": {tid: _rate(sum(r["pass"] for r in rows), len(rows))
                         for tid, rows in by.items()},
            "per_topic": per_topic}


def score(d: Path, runner: Runner | None = None, tasks=SUITE, workers: int = 8,
          budget_ms: int = 30_000) -> dict:
    """Judge DIR/samples.jsonl: 3 emulator runs per sample (DIR/runs/<task>/t<j>/), then
    DIR/judged.jsonl and DIR/metrics.json. Without a runner, every batch runs on one pool of
    `workers` emulator processes (pooled_runner)."""
    d = Path(d)
    meta = json.loads((d / "sample.json").read_text())
    if meta.get("suite_sha256") != suite_sha256(tasks):
        raise ValueError(f"{d}: samples were drawn for another suite (suite_sha256 differs)")
    rows = [json.loads(x) for x in (d / "samples.jsonl").read_text().splitlines() if x.strip()]
    unknown = {r["task"] for r in rows} - {t.id for t in tasks}
    if unknown:
        raise ValueError(f"{d}: samples for unknown tasks {sorted(unknown)}")
    keys = [(r["task"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{d}: a (task, seed) appears twice")
    judged = []
    session = pooled_runner(workers, budget_ms) if runner is None else contextlib.nullcontext(
        runner)
    with session as run:
        pending = []      # every task's batches are queued before any record is awaited
        for t in tasks:
            mine = [r for r in rows if r["task"] == t.id]
            pending.append((t, mine, submit_programs(t, [r["ids"] for r in mine], run,
                                                     d / "runs")))
        for t, mine, collect in pending:
            refs = t.expected()
            judged += [{"task": t.id, "seed": r["seed"], **judge(rec, refs)}
                       for r, rec in zip(mine, collect())]
    m = task_metrics(judged, tasks)
    m |= {"samples_sha256": hashlib.sha256((d / "samples.jsonl").read_bytes()).hexdigest(),
          "k": meta.get("k"), "export": meta.get("export"), "budget_ms": budget_ms}
    (d / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="validate the suite: tokens, echo guard, gold and mutants")
    b.add_argument("--out", type=Path, default=ROOT / "runs" / "tasks" / "build")
    s = sub.add_parser("sample", help="k samples per task with v2.1 decoding")
    s.add_argument("--export", required=True, type=Path)
    s.add_argument("--k", type=int, default=25)
    s.add_argument("--seed0", type=int, default=1)
    s.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("score", help="judge DIR/samples.jsonl in CEmu -> DIR/metrics.json")
    c.add_argument("dir", type=Path)
    c.add_argument("--budget-ms", type=int, default=30_000)
    g = sub.add_parser("gold", help="write the gold completions as samples")
    g.add_argument("--out", required=True, type=Path)
    b2 = sub.add_parser("build2", help="task suite v2 (lab.tasks_v2): validate in CEmu")
    b2.add_argument("--out", type=Path, default=ROOT / "runs" / "tasks" / "build2")
    c2 = sub.add_parser("score2", help="task suite v2: judge DIR's samples under the hardened "
                                       "harness (v1 samples: the dev tasks, into DIR/v2/)")
    c2.add_argument("dir", type=Path)
    c2.add_argument("--out", type=Path, default=None)
    s2 = sub.add_parser("sample2", help="task suite v2: k samples per task of a split, scored")
    s2.add_argument("--export", required=True, type=Path)
    s2.add_argument("--k", type=int, default=25)
    s2.add_argument("--seed0", type=int, default=1)
    s2.add_argument("--tasks", choices=("heldout", "dev", "all"), default="heldout")
    s2.add_argument("--out", required=True, type=Path)
    for p in (c2, s2):
        p.add_argument("--budget-ms", type=int, default=30_000)
    for p in (b, s, c, b2, c2, s2):
        p.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd in ("build2", "score2", "sample2"):
        return _main_v2(a)
    if a.cmd == "build":
        rep = build(a.out, emulator_runner(a.workers))
        bad = [r["task"] for r in rep["tasks"] if not r["ok"]]
        print(json.dumps({"ok": rep["ok"], "n_tasks": rep["n_tasks"], "failed": bad,
                          "suite_sha256": rep["suite_sha256"]}))
        return 0 if rep["ok"] else 1
    if a.cmd == "sample":
        sample(a.export, a.k, a.seed0, a.out, a.workers)
    elif a.cmd == "gold":
        gold_samples(a.out)
    else:
        m = score(a.dir, workers=a.workers, budget_ms=a.budget_ms)
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_task",)}, indent=1))
    return 0


def _main_v2(a) -> int:
    from lab import tasks_v2 as V

    if a.cmd == "build2":
        rep = V.build2(a.out, V.emulator_runner2(a.workers))
        bad = [r["task"] for r in rep["tasks"] if not r["ok"]]
        print(json.dumps({"ok": rep["ok"], "n_tasks": rep["n_tasks"], "failed": bad,
                          "suite_sha256": rep["suite_sha256"]}))
        return 0 if rep["ok"] else 1
    if a.cmd == "score2":
        m = V.score2(a.dir, out=a.out, workers=a.workers, budget_ms=a.budget_ms)
    else:
        m = V.sample2(a.export, k=a.k, seed0=a.seed0, out=a.out, workers=a.workers,
                      split=a.tasks, budget_ms=a.budget_ms)
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_task",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
