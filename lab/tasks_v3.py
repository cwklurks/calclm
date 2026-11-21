"""Task suite v3 (lab/TASKS_V3.md): v2's judgment (lab.tasks_v2.judge2, unchanged) plus two
stricter rules, applied offline to saved hardened-harness records. A SECONDARY metric: the
milestone target stays on v2 (decision of 2026-10-28).

A sample passes v3 iff it passes v2 and, on every tuple:
1. the run's outcome is OK, not TIMEOUT (a program that never ends does not pass);
2. the tuple's reference is the LAST number shown (lab.tasks.matches, v2's tolerance).
pass_v3_labelled: pass_v3 and v2's label check (pass_labelled).

The last number shown (last_number): v2's screens (lab.tasks_v2.screens: the yield screens
captured after the last input was answered, and the final home screen, blank lines dropped) put
in display order, yield screens in capture order and the final home screen last. Each screen is
read top to bottom, each line left to right, with v2's parser (lab.tasks.numbers). The last
number is the last one on the latest screen that shows any number; screens repeat what is
still on the display, so a cleared final screen falls back to the last Pause screen. No new
parser: digits inside a shown string count as v2 counts them.

judge3 also reports, per tuple, timeouts, last and, where the reference is not last, what the
last number is (not_last_kinds): "typed" (one of the tuple's inputs or the decoy, an echo),
"prefill" (a stale prefilled variable) or "other"; and lost: why a v2 pass fails v3
("timeout", "not_last" or "both"; None otherwise).

    python -m lab.tasks_v3 rescore [--runs R] [--out R/v3]   # saved E10/E11/E13/E14 runs
    python -m lab.tasks_v3 report [--runs R] [--out R/v3]    # tables -> OUT/<exp>/table.*
    python -m lab.tasks_v3 build3 [--out R/v3/build]         # golds and mutants in CEmu

rescore re-judges every saved dir with v2 first and refuses any dir whose judged.jsonl it does
not reproduce exactly. It never writes inside a source dir. The v3 loophole mutants below are
hand-written for this suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lab import tasks as T
from lab import tasks_v2 as V
from lab.stats import paired_greater

RUNS = Path(os.environ.get("CALCLM_RUNS", T.ROOT / "runs"))
DECISION_SEEDS = tuple(range(11, 27))
ALL_SEEDS = tuple(range(10, 27))

# Programs v2 passes and v3 must fail (the metric's own tests, never samples).
V3_LOOPHOLES: dict[str, tuple[tuple[str, str], ...]] = {
    "geo_circle_area": (("timeout", "Disp piR^2\nWhile 1\nEnd"),
                        ("spray", "Disp piR^2\nDisp 2piR"), ("prefill", "Disp piR^2\nDisp Q")),
    "geo_cylinder_volume": (("spray", "Disp piR^2H\nDisp piR^2H/3"),),
    "phy_ohm_voltage": (("timeout", "Disp IR\nWhile 1\nEnd"),),
}


# ---- judging --------------------------------------------------------------------------------

def screens3(r: dict) -> list[list[str]]:
    """v2's screens in display order: yield screens after the last input, then the final home
    screen."""
    home, *ys = V.screens(r)
    return [*ys, home]


def last_number(r: dict) -> float | None:
    """The last number on the latest screen that shows one (None: no screen does)."""
    for lines in reversed(screens3(r)):
        xs = [x for line in lines for x in T.numbers(line)]
        if xs:
            return xs[-1]
    return None


def _kind(r: dict, x: float | None, typed) -> str:
    if x is None:
        return "none"
    if any(T.matches(x, T.value(v)) for v in (*typed, *([r["decoy"]] if r.get("decoy") else []))):
        return "typed"
    if any(T.matches(x, T.value(v)) for v in (r.get("prefill") or {}).values()):
        return "prefill"
    return "other"


def judge3(records: list[dict] | None, task: T.Task) -> dict:
    """judge2's fields unchanged, plus timeouts, last, not_last_kinds, pass_v3,
    pass_v3_labelled and lost."""
    j = V.judge2(records, task)
    n = len(task.inputs)
    if records is None:
        return {**j, "timeouts": [False] * n, "last": [False] * n,
                "not_last_kinds": [None] * n, "pass_v3": False, "pass_v3_labelled": False,
                "lost": None}
    refs = task.expected()
    lasts = [last_number(r) for r in records]
    last = [x is not None and T.matches(x, ref) for x, ref in zip(lasts, refs)]
    kinds = [None if ok else _kind(r, x, tup)
             for ok, r, x, tup in zip(last, records, lasts, task.inputs)]
    timeouts = [r["outcome"] == "TIMEOUT" for r in records]
    v3 = j["pass"] and not any(timeouts) and all(last)
    lost = None
    if j["pass"] and not v3:
        lost = ("both" if any(timeouts) and not all(last)
                else "timeout" if any(timeouts) else "not_last")
    return {**j, "timeouts": timeouts, "last": last, "not_last_kinds": kinds, "pass_v3": v3,
            "pass_v3_labelled": v3 and j["pass_labelled"], "lost": lost}


# ---- saved records (read-only) --------------------------------------------------------------

def saved_records(d: Path, base: Path) -> list[tuple[T.Task, int, list | None]]:
    """(task, sampling seed, its records) for every sample of a saved score2 dir, in
    judged.jsonl order: score2 itself (lab.tasks.run_programs: the same line trim, empties and
    record order) with the emulator replaced by DIR/runs/<task>/t<j>/results.jsonl. base is the
    checkout that holds runs/ (metrics.json's samples_dir is relative to it)."""
    d = Path(d)
    m = json.loads((d / "metrics.json").read_text())
    sd = Path(base) / m["samples_dir"]
    meta = json.loads((sd / "sample.json").read_text())
    if m["source"] == "v1":
        tasks = T.SUITE
    else:
        tasks = tuple(V.TASKS_V2[i] for i in meta["tasks"])
    rows = V._rows(sd, tasks)

    def saved(programs, values, out):
        recs = [json.loads(x) for x in (Path(out) / "results.jsonl").read_text().splitlines()
                if x.strip()]
        if len(recs) != len(programs):
            raise ValueError(f"{out}: {len(recs)} records for {len(programs)} programs")
        return recs
    out = []
    for t in tasks:
        mine = [r for r in rows if r["task"] == t.id]
        recs = T.run_programs(t, [r["ids"] for r in mine], saved, d / "runs")
        out += [(t, r["seed"], rec) for r, rec in zip(mine, recs)]
    return out


def rejudge_v2(d: Path, base: Path) -> list[dict]:
    """judged.jsonl's rows recomputed offline from the saved records."""
    return [{"task": t.id, "seed": s, **V.judge2(rec, t)} for t, s, rec in saved_records(d, base)]


_KEEP = ("task", "seed", "empty", "outcomes", "pass", "pass_labelled", "timeouts", "last",
         "not_last_kinds", "pass_v3", "pass_v3_labelled", "lost")


def rescore_dir(d: Path, base: Path) -> list[dict]:
    """judge3 rows (compact) for a saved dir; raises unless v2 reproduces judged.jsonl."""
    d = Path(d)
    saved = [json.loads(x) for x in (d / "judged.jsonl").read_text().splitlines() if x.strip()]
    samples = saved_records(d, base)
    v2 = [{"task": t.id, "seed": s, **V.judge2(rec, t)} for t, s, rec in samples]
    if v2 != saved:
        bad = sum(a != b for a, b in zip(v2, saved)) + abs(len(v2) - len(saved))
        raise ValueError(f"{d}: offline v2 differs from judged.jsonl on {bad} rows")
    rows = [{"task": t.id, "seed": s, **judge3(rec, t)} for t, s, rec in samples]
    return [{k: r[k] for k in _KEEP} for r in rows]


# ---- metrics ----------------------------------------------------------------------------------

def metrics3(rows: list[dict], max_seed: int | None = None) -> dict:
    """Rates over one split's judge3 rows (sampling seeds up to max_seed, when given)."""
    rows = [r for r in rows if max_seed is None or r["seed"] <= max_seed]
    n = len(rows)

    def count(k):
        return sum(bool(r[k]) for r in rows)
    lost = Counter(r["lost"] for r in rows if r["lost"])
    kinds = Counter(k for r in rows if r["lost"] in ("not_last", "both")
                    for k in r["not_last_kinds"] if k)
    return {"n_samples": n, "n_pass": count("pass"), "n_pass_v3": count("pass_v3"),
            "n_pass_labelled": count("pass_labelled"),
            "n_pass_v3_labelled": count("pass_v3_labelled"),
            "task_correct": T._rate(count("pass"), n),
            "task_correct_v3": T._rate(count("pass_v3"), n),
            "task_correct_labelled": T._rate(count("pass_labelled"), n),
            "task_correct_v3_labelled": T._rate(count("pass_v3_labelled"), n),
            "lost": {k: lost.get(k, 0) for k in ("timeout", "not_last", "both")},
            "not_last_kinds": dict(kinds), "max_seed": max_seed}


def compare(a, b, two_sided: bool) -> dict:
    """lab.stats.paired_greater (H1 mean(a - b) > 0); two_sided doubles the smaller tail."""
    r = paired_greater(a, b)
    return r | {"p": 2 * min(r["p"], 1 - r["p"])} if two_sided else r


# ---- the experiments ----------------------------------------------------------------------------

# arms in table order, the paired comparisons (a, b) as each write-up tested them, sidedness.
EXPERIMENTS = {
    "e10": {"arms": ("W", "L2", "L4"), "pairs": (("L2", "W"), ("L4", "W")), "two_sided": False},
    "e11": {"arms": ("L4", "L8"), "pairs": (("L8", "L4"),), "two_sided": False},
    "e13": {"arms": ("L4", "BR"), "pairs": (("BR", "L4"),), "two_sided": False},
    "e14": {"arms": ("V21", "LD2"), "pairs": (("LD2", "V21"),), "two_sided": True},
}
SPLITS = ("dev", "heldout")


def source(runs: Path, exp: str, arm: str, seed: int, split: str) -> tuple[Path, int | None]:
    """(saved score2 dir, sampling-seed bound) of one arm, seed and split."""
    runs = Path(runs)
    if exp == "e13" and arm == "L4":          # E13's L4 is E11's retrained L4 (resolution 17)
        return source(runs, "e11", "L4", seed, split)
    if exp == "e14" and arm == "V21":         # E10's L4 judgments; held-out k 100 at seeds 1-50
        return source(runs, "e10", "L4", seed, split)[0], (50 if split == "heldout" else None)
    if split == "dev":
        return runs / exp / f"s{seed}" / arm / "tasks" / "v2", None
    if exp == "e10" and arm == "W" and 11 <= seed <= 22:   # reused W: E8's held-out (lab.e10)
        return runs / "e8" / f"s{seed}" / "W" / "heldout100", None
    return runs / exp / f"s{seed}" / arm / ("heldout100" if exp == "e10" else "heldout50"), None


def check_out(out: Path, runs: Path) -> None:
    """OUT must not lie inside a source experiment dir (runs/e*)."""
    o, r = Path(out).resolve(), Path(runs).resolve()
    if o == r:
        raise ValueError(f"{out} is the runs dir; v3 would write inside its e* dirs")
    if r in o.parents:
        top = o.relative_to(r).parts[0]
        if top[:1] == "e" and top[1:2].isdigit():
            raise ValueError(f"{out} is inside {r / top}; v3 writes nothing there")


def _units(runs: Path):
    for exp, spec in EXPERIMENTS.items():
        for arm in spec["arms"]:
            for seed in ALL_SEEDS:
                for split in SPLITS:
                    src, bound = source(runs, exp, arm, seed, split)
                    if (src / "judged.jsonl").exists():
                        yield exp, arm, seed, split, src, bound


def _rescore_one(args):
    src, base = args
    return str(src), rescore_dir(src, base)


def rescore(runs: Path, out: Path, workers: int = 8) -> dict:
    """Every present (exp, arm, seed, split) -> OUT/<exp>/s<seed>/<arm>/<split>/{judged3.jsonl,
    metrics3.json}; each source dir is judged once."""
    check_out(out, runs)
    units = list(_units(runs))
    srcs = sorted({u[4] for u in units})
    with ProcessPoolExecutor(workers) as pool:
        done = dict(pool.map(_rescore_one, [(s, Path(runs).parent) for s in srcs]))
    for exp, arm, seed, split, src, bound in units:
        rows = [r for r in done[str(src)] if bound is None or r["seed"] <= bound]
        d = Path(out) / exp / f"s{seed}" / arm / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "judged3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        m = metrics3(rows, bound) | {"source": str(src), "v2_reproduced": True,
                                     "suite_v2_sha256": V.SUITE_V2_SHA256}
        (d / "metrics3.json").write_text(json.dumps(m, indent=1))
    return {"units": len(units), "source_dirs": len(srcs)}


# ---- tables -------------------------------------------------------------------------------------

_RATES = ("task_correct", "task_correct_v3", "task_correct_labelled", "task_correct_v3_labelled")


def table(out: Path, exp: str) -> dict:
    """Means over the decision seeds, paired comparisons under v2 and v3, and losses pooled."""
    spec = EXPERIMENTS[exp]
    m = {(a, s, sp): json.loads((Path(out) / exp / f"s{s}" / a / sp / "metrics3.json")
                                .read_text())
         for a in spec["arms"] for s in DECISION_SEEDS for sp in SPLITS}
    res = {"exp": exp, "seeds": list(DECISION_SEEDS), "two_sided": spec["two_sided"],
           "means": {}, "pairs": [], "losses": {}}
    for a in spec["arms"]:
        for sp in SPLITS:
            rows = [m[a, s, sp] for s in DECISION_SEEDS]
            res["means"][f"{a}/{sp}"] = {k: sum(r[k] for r in rows) / len(rows) for k in _RATES}
            tot = Counter()
            for r in rows:
                tot.update({"n_pass": r["n_pass"], "n_pass_v3": r["n_pass_v3"], **r["lost"]})
                tot.update({f"kind_{k}": v for k, v in r["not_last_kinds"].items()})
            res["losses"][f"{a}/{sp}"] = dict(tot)
    for a, b in spec["pairs"]:
        for sp in SPLITS:
            row = {"a": a, "b": b, "split": sp}
            for k in ("task_correct", "task_correct_v3"):
                col = [[m[x, s, sp][k] for s in DECISION_SEEDS] for x in (a, b)]
                row[k] = compare(*col, spec["two_sided"])
            res["pairs"].append(row)
    return res


def _pct(x):
    return f"{x:.2f}"


def table_md(t: dict) -> str:
    exp, side = t["exp"].upper(), "two-sided" if t["two_sided"] else "one-sided"
    lines = [f"### {exp} (seeds 11-26, paired, df 15, {side} p)", "",
             "| arm | dev v2 | dev v3 | dev v3 labelled | held-out v2 | held-out v3 |",
             "|---|---:|---:|---:|---:|---:|"]
    arms = dict.fromkeys(k.split("/")[0] for k in t["means"])
    for a in arms:
        d, h = t["means"][f"{a}/dev"], t["means"][f"{a}/heldout"]
        lines.append(f"| {a} | {_pct(d['task_correct'])} | {_pct(d['task_correct_v3'])} | "
                     f"{_pct(d['task_correct_v3_labelled'])} | {h['task_correct']:.3f} | "
                     f"{h['task_correct_v3']:.3f} |")
    lines += ["", "| comparison | split | v2 diff | v2 p | v3 diff | v3 p |",
              "|---|---|---:|---:|---:|---:|"]
    for p in t["pairs"]:
        v2, v3 = p["task_correct"], p["task_correct_v3"]
        lines.append(f"| {p['a']} − {p['b']} | {p['split']} | {v2['diff']:+.3f} | "
                     f"{v2['p']:.3g} | {v3['diff']:+.3f} | {v3['p']:.3g} |")
    lines += ["", "| arm | split | v2 passes | lost: TIMEOUT only | lost: not last only | "
              "lost: both | share lost | v3 passes |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in t["losses"].items():
        a, sp = k.split("/")
        lost = v.get("timeout", 0) + v.get("not_last", 0) + v.get("both", 0)
        share = 100 * lost / v["n_pass"] if v.get("n_pass") else 0.0
        lines.append(f"| {a} | {sp} | {v.get('n_pass', 0)} | {v.get('timeout', 0)} | "
                     f"{v.get('not_last', 0)} | {v.get('both', 0)} | {share:.1f}% | "
                     f"{v.get('n_pass_v3', 0)} |")
    return "\n".join(lines) + "\n"


def report(out: Path, runs: Path) -> str:
    check_out(out, runs)
    md = []
    for exp in EXPERIMENTS:
        t = table(out, exp)
        (Path(out) / exp / "table.json").write_text(json.dumps(t, indent=1))
        (Path(out) / exp / "table.md").write_text(table_md(t))
        md.append(table_md(t))
    return "\n".join(md)


# ---- validating the rules in CEmu ---------------------------------------------------------------

def validate3(task: T.Task, runner, out: Path) -> dict:
    """Gold passes v3; every mutant and v2 loophole mutant fails; every v3 loophole mutant
    passes v2 and fails v3 (all under the v2 harness)."""
    loops = [b for _, b in V.LOOPHOLES.get(task.id, ())]
    v3l = [b for _, b in V3_LOOPHOLES.get(task.id, ())]
    progs = [T.program_ids(task, x) for x in (task.gold, *task.mutants, *loops, *v3l)]
    gold, *rest = (judge3(r, task) for r in V.run_programs2(task, progs, runner, out))
    n, k = len(task.mutants), len(loops)
    mut, lp, v3 = rest[:n], rest[n:n + k], rest[n + k:]
    ok = (gold["pass_v3"] and not any(x["pass"] for x in mut + lp)
          and all(x["pass"] and not x["pass_v3"] for x in v3))
    return {"task": task.id, "gold": gold, "mutants": mut, "loopholes": lp, "v3_loopholes": v3,
            "ok": ok}


def build3(out: Path, runner, tasks=V.SUITE_V2) -> dict:
    res = [validate3(t, runner, Path(out) / "runs") for t in tasks]
    golds = [r["gold"] for r in res]
    report = {"suite_v2_sha256": V.SUITE_V2_SHA256, "n_tasks": len(tasks),
              "ok": all(r["ok"] for r in res),
              "golds_pass_v3": sum(g["pass_v3"] for g in golds),
              "gold_tuples_last": sum(sum(g["last"]) for g in golds),
              "gold_tuples": sum(len(g["last"]) for g in golds),
              "gold_timeouts": sum(sum(g["timeouts"]) for g in golds),
              "golds_pass_v3_labelled": sum(g["pass_v3_labelled"] for g in golds),
              "mutants": sum(len(r["mutants"]) for r in res),
              "mutants_failing": sum(not m["pass"] for r in res for m in r["mutants"]),
              "loopholes": sum(len(r["loopholes"]) for r in res),
              "loopholes_failing": sum(not m["pass"] for r in res for m in r["loopholes"]),
              "v3_loopholes": sum(len(r["v3_loopholes"]) for r in res),
              "v3_loopholes_caught": sum(m["pass"] and not m["pass_v3"]
                                         for r in res for m in r["v3_loopholes"]),
              "tasks": res}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "build3.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rescore", help="re-judge saved E10/E11/E13/E14 dirs under v3 (offline)")
    p = sub.add_parser("report", help="tables from OUT's rescored dirs")
    b = sub.add_parser("build3", help="golds, mutants and loophole mutants under v3 in CEmu")
    for x in (r, p):
        x.add_argument("--runs", type=Path, default=RUNS)
        x.add_argument("--out", type=Path, default=RUNS / "v3")
    b.add_argument("--out", type=Path, default=RUNS / "v3" / "build")
    for x in (r, b):
        x.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "rescore":
        print(json.dumps(rescore(a.runs, a.out, a.workers)))
    elif a.cmd == "report":
        print(report(a.out, a.runs))
    else:
        from lab.emulator_lock import emulator_stage

        with emulator_stage("suite v3 build3"):
            rep = build3(a.out, V.emulator_runner2(a.workers))
        print(json.dumps({k: v for k, v in rep.items() if k != "tasks"}, indent=1))
        return 0 if rep["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
