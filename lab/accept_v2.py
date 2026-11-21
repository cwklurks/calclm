"""CalcLM v2 acceptance (PLAN-v2.md, "Shipping rules"): criteria M2-M5 and the tables, from
lab.score's score.json files.

    uv run python -m lab.accept_v2 --dir runs/v2/accept

Inputs (`<dir>/<name>.jsonl` from eval.gen_intref, `<dir>/<name>/score.json` from lab.score):
`v1-256` (runs/v1/export through intref, cap 256, no masks), `v2-256` and `v2-1024` (the shipped
v2 export through intref, masks on, caps 256 and 1024); all seeds 1..1000, T 0.8 / top-k 20, on
the fixed harness. main() checks those settings in the sample records and the cut length in each
score file. The human reference is G2a's runs/lab/REF-val-1024 (val programs cut at 1024).

- M2: strict OK higher than v1 with non-overlapping Wilson 95% intervals (regime (i), cap 256).
- M3: LABEL + UNDEFINED + SYNTAX (rates over the programs run, summed) at most half of v1's.
- M4: the share of samples ending with <eos> within 1024 tokens is at least 80% of the share of
  val programs that end within 1024 tokens (regime (ii)).
- M5: 32-gram train overlap (mean seen fraction) at most 1 point above v1's (regime (i)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "runs" / "lab" / "REF-val-1024" / "result.json"

RUNS = {"v1-256": (256, False), "v2-256": (256, True), "v2-1024": (1024, True)}   # cap, masks
EPS = 1e-9          # the thresholds are inclusive; rates are ratios of counts

COLUMNS = ["complete %", "no-ERR %", "strict OK % (95% CI)", "SYNTAX %", "LABEL %",
           "UNDEFINED %", "L+U+S %", "distinct-4", "overlap %", "length"]


def lus(s: dict) -> float:
    return s["err_syntax"]["p"] + s["err_label"]["p"] + s["err_undefined"]["p"]


def criteria(v1: dict, v2: dict, v2_long: dict, val_complete: float) -> dict:
    """v1, v2: cap-256 scores; v2_long: v2 at cap 1024; val_complete: the val share ending
    within 1024 tokens (REF-val-1024's complete rate)."""
    a, b = v1["strict_ok"], v2["strict_ok"]
    m3 = {"v1": lus(v1), "v2": lus(v2), "threshold": 0.5 * lus(v1)}
    m4 = {"v2": v2_long["complete"]["p"], "val": val_complete, "threshold": 0.8 * val_complete}
    m5 = {"v1": v1["overlap_32"]["mean_seen"], "v2": v2["overlap_32"]["mean_seen"]}
    m5["difference_points"] = 100 * (m5["v2"] - m5["v1"])
    return {
        "M2": {"v1": a, "v2": b, "pass": b["lo"] > a["hi"]},
        "M3": {**m3, "relative_change": m3["v2"] / m3["v1"] - 1,
               "pass": m3["v2"] <= m3["threshold"] + EPS},
        "M4": {**m4, "ratio": m4["v2"] / val_complete, "pass": m4["v2"] >= m4["threshold"] - EPS},
        "M5": {**m5, "pass": m5["difference_points"] <= 1.0 + EPS},
    }


def table_row(name: str, s: dict, length_key: str = "length") -> str:
    pct = lambda r: f"{100 * r['p']:.1f}"  # noqa: E731
    st = s["strict_ok"]
    cells = [pct(s["complete"]), pct(s["no_err"]),
             f"{100 * st['p']:.1f} ({100 * st['lo']:.1f}-{100 * st['hi']:.1f})",
             pct(s["err_syntax"]), pct(s["err_label"]), pct(s["err_undefined"]),
             f"{100 * lus(s):.1f}", f"{s['distinct_4']:.3f}",
             f"{100 * s['overlap_32']['mean_seen']:.1f}", f"{s[length_key]['mean']:.1f}"]
    return f"| {name} | " + " | ".join(cells) + " |"


def tables(v1: dict, v2: dict, v2_long: dict, ref: dict) -> str:
    head = "| run | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 1)
    return "\n".join([head, table_row("v1, cap 256", v1), table_row("v2, cap 256", v2),
                      table_row("v2, cap 1024", v2_long),
                      table_row("human val, cut at 1024", ref, "trimmed_length")])


def load_run(d: Path, name: str) -> dict:
    """A run's score.json, after checking its samples and cut against the shipping setting."""
    cap, masks = RUNS[name]
    recs = [json.loads(line) for line in (d / f"{name}.jsonl").read_text().splitlines() if line]
    if [r["seed"] for r in recs] != list(range(1, 1001)):
        raise ValueError(f"{name}: sampling seeds are not 1..1000")
    for r in recs:
        if (r["temp"], r["top_k"], r["cap"], r["masks"]) != (0.8, 20, cap, masks):
            raise ValueError(f"{name}: seed {r['seed']} was not sampled at T 0.8 / top-k 20, "
                             f"cap {cap}, masks {masks}")
    score = json.loads((d / name / "score.json").read_text())
    if score.get("max_len") != cap:
        raise ValueError(f"{name}: scored with cut length {score.get('max_len')}, not {cap}")
    return score


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "v2" / "accept")
    ap.add_argument("--ref", type=Path, default=REF)
    a = ap.parse_args(argv)
    v1, v2, v2_long = (load_run(a.dir, name) for name in RUNS)
    ref_run = json.loads(a.ref.read_text())
    if ref_run.get("cap") != 1024:
        raise ValueError("the human reference must be val programs cut at 1024")
    ref = ref_run["score"]
    crit = criteria(v1, v2, v2_long, ref["complete"]["p"])
    (a.dir / "criteria.json").write_text(json.dumps(crit, indent=1))
    print(tables(v1, v2, v2_long, ref))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in crit.items()}, indent=1))
    return crit


if __name__ == "__main__":
    main()
