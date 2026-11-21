"""E20 spike: the M1-M5 numbers from runs/e20's records (PLAN-E20.md Measurements, "How O1 is
read"), as JSON and a short Markdown table.

    python -m ce.tools.e20_summary runs/e20 > runs/e20/summary.md   (writes runs/e20/summary.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HZ = 48_000_000
RC2_BUDGET = 1.35            # lab.budget: rc2 (L4's shape) = 1.35x v2.1
R_RANGE = (0.994, 1.0)       # CEmu-to-hardware ratio: rc2, v2.2
CODE_FACTOR = 0.68           # CLMCODE flash against RAM code fetch (ce/BENCH.md, an estimate)


def _jobs(path: Path) -> list[dict]:
    return json.loads(path.read_text())["jobs"] if path.exists() else []


def _overhead(ld2_jobs: list[dict]) -> dict:
    """Per-token cycles of the mask and sample phases (tracker, masks, LD2 rules, sampling) from
    the hooked generations, by (cfg, ld2)."""
    out = {}
    for j in ld2_jobs:
        ph = j["phases"]
        out[(j["cfg"], j["ld2"])] = {
            "tokens": j["tokens"],
            "mask": ph["mask"]["cycles"] / j["tokens"],
            "sample": ph["sample"]["cycles"] / j["tokens"],
            "per_token": (ph["mask"]["cycles"] + ph["sample"]["cycles"]) / j["tokens"]}
    return out


def summarize(d: Path) -> dict:
    m1 = {(j["cfg"], j["horizons"], j["seq"]): j for j in _jobs(d / "m1.json")}
    mean = {k: float(np.mean(v["cycles"])) for k, v in m1.items()}
    ov = _overhead(_jobs(d / "m3_ld2.json"))
    s: dict = {"m1_mean_forward": {f"{c} {'hz' if h else 'nohz'} {q}": v
                                   for (c, h, q), v in sorted(mean.items())}}
    s["m3_per_token_overhead"] = {f"{c} {'ld2' if l else 'no-ld2'}": v
                                  for (c, l), v in sorted(ov.items())}

    def cpt(cfg, ld2):                     # O1 primary: sequence A, horizons on, + overhead
        f = mean.get((cfg, True, "A"))
        o = ov.get((cfg, ld2))
        return None if f is None or o is None else f + o["per_token"]

    o1 = {}
    for cfg in ("W8k4B", "W8k4A", "rc2", "v2"):
        o1[cfg] = {"with_ld2": cpt(cfg, True), "without_ld2": cpt(cfg, False)}
    v22, v21 = o1["v2"]["with_ld2"], o1["v2"]["without_ld2"]
    for cfg, r in o1.items():
        w = r["with_ld2"]
        if w:
            r["x_v2.2"] = w / v22 if v22 else None
            r["x_v2.1_ld2_in_neither"] = r["without_ld2"] / v21 if v21 and r["without_ld2"] else None
            r["lab_budget_units"] = (w / o1["rc2"]["with_ld2"] * RC2_BUDGET
                                     if o1["rc2"]["with_ld2"] else None)
            r["hw_tok_s_r1"] = HZ / w
            r["hw_tok_s_range"] = [HZ / w * x for x in R_RANGE]
            r["minutes_316_tokens"] = w * 316 / HZ / 60
    s["o1"] = o1
    g = lambda c, h=True, q="A": mean.get((c, h, q))            # noqa: E731
    if g("W8k4A") and g("L4k4"):
        slope = (g("W8k4A") - g("L4k4")) / 2
        s["per_layer_slope_A"] = slope
        s["w7_interpolated_A"] = g("L4k4") + slope
    s["4bit_overhead_L4_A"] = (g("L4k4") - g("L4i8")) if g("L4k4") and g("L4i8") else None
    s["horizon_savings"] = {c: {q: (g(c, False, q) - g(c, True, q))
                                for q in ("A", "B") if g(c, False, q) and g(c, True, q)}
                            for c in ("L4i8", "L4k4", "W8k4A")}
    s["placement_B_minus_A"] = {q: g("W8k4B", True, q) - g("W8k4A", True, q)
                                for q in ("A", "B") if g("W8k4B", True, q) and g("W8k4A", True, q)}
    s["checkpoint_spread_A"] = (g("W8s11k4A") - g("W8k4A")) if g("W8s11k4A") and g("W8k4A") else None
    m2 = _jobs(d / "m2.json")
    pooled = {}
    for cfg in sorted({j["cfg"] for j in m2}):
        js = [j for j in m2 if j["cfg"] == cfg]
        cyc, tok = sum(j["cycles"] for j in js), sum(j["tokens"] for j in js)
        pooled[cfg] = {"programs": len(js), "tokens": tok, "cycles_per_token": cyc / tok,
                       "tok_s_r1": HZ * tok / cyc, "minutes": cyc / HZ / 60,
                       "per_program": {j["seed"]: [j["tokens"], j["cycles_per_token"]] for j in js}}
    s["m2_pooled"] = pooled
    m3 = _jobs(d / "m3.json")
    prof = {}
    for j in m3:
        rows = {}
        for r in j["rows"]:
            ph = r["phases"]
            raw = sum(v["cycles"] for k, v in ph.items() if k != "_pre")
            scale = r["cycles"] / raw if raw else 1.0             # hook overhead, proportionally
            rows[r["pos"]] = {"cycles": r["cycles"],
                              "kv4": ph["kv4"]["cycles"] * scale,
                              "kv4_flash": [ph["kv4"]["flash_accesses"], ph["kv4"]["flash_misses"]],
                              "attention": sum(ph[p]["cycles"] for p in ("scores", "softmax", "pv")) * scale,
                              "matvec": sum(ph[p]["cycles"] for p in ("build", "rows")) * scale}
        rec = {"rows": rows}
        if "unpack_full_window" in j:
            u = j["unpack_full_window"]
            asm, c = sum(x["cycles"] for x in u["build"]), sum(x["cycles"] for x in u["c"])
            rec["unpack_full_window"] = {"asm": asm, "c": c, "c_over_asm": c / asm,
                                         "asm_clmcode_estimate": asm * CODE_FACTOR,
                                         "flash_accesses": sum(x["flash_accesses"] for x in u["build"]),
                                         "flash_misses": sum(x["flash_misses"] for x in u["build"])}
        prof[j["cfg"]] = rec
    s["m3_profile"] = prof
    return s


def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    d = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else d / "summary.json"
    s = summarize(d)
    out.write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
