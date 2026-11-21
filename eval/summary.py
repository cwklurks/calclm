"""Summary of a run: outcome counts and percentages, error-name histogram, throughput."""
from __future__ import annotations

from collections import Counter

OUTCOMES = ("OK", "ERR", "TIMEOUT", "REJECTED")
AFTER_INPUT_MS = 2000   # an ERR this soon after typing the default answer may be our input


def _pct(k: int, n: int) -> float:
    return round(100.0 * k / n, 2) if n else 0.0


def _static(r: dict, key: str) -> bool:
    return bool(r.get("static", {}).get(key))


SUBSETS = {
    "all": lambda r: True,
    "self_contained": lambda r: not (_static(r, "prgm_call") or _static(r, "asm")
                                     or "lib_calls" in (r.get("meta") or {}).get("tags", ())),
    "calls_prgm": lambda r: _static(r, "prgm_call"),
    "uses_asm": lambda r: _static(r, "asm"),
    "uses_getkey": lambda r: _static(r, "getkey"),
    "uses_input": lambda r: _static(r, "input"),
    "uses_menu": lambda r: _static(r, "menu"),
    "uses_draw": lambda r: _static(r, "draw"),
    "le_256_tokens": lambda r: r.get("n_tokens", 0) <= 256,   # fits CalcLM's context
}


def subsets(results: list[dict]) -> dict:
    """Outcome percentages within subsets defined by static program features."""
    out = {}
    for name, pred in SUBSETS.items():
        sel = [r for r in results if pred(r)]
        c = Counter(r["outcome"] for r in sel)
        out[name] = {"n": len(sel), "pct": {o: _pct(c.get(o, 0), len(sel)) for o in OUTCOMES}}
    return out


def summarize(results: list[dict], wall_s: float, workers: int) -> dict:
    n = len(results)
    counts = Counter(r["outcome"] for r in results)
    errors = Counter(r["error"] for r in results if r["outcome"] == "ERR")
    tstates = Counter(r["timeout_state"] for r in results if r["outcome"] == "TIMEOUT")
    ran = [r for r in results if r["outcome"] != "REJECTED"]
    return {
        "n": n,
        "outcomes": {o: {"n": counts.get(o, 0), "pct": _pct(counts.get(o, 0), n)}
                     for o in OUTCOMES + tuple(sorted(set(counts) - set(OUTCOMES)))},
        "errors": dict(errors.most_common()),
        "timeout_states": dict(tstates.most_common()),
        "tokenize_pct": _pct(len(ran), n),
        "no_err_pct": _pct(counts.get("OK", 0) + counts.get("TIMEOUT", 0), n),
        "graph": sum(bool(r.get("graph")) for r in results),
        "subsets": subsets(results),
        "err_after_input": sum(r["outcome"] == "ERR" and r.get("last_action") == "input"
                               and (r.get("ms_since_action") or 0) <= AFTER_INPUT_MS
                               for r in results),
        "emu_s_total": round(sum(r.get("emu_ms", 0) for r in results) / 1000, 1),
        "wall_s": round(wall_s, 1),
        "workers": workers,
        "programs_per_min": round(60.0 * n / wall_s, 1) if wall_s > 0 else 0.0,
    }


def to_markdown(s: dict, title: str) -> str:
    lines = [f"# {title}", "", f"{s['n']} programs, {s['workers']} workers, {s['wall_s']} s wall, "
             f"{s['programs_per_min']} programs/min, {s['emu_s_total']} s emulated", "",
             "| outcome | n | % |", "|---|---:|---:|"]
    for o, v in s["outcomes"].items():
        lines.append(f"| {o} | {v['n']} | {v['pct']} |")
    lines += ["", f"tokenize (not REJECTED): {s['tokenize_pct']}%; run without ERR "
              f"(OK + TIMEOUT): {s['no_err_pct']}%; showed the graph screen: {s['graph']}; "
              f"ERR within {AFTER_INPUT_MS} ms of a driven Input answer: {s['err_after_input']}",
              ""]
    lines += ["| error | n |", "|---|---:|"]
    lines += [f"| ERR:{k} | {v} |" for k, v in s["errors"].items()]
    lines += ["", "| timeout state | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in s["timeout_states"].items()]
    lines += ["", "| subset (static features) | n | " + " | ".join(f"{o} %" for o in OUTCOMES)
              + " |", "|---|---:|" + "---:|" * len(OUTCOMES)]
    for name, v in s.get("subsets", {}).items():
        lines.append(f"| {name} | {v['n']} | "
                     + " | ".join(str(v["pct"][o]) for o in OUTCOMES) + " |")
    return "\n".join(lines) + "\n"
