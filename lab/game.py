"""The game tier: does a generated program behave like an interactive game?

Judged from a `lab.score --yields --game` run, where the harness feeds every getKey loop the
arrows and then Clear (eval.driver.GAME_KEYS) and records, per fed key, whether the screen
changed (eval.driver Config.key_probe). Per program:
- getkey_loop: getKey runs inside a While/Repeat/For body or condition, or between a Lbl and a
  later Goto (or Menu) back to it (static, from the trimmed tokens).
- reacts: some arrow key changed the screen.
- quits_on_clear: the run ended OK right after the harness pressed Clear.
- any_key: the run ended OK on the first fed key (a "press any key" program, not a quit key).
- interactive: getkey_loop and reacts, and the run did not crash: it ended OK or was still
  answering keys at the budget (TIMEOUT in a getKey wait). On human val programs, about as many
  reacting getKey loops are still running at the budget as end, so this is the main tier.
- game: getkey_loop and reacts and quits_on_clear (stricter: Clear is only one of the quit
  keys human programs use).
"""
from __future__ import annotations

from pathlib import Path

from eval.driver import GAME_KEYS
from lab.function import _ID, _R, _checked_run, aligned, menu_labels, name_of, statements

GETKEY = _ID["AD"]
ARROWS = GAME_KEYS[:4]
LOOPS = ("for", "while", "repeat")
FLAGS = ("getkey_loop", "reacts", "quits_on_clear", "any_key", "interactive", "game")


def getkey_in_loop(ids) -> bool:
    stack, labels, seen_gk = [], {}, []
    if_pending = False
    for k, (_, _, toks) in enumerate(statements(ids)):
        if not toks:
            continue
        r0 = _R[toks[0]][0]
        if if_pending:
            if_pending = False
            if r0 == "then":
                stack.append("if")
                continue
        has_gk = GETKEY in toks
        if has_gk and ("loop" in stack or r0 in LOOPS):
            return True
        if has_gk:
            seen_gk.append(k)
        if r0 == "if":
            if_pending = True
        elif r0 in LOOPS:
            stack.append("loop")
        elif r0 == "end" and stack:
            stack.pop()
        elif r0 == "lbl":
            labels.setdefault(name_of(toks[1:]), k)
        elif r0 in ("goto", "menu"):
            names = [name_of(toks[1:])] if r0 == "goto" else menu_labels(toks[1:])
            if any(n in labels and any(labels[n] <= g for g in seen_gk) for n in names):
                return True
    return False


def classify(ids, r: dict) -> dict:
    """ids: the trimmed program; r: its key-probe run record."""
    probes = r.get("key_probes") or []
    ended_ok = r["outcome"] == "OK" and r.get("last_action") == "getkey" and bool(probes)
    c = {"getkey_loop": getkey_in_loop(ids),
         "reacts": any(changed for key, changed, _ in probes if key in ARROWS),
         "quits_on_clear": ended_ok and probes[-1][0] == "Clear",
         "any_key": ended_ok and len(probes) == 1}
    alive = r["outcome"] == "OK" or (r["outcome"] == "TIMEOUT"
                                     and r.get("timeout_state") == "getkey")
    c["interactive"] = c["getkey_loop"] and c["reacts"] and alive
    c["game"] = c["getkey_loop"] and c["reacts"] and c["quits_on_clear"]
    return c


def game_metrics(samples: list[list[int]], results_dir: Path, cap: int = 1024) -> dict:
    """Percentages of the programs run (n_run) for every flag, and the count of games."""
    _, config = _checked_run(results_dir, cap)
    cfg = config.get("config", {})
    if not cfg.get("key_probe") or tuple(cfg.get("feed_keys", ())) != GAME_KEYS:
        raise ValueError(f"{results_dir}: not a key-probe run (lab.score --yields --game)")
    flags = [classify(t, r) for _, t, r in aligned(samples, results_dir, cap).values()]
    n = len(flags)
    out = {"n_run": n, "n_game": sum(f["game"] for f in flags),
           "n_interactive": sum(f["interactive"] for f in flags)}
    out |= {k: (100 * sum(f[k] for f in flags) / n if n else None) for k in FLAGS}
    return out
