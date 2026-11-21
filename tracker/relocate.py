"""Re-run ERR programs with a patient error locator (for T4 line-level matching).

eval/driver.locate_error presses 2:Goto on the ERR: menu, runs 500 ms of emulated time and reads
the program editor's cursor. On long programs the editor is still moving the cursor at that
point, so the recorded error position stops short of the real one (no eval run reports an
error past line ~28). This script leaves eval/ untouched: in this process only, it swaps in a
locator that waits until the cursor has stopped moving, re-runs the chosen programs through the
normal harness (single process) and writes a separate run directory.

    uv run python -m tracker.relocate val-full     # -> eval/runs/w2-val-full-loc/
"""
from __future__ import annotations

import argparse
import json
import sys

from eval import driver, inputs, osram, run_programs
from . import validate as V

RUNS = V.RUNS


def patient_locate_error(emu, body: bytes) -> int | None:
    """driver.locate_error, but wait (up to 30 s emulated) until the cursor stops moving."""
    if not emu.read(osram.ERR_NO, 1)[0] & osram.ERR_GOTO:
        return None
    emu.inject_key("2")
    emu.run(500)
    last, still = None, 0
    for _ in range(300):
        cur = emu.u24(osram.EDIT_CURSOR) - emu.u24(osram.EDIT_TOP)
        still = still + 1 if cur == last else 0
        if still >= 3:
            break
        last = cur
        emu.run(100)
    top = emu.u24(osram.EDIT_TOP)
    off = emu.u24(osram.EDIT_CURSOR) - top
    if not 0 <= off <= len(body) or emu.read(top, off) != body[:off]:
        return None
    return off


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set", choices=list(V.SETS))
    p.add_argument("--limit", type=int, default=0, help="only the first N ERR programs")
    a = p.parse_args(argv)
    ids_path, run = V.SETS[a.set]
    recs = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines() if x]
    ids = V.load_ids(ids_path)
    err = [i for i, r in enumerate(recs) if r["outcome"] == "ERR"]
    if a.limit:
        err = err[:a.limit]
    out = RUNS / f"w2-{a.set}-loc"
    specs = inputs.from_token_ids([ids[i] for i in err], V.ROOT / "corpus" / "vocab.json",
                                  out / "inputs")
    for spec, i in zip(specs, err):          # keep the original indices
        spec.index = i
    driver.locate_error = patient_locate_error
    res = run_programs.run_specs(specs, out, workers=1, shots=0, title=f"{a.set} relocated")
    moved = sum(1 for r in res if r.get("error_offset") != recs[r["index"]].get("error_offset"))
    same = sum(1 for r in res if r["label"] == recs[r["index"]]["label"])
    print(f"{len(res)} ERR programs re-run: same outcome {same}, error position changed {moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
