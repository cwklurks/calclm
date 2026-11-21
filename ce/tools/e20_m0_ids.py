"""E20 M0's ids check (PLAN-E20 Measurements, M0): clm_bench --program's exact call order on rc2
(runs/e10/export/L4-s23), seed 20, with LD2 from its own fit, keeping the generated ids: the
cycles must equal clm_bench's record (22,505,577,473) and the ids CHECK.md's ids_sha256.

    python -m ce.tools.clm_bench_kv4 stage "E20 M0 ids" -- python -m ce.tools.e20_m0_ids OUT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from ce.calclm.tools.make_ld2 import payload
from ce.tools.clm import ROOT, EmuCore, Export

RC2 = ROOT / "runs" / "e10" / "export" / "L4-s23"
OFFSETS = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"
CHECK_IDS = "219af06a7ca3975814185c646a7ca9b5a3882e65d498b26ac87b5db9299aad9e"
CHECK_CYCLES = 22_505_577_473


def main(out: str) -> int:
    t0 = time.time()
    ex = Export.load(RC2)
    core = EmuCore(ex, lcd_dma=True, flash=True, horizons=True)
    assert core.load()[0] == 0                       # reset, binary, AppVars, LOAD, ATTACH, HZ
    assert core.program_decoder(payload(ex.path, OFFSETS)) == 0
    assert core.reset(20, 205, 20, cap=1024, masks=True) == 0
    t1 = time.time()
    toks = core.generate(1024)
    t2 = time.time()
    cycles = core.last.cycles
    ids = b"".join(int(t).to_bytes(2, "little") for t in [1] + toks)
    rec = {"tokens": len(toks), "cycles": cycles, "ids_sha256": hashlib.sha256(ids).hexdigest(),
           "cycles_match": cycles == CHECK_CYCLES,
           "ids_match": hashlib.sha256(ids).hexdigest() == CHECK_IDS,
           "flash_accesses": core.last.flash_accesses, "flash_misses": core.last.flash_misses,
           "generate_wall_s": t2 - t1, "total_wall_s": t2 - t0,
           "emulated_cycles_per_wall_s": cycles / (t2 - t1),
           "x_real_time_at_48MHz": cycles / (t2 - t1) / 48e6}
    Path(out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0 if rec["cycles_match"] and rec["ids_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
