"""E20 spike measurements in CEmu (PLAN-E20.md Measurements M1-M3), one job per process; the
parallel driver and the emulator lock are in ce/tools/clm_bench_kv4.py.

Every core runs as the calculator does: 48 MHz, LCD DMA on, CLMCODE attached, the flash cache
not flushed between calls (EmuCore's calls). Jobs:

    m1 CFG HZ SEQ      teacher-forced FORWARD over sequence A (rc2's check ids without the final
                       <eos>: the 316 forward passes of that program) or B (1,024 tokens): cycles,
                       flash accesses and misses per position
    m2 CFG SEED        one whole generation at CHECK.md's settings (LD2: rc2's own fit through
                       the program's load policy; the kv4 builds rc2's offsets, borrowed, through
                       the driver's E command): cycles, tokens, ids hash
    m3 CFG             sequence B fed to position 600; at 0, 128, 255, 256 and 600 the plain
                       FORWARD (total, flash stats) and the hooked one (cycles and flash stats per
                       profiler phase); kv4 builds also time KV4_UNPACK of every layer at the full
                       window, with the build's unpack and the C one
    m3ld2 CFG LD2      a hooked whole generation of seed 20 with LD2 on or off: per-token cycles
                       of the mask and sample phases (tracker, masks, LD2 rules, sampling)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ce.tools.clm import BLK_ADDR, FORWARD, GENERATE, OUT_TOK, ROOT, Export

EXPORTS = {"W8-s10": "runs/e16/export/W8-s10", "W8-s11": "runs/e16/export/W8-s11",
           "L4-s10": "runs/e15/export/L4-s10", "rc2": "runs/e10/export/L4-s23",
           "v2": "runs/v2/export"}
# name: (export, kv4, ram tables)
CONFIGS = {"L4i8": ("L4-s10", False, False), "L4k4": ("L4-s10", True, False),
           "W8k4A": ("W8-s10", True, False), "W8k4B": ("W8-s10", True, True),
           "W8s11k4A": ("W8-s11", True, False), "rc2": ("rc2", False, False),
           "v2": ("v2", False, False)}
RC2_FIT = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"
V2_FIT = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_full.json"
PHASES = ["embed", "norm", "build", "rows", "scale_out", "scores", "softmax", "pv", "quant",
          "ffn_act", "sample", "end", "mask", "kv4"]
M3_POSITIONS = [0, 128, 255, 256, 600]


def make_core(cfg: str, horizons: bool):
    from ce.tools.clm_kv4 import EmuInt8Core, EmuKv4Core, Kv4Export
    name, kv4, ram = CONFIGS[cfg]
    path = ROOT / EXPORTS[name]
    if kv4:
        return EmuKv4Core(Kv4Export.load(path), lcd_dma=True, flash=True, horizons=horizons,
                          ram_tables=ram)
    return EmuInt8Core(Export.load(path), lcd_dma=True, flash=True, horizons=horizons)


def sequence(name: str) -> list[int]:
    from ce.tests.kv4_cases import sequence_a, sequence_b
    if name == "A":
        a = sequence_a()
        return a[:-1] if a[-1] == 2 else a
    return sequence_b()


def _ld2_on(core, cfg: str) -> None:
    """LD2 as M0 (rc2: its own CLMLD2 through the program's load policy; v2: v2.2's) or, for the
    kv4 builds, rc2's offsets borrowed through the driver's E command."""
    from ce.calclm.tools.make_ld2 import payload
    name = CONFIGS[cfg][0]
    if name in ("rc2", "v2"):
        assert core.program_decoder(payload(core.ex.path, RC2_FIT if name == "rc2" else V2_FIT)) == 0
    else:
        assert core.load_E(json.loads(RC2_FIT.read_text())["E"]) == 0
        assert core.set_ld2(True) == 0


def job_m1(cfg: str, horizons: bool, seq: str) -> dict:
    core = make_core(cfg, horizons)
    toks = sequence(seq)
    assert core.load()[0] == 0
    cyc, acc, miss = [], [], []
    t0 = time.time()
    for pos, tok in enumerate(toks):
        core.forward(tok, pos, copy=False)
        cyc.append(core.last.cycles)
        acc.append(core.last.flash_accesses)
        miss.append(core.last.flash_misses)
    return {"cfg": cfg, "horizons": horizons, "seq": seq, "positions": len(toks),
            "cycles": cyc, "flash_accesses": acc, "flash_misses": miss,
            "mean_cycles": float(np.mean(cyc)), "wall_s": time.time() - t0}


def job_m2(cfg: str, seed: int) -> dict:
    core = make_core(cfg, True)
    assert core.load()[0] == 0
    _ld2_on(core, cfg)
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == 0
    t0 = time.time()
    toks = core.generate(1024)
    r = core.last
    ids = b"".join(int(t).to_bytes(2, "little") for t in [1] + toks)
    return {"cfg": cfg, "seed": seed, "tokens": len(toks), "ends_with_eos": bool(toks[-1] == 2),
            "cycles": r.cycles, "cycles_per_token": r.cycles / len(toks),
            "flash_accesses": r.flash_accesses, "flash_misses": r.flash_misses,
            "ids_sha256": hashlib.sha256(ids).hexdigest(), "wall_s": time.time() - t0}


def hooked(core, phases=PHASES) -> dict:
    """Run the command in core.blk with the profiling hook on: cycles and flash stats per phase
    (the hook HALTs at the start of each phase; the harness resumes, like a debugger)."""
    b = core.bench
    b.write_ram(BLK_ADDR, bytes(core.blk.buf))
    addr = core.syms["drv_phase"]
    stats, done = b.start(core.syms["drv_cmd"], [BLK_ADDR], flush_cache=False,
                          max_cycles=3_000_000_000 * 1100)
    out = {p: [0, 0, 0] for p in phases}
    out["_pre"] = [stats[0], stats[4], stats[5]]
    hooks = 0
    while not done:
        cur = phases[b.read_ram(addr, 1)[0]]
        hooks += 1
        stats, done = b.resume(max_cycles=3_000_000_000 * 1100)
        for i, k in enumerate((0, 4, 5)):
            out[cur][i] += stats[k]
    core.blk.buf[:] = b.read_ram(BLK_ADDR, len(core.blk.buf))
    return {"phases": {p: {"cycles": v[0], "flash_accesses": v[1], "flash_misses": v[2]}
                       for p, v in out.items()}, "hooks": hooks}


def job_m3(cfg: str) -> dict:
    core = make_core(cfg, True)
    toks = sequence("B")
    assert core.load()[0] == 0
    rows = []
    for pos in range(max(M3_POSITIONS) + 1):
        core.forward(toks[pos], pos, copy=False)
        if pos not in M3_POSITIONS:
            continue
        plain = core.last                     # re-running FORWARD at pos rewrites the same slot
        core.set_prof(True)
        core.blk.clear(FORWARD)
        core.blk.put(4, toks[pos], 2)
        core.blk.put(6, pos, 2)
        h = hooked(core)
        core.set_prof(False)
        rows.append({"pos": pos, "cycles": plain.cycles, "flash_accesses": plain.flash_accesses,
                     "flash_misses": plain.flash_misses, **h})
    rec = {"cfg": cfg, "horizons": True, "rows": rows}
    if CONFIGS[cfg][1]:
        unpack = {}
        for c_only in (False, True):
            per = []
            for layer in range(core.ex.n_layers):
                assert core.kv4_unpack(layer, 256, c_only) == 0
                per.append({"cycles": core.last.cycles, "flash_accesses": core.last.flash_accesses,
                            "flash_misses": core.last.flash_misses})
            unpack["c" if c_only else "build"] = per
        rec["unpack_full_window"] = unpack
    return rec


def job_m3ld2(cfg: str, ld2: bool) -> dict:
    core = make_core(cfg, True)
    assert core.load(prof=True)[0] == 0
    if ld2:
        _ld2_on(core, cfg)
    core.set_prof(False)
    assert core.reset(20, 205, 20, cap=1024, masks=True) == 0
    core.set_prof(True)
    core.blk.clear(GENERATE)
    core.blk.put(4, 1024, 2)
    core.blk.ptr(8, OUT_TOK)
    h = hooked(core)
    toks = []                                 # GENERATE's reply is in HL: count the output
    for i in range(1023):                     # cap 1024: at most 1,023 tokens after <bos>
        t = int.from_bytes(core.bench.read_ram(OUT_TOK + 2 * i, 2), "little")
        toks.append(t)
        if t == 2:
            break
    return {"cfg": cfg, "ld2": ld2, "seed": 20, "tokens": len(toks), **h}


JOBS = {"m1": job_m1, "m2": job_m2, "m3": job_m3, "m3ld2": job_m3ld2}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    kind, out, args = argv[0], Path(argv[1]), json.loads(argv[2])
    rec = JOBS[kind](*args)
    out.write_text(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
