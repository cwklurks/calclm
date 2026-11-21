"""Cycle benchmark of the CalcLM core in CEmu: cycles per token by position, broken down by phase.

    uv run python -m ce.tools.clm_bench                  # positions 0 64 128 255, LCD DMA on
    uv run python -m ce.tools.clm_bench --pos 0 --no-lcd
    uv run python -m ce.tools.clm_bench --v2 --flash     # format 2: past the window, + masks
    uv run python -m ce.tools.clm_bench --v2 --flash --program 89 --cap 1024 --masks
                                                         # one whole generation, measured
    uv run python -m ce.tools.clm_bench --v2 --flash --horizons ...
                                                         # with ALiBi horizons (PLAN-v2.1.md E1)
    uv run python -m ce.tools.clm_bench --v2 --flash --export runs/e7/export/W-s11 \\
        --mask-samples runs/e7/s11/W/samples.jsonl      # the wide model W (v2.3), its core

Method (emulator instrumentation only, like a debugger): the exact test binary of the tests
runs one STEP (forward pass at position `pos`, then sampling). The KV cache rows before `pos`
hold random int8 values: costs do not depend on the values (the kernels are data-independent;
attention loops run pos+1 times whatever the scores are). Totals come from a run without
hooks; the breakdown from a second run where the core's profiling hook HALTs at the start of
each phase and the harness reads CEmu's cycle counter and resumes. The hook's own cost is
measured and removed. 48 MHz, LCD DMA on (8bpp not modeled: the booted 16bpp home screen
DMA is the pessimistic case), flash cache warm as in steady generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bench import CPU_HZ
from .clm import (
    BLK_ADDR, EXPORT_DIR, FORWARD, KC_ADDR, KROW, MASK, ROOT, STEP, TRACK, TRACK_IN,
    V2_EXPORT, EmuCore, Export, out_dir, shape_tag,
)

PHASES = ["embed", "norm", "build", "rows", "scale_out", "scores", "softmax", "pv", "quant",
          "ffn_act", "sample", "end", "mask"]
GROUPS = {
    "matvec (kernel build+rows)": ["build", "rows"],
    "scale_out (dequant)": ["scale_out"],
    "attention (scores, softmax, PV)": ["scores", "softmax", "pv"],
    "norms + requant": ["norm", "quant"],
    "FFN SiLU/w3 path": ["ffn_act"],
    "embedding": ["embed"],
    "sampling": ["sample"],
}


def prepare(core: EmuCore, seed: int = 0) -> None:
    """Load, reset, and fill the KV cache with random history (costs do not depend on it)."""
    assert core.load()[0] == 0
    assert core.reset(seed=1, temperature_q8=205, top_k=20) == 0   # the default cap
    rng = np.random.default_rng(seed)
    n = core.ex.n_layers * 256 * 32
    core.bench.write_ram(KC_ADDR, rng.integers(0, 256, n * KROW // 32, dtype=np.uint8).tobytes())
    core.bench.write_ram(core.vc_addr, rng.integers(0, 256, n, dtype=np.uint8).tobytes())


def run(core: EmuCore, cmd: int, pos: int, profile: bool) -> tuple[int, dict[str, int], int]:
    """One FORWARD at `pos` (token 60) or one STEP (forward at `pos` + sampling).
    Returns (total cycles, cycles per phase, number of hook calls)."""
    b = core.bench
    core.set_prof(profile)
    if cmd == STEP:
        assert core.seek(pos) == 0
        core.blk.clear(STEP)
    else:
        core.blk.clear(FORWARD)
        core.blk.put(4, 60, 2)
        core.blk.put(6, pos, 2)              # logits pointer 0: no copy-out
    b.write_ram(BLK_ADDR, bytes(core.blk.buf))
    phase_addr = core.syms["drv_phase"]
    stats, done = b.start(core.syms["drv_cmd"], [BLK_ADDR], flush_cache=False,
                          max_cycles=3_000_000_000)
    total, phases, hooks = stats[0], {p: 0 for p in PHASES}, 0
    while not done:
        cur = PHASES[b.read_ram(phase_addr, 1)[0]]
        hooks += 1
        stats, done = b.resume(max_cycles=3_000_000_000)
        total += stats[0]
        phases[cur] += stats[0]
    return total, phases, hooks


def measure(core: EmuCore, pos: int, sample_cycles: int) -> dict:
    """Forward pass at `pos` (plain run for the total, hooked run for the breakdown) plus the
    sampling cost (position-independent, measured once)."""
    prepare(core)
    fwd, _, _ = run(core, FORWARD, pos, profile=False)
    prepare(core)
    _, phases, hooks = run(core, FORWARD, pos, profile=True)
    phases["sample"] = sample_cycles
    total = fwd + sample_cycles
    return {"pos": pos, "cycles": total, "forward": fwd, "ms": total / CPU_HZ * 1e3,
            "tokens_per_s": CPU_HZ / total, "hooks": hooks, "phases_raw": phases}


def sampling_cost(core: EmuCore) -> int:
    prepare(core)
    step, _, _ = run(core, STEP, 0, profile=False)
    prepare(core)
    fwd, _, _ = run(core, FORWARD, 0, profile=False)
    return step - fwd


def mask_cost(core: EmuCore, programs: list[list[int]]) -> dict:
    """Cycles per token of the masks (MASK) and the tracker reading the token (TRACK) over
    every position of `programs` (lists of ids from <bos>)."""
    assert core.load()[0] == 0
    masks, tracks = [], []
    for prog in programs:
        core.track([], reset=True)
        for tok in prog[1:]:
            core.blk.clear(MASK)                     # out NULL: the core's work only
            core._call()
            masks.append(core.last.cycles)
            core.bench.write_ram(TRACK_IN, int(tok).to_bytes(2, "little"))
            core.blk.clear(TRACK)
            core.blk.put(4, 1, 2)
            core.blk.ptr(8, TRACK_IN)
            core._call()
            tracks.append(core.last.cycles)
    both = np.array(masks) + np.array(tracks)
    stat = lambda a: {"mean": float(np.mean(a)), "p99": float(np.percentile(a, 99)),  # noqa: E731
                      "max": int(np.max(a))}
    return {"positions": len(both), "programs": len(programs), "mask": stat(masks),
            "track": stat(tracks), "total": stat(both)}


def program_cost(core: EmuCore, seed: int, cap: int, masks: bool, tq: int = 205,
                 top_k: int = 20, ld2_payload: bytes | None = None) -> dict:
    """One whole generation from <bos> (the driver's GENERATE: forward pass, masks, sampling
    and tracker for every token), measured in one run: cycles per generated token."""
    assert core.load()[0] == 0
    decoder_cycles = None
    if ld2_payload is not None:
        assert core.program_decoder(ld2_payload) == 0
        decoder_cycles = core.last.cycles
    assert core.reset(seed, tq, top_k, cap=cap, masks=masks) == 0
    toks = core.generate(cap)
    cycles = core.last.cycles
    return {"seed": seed, "cap": cap, "masks": masks, "tokens": len(toks),
            "ends_with_eos": bool(toks and toks[-1] == 2), "cycles": cycles,
            "cycles_per_token": cycles / len(toks), "tokens_per_s": CPU_HZ * len(toks) / cycles,
            "minutes": cycles / CPU_HZ / 60,
            "decoder_load_cycles": decoder_cycles}


def report(rows: list[dict]) -> str:
    lines = ["| position | cycles/token | ms @ 48 MHz | tokens/s | " +
             " | ".join(GROUPS) + " |", "|" + "---|" * (4 + len(GROUPS))]
    for r in rows:
        raw = r["phases_raw"]
        fwd_raw = sum(v for k, v in raw.items() if k != "sample")
        scale = r["forward"] / max(fwd_raw, 1)              # hook overhead spread proportionally
        raw = {k: (v * scale if k != "sample" else v) for k, v in raw.items()}
        scale = 1.0
        cells = [f"{sum(raw[p] for p in ps) * scale / 1e6:.2f} M" for ps in GROUPS.values()]
        lines.append(f"| {r['pos']} | {r['cycles']:,} | {r['ms']:.0f} | {r['tokens_per_s']:.3f} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pos", type=int, nargs="*", default=[0, 64, 128, 255])
    ap.add_argument("--no-lcd", action="store_true", help="LCD DMA off")
    ap.add_argument("--flash", action="store_true",
                    help="attach the flash-resident routines (CLMCODE) as the program does")
    ap.add_argument("--json", default=None)
    ap.add_argument("--v2", action="store_true",
                    help="the format 2 export (runs/v2/export): positions past the window "
                         "and the masks' cost on the shipped model's masked samples")
    ap.add_argument("--mask-programs", type=int, default=20)
    ap.add_argument("--program", type=int, default=None, metavar="SEED",
                    help="measure one whole generation with this seed instead (T 0.8, top-k 20)")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--horizons", action="store_true",
                    help="ALiBi horizons on (format 2; PLAN-v2.1.md E1)")
    ap.add_argument("--ld2", action="store_true", help="LD2 rules 1-7 for --program")
    ap.add_argument("--ld2-offsets", type=Path,
                    default=ROOT / "ce/tests/fixtures/ld2_offsets_full.json")
    ap.add_argument("--mask-samples", type=Path,
                    default=ROOT / "runs" / "lab" / "S0-s10" / "samples.jsonl",
                    help="masked samples of the shipped v2 model (G2b's S0 seed 10)")
    ap.add_argument("--export", type=Path, default=None,
                    help="with --v2: this format 2 export instead of runs/v2/export, e.g. "
                         "the wide model W (the core is built for its shape)")
    args = ap.parse_args(argv)
    if args.ld2 and not (args.v2 and args.horizons and args.masks and args.program is not None):
        ap.error("--ld2 needs --v2 --horizons --masks --program SEED")
    if args.export is not None and not args.v2:
        ap.error("--export needs --v2")
    ex = Export.load(args.export or (V2_EXPORT if args.v2 else EXPORT_DIR))
    pos = args.pos if args.pos != [0, 64, 128, 255] or not args.v2 else \
        [0, 128, 255, 256, 300, 600, 1000]
    core = EmuCore(ex, lcd_dma=not args.no_lcd, flash=args.flash, horizons=args.horizons)
    if args.program is not None:
        ld2_payload = None
        if args.ld2:
            from ce.calclm.tools.make_ld2 import payload
            ld2_payload = payload(ex.path, args.ld2_offsets)
        res = {"export_sha256": ex.meta["sha256"], "lcd_dma": not args.no_lcd, "flash": args.flash,
               "horizons": args.horizons, "ld2": args.ld2,
               **program_cost(core, args.program, args.cap, args.masks, ld2_payload=ld2_payload)}
        print(json.dumps(res, indent=1))
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
        return
    samp = sampling_cost(core)
    rows = [measure(core, p, samp) for p in pos]
    print(report(rows))
    out = {"lcd_dma": not args.no_lcd, "flash": args.flash, "horizons": args.horizons,
           "export": str(ex.path), "export_sha256": ex.meta["sha256"], "rows": rows}
    if args.v2:
        progs = [json.loads(line)["ids"] for line in args.mask_samples.read_text().splitlines()]
        out["masks"] = mask_cost(core, progs[: args.mask_programs])
        print(json.dumps(out["masks"], indent=1))
    name = args.json or str(out_dir(shape_tag(ex)) / ("clm_bench" + ("_v2" if args.v2 else "") +
                                   ("_hz" if args.horizons else "") + ".json"))
    path = name.replace(".json", "_flash.json") if args.flash else name
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
