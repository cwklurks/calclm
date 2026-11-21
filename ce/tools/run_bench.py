"""Benchmark the ternary LUT matvec kernels in headless CEmu and project per-token cost.

    uv run python -m ce.tools.run_bench            # prints Markdown tables, writes ce/build/bench.json

Every number is an exact CEmu CPU cycle count (48 MHz), cold flash cache at the start of each
call, interrupts disabled. Rows and table build are separate calls (`build` once per input
vector, shared by every matrix that reads that vector).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from model.pack import lut_matvec, pack_ternary

from .bench import BUILD, CPU_HZ, Bench
from .tmv import run_asm, run_c

SHAPES = [(192, 192), (192, 512), (512, 192)]
# SPEC v0 (d=192, 6 layers, MQA head_dim 32, SwiGLU 512, vocab <= 1024): per layer
LAYER = {"q": (192, 192), "k": (192, 32), "v": (192, 32), "o": (192, 192),
         "w1": (192, 512), "w3": (192, 512), "w2": (512, 192)}
HEAD = (192, 1024)
N_LAYERS = 6
# distinct input vectors per layer (one table build each): x_attn (q,k,v), attn (o),
# x_ffn (w1,w3), h (w2); plus the final x for the head
BUILDS_PER_LAYER = {192: 3, 512: 1}


def case(n_in: int, n_out: int, seed: int = 0):
    rng = np.random.default_rng(seed + 7 * n_in + n_out)
    w = rng.integers(-1, 2, size=(n_out, n_in), dtype=np.int8)
    x = rng.integers(-127, 128, size=n_in, dtype=np.int8)
    return pack_ternary(w), x


def measure(bench: Bench, n_in: int, n_out: int, impl: str = "asm", **kw) -> dict:
    packed, x = case(n_in, n_out)
    res = (run_c(bench, packed, x, **kw) if impl == "c" else run_asm(bench, packed, x, **kw))
    if not np.array_equal(res.y, lut_matvec(packed, x)):
        raise AssertionError(f"kernel output mismatch for {impl} {n_in}->{n_out} {kw}")
    n_w = n_in * n_out
    return {"impl": impl, **kw, "n_in": n_in, "n_out": n_out, "G": res.G,
            "build": res.build.cycles, "rows": res.rows.cycles,
            "rows_dma": res.rows.dma_cycles, "build_dma": res.build.dma_cycles,
            "flash_misses": res.rows.flash_misses,
            "rows_per_weight": res.rows.cycles / n_w,
            "rows_per_byte": res.rows.cycles / (n_out * res.G),
            "total_per_weight": res.cycles / n_w,
            "build_per_group": res.build.cycles / res.G,
            "ms": res.cycles / CPU_HZ * 1e3}


def projection(bench: Bench, code: str, weights: str = "flash") -> dict:
    rows = {name: measure(bench, *shape, code=code, weights=weights)
            for name, shape in {**LAYER, "head": HEAD}.items()}
    build = {n_in: measure(bench, n_in, 2, code=code, weights=weights)["build"]
             for n_in in BUILDS_PER_LAYER}
    layer_rows = sum(rows[n]["rows"] for n in LAYER)
    layer_build = sum(build[n_in] * k for n_in, k in BUILDS_PER_LAYER.items())
    total_rows = N_LAYERS * layer_rows + rows["head"]["rows"]
    total_build = N_LAYERS * layer_build + build[192]
    n_weights = N_LAYERS * sum(a * b for a, b in LAYER.values()) + HEAD[0] * HEAD[1]
    total = total_rows + total_build
    return {"code": code, "weights": weights, "lcd_dma": bench.lcd_dma,
            "per_matrix_rows": {n: r["rows"] for n, r in rows.items()},
            "build_192": build[192], "build_512": build[512],
            "layer_rows": layer_rows, "layer_build": layer_build,
            "token_rows": total_rows, "token_build": total_build, "token_total": total,
            "token_ms": total / CPU_HZ * 1e3, "tokens_per_s": CPU_HZ / total,
            "n_weights": n_weights, "cycles_per_weight": total / n_weights}


def token_sim(bench: Bench, code: str, seed: int = 0) -> dict:
    """One token's worth of SPEC v0 matvecs, run back to back without resetting the emulator:
    all 43 matrices packed into distinct flash addresses (like the exported model), the kernels
    for G=39 and G=103 resident at once, the flash cache and the LCD's DMA phase carried over
    from call to call. Every output is checked against lut_matvec."""
    from . import tmv
    rng = np.random.default_rng(seed)
    bench.reset()
    kern = {}
    for i, G in enumerate((39, 103)):
        if code == "ram":
            k = tmv.asm_kernel(G, 0xD1B000 + 0x1800 * i)
        else:   # position-independent, executing in place from flash; vars/trampolines in RAM
            k = tmv.asm_kernel(G, tmv.FLASH_CODE_ORG + 0x2000 * i,
                               vars_addr=tmv.FLASH_VARS + 0x40 * i, pic=True)
        kern[G] = tmv.load_kernel(bench, k)
    addr = 0x100000                         # erased archive flash in the booted state
    mats = []
    for layer in range(N_LAYERS):
        for name, (n_in, n_out) in LAYER.items():
            mats.append((f"L{layer}.{name}", n_in, n_out))
    mats.append(("head", *HEAD))
    placed = {}
    for name, n_in, n_out in mats:
        p = pack_ternary(rng.integers(-1, 2, size=(n_out, n_in), dtype=np.int8))
        bench.write_flash(addr, p.tobytes())
        placed[name] = (addr, p)
        addr += p.size
    totals = {"build": 0, "rows": 0, "dma": 0, "misses": 0, "calls": 0}

    def build(n_in):
        x = rng.integers(-127, 128, size=n_in, dtype=np.int8)
        G = -(-n_in // 5)
        xp = np.zeros(5 * G, np.int8)
        xp[:n_in] = x
        bench.write_ram(tmv.X_ADDR, xp.tobytes())
        r = bench.call(kern[G]["tmv_build"], [tmv.X_ADDR, tmv.TBL_ADDR], flush_cache=False)
        totals["build"] += r.cycles
        totals["dma"] += r.dma_cycles
        totals["calls"] += 1
        return x

    def rows(name, x):
        a, p = placed[name]
        G = p.shape[1]
        r = bench.call(kern[G]["tmv_rows"], [a, tmv.Y_ADDR, p.shape[0], tmv.TBL_ADDR],
                       flush_cache=False)
        y = tmv._read_y(bench, p.shape[0])
        if not np.array_equal(y, lut_matvec(p, x)):
            raise AssertionError(f"token sim mismatch at {name}")
        totals["rows"] += r.cycles
        totals["dma"] += r.dma_cycles
        totals["misses"] += r.flash_misses
        totals["calls"] += 1

    for layer in range(N_LAYERS):
        x = build(192)
        for m in ("q", "k", "v"):
            rows(f"L{layer}.{m}", x)
        x = build(192)
        rows(f"L{layer}.o", x)
        x = build(192)
        for m in ("w1", "w3"):
            rows(f"L{layer}.{m}", x)
        x = build(512)
        rows(f"L{layer}.w2", x)
    x = build(192)
    rows("head", x)
    total = totals["build"] + totals["rows"]
    n_weights = N_LAYERS * sum(a * b for a, b in LAYER.values()) + HEAD[0] * HEAD[1]
    return {"code": code, "lcd_dma": bench.lcd_dma, **totals, "total": total,
            "ms": total / CPU_HZ * 1e3, "tokens_per_s": CPU_HZ / total,
            "weight_bytes": addr - 0x100000, "n_weights": n_weights,
            "cycles_per_weight": total / n_weights}


def md_table(results: list[dict]) -> str:
    head = ("| shape | code | weights | LCD DMA | build cyc | rows cyc | rows c/weight | "
            "rows c/byte | total c/weight | ms @48MHz |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in results:
        lines.append(
            f"| {r['n_in']}->{r['n_out']} | {r.get('code', 'ram')} | {r.get('weights', 'flash')} | "
            f"{'on' if r['lcd_dma'] else 'off'} | {r['build']:,} | {r['rows']:,} | "
            f"{r['rows_per_weight']:.2f} | {r['rows_per_byte']:.1f} | {r['total_per_weight']:.2f} | "
            f"{r['ms']:.2f} |")
    return "\n".join(lines)


def main() -> int:
    out: dict = {"hw": None, "shapes": [], "variants": [], "projection": [], "token_sim": []}
    for lcd_dma in (False, True):
        bench = Bench(lcd_dma=lcd_dma)
        out["hw"] = bench.hw_info()
        for code in ("ram", "flash"):
            for weights in ("flash", "ram"):
                for n_in, n_out in SHAPES:
                    r = measure(bench, n_in, n_out, code=code, weights=weights)
                    r["lcd_dma"] = lcd_dma
                    out["shapes"].append(r)
            p = projection(bench, code)
            out["projection"].append(p)
            out["token_sim"].append(token_sim(bench, code))
        for n_in, n_out in SHAPES:
            for impl, kw in (("c", {}), ("asm", {"variant": "de"}), ("asm", {"variant": "pop"})):
                r = measure(bench, n_in, n_out, impl=impl, **kw)
                r["lcd_dma"] = lcd_dma
                out["variants"].append(r)
    (BUILD / "bench.json").write_text(json.dumps(out, indent=1, default=str))

    print("## Hardware (CEmu, booted home screen)\n")
    print(json.dumps(out["hw"], default=str))
    print("\n## Shapes\n")
    print(md_table(out["shapes"]))
    print("\n## Implementations (code in RAM, weights in flash)\n")
    print("| shape | impl | LCD DMA | build cyc | build/group | rows cyc | rows c/weight |\n"
          "|---|---|---|---|---|---|---|")
    for r in out["variants"]:
        name = "C (ez80-clang -O2)" if r["impl"] == "c" else f"asm {r['variant']}"
        print(f"| {r['n_in']}->{r['n_out']} | {name} | {'on' if r['lcd_dma'] else 'off'} | "
              f"{r['build']:,} | {r['build_per_group']:.0f} | {r['rows']:,} | "
              f"{r['rows_per_weight']:.2f} |")
    print("\n## SPEC v0 per-token matvec projection (weights in flash)\n")
    print("| code | LCD DMA | build 192 | build 512 | layer rows | layer build | token rows | "
          "token build | token total | ms/token | tokens/s | c/weight |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in out["projection"]:
        print(f"| {p['code']} | {'on' if p['lcd_dma'] else 'off'} | {p['build_192']:,} | "
              f"{p['build_512']:,} | {p['layer_rows']:,} | {p['layer_build']:,} | "
              f"{p['token_rows']:,} | {p['token_build']:,} | {p['token_total']:,} | "
              f"{p['token_ms']:.0f} | {p['tokens_per_s']:.2f} | {p['cycles_per_weight']:.2f} |")
    print("\n## Measured token: all 43 SPEC v0 matvecs back to back (no resets, weights in flash)\n")
    print("| code | LCD DMA | build cyc | rows cyc | total cyc | DMA stall | flash misses | ms/token "
          "| tokens/s | c/weight |\n|---|---|---|---|---|---|---|---|---|---|")
    for t in out["token_sim"]:
        print(f"| {t['code']} | {'on' if t['lcd_dma'] else 'off'} | {t['build']:,} | {t['rows']:,} | "
              f"{t['total']:,} | {t['dma']:,} | {t['misses']:,} | {t['ms']:.0f} | "
              f"{t['tokens_per_s']:.2f} | {t['cycles_per_weight']:.2f} |")
    print(f"\nweights: {out['token_sim'][0]['n_weights']:,} ternary, "
          f"{out['token_sim'][0]['weight_bytes']:,} packed bytes")
    for p in out["projection"]:
        print(f"\nper-matrix rows cycles ({p['code']} code, LCD DMA {p['lcd_dma']}): "
              + ", ".join(f"{k} {v:,}" for k, v in p["per_matrix_rows"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
