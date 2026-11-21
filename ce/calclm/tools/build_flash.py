#!/usr/bin/env python3
"""Build CLMCODE: a flash-resident, position-independent copy of the core's assembly routines
(the two M2 matvec kernels and ce/calclm/core/ez80/*.s), executed in place from an archived
AppVar. On the CE, code fetched from flash runs about 2 cycles per byte (serial-flash cache)
against 4 from RAM, so the same routines run faster there (ce/BENCH.md).

All mutable state of the flash copy (its statics, the kernels' variables and the jump
trampolines) lives in a fixed RAM window FRAM .. FRAM + 512 (VRAM below the visible screen,
see src/memmap.h). The loader (core/clm_code.c) checks the payload, writes `jp <flash
address>` into every trampoline and redirects the RAM routines' patchable entry jumps.

Payload (little-endian):
    0  "CC", u8 format (1), u8 entry count E, u16 trampoline count T, u24 code length,
       u32 build id (must equal CLM_CODE_ID compiled into core.o)
    13 E x u24 entry offsets (order: ENTRIES), T x (u24 RAM trampoline, u24 code offset), code

    python build_flash.py <out_dir> <build_id> [--shape clm_shape.h]   -> out_dir/clmcode.bin
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALCLM = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CALCLM.parent / "kernel"))
import gen_kernels  # noqa: E402
import gen_tmv  # noqa: E402

FRAM = 0xD52A00               # also CLM_FRAM in core/clm_code.h and MM_FRAM in src/memmap.h
FRAM_SIZE = 512
KVARS = {"d": FRAM, "h": FRAM + 16}      # 12 bytes of kernel variables each
FORMAT = 1
ENTRIES = ["_clm_a_quant", "_clm_a_scale", "_clm_a_mulnorm", "_clm_a_t3", "_clm_a_maxabs32",
           "_clm_a_scores", "_clm_a_softmax", "_clm_a_pv", "_clm_a_havg", "_clm_a_silu",
           "_clm_a_maxabs24", "_clm_a_compact", "_clm_a_emb", "_tmv_d_build", "_tmv_d_rows",
           "_tmv_h_build", "_tmv_h_rows"]
SOURCES = ["clm_quant.s", "clm_scale.s", "clm_vec.s", "clm_attn.s", "clm_misc.s"]
BU = Path.home() / "CEdev" / "binutils" / "bin"

LINK = f"""SECTIONS
{{
    . = 0;
    .text : {{ *(.text .text.*) }}
    .clmtab 0x400000 : {{ *(.clmtab) }}
    . = {FRAM + 32:#x};
    .bss (NOLOAD) : {{ *(.bss .bss.*) *(.clmtramp) }}
    /DISCARD/ : {{ *(.comment) *(.note*) *(.debug*) }}
}}
"""


def run(*cmd):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"{' '.join(map(str, cmd))}\n{res.stdout}{res.stderr}")
    return res.stdout


def kernel_source(G: int, tag: str) -> str:
    src = gen_tmv.generate(G, 0, "pop", var_base=KVARS[tag])
    for old, new in {"tmv_build": f"_tmv_{tag}_build", "tmv_rows": f"_tmv_{tag}_rows",
                     "tmv_end": f"_tmv_{tag}_end"}.items():
        src = re.sub(rf"\b{old}\b", new, src)
    for var, _ in gen_tmv.VARS:
        src = re.sub(rf"\b{var}\b", f"{var}_{tag}", src)
    return re.sub(r"(\.L\w+)", rf"\1_{tag}", src).replace(
        "\t.section .text", f"\t.section .text.tmv_{tag}")


def build(out: Path, build_id: int, shape_h: Path = CALCLM / "core" / "clm_shape.h") -> bytes:
    obj = out / "flash"
    obj.mkdir(parents=True, exist_ok=True)
    shape = gen_kernels.shape(shape_h)
    gd, gh = -(-shape["CLM_DIM"] // 5), -(-shape["CLM_HIDDEN"] // 5)
    srcs = []
    for tag, G in (("d", gd), ("h", gh)):
        p = obj / f"tmv_{tag}.s"
        p.write_text(kernel_source(G, tag))
        srcs.append(p)
    srcs += [CALCLM / "core" / "ez80" / s for s in SOURCES]
    objs = []
    for s in srcs:
        placed = obj / (s.stem + "_flash.s")
        run(sys.executable, HERE / "asm_place.py", "flash", s, "-o", placed, "--tag", s.stem)
        o = obj / (s.stem + ".o")
        run(BU / "z80-none-elf-as", "-march=ez80+full", placed, "-o", o)
        objs.append(o)
    (obj / "flash.ld").write_text(LINK)
    elf = obj / "clmcode.elf"
    run(BU / "z80-none-elf-ld", "-static", "--no-warn-rwx-segments", "--no-warn-execstack",
        "-T", obj / "flash.ld", *objs, "-o", elf)
    run(BU / "z80-none-elf-objcopy", "-O", "binary", "-j", ".text", elf, obj / "code.bin")
    run(BU / "z80-none-elf-objcopy", "-O", "binary", "-j", ".clmtab", elf, obj / "tab.bin")
    code = (obj / "code.bin").read_bytes()
    tab = (obj / "tab.bin").read_bytes() if (obj / "tab.bin").exists() else b""
    syms = {}
    for line in run(BU / "z80-none-elf-nm", elf).splitlines():
        parts = line.split()
        if len(parts) == 3:
            syms[parts[2]] = int(parts[0], 16)
    bss_end = max(v for k, v in syms.items() if k.startswith("__tr_")) + 4
    if bss_end > FRAM + FRAM_SIZE:
        sys.exit(f"flash RAM window overflow: {bss_end:#x}")
    pairs = [(int.from_bytes(tab[i:i + 3], "little"), int.from_bytes(tab[i + 3:i + 6], "little"))
             for i in range(0, len(tab), 6)]
    for ram, off in pairs:
        if not FRAM <= ram < FRAM + FRAM_SIZE or off >= len(code):
            sys.exit(f"bad trampoline {ram:#x} -> {off:#x}")
    head = (b"CC" + bytes([FORMAT, len(ENTRIES)]) + len(pairs).to_bytes(2, "little")
            + len(code).to_bytes(3, "little") + build_id.to_bytes(4, "little"))
    head += b"".join(syms[e].to_bytes(3, "little") for e in ENTRIES)
    head += b"".join(r.to_bytes(3, "little") + o.to_bytes(3, "little") for r, o in pairs)
    return head + code


def appvar_8xv(name: str, data: bytes, comment: bytes = b"CalcLM") -> bytes:
    """An archived AppVar as a .8xv file (the same layout tivars writes)."""
    if len(data) > 65505 or len(name) > 8:
        raise ValueError("AppVar too large or name too long")
    size = len(data) + 2
    entry = (b"\x0d\x00" + size.to_bytes(2, "little") + b"\x15" + name.encode().ljust(8, b"\0")
             + b"\x00\x80" + size.to_bytes(2, "little") + len(data).to_bytes(2, "little") + data)
    head = b"**TI83F*\x1a\x0a\x13" + comment.ljust(42, b"\0") + len(entry).to_bytes(2, "little")
    return head + entry + (sum(entry) & 0xFFFF).to_bytes(2, "little")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("build_id", type=lambda s: int(s, 0))
    ap.add_argument("--appvar", type=Path, default=None, help="also write CLMCODE.8xv here")
    ap.add_argument("--shape", type=Path, default=CALCLM / "core" / "clm_shape.h",
                    help="model shape header the kernels are generated for")
    args = ap.parse_args(argv)
    data = build(args.out, args.build_id, args.shape)
    (args.out / "clmcode.bin").write_bytes(data)
    if args.appvar:
        args.appvar.parent.mkdir(parents=True, exist_ok=True)
        args.appvar.write_bytes(appvar_8xv("CLMCODE", data))
    print(f"clmcode={args.out / 'clmcode.bin'} bytes={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
