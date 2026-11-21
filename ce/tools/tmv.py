"""Run the ternary LUT matvec kernels (ce/kernel) inside the CEmu harness (ce/tools/bench.py).

    from ce.tools.bench import Bench
    from ce.tools.tmv import run_asm, run_c
    b = Bench()
    res = run_asm(b, packed, x)            # packed: (n_out, G) uint8 from model.pack.pack_ternary
    res.y, res.build.cycles, res.rows.cycles

Memory map (emulated RAM unless noted; the OS is suspended with interrupts disabled):
    0xD1B000  assembly kernel (generated for G)       0xD1E000  C reference kernel
    0xD1F000  x (int8, padded to 5G)                  0xD1F400  y (int24 per row)
    0xD30000  tables (asm: b-major bank 0xD3; C: int16[G][243])
    0x300000  packed weights in flash (archive area, erased in the booted state)
    0xD40000  packed weights in RAM (comparison runs)
    0x380000  flash-resident kernel (position-independent; run_asm(load_at=...) relocates it),
              its variables and jp trampolines in RAM at 0xD1AA00
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bench import BUILD, CE, Bench, BenchError, CallResult

KDIR = CE / "kernel"
KOUT = BUILD / "kernel"

ASM_ORG = 0xD1B000
C_ORG = 0xD1E000
X_ADDR = 0xD1F000
Y_ADDR = 0xD1F400
TBL_ADDR = 0xD30000
W_FLASH = 0x300000
W_RAM = 0xD40000
FLASH_CODE_ORG = 0x380000
FLASH_VARS = 0xD1AA00


@dataclass(frozen=True)
class Kernel:
    path: Path
    org: int
    code: bytes
    symbols: dict[str, int]

    def __getitem__(self, name: str) -> int:
        return self.symbols[name]


def _build(args: list[str]) -> Path:
    """Run ce/kernel/build.sh (cheap: generate + assemble + link); cached per process."""
    env = dict(os.environ, PYTHON=sys.executable, KERNEL_OUT=str(KOUT))
    out = subprocess.run([str(KDIR / "build.sh"), *args], check=True, capture_output=True,
                         text=True, env=env)
    return Path(out.stdout.strip().splitlines()[-1])


def _load(binpath: Path, org: int) -> Kernel:
    syms = {}
    for line in binpath.with_suffix(".sym").read_text().splitlines():
        addr, _, name = line.split()
        syms[name.lstrip("_")] = int(addr, 16)
    return Kernel(binpath, org, binpath.read_bytes(), syms)


_cache: dict[tuple, Kernel] = {}


def _hex(v: int) -> str:
    return f"{v:#X}".replace("0X", "0x")


def asm_kernel(G: int, org: int = ASM_ORG, variant: str = "pop", vars_addr: int | None = None,
               c0: int = 0, pic: bool = False) -> Kernel:
    key = ("asm", G, org, variant, vars_addr, c0, pic)
    if key not in _cache:
        args = ["asm", str(G), _hex(org), _hex(vars_addr) if vars_addr is not None else "",
                variant, str(c0), "1" if pic else "0"]
        _cache[key] = _load(_build(args), org)
    return _cache[key]


def c_kernel(org: int = C_ORG) -> Kernel:
    key = ("c", org)
    if key not in _cache:
        _cache[key] = _load(_build(["c", _hex(org)]), org)
    return _cache[key]


TRAMPOLINES = {"T_build_group": "build_group", "T_row_iter": "row_iter",
               "T_rows_tail": "rows_tail", "T_rows_done": "rows_done"}


def load_kernel(bench: Bench, k: Kernel, base: int | None = None) -> dict[str, int]:
    """Place a kernel and return its entry points. Position-independent kernels (they export
    T_* trampolines) may be loaded at any flash/RAM `base`: like the M4 loader would after
    finding its archived AppVar, write `jp base+offset` into each RAM trampoline."""
    base = k.org if base is None else base
    if base < 0xD00000:
        bench.write_flash(base, k.code)
    else:
        bench.write_ram(base, k.code)
    reloc = base - k.org
    if "T_row_iter" in k.symbols:
        for tramp, label in TRAMPOLINES.items():
            bench.write_ram(k[tramp], b"\xc3" + (k[label] + reloc).to_bytes(3, "little"))
    elif reloc:
        raise BenchError("only position-independent kernels can be relocated")
    return {name: addr + reloc for name, addr in k.symbols.items()}


@dataclass(frozen=True)
class TmvResult:
    y: np.ndarray        # int32 (sign-extended int24)
    build: CallResult    # table build (once per input vector)
    rows: CallResult     # row loop
    n_out: int
    G: int

    @property
    def cycles(self) -> int:
        return self.build.cycles + self.rows.cycles


def _pad_x(x: np.ndarray, G: int) -> bytes:
    x = np.asarray(x)
    if x.size > 5 * G:
        raise ValueError(f"x has {x.size} > 5G = {5 * G} entries")
    if x.size and (x.min() < -127 or x.max() > 127):
        raise ValueError("x must be int8 in [-127, 127] (SPEC)")
    xp = np.zeros(5 * G, dtype=np.int8)
    xp[: x.size] = x
    return xp.tobytes()


def _place_weights(bench: Bench, packed: np.ndarray, weights: str) -> int:
    data = np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    if weights == "flash":
        bench.write_flash(W_FLASH, data + b"\0\0")   # kernel may read 2 bytes past the end
        return W_FLASH
    if weights == "ram":
        bench.write_ram(W_RAM, data + b"\0\0")
        return W_RAM
    raise ValueError(weights)


def _read_y(bench: Bench, n_out: int) -> np.ndarray:
    raw = np.frombuffer(bench.read_ram(Y_ADDR, 3 * n_out), dtype=np.uint8).reshape(n_out, 3)
    v = raw[:, 0].astype(np.int32) | raw[:, 1].astype(np.int32) << 8 | raw[:, 2].astype(np.int32) << 16
    return np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.int32)


def table_ptr(base: int, c0: int = 0) -> int:
    """The kernel's tbl argument: a 64 KB-aligned bank plus the generation-time column C0
    (the kernel takes only the bank byte from it; a mismatched pointer is a caller bug)."""
    if base & 0xFFFF or not 0 <= c0 < 256:
        raise ValueError(f"table bank must be 64 KB aligned: {base:#x}, C0={c0}")
    return base | c0


def run_asm(bench: Bench, packed: np.ndarray, x: np.ndarray, weights: str = "flash",
            code: str = "ram", variant: str = "pop", load_at: int | None = None,
            c0: int = 0) -> TmvResult:
    """Run the assembly kernel: tmv_build(x, tbl) then tmv_rows(W, y, n_out, tbl).
    code="ram": kernel in RAM at ASM_ORG. code="flash": position-independent kernel in flash
    (linked at FLASH_CODE_ORG, loaded at `load_at` if given), variables and trampolines in RAM."""
    n_out, G = packed.shape
    if code == "ram":
        k = asm_kernel(G, ASM_ORG, variant, c0=c0)
    elif code == "flash":
        k = asm_kernel(G, FLASH_CODE_ORG, variant, vars_addr=FLASH_VARS, c0=c0, pic=True)
    else:
        raise ValueError(code)
    bench.reset()
    entry = load_kernel(bench, k, load_at)
    bench.write_ram(X_ADDR, _pad_x(x, G))
    w_addr = _place_weights(bench, packed, weights)
    tbl = table_ptr(TBL_ADDR, c0)
    rb = bench.call(entry["tmv_build"], [X_ADDR, tbl])
    rr = bench.call(entry["tmv_rows"], [w_addr, Y_ADDR, n_out, tbl])
    return TmvResult(_read_y(bench, n_out), rb, rr, n_out, G)


def run_c(bench: Bench, packed: np.ndarray, x: np.ndarray, weights: str = "flash") -> TmvResult:
    """Run the C reference: tmv_c_build(x, G, tbl) then tmv_c_rows(W, y, n_out, G, tbl)."""
    n_out, G = packed.shape
    k = c_kernel()
    if 486 * G > 0x10000:
        raise BenchError("C reference tables exceed the 64 KB table area")
    bench.reset()
    bench.write_ram(k.org, k.code)
    bench.write_ram(X_ADDR, _pad_x(x, G))
    w_addr = _place_weights(bench, packed, weights)
    rb = bench.call(k["tmv_c_build"], [X_ADDR, G, TBL_ADDR])
    rr = bench.call(k["tmv_c_rows"], [w_addr, Y_ADDR, n_out, G, TBL_ADDR])
    return TmvResult(_read_y(bench, n_out), rb, rr, n_out, G)
