#!/usr/bin/env python3
"""Generate the eZ80 (ADL mode) ternary LUT matvec kernel, GNU as syntax, for a fixed G.

Contract (SPEC.md): x is int8 padded to 5*G; packed weights W are n_out rows of G bytes,
b = sum_j (w_j + 1) * 3**j; y[r] = sum_g T_g[W[r][g]], T_g[b] = sum_j (digit_j(b) - 1) x[5g+j].

Table layout ("b-major", chosen for speed): one 64 KB-aligned bank U. The entry for group g and
packed byte b is the 16-bit little-endian value V = T_g[b] + 635 (always 0..1270) at
    lo: U:b:(C0 + 2g)     hi: U:b:(C0 + 2g + 1)
so a weight byte loaded into H addresses its entry directly (`ld h,e`) and L walks the
columns (`inc l`). 243 pages x (2G + 1) columns (+1: the build's 24-bit store spills one byte).
Rows start the accumulator at -635*G, so the 24-bit sum of biased entries is exact.

C ABI (CE toolchain): args on the stack (3 bytes each), IX/IY preserved. Both functions use SP
as a data register, so they disable interrupts on entry and restore the caller's state on exit.

    void tmv_build(const int8_t *x, uint8_t *tbl);              // tbl must be U:00:C0
    void tmv_rows(const uint8_t *W, int24_t *y, unsigned n_out, const uint8_t *tbl);

Variants for the row loop:
    pop  SP = weight pointer; `pop de; dec sp` yields two weight bytes for 2 opcode bytes.
    de   DE = weight pointer; `ld a,(de); inc de; ld h,a` per group (no SP tricks).
"""
from __future__ import annotations

import argparse
import sys

BIAS = 5 * 127  # max |T_g[b]| for int8 x in [-127, 127]


class Asm:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *ins: str) -> None:
        self.lines.extend("\t" + i for i in ins)

    def label(self, name: str) -> None:
        if Jumps.pic and name in TRAMPOLINES:
            self.lines.append(f"{name[2:]}:")        # exported: the loader patches T_* with it
        self.lines.append(f"{name}:")

    def comment(self, text: str) -> None:
        self.lines.append(f"\t; {text}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


VARS = [("v_sp", 3), ("v_cnt", 3), ("v_tail", 3), ("v_d2", 3)]   # RAM variables (12 bytes)
# PIC mode: long jumps go through 4-byte `jp nnnnnn` trampolines in RAM that the loader patches
# with (load address + label offset); label -> trampoline variable
TRAMPOLINES = {".Lbuild_group": "T_build_group", ".Lrow_iter": "T_row_iter",
               ".Lrows_tail": "T_rows_tail", ".Lrows_done": "T_rows_done"}
PIC_VARS = [(t, 4) for t in TRAMPOLINES.values()]


class Jumps:
    """Absolute jumps to code labels, or via RAM trampolines when the code is relocatable."""
    pic = False

    @classmethod
    def jp(cls, cond: str, label: str) -> str:
        target = TRAMPOLINES[label] if cls.pic else label
        return f"jp {cond}, {target}" if cond else f"jp {target}"


def trailing_zeros3(b: int) -> int:
    k = 0
    while b and b % 3 == 0:
        b //= 3
        k += 1
    return k


def gen_build(a: Asm, G: int, c0: int) -> None:
    """Descending build: V[242] = 635 + sum(x); V[b-1] = V[b] - d_k, k = trailing base-3
    zeros of b, d_k = x_k - 2*(x_0+..+x_{k-1}). All arithmetic is mod 2^16 (the table keeps
    16 bits; the upper byte of each 24-bit store spills into the next column)."""
    a.comment("void tmv_build(const int8_t *x, uint8_t *tbl)")
    a.label("tmv_build")
    a("ld a, i", "push af", "di")                  # P/V = IEF1; SP becomes a data register
    a("push ix", "push iy", "ld iy, 0", "add iy, sp")
    a("ld hl, (iy+15)", "ld iy, (iy+12)", "ld (v_sp), sp", f"ld a, {G}", "ld (v_cnt), a",
      "ld h, 242", f"ld l, {c0}")                  # only HLU (the bank) comes from tbl
    a.label(".Lbuild_group")
    a("exx")                                         # set B: arithmetic; HL (set A) = pointer
    a.comment("HL = 2*prefix; push -d_j = 2*prefix - x_j (d_2 goes to memory, it becomes SP)")
    a("or a, a", "sbc hl, hl")
    for j in range(5):
        a(f"ld a, (iy+{j})", "ld c, a", "add a, a", "sbc a, a", "ld b, a")  # BC = sext16(x_j)
        a("or a, a", "sbc hl, bc")
        a("ld (v_d2), hl" if j == 2 else "push hl")
        a("add hl, bc", "add hl, bc", "add hl, bc")
    a.comment("V[242] = 635 + (x0+..+x4) = 635 + HL/2")
    a("sra h", "rr l", f"ld bc, {BIAS}", "add hl, bc", "push hl", "pop ix")
    a("pop de", "pop bc", "exx", "pop de", "pop bc", "ld sp, (v_d2)", "lea iy, iy+5")
    a.comment("243 entries, b = 242 .. 0: 24-bit store (3rd byte spills into the next column)")
    add = {0: ["add ix, bc"], 1: ["add ix, de"], 2: ["add ix, sp"],
           3: ["exx", "add ix, bc", "exx"], 4: ["exx", "add ix, de", "exx"]}
    for b in range(242, -1, -1):
        a("ld (hl), ix", "dec h")
        if b > 0:
            a(*add[trailing_zeros3(b)])
    a("ld sp, (v_sp)", "ld h, 242", "inc l", "inc l",
      "ld a, (v_cnt)", "dec a", "ld (v_cnt), a", Jumps.jp("nz", ".Lbuild_group"))
    a("pop iy", "pop ix", "pop af", "ret po", "ei", "ret")


def slot(a: Asm, G: int, c0: int, r: int, g: int, src: str, variant: str) -> None:
    """One group lookup for row r (within the loop body), weight byte in register `src`."""
    init = (-BIAS * G) & 0xFFFFFF
    if g == 0:
        a(f"ld ix, {init:#08x}", f"ld l, {c0}")
    if variant == "de":
        a("ld a, (de)", "inc de", "ld h, a")
    else:
        a(f"ld h, {src}")
    a("ld c, (hl)", "inc l", "ld b, (hl)")
    if g < G - 1:
        a("inc l")
    a("add ix, bc")
    if g == G - 1:
        a(f"ld (iy+{3 * r}), ix")


def body_pop(a: Asm, G: int, c0: int, rows: int) -> None:
    slots = [(r, g) for r in range(rows) for g in range(G)]
    for p in range(0, len(slots) - 1, 2):
        a("pop de", "dec sp")
        slot(a, G, c0, *slots[p], "e", "pop")
        slot(a, G, c0, *slots[p + 1], "d", "pop")
    if len(slots) % 2:
        a("pop de", "dec sp", "dec sp")
        slot(a, G, c0, *slots[-1], "e", "pop")


def gen_rows(a: Asm, G: int, c0: int, variant: str) -> None:
    a.comment("void tmv_rows(const uint8_t *W, int24_t *y, unsigned n_out, const uint8_t *tbl)")
    a.label("tmv_rows")
    a("ld a, i", "push af", "di")                  # P/V = IEF1; SP walks the weights
    a("push ix", "push iy", "ld iy, 0", "add iy, sp", "ld (v_sp), sp")
    R = 2 if (variant == "pop" and G % 2) else 1   # rows per iteration (even number of slots)
    a("ld hl, (iy+18)")                            # n_out (<= 65280: outer count is 8-bit)
    if R == 2:
        a("ld a, l", "and a, 1", "ld (v_tail), a", "srl h", "rr l")
    a.comment("iterations N = HL: inner count = N mod 256 (0 means 256), outer = ceil(N/256)")
    a("ld a, l", "ld (v_cnt), a", "or a, a", "jr z, 1f", "inc h")
    a.label("1")
    a("ld a, h", "exx", "ld b, a", "exx")
    if variant == "pop":
        a("ld hl, (iy+12)", "ld sp, hl")           # SP walks the weights (interrupts off)
    else:
        a("ld de, (iy+12)")
    a("ld hl, (iy+21)", "ld iy, (iy+15)")          # HLU = table bank; IY = y
    a("exx", "ld a, b", "exx", "ld bc, 0", "or a, a", Jumps.jp("z", ".Lrows_tail"),
      "ld a, (v_cnt)")
    a.label(".Lrow_iter")
    if variant == "pop":
        body_pop(a, G, c0, R)
        a(f"lea iy, iy+{3 * R}", "dec a", Jumps.jp("nz", ".Lrow_iter"))
    else:
        a("ld (v_cnt), a")
        for g in range(G):
            slot(a, G, c0, 0, g, "", "de")
        a("lea iy, iy+3", "ld a, (v_cnt)", "dec a", Jumps.jp("nz", ".Lrow_iter"))
    a("exx", "dec b", "exx", Jumps.jp("nz", ".Lrow_iter"))
    a.label(".Lrows_tail")
    if R == 2:
        a("ld a, (v_tail)", "or a, a", Jumps.jp("z", ".Lrows_done"))
        body_pop(a, G, c0, 1)
    a.label(".Lrows_done")
    a("ld sp, (v_sp)", "pop iy", "pop ix", "pop af", "ret po", "ei", "ret")


def generate(G: int, c0: int = 0, variant: str = "pop", var_base: int | None = None,
             pic: bool = False) -> str:
    if not 1 <= G or c0 + 2 * G + 1 > 256:
        raise ValueError(f"G={G}, C0={c0}: need C0 + 2G + 1 <= 256")
    if variant not in ("pop", "de"):
        raise ValueError(variant)
    if pic and var_base is None:
        raise ValueError("position-independent code needs --vars (RAM for variables/trampolines)")
    Jumps.pic = pic
    a = Asm()
    a.lines += [f"; generated by ce/kernel/gen_tmv.py G={G} C0={c0} variant={variant} pic={pic}",
                "\t.assume adl=1", "\t.section .text", "\t.global tmv_build", "\t.global tmv_rows",
                "\t.global tmv_end"]
    if pic:
        a.lines += [f"\t.global {lab[2:]}" for lab in TRAMPOLINES]
        a.lines += [f"\t.global {t}" for t in TRAMPOLINES.values()]
    if var_base is not None:
        off = 0
        for name, size in VARS + (PIC_VARS if pic else []):
            a.lines.append(f"\t.equ {name}, {var_base + off:#08x}")
            off += size
    gen_build(a, G, c0)
    gen_rows(a, G, c0, variant)
    if var_base is None:
        for name, size in VARS:
            a.label(name)
            a(f".skip {size}")
    a.label("tmv_end")
    return a.text()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--groups", "-G", type=int, required=True)
    p.add_argument("--c0", type=int, default=0, help="first table column (low address byte)")
    p.add_argument("--variant", default="pop", choices=["pop", "de"])
    p.add_argument("--vars", type=lambda s: int(s, 0), default=None,
                   help="RAM address for variables (required when the code lives in flash)")
    p.add_argument("--pic", action="store_true",
                   help="position-independent: long jumps via RAM trampolines (flash-resident code)")
    p.add_argument("-o", "--output", default="-")
    args = p.parse_args(argv)
    src = generate(args.groups, args.c0, args.variant, args.vars, args.pic)
    if args.output == "-":
        sys.stdout.write(src)
    else:
        with open(args.output, "w") as f:
            f.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
