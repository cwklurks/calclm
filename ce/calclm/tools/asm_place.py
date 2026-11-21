#!/usr/bin/env python3
"""Prepare the core's eZ80 assembly for one of two placements (sources stay unchanged).

ram:   the routines run where they are linked (inside the program, in RAM). Each global entry
       point gets a patchable `jp` to its body, so the loader (core/clm_code.c) can redirect
       the C calls to a flash-resident copy of the same routine.
flash: position-independent copy for the CLMCODE AppVar, executed in place from archive.
       Every `jp` to a label of the file goes through a 4-byte RAM trampoline (`jp nn`, patched
       by the loader once the AppVar's address is known). Numeric local labels (1:, 1f, 1b)
       are renamed to unique names first. The trampolines live in section .clmtramp and the
       (trampoline, target) pairs in section .clmtab (.d24 pairs), both read by
       tools/build_flash.py.

    python asm_place.py ram|flash in.s -o out.s [--tag NAME]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ENTRY = re.compile(r"^(_clm_a_\w+|_tmv_[dh]_\w+):\s*(;.*)?$")
LABEL = re.compile(r"^([.A-Za-z_][\w.]*):")
NUMLABEL = re.compile(r"^(\d+):")
JP = re.compile(r"^(\s+)jp\s+(?:(nz|z|nc|c|po|pe|p|m)\s*,\s*)?([^\s;()]+)\s*(;.*)?$")
NUMREF = re.compile(r"\b(\d+)([fb])\b")


def expand_includes(text: str, base: Path) -> str:
    out = []
    for line in text.splitlines():
        m = re.match(r'^\s*\.include\s+"([^"]+)"', line)
        if m:
            out.append(expand_includes((base / m.group(1)).read_text(), base))
        else:
            out.append(line)
    return "\n".join(out)


def macro_free(text: str) -> bool:
    return not re.search(r"^\s*\.(macro|irp)\b", text, re.M)


def rename_numeric(lines: list[str], tag: str) -> list[str]:
    """Give every numeric local label a unique name and resolve Nf / Nb references."""
    defs: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        m = NUMLABEL.match(line)
        if m:
            defs.setdefault(m.group(1), []).append(i)
    out = []
    for i, line in enumerate(lines):
        m = NUMLABEL.match(line)
        if m:
            k = defs[m.group(1)].index(i)
            line = f".Ln_{tag}_{m.group(1)}_{k}:" + line[m.end():]

        def ref(mm, i=i):
            num, d = mm.group(1), mm.group(2)
            if num not in defs:
                return mm.group(0)
            idx = defs[num]
            if d == "f":
                k = next(n for n, j in enumerate(idx) if j > i)
            else:
                k = max(n for n, j in enumerate(idx) if j < i)
            return f".Ln_{tag}_{num}_{k}"
        code = line.split(";", 1)
        code[0] = NUMREF.sub(ref, code[0]) if not NUMLABEL.match(code[0]) else code[0]
        out.append(";".join(code))
    return out


def place_ram(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        m = ENTRY.match(line)
        if m:
            name = m.group(1)
            out += [f"{name}:", f"\tjp\t.Lbody{name}\t\t; patchable entry (core/clm_code.c)",
                    f".Lbody{name}:"]
        else:
            out.append(line)
    return out


def place_flash(lines: list[str], tag: str) -> list[str]:
    labels = {LABEL.match(l).group(1) for l in lines if LABEL.match(l)}
    lines = rename_numeric(lines, tag)
    labels |= {LABEL.match(l).group(1) for l in lines if LABEL.match(l)}
    tramps: dict[str, str] = {}
    out = []
    for line in lines:
        m = JP.match(line)
        if m and m.group(3) in labels:
            target = m.group(3)
            t = tramps.setdefault(target, f"__tr_{tag}_{len(tramps)}")
            cond = f"{m.group(2)}, " if m.group(2) else ""
            out.append(f"{m.group(1)}jp\t{cond}{t}\t; -> {target} via RAM trampoline")
        else:
            out.append(line)
    if tramps:
        out.append('\t.section .clmtramp, "aw", @nobits')
        out += [f"{t}:\t.skip\t4" for t in tramps.values()]
        out.append('\t.section .clmtab, "a", @progbits')
        out += [f"\t.d24\t{t}, {target}" for target, t in tramps.items()]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["ram", "flash"])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    text = expand_includes(args.src.read_text(), args.src.parent)
    if args.mode == "flash":
        text = expand_macros(text)       # every jump becomes a line of its own
    lines = text.splitlines()
    tag = args.tag or re.sub(r"\W", "_", args.src.stem)
    out = place_ram(lines) if args.mode == "ram" else place_flash(lines, tag)
    args.output.write_text("\n".join(out) + "\n")
    return 0


def expand_macros(text: str) -> str:
    """Expand .macro/.endm (with \\param and \\@-free bodies) and .irp in Python, so every
    jump is visible as a line of its own."""
    macros: dict[str, tuple[list[str], list[str]]] = {}
    lines = text.splitlines()
    body: list[str] = []
    i = 0
    out_lines: list[str] = []
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*\.macro\s+(\w+)\s*(.*)$", line)
        if m:
            name, params = m.group(1), [p.strip() for p in re.split(r"[,\s]+", m.group(2)) if p.strip()]
            body = []
            i += 1
            while not re.match(r"^\s*\.endm\b", lines[i]):
                body.append(lines[i])
                i += 1
            macros[name] = (params, body)
            i += 1
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(_select_if(_expand(out_lines, macros)))


def _select_if(lines: list[str]) -> list[str]:
    """Resolve `.if 0` / `.if 1` ... `.else` ... `.endif` left by macro arguments."""
    out, stack = [], []
    for line in lines:
        code = line.split(";", 1)[0].strip()
        m = re.match(r"^\.if\s+(\d+)$", code)
        if m:
            stack.append(int(m.group(1)) != 0)
            continue
        if code == ".else" and stack:
            stack[-1] = not stack[-1]
            continue
        if code == ".endif" and stack:
            stack.pop()
            continue
        if all(stack):
            out.append(line)
    return out


def _split_labels(lines: list[str]) -> list[str]:
    """`label: instr` -> `label:` and `\tinstr` (macro calls may follow a label)."""
    out = []
    for line in lines:
        m = re.match(r"^([.A-Za-z_\d][\w.]*):\s*([^;\s].*)$", line)
        if m and not line.lstrip().startswith(";"):
            out += [f"{m.group(1)}:", f"\t{m.group(2)}"]
        else:
            out.append(line)
    return out


def _expand(lines: list[str], macros) -> list[str]:
    lines = _split_labels(lines)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*\.irp\s+(\w+)\s*,\s*(.*)$", line)
        if m:
            var, vals = m.group(1), [v.strip() for v in m.group(2).split(",")]
            body = []
            i += 1
            while not re.match(r"^\s*\.endr\b", lines[i]):
                body.append(lines[i])
                i += 1
            for v in vals:
                out += _expand([b.replace(f"\\{var}", v) for b in body], macros)
            i += 1
            continue
        code = line.split(";", 1)[0]
        m = re.match(r"^(\s*)(\w+)\b\s*(.*)$", code)
        if m and m.group(2) in macros and not code.lstrip().startswith("."):
            params, body = macros[m.group(2)]
            args = [a.strip() for a in m.group(3).split(",")] if m.group(3).strip() else []
            sub = []
            for b in body:
                for p, a in zip(params, args):
                    b = b.replace(f"\\{p}", a)
                sub.append(b)
            out += _expand(sub, macros)
        else:
            out.append(line)
        i += 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
