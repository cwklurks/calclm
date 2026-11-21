#!/usr/bin/env python3
"""Emit the two M2 matvec kernels the core links (ce/kernel/gen_tmv.py, unchanged) as one
assembly file, with their entry points renamed to C symbols per input width:

    tmv_build / tmv_rows for G = ceil(dim/5)     ->  _tmv_d_build / _tmv_d_rows
    tmv_build / tmv_rows for G = ceil(hidden/5)  ->  _tmv_h_build / _tmv_h_rows

Both use the "pop" row loop, table column C0 = 0 and keep their variables inside their own
code (the CE program runs from RAM). Variables and .L labels get a per-kernel suffix.

    python ce/calclm/tools/gen_kernels.py ce/calclm/core/clm_shape.h -o out.s
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kernel"))
import gen_tmv  # noqa: E402


def shape(header: Path) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r"#define (CLM_\w+) (\d+)", header.read_text())}


def kernel(G: int, tag: str) -> str:
    src = gen_tmv.generate(G, 0, "pop")
    rename = {"tmv_build": f"_tmv_{tag}_build", "tmv_rows": f"_tmv_{tag}_rows",
              "tmv_end": f"_tmv_{tag}_end"}
    for old, new in rename.items():
        src = re.sub(rf"\b{old}\b", new, src)
    # private variables and section per kernel (labels v_* would clash between the two)
    for var, _ in gen_tmv.VARS:
        src = re.sub(rf"\b{var}\b", f"{var}_{tag}", src)
    src = re.sub(r"(\.L\w+)", rf"\1_{tag}", src)
    return src.replace("\t.section .text", f"\t.section .text.tmv_{tag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shape_h", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args(argv)
    s = shape(args.shape_h)
    gd, gh = -(-s["CLM_DIM"] // 5), -(-s["CLM_HIDDEN"] // 5)
    text = kernel(gd, "d") + kernel(gh, "h") + '\t.section .note.GNU-stack,"",%progbits\n'
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
