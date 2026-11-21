"""Stage an export for the calculator: CLM00..CLMnn.8xv and CLMVOC.8xv next to links to its
model.json and model.bin, in a directory of their own (exports under runs/ stay read-only).

    uv run python -m ce.tools.stage_export \\
        runs/e7/export/W-s11 ce/build/w/export

The AppVars come from model/export.py's write_appvars (as v2's), the token table from
ce/calclm/tools/make_vocab.py; the staged directory is what ce.tools.bundle --export takes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ce.calclm.tools.make_vocab import build as build_vocab
from ce.calclm.tools.make_vocab import main as make_vocab

APPVAR_MAX = 65505              # data bytes of one AppVar (build_flash.appvar_8xv)
_NONZERO = np.array([[(b // 3 ** j) % 3 != 1 for j in range(5)] for b in range(256)])


def accumulator_bound(meta: dict, blob: bytes) -> int:
    """The largest |matvec accumulator| any int8 input in [-127, 127] can give: 127 times the
    most nonzero ternary weights in one row of any matrix. INTREF.md's |acc| < 2^16 holds by
    construction when this is below 2^16 (for v2.2's shape always: n_in <= 512; W's w2 has
    576 inputs, so it depends on the export)."""
    cfg, worst = meta["config"], 0
    for name, info in meta["tensors"].items():
        if not name.endswith(".w") or len(info["shape"]) != 2:
            continue
        rows, groups = info["shape"]
        n_in = cfg["hidden_dim"] if groups == -(-cfg["hidden_dim"] // 5) else cfg["dim"]
        w = np.frombuffer(blob, np.uint8, rows * groups, info["offset"]).reshape(rows, groups)
        nz = _NONZERO[w].reshape(rows, 5 * groups)[:, :n_in].sum(axis=1)
        worst = max(worst, 127 * int(nz.max()))
    return worst


def stage(export: Path, out: Path) -> dict:
    """Write out/; returns {file name: payload bytes} for the model's AppVars and CLMVOC."""
    from model.export import write_appvars

    export, out = export.resolve(), out.resolve()
    if out == export:
        raise ValueError("the staging directory must differ from the export")
    meta = json.loads((export / "model.json").read_text())
    blob = (export / "model.bin").read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise ValueError(f"{export}/model.bin does not match model.json")
    bound = accumulator_bound(meta, blob)
    if bound >= 1 << 16:
        raise ValueError(f"a matvec accumulator can reach {bound}, not below 2^16 (INTREF.md)")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.json", "model.bin"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(export / name)
    sizes = {}
    for path in write_appvars(out):
        sizes[path.name] = len(path.read_bytes()) - 76      # 74-byte head, 2-byte checksum
    make_vocab([str(export), "-o", str(out / "CLMVOC.8xv")])    # as v2's (tivars)
    sizes["CLMVOC.8xv"] = len(build_vocab(export)[0])
    too_big = {n: s for n, s in sizes.items() if s > APPVAR_MAX}
    if too_big:
        raise ValueError(f"AppVars over {APPVAR_MAX} bytes: {too_big}")
    return sizes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    for name, size in stage(a.export, a.out).items():
        print(f"{name} {size}")


if __name__ == "__main__":
    main()
