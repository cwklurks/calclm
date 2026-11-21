"""Build a small proxy dataset in the M1 data/tok format from local .8xp programs.

Used until the real corpus (data/tok, corpus/vocab.json) exists. --src names a folder of
TI-BASIC programs (read-only), searched recursively and deduplicated by body bytes.

    uv run python -m model.make_proxy --src DIR --out runs/proxy
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from model.data import write_split
from model.tokens import encode_body, sheet_vocab


def read_body(path: Path) -> bytes:
    from tivars import TIProgram

    return bytes(TIProgram.open(str(path)).data)


def split_programs(bodies: list[bytes], val_frac: float) -> tuple[list[bytes], list[bytes]]:
    """Deterministic split by body hash; guarantees at least one program on each side."""
    keyed = sorted(bodies, key=lambda b: hashlib.sha256(b).digest())
    n_val = min(max(1, round(len(keyed) * val_frac)), len(keyed) - 1)
    return keyed[n_val:], keyed[:n_val]


def build_proxy(files, out_dir: str | Path, val_frac: float = 0.1) -> dict:
    out = Path(out_dir)
    bodies = sorted({read_body(Path(f)) for f in files})
    bodies = [b for b in bodies if b]
    train, val = split_programs(bodies, val_frac)
    vocab = sheet_vocab()
    index = {tok: i for i, tok in enumerate(vocab)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab))
    meta = {"vocab_size": len(vocab), "source": "proxy"}
    for name, progs in (("train", train), ("val", val)):
        stream = np.array([t for b in progs for t in encode_body(b, index)], dtype=np.uint16)
        write_split(out / "tok" / f"{name}.bin", stream)
        meta[f"n_{name}_programs"] = len(progs)
        meta[f"n_{name}_tokens"] = int(len(stream))
    (out / "tok" / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="folder of .8xp programs")
    ap.add_argument("--out", type=Path, default=Path("runs/proxy"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    meta = build_proxy(sorted(args.src.rglob("*.8xp")), args.out, args.val_frac)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
