"""Build CLMKV4.8xv: the 4-bit K/V tables of one export for the E20 spike (PLAN-E20.md Design 6).

    uv run python -m ce.calclm.tools.make_kv4 --export runs/e16/export/W8-s10 \\
        --tables runs/e19/kv4/W8-s10.json --out ce/build/e20/W8-s10/CLMKV4.8xv

Payload (little-endian), bound to the export like CLMLD2 (core/clm_kv4.h checks every field):

    0   "KV", format 1, layers (u8), head_dim (u8)
    5   the first 6 bytes of the export's SHA-256 (as CLMVOC carries them)
    11  the export's blob size (u24, the end of the last CLMnn slice)
    14  FNV-1a (u32) of every byte after this 18-byte header
    18  the reciprocals M[l][kind][c] (uint16), kind 0 = K, 1 = V
        the biased tables T[l][kind][c][16]: T[j] + 128 at nibble j + 7 (j = -7..7), entry 15
        a pad (0x80, never indexed: codes are 0..14)

The tables come from model/kv4.py's read_tables(path, export), which refuses a tables file
calibrated on another export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ce.calclm.tools.build_flash import appvar_8xv
from ce.calclm.tools.make_ld2 import checksum

HEADER = 18
FORMAT = 1
PAD = 0x80


def payload_size(n_layers: int, head_dim: int = 32) -> int:
    n = n_layers * 2 * head_dim
    return HEADER + 2 * n + 16 * n


def build(meta: dict, m: np.ndarray, t: np.ndarray) -> bytes:
    """The CLMKV4 payload from an export's model.json and the tables (M (L, 2, hd), T (L, 2, hd,
    15) as model/kv4.py stores them)."""
    m, t = np.asarray(m, dtype=np.int64), np.asarray(t, dtype=np.int64)
    n_layers, kinds, hd = m.shape
    if kinds != 2 or t.shape != (n_layers, 2, hd, 15):
        raise ValueError(f"tables of shape {m.shape} / {t.shape}")
    if m.min() < 1 or m.max() > 0xFFFF or np.abs(t).max() > 127:
        raise ValueError("a reciprocal or table entry out of range")
    cfg = meta["config"]
    if (n_layers, hd) != (cfg["n_layers"], cfg["dim"] // cfg["n_heads"]):
        raise ValueError(f"tables for {n_layers} layers / head_dim {hd}, export has "
                         f"{cfg['n_layers']} / {cfg['dim'] // cfg['n_heads']}")
    biased = np.full((n_layers, 2, hd, 16), PAD, dtype=np.uint8)
    biased[..., :15] = t + 128
    data = m.astype("<u2").tobytes() + biased.tobytes()
    head = (b"KV" + bytes([FORMAT, n_layers, hd]) + bytes.fromhex(meta["sha256"])[:6]
            + meta["blob_bytes"].to_bytes(3, "little") + checksum(data).to_bytes(4, "little"))
    return head + data


def payload(export: Path, tables: Path) -> bytes:
    from model.kv4 import read_tables
    export = Path(export)
    meta = json.loads((export / "model.json").read_text())
    tab = read_tables(tables, export)           # refuses another export's tables
    return build(meta, tab.m, tab.t)


def parse(data: bytes) -> dict:
    """The fields of a CLMKV4 payload (the inverse of build, for tests)."""
    if data[:2] != b"KV" or data[2] != FORMAT:
        raise ValueError("not a CLMKV4 format 1 payload")
    n_layers, hd = data[3], data[4]
    if len(data) != payload_size(n_layers, hd):
        raise ValueError("CLMKV4 payload size")
    if checksum(data[HEADER:]) != int.from_bytes(data[14:18], "little"):
        raise ValueError("CLMKV4 checksum")
    n = n_layers * 2 * hd
    m = np.frombuffer(data, "<u2", n, HEADER).astype(np.int64).reshape(n_layers, 2, hd)
    biased = np.frombuffer(data, np.uint8, 16 * n, HEADER + 2 * n).reshape(n_layers, 2, hd, 16)
    return {"layers": n_layers, "head_dim": hd, "sha6": data[5:11].hex(),
            "blob_bytes": int.from_bytes(data[11:14], "little"), "M": m,
            "T": biased[..., :15].astype(np.int64) - 128, "pad": biased[..., 15]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    data = payload(a.export, a.tables)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(appvar_8xv("CLMKV4", data))
    print(f"CLMKV4 {len(data)} bytes ({a.out})")


if __name__ == "__main__":
    main()
