"""Build CLMLD2.8xv: LD2 version 2, cheap export binding, checked offsets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ce.calclm.tools.build_flash import appvar_8xv


def checksum(data: bytes) -> int:
    """FNV-1a integrity check of the small payload, independent of the model hash."""
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def payload(export: Path, offsets: Path) -> bytes:
    fit = json.loads(offsets.read_text())
    digest = hashlib.sha256((export / "model.bin").read_bytes()).hexdigest()
    meta = json.loads((export / "model.json").read_text())
    if digest != meta["sha256"] or digest != fit.get("export_sha256"):
        raise ValueError("LD2 offsets/export SHA-256 mismatch")
    if not fit.get("hazard_full") or not fit.get("horizons"):
        raise ValueError("LD2 needs a full-vocabulary fit with horizons")
    E = fit["E"]
    if len(E) != 16 or any(type(x) is not int or not 0 <= x <= 0xFFFFFFFF for x in E):
        raise ValueError("LD2 needs 16 uint32 offsets")
    data = (b"LD2\x02" + len(meta["vocab"]).to_bytes(2, "little")
            + meta["blob_bytes"].to_bytes(3, "little") + bytes.fromhex(digest)[:6]
            + b"".join(x.to_bytes(4, "little") for x in E))
    return data + checksum(data).to_bytes(4, "little")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(appvar_8xv("CLMLD2", payload(args.export, args.offsets)))


if __name__ == "__main__":
    main()
