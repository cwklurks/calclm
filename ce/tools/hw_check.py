"""Reference for a CalcLM hardware check: the program the calculator must write for one seed at
INSTALL check 3's settings (T 0.8, top-k 20, sampled, masks on, length 1024, ALiBi horizons and
LD2 as the program's load policy sets them), from model.linker on the host (no emulator).

    uv run python -m ce.tools.hw_check \\
        --export runs/e7/export/W-s11 --offsets ce/tests/fixtures/ld2_offsets_W-s11.json \\
        --seed 19 --out dist/CalcLM-v2.3-rc1-reference

Writes OUT.json (ids, counts, hashes, the prgmGEN body bytes as hex) and OUT.txt (the program
as the editor shows it, without the editor's leading ':'), and prints the machine-readable line
of a hardware-check sheet (hw/checks/*/CHECK.md; test_calclm_w_ld2.py checks it against the
core). Generated program text stays out of git: write OUT under dist/ or ce/build/.
--mask-level 2: the v2.4 program's masks (model/safety.py); the default 1 is v2.3's.
--topic KEY: a v2.5 topic (model/topics.py) at the topic settings (mask level 2, cap 256, the
seed mixed by fmix32, the opening fed first); prgmGEN's body then starts with the opening. The
sheet line names topic= and seed_mix=. One sheet per (export, mask level) holds the Random line
and the topic lines of that level (test_calclm_w_ld2.check_sheets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BOS, EOS = 1, 2


def reference(export: Path, offsets: Path, seed: int, mask_level: int | None = None,
              topic: str | None = None) -> dict:
    """mask_level: 1 (the default) or 2; a topic is level 2 only (None or 2)."""
    from model.intref import IntModel, render
    from model.linker import ARMS, generate

    fit = json.loads(offsets.read_text())
    meta = json.loads((export / "model.json").read_text())
    if fit.get("export_sha256") != meta["sha256"]:
        raise ValueError(f"{offsets} was not fitted on {export}")
    m = IntModel.load(export, horizons=True)
    cap, extra = 1024, {}
    if topic is not None:
        from model import topics as TP
        if mask_level not in (None, TP.MASK_LEVEL):
            raise ValueError(f"a topic runs at mask level {TP.MASK_LEVEL} only")
        cap, mask_level = TP.CAP, TP.MASK_LEVEL
        got = TP.generate_topic(m, fit["E"], topic, seed)["ids"][1:]
        extra = {"topic": topic, "seed_mix": "fmix32"}
    else:
        mask_level = mask_level or 1
        got = generate(m, ARMS["LD2"], seed, E=fit["E"], cap=cap,
                       mask_level=mask_level)["ids"][1:]
    text = render(m, got)
    body = b"".join(bytes.fromhex(meta["vocab"][t]) for t in got if t > 3)   # no specials
    ids = b"".join(t.to_bytes(2, "little") for t in [BOS] + got)
    return {"seed": seed, **extra, "temperature_q8": 205, "top_k": 20, "cap": cap,
            "masks": True,
            **({"mask_level": mask_level} if mask_level != 1 else {}), "horizons": True, "ld2": True, "export_sha256": meta["sha256"],
            "offsets_sha256": hashlib.sha256(offsets.read_bytes()).hexdigest(),
            "tokens": len(got), "ends_with_eos": bool(got and got[-1] == EOS),
            "lines": len(text.split("\n")), "ids": [BOS] + got,
            "ids_sha256": hashlib.sha256(ids).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_hex": body.hex(), "text": text}


def sheet_line(ref: dict) -> str:
    masks = f" masks={ref['mask_level']}" if ref.get("mask_level", 1) != 1 else ""
    if "topic" in ref:
        masks = f" topic={ref['topic']} seed_mix={ref['seed_mix']}" + masks
    return (f"<!-- hardware-check seed={ref['seed']}{masks} tokens={ref['tokens']} lines={ref['lines']} "
            f"export_sha256={ref['export_sha256'][:16]} ids_sha256={ref['ids_sha256']} "
            f"text_sha256={ref['text_sha256']} body_sha256={ref['body_sha256']} -->")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--out", type=Path, required=True, help="OUT.json and OUT.txt")
    ap.add_argument("--mask-level", type=int, choices=(1, 2), default=None,
                    help="1 (default): v2.3's masks; 2: v2.4's safety masks (a topic: 2 only)")
    ap.add_argument("--topic", default=None,
                    help="a v2.5 topic key (model/topics.py: CYL, CIRC, ...); level 2, cap 256")
    a = ap.parse_args(argv)
    ref = reference(a.export, a.offsets, a.seed, a.mask_level, a.topic)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stem = a.out.parent / a.out.name          # not with_suffix: names like v2.3-rc1 have dots
    Path(f"{stem}.json").write_text(json.dumps(ref, indent=1) + "\n")
    Path(f"{stem}.txt").write_text(ref["text"] + "\n")
    print(sheet_line(ref))
    print(f"tokens {ref['tokens']} (eos {ref['ends_with_eos']}), lines {ref['lines']}, "
          f"prgmGEN body {ref['body_bytes']} bytes")


if __name__ == "__main__":
    main()
