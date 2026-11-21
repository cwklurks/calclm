"""The Python reference grid for the browser demo's check (docs/agents/phase6-browser-demo.md A1):
every topic x seeds {1, 2, 3, 48} x mask levels {1, 2, 3}, and Random at the same seeds and
levels, from the integer reference the hardware-check sheets use (ce/tools/hw_check.py). Writes
demo/build/reference.json (git-ignored: it holds generated programs); demo/tests/check_grid.js
runs the same cases through the WebAssembly module.

    uv run python -m demo.tools.reference [--export DIR]

One process, numpy on one thread (about two CPU minutes); no GPU, no emulator.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from demo.tools.build_model import FIT, ROOT, default_export  # noqa: E402

OUT = ROOT / "demo" / "build" / "reference.json"
SEEDS = (1, 2, 3, 48)
LEVELS = (1, 2, 3)
TOPIC_CAP, RANDOM_CAP, TEMP_Q8, TOP_K = 256, 1024, 205, 20


def topic_ids(m, E, key: str, seed: int, level: int) -> list[int]:
    """A topic run: level 2 is the calculator's (model.topics.generate_topic, = hw_check
    --topic); levels 1 and 3 change only the masks and LD2 (level 3: no LD2, plain sampling)."""
    from lab import e14
    from model import topics as TP
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    from model.linker import ARMS
    if level == 2:
        return TP.generate_topic(m, E, key, seed)["ids"]
    p, s = TP.prompt(key), TP.mix_seed(seed)
    if level == 1:
        return e14.generate_prompted(m, ARMS["LD2"], s, E, p, cap=TOPIC_CAP, mask_level=1)["ids"]
    return generate(m, p, TOPIC_CAP, SamplerConfig(TEMP_Q8, TOP_K, False, s), cap=TOPIC_CAP,
                    mask=IntMask(m.vocab, level=3))


def random_ids(m, E, seed: int, level: int) -> list[int]:
    """A Random run at the calculator's defaults: levels 1 and 2 are LD2 with those masks
    (hw_check, --mask-level); level 3 is the Masks row off, plain sampling with IntMask(3)
    (test_calclm_safety.test_level_3_generation_matches_intref)."""
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    from model.linker import ARMS, generate as ld2_generate
    if level in (1, 2):
        return ld2_generate(m, ARMS["LD2"], seed, E=E, cap=RANDOM_CAP, mask_level=level)["ids"]
    return generate(m, [], RANDOM_CAP, SamplerConfig(TEMP_Q8, TOP_K, False, seed),
                    cap=RANDOM_CAP, mask=IntMask(m.vocab, level=3))


def grid(export: Path, fit: Path = FIT, seeds=SEEDS, levels=LEVELS, topics=None) -> dict:
    from model import topics as TP
    from model.intref import IntModel, render
    meta = json.loads((export / "model.json").read_text())
    E = json.loads(fit.read_text())["E"]
    m = IntModel.load(export, horizons=True)
    if topics is None:            # the menu's topics (topics.h): v2.6-rc2 offers all 10
        topics = [e["key"] for e in TP.menu_entries(TP.load_fixture())[:-1]]
    keys = list(topics)
    cases = []
    for key in [*keys, None]:
        for seed in seeds:
            for level in levels:
                ids = (random_ids(m, E, seed, level) if key is None
                       else topic_ids(m, E, key, seed, level))
                ids = [int(t) for t in ids]
                cases.append({"topic": key, "seed": seed, "level": level, "ids": ids,
                              "text": render(m, ids[1:])})
    return {"export_sha256": meta["sha256"], "fit": fit.name, "seeds": list(seeds),
            "levels": list(levels), "cases": cases}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--fit", type=Path, default=FIT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    export = a.export or default_export()
    if export is None:
        raise SystemExit("no export: pass --export or set CALCLM_DEMO_EXPORT")
    t0 = time.time()
    ref = grid(export, a.fit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(ref) + "\n")
    n = sum(len(c["ids"]) - 1 for c in ref["cases"])
    print(f"{len(ref['cases'])} reference cases, {n} tokens, {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
