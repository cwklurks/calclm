"""E14 (PLAN-E14.md): LD2 decoding from a task opening, scored like v2.1's task samples.

model.linker.generate has no prompt and is hashed in earlier experiments' snapshots, so it is
left untouched: `generate_prompted` is a copy of its loop (everything else is imported from
model.linker) behind a prompt-feeding step.

Prompt feeding (the device after typed input: the opening is program text already written):
- tokens = [<bos>, *opening]; for each (pos, tok): logits = model.step(tok, pos, cache), then
  the masks, XT and Row read tok (Row skips <bos>, as every LD2 arm's row_bos_fix does).
- Nothing else happens during the prompt: no bans are asked, no token is forced, no hazard
  draw or xorshift32 step is taken, and the stats start at zero. The PRNG state at the first
  decided token is the seed's (2463534242 for seed 0).
- Then linker.generate's loop runs unchanged from `queued = []`. len(tokens) includes the
  prompt, so:
  - cap counts it: 1024 tokens in all, <bos> and the opening included (v2.1's cap counts it
    the same way);
  - the deadline's R = cap - len(tokens) and rule 3's late window (R <= D + 3 + BODY) see it;
  - references the prompt leaves pending would enter P, D (3 + name length + BODY 24 each) and
    Row's pending list like generated ones (task openings leave none);
  - rule 7's cell uses the absolute position len(tokens) - 1 (bins at 256/512/1024 count the
    prompt) and the opening's last token as `prev`; its `len(tokens) > 1` guard is already true,
    so the first decided token can be a hazard stop.

Pairing with v2.1 at the same seed: both start from the seed's xorshift32 state, but LD2 takes
one extra xorshift32 step (the hazard draw) at every step where rule 7 is eligible, and none for
forced tokens, so the two random streams part at the first decided token. LD2 and v2.1 samples of
one (task, seed) are paired by the job, not by shared random numbers.

    python -m lab.e14 sample --export EX --offsets FIT --split dev --k 50 --seed0 1 \
        --workers 8 --out DIR        # then: python -m lab.tasks score2 DIR

CalcLM v2.5's topics (PHASE3-topic-menu.md, model/topics.py) use three more options: --tasks
(task ids of either split instead of --split; sample.json then has sample2's v2 keys with
split "tasks"), --cap (tokens in all, the opening included) and --seed-mix fmix32 (the decoder
gets model.topics.mix_seed(seed); rows keep the raw seed). Their defaults reproduce E14 and
rc2's release review id for id.

Rows are {task, seed, ids, stats} in lab.tasks.sample's job order; sample.json has
lab.tasks.sample's keys (sample2's for --split heldout) plus decoder "LD2", the offsets and
their sha256, the rules and E, so `lab.tasks score2` judges the samples unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from model.intmask import IntMask
from model.intref import SamplerConfig, choose
from model.linker import (ARMS, BODY, CAP, N_CELLS, TEMP_Q8, TOP_K, XT, Row, Rules, hazard16,
                          hazard_cell, hazard_draw, hazard_weights, rule_bans, vocab_ids)
from model.tokens import BOS, EOS, PAD, UNK

DECODER = "LD2"


# ---- the decoder ----------------------------------------------------------------------------------

@dataclass
class Fed:
    """State after the prompt: what linker.generate holds before its first decision."""
    tokens: list[int]
    cache: object
    logits: np.ndarray
    mask: IntMask
    sp: XT
    row: Row


def check_prompt(prompt, cap: int, vocab_size: int | None = None) -> list[int]:
    """The opening's ids (no <bos>): program tokens only (ids 4 up, below vocab_size when
    given), leaving room under the cap."""
    p = [int(t) for t in prompt]
    if any(t < 4 or (vocab_size is not None and t >= vocab_size) for t in p):
        raise ValueError("prompt must hold program tokens only (no <pad>/<bos>/<eos>/<unk>, "
                         "no id outside the vocab)")
    if 1 + len(p) >= cap:
        raise ValueError(f"<bos> + a prompt of {len(p)} tokens leaves no room under the cap {cap}")
    return p


def feed_prompt(model, rules: Rules, prompt, mask_level: int = 1) -> Fed:
    """<bos> + prompt through the model, the masks, XT and Row (Row skips <bos> under
    row_bos_fix); no bans, draws or forced tokens. mask_level: IntMask's level."""
    ids = vocab_ids(model.vocab)
    mask, sp, row = IntMask(model.vocab, mask_level), XT(ids), Row(ids)
    tokens, cache, logits = [BOS, *prompt], model.new_cache(), None
    mask.reset()
    for pos, tok in enumerate(tokens):
        logits = model.step(tok, pos, cache)
        mask.step(tok)
        sp.step(tok)
        if pos >= 1 or not rules.row_bos_fix:
            row.step(tok)
    return Fed(tokens, cache, logits, mask, sp, row)


def generate_prompted(model, rules: Rules, seed: int, E: list[int] | None, prompt,
                      cap: int = CAP, temperature_q8: int = TEMP_Q8, top_k: int = TOP_K,
                      body: int = BODY, mask_level: int = 1) -> dict:
    """linker.generate after feed_prompt (the module docstring has the rules). prompt []: the
    unprompted decoder. mask_level: 1 v2.3's masks, 2 v2.4's safety masks. Returns
    {"ids": [BOS, *prompt, ..., (EOS)], "stats": ...}."""
    if rules.hazard and (E is None or len(E) != N_CELLS):
        raise ValueError(f"rule 7 needs {N_CELLS} integer offsets E (offsets_to_E)")
    f = feed_prompt(model, rules, check_prompt(prompt, cap, len(model.vocab)), mask_level)
    ids, cfg = vocab_ids(model.vocab), SamplerConfig(temperature_q8, top_k, False, seed)
    mask, sp, row, tokens, cache, logits = f.mask, f.sp, f.row, f.tokens, f.cache, f.logits
    state = seed if seed else 2463534242
    # ---- from here: model.linker.generate's loop, unchanged ----
    queued: list[int] = []
    st = dict(forced_lbl=0, hazard_stop=0, r1=0, r2=0, eos_sub=0, hazard_draws=0)
    if rules.hazard_full:            # E2b's stop rate by cell (LD's stats stay as E2 wrote them)
        st |= dict(draws_c=[0] * N_CELLS, stops_c=[0] * N_CELLS)
    while len(tokens) < cap:
        b = mask.banned()
        R, P = cap - len(tokens), sorted(sp.unres)
        D = sum(3 + len(x) + body for x in P) + 1
        lg = rule_bans(rules, ids, sp, P, R, D, b, logits, body)
        at_start = (sp.at_start and not sp.in_str and sp.name is None and sp.pending is None
                    and sp.if_state == 0)
        closure = row.closure()
        if queued:
            tok = queued.pop(0)
        elif rules.r1 and closure and not P and not row.pending:
            tok = EOS
            st["r1"] += 1
        elif rules.old and P and at_start and R <= D:
            tok = ids.lbl
            st["forced_lbl"] += 1
        elif rules.r2 and closure and (P or row.pending) and at_start:
            tok = ids.lbl                           # rule 1 then picks a pending name
            st["r2"] += 1
        else:
            stop = False
            if rules.hazard and not (rules.old and P) and not b[EOS] and len(tokens) > 1:
                cand, w = hazard_weights(model, lg, cfg, b, rules.hazard_full)
                w_e = w[cand.index(EOS)] if EOS in cand else 0
                c = hazard_cell(ids, bool(P), len(sp.blocks), tokens[-1], len(tokens) - 1)
                stop, state = hazard_draw(state, hazard16(w_e, sum(w), E[c]))
                st["hazard_draws"] += 1
                if rules.hazard_full:
                    st["draws_c"][c] += 1
                    st["stops_c"][c] += stop
            if stop:
                tok = EOS
                st["hazard_stop"] += 1
            else:
                if rules.hazard or (rules.old and P):   # (kept) hazard on: <eos> only by rule 7
                    b[EOS] = True
                allowed = ~b
                allowed[[PAD, BOS, UNK]] = False
                if not allowed.any():
                    b[EOS] = False
                tok, state = choose(model, lg, cfg, state, b)
                if rules.old and tok == EOS and P and R > 4:   # rule 2: <eos> becomes Lbl
                    st["eos_sub"] += 1
                    tok, queued = (ids.lbl, []) if at_start else (ids.nl, [ids.lbl])
        tokens.append(tok)
        if tok == EOS:
            break
        mask.step(tok)
        sp.step(tok)
        row.step(tok)
        logits = model.step(tok, len(tokens) - 1, cache)
    return {"ids": [int(t) for t in tokens], "stats": st}


# ---- the fit ----------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_offsets(offsets, export) -> dict:
    """An LD2 fit of this export (model.linker fit --hazard-full --horizons): its export path
    (and model.bin sha256 when the fit records one), hazard_full and horizons on, 16 offsets.
    Returns the record sample.json keeps."""
    raw = Path(offsets).read_bytes()
    off = json.loads(raw)
    if "export" not in off or Path(off["export"]).resolve() != Path(export).resolve():
        raise ValueError(f"{offsets} was fitted on export {off.get('export')!r}, not {export}")
    ex_sha = off.get("export_sha256")
    if ex_sha is not None and ex_sha != _sha256(Path(export) / "model.bin"):
        raise ValueError(f"{offsets}: export_sha256 is not {export}/model.bin's")
    if off.get("hazard_full") is not True or off.get("horizons") is not True:
        raise ValueError(f"{offsets}: LD2 needs a fit with hazard_full and horizons on")
    E = [int(x) for x in off.get("E", [])]
    if len(E) != N_CELLS:
        raise ValueError(f"{offsets}: needs {N_CELLS} integer offsets E")
    return {"decoder": DECODER, "offsets": str(offsets),
            "offsets_sha256": hashlib.sha256(raw).hexdigest(),
            "rules": asdict(ARMS[DECODER]), "body": BODY, "E": E}


# ---- sampling ---------------------------------------------------------------------------------------

_E: list[int] | None = None
_LEVEL = 1
_CAP = CAP
_MIX = "none"


def _load(export: str, E: list[int], mask_level: int = 1, cap: int = CAP,
          seed_mix: str = "none") -> None:
    from lab import tasks as T
    global _E, _LEVEL, _CAP, _MIX
    T._load(export)                      # ALiBi horizons on; refuses another vocab
    _E, _LEVEL, _CAP, _MIX = list(E), mask_level, cap, seed_mix


def _draw_one(job: tuple[str, int]) -> dict:
    from lab import tasks as T
    from lab.tasks_v2 import TASKS_V2
    from model.topics import seeded
    task_id, seed = job
    kw = {"mask_level": _LEVEL} | ({"cap": _CAP} if _CAP != CAP else {})
    return generate_prompted(T._model, ARMS[DECODER], seeded(seed, _MIX), _E,
                             TASKS_V2[task_id].opening_ids(), **kw)


def draw(export, jobs: list[tuple[str, int]], E: list[int], workers: int,
         mask_level: int = 1, cap: int = CAP, seed_mix: str = "none") -> list[dict]:
    """LD2 from each (task id, seed) job's opening, in job order: {ids, stats}. cap: tokens
    in all; seed_mix "fmix32": the decoder's seed is model.topics.mix_seed(seed)."""
    from model.topics import SEED_MIXES
    if seed_mix not in SEED_MIXES:
        raise ValueError(f"seed mix {seed_mix!r}: one of {SEED_MIXES}")
    args = (str(export), list(E), mask_level, cap, seed_mix)
    _load(*args)                         # here too: a bad export fails before the pool starts
    if workers <= 1:
        return [_draw_one(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_load,
                             initargs=args) as pool:
        return list(pool.map(_draw_one, jobs, chunksize=1))


def split_tasks(split: str):
    from lab import tasks as T
    from lab import tasks_v2 as V
    return {"dev": T.SUITE, "heldout": V.HELDOUT}[split]


def named_tasks(ids: list[str]):
    from lab.tasks_v2 import TASKS_V2
    unknown = [t for t in ids if t not in TASKS_V2]
    if unknown or not ids or len(set(ids)) != len(ids):
        raise ValueError(f"--tasks: unknown or repeated task ids {unknown or ids}")
    return tuple(TASKS_V2[t] for t in ids)


def sample_meta(export, k: int, seed0: int, n: int, split: str, tasks=None) -> dict:
    """lab.tasks.sample's sample.json keys (dev: suite v1, re-judged by score2 into DIR/v2), or
    lab.tasks_v2.sample2's (heldout; named tasks: split "tasks"), in their order."""
    from lab import tasks as T
    from lab import tasks_v2 as V
    tasks = split_tasks(split) if tasks is None else tasks
    ex = Path(export) / "model.bin"
    meta = {"export": str(export), "k": k, "seed0": seed0, "n": n}
    if split == "dev":
        meta["suite_sha256"] = T.suite_sha256(tasks)
    else:
        meta |= {"suite_sha256": V.SUITE_V2_SHA256, "split": split,
                 "tasks": [t.id for t in tasks]}
    return meta | {"temp": T.TEMP, "top_k": T.TOP_K, "cap": T.CAP, "masks": True,
                   "horizons": True, "export_sha256": _sha256(ex) if ex.exists() else None}


def sample(export, offsets, split: str | None, k: int, seed0: int, out: Path, workers: int = 8,
           mask_level: int = 1, tasks: list[str] | None = None, cap: int = CAP,
           seed_mix: str = "none") -> None:
    """split "dev"/"heldout", or tasks: task ids of either split (split "tasks")."""
    named = named_tasks(tasks) if tasks is not None else None
    if named is not None:
        split = "tasks"
    fit = load_offsets(offsets, export)
    jobs = [(t.id, s) for t in (named or split_tasks(split)) for s in range(seed0, seed0 + k)]
    extra = ({"cap": cap} if cap != CAP else {}) | (
        {"seed_mix": seed_mix} if seed_mix != "none" else {})
    outs = draw(export, jobs, fit["E"], workers, mask_level=mask_level, **extra)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(
        json.dumps({"task": tid, "seed": s, "ids": o["ids"], "stats": o["stats"]}) + "\n"
        for (tid, s), o in zip(jobs, outs, strict=True)))
    meta = sample_meta(export, k, seed0, len(jobs), split, named) | fit
    meta["cap"] = cap                    # the default (1024) leaves sample.json unchanged
    if mask_level != 1:                  # level 1: v2.3's sample.json, unchanged
        meta["mask_level"] = mask_level
    if seed_mix != "none":
        meta["seed_mix"] = seed_mix
    (out / "sample.json").write_text(json.dumps(meta, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="k LD2 samples per task of a split, from each opening")
    s.add_argument("--export", required=True, type=Path)
    s.add_argument("--offsets", required=True, type=Path, help="the export's LD2 fit")
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--split", choices=("dev", "heldout"))
    which.add_argument("--tasks", type=lambda x: x.split(","),
                       help="comma-separated task ids (lab.tasks_v2.TASKS_V2) instead of a split")
    s.add_argument("--k", type=int, required=True)
    s.add_argument("--seed0", type=int, default=1)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--mask-level", type=int, choices=(1, 2), default=1,
                   help="1: v2.3's masks; 2: v2.4's safety masks (model/safety.py)")
    s.add_argument("--cap", type=int, default=CAP, help="tokens in all, the opening included")
    s.add_argument("--seed-mix", choices=("none", "fmix32"), default="none",
                   help="fmix32: the decoder's seed is model.topics.mix_seed(seed) (v2.5 topics)")
    a = ap.parse_args(argv)
    try:
        sample(a.export, a.offsets, a.split, a.k, a.seed0, a.out, a.workers, a.mask_level,
               a.tasks, a.cap, a.seed_mix)
    except ValueError as e:
        print(f"lab.e14: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
