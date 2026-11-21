"""E3 behaviour tags (PLAN-v3-E3.md, "Tags"): run every human train and val program through the
judging path (lab.score --yields --max-len 1024), tag it ERR / BLANK / LITERAL / COMPUTED, and
write the per-token tag features: data/feats/tags, and data/feats/tags-shuf with the tags
permuted across programs.

    uv run python -m lab.tags --workers 12      # resumable: shards with a score.json are kept
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from lab.function import _checked_run, aligned, computed2, vis_lines
from lab.reference import programs
from model.data import load_split, write_features
from model.tokens import BOS

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("ERR", "BLANK", "LITERAL", "COMPUTED")      # the feature value is the index
VALUE = {t: i for i, t in enumerate(TAGS)}
CAP = 1024
SHUFFLE_SEED = 0


def classify(ids, r: dict) -> str:
    """ids: the trimmed program; r: its yield-run record."""
    if r["outcome"] not in ("OK", "TIMEOUT"):
        return "ERR"
    if computed2(ids, r):
        return "COMPUTED"
    return "LITERAL" if vis_lines(r) else "BLANK"


def program_tags(progs: list[list[int]], results_dir: Path, cap: int = CAP) -> list[str]:
    """Tag of every program of one lab.score run. A program with nothing left after the line
    cut is not run, and is ERR."""
    _checked_run(results_dir, cap)
    A = aligned(progs, results_dir, cap)
    return [classify(A[i][1], A[i][2]) if i in A else "ERR" for i in range(len(progs))]


def judge_counts(progs: list[list[int]], results_dir: Path, cap: int = CAP) -> dict:
    """What the judge could not judge cleanly: programs not run (nothing left after the line
    cut), harness errors and wall-limit hits (both tagged ERR; lab.function counts them the same
    way)."""
    _, config = _checked_run(results_dir, cap)
    recs = [r for _, _, r in aligned(progs, results_dir, cap).values()]
    limit = config.get("config", {}).get("wall_limit_s", float("inf"))
    return {"programs": len(progs), "not_run": len(progs) - len(recs),
            "harness_errors": sum(r["outcome"] == "HARNESS_ERROR" for r in recs),
            "wall_limit_hits": sum((r.get("wall_s") or 0) > limit for r in recs)}


def token_tags(stream, tags: list[str]) -> np.ndarray:
    """(N, 1) uint8: every stream position carries its program's tag (programs start at <bos>)."""
    s = np.asarray(stream)
    starts = s == BOS
    if not len(s) or not starts[0]:
        raise ValueError("the stream must start with <bos>")
    if int(starts.sum()) != len(tags):
        raise ValueError(f"{int(starts.sum())} programs but {len(tags)} tags")
    return np.array([VALUE[t] for t in tags], dtype=np.uint8)[np.cumsum(starts) - 1][:, None]


def write_tag_dirs(true_dir: Path, shuf_dir: Path, streams: dict, tags: dict,
                   extra: dict | None = None) -> None:
    """The true tags, and the same tags permuted across programs within each split
    (numpy default_rng(SHUFFLE_SEED), train first, then val). extra goes into both feats.json."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = {s: [tags[s][i] for i in rng.permutation(len(tags[s]))] for s in ("train", "val")}
    for d, tg, shuffled in ((true_dir, tags, False), (shuf_dir, shuf, True)):
        meta = {"cats": [len(TAGS)], "bag": 0, "source": "E3 behaviour tags (lab.tags)",
                "tags": list(TAGS), "shuffled": shuffled,
                "shuffle_seed": SHUFFLE_SEED if shuffled else None,
                "counts": {s: {t: tg[s].count(t) for t in TAGS} for s in ("train", "val")},
                **(extra or {})}
        for split in ("train", "val"):
            cats = token_tags(streams[split], tg[split])
            write_features(d, split, cats, np.zeros(len(cats), dtype=np.uint64), meta)


def score_shard(progs: list[list[int]], d: Path, workers: int) -> None:
    if (d / "score.json").exists():
        return
    shutil.rmtree(d, ignore_errors=True)            # a shard cut short is scored again
    d.mkdir(parents=True)
    samples = d.parent / f"{d.name}.samples.jsonl"
    samples.write_text("".join(json.dumps({"ids": p}) + "\n" for p in progs))
    subprocess.run([sys.executable, "-m", "lab.score", "--samples", str(samples), "--out", str(d),
                    "--yields", "--max-len", str(CAP), "--workers", str(workers)],
                   check=True, cwd=ROOT)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "tok")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "e3" / "tags")
    ap.add_argument("--feats", type=Path, default=ROOT / "data" / "feats")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=5000)
    a = ap.parse_args(argv)
    streams, tags, sha, judge = {}, {}, {}, {}
    for split in ("val", "train"):                  # val first: small, and it checks the path
        path = a.data / f"{split}.bin"
        sha[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        streams[split] = load_split(path)
        progs = programs(streams[split])
        tags[split], judge[split] = [], {}
        for k in range(0, len(progs), a.shard):
            d = a.out / split / f"{k // a.shard:03d}"
            score_shard(progs[k : k + a.shard], d, a.workers)
            tags[split] += program_tags(progs[k : k + a.shard], d)
            for key, n in judge_counts(progs[k : k + a.shard], d).items():
                judge[split][key] = judge[split].get(key, 0) + n
            print(f"{split} {len(tags[split])}/{len(progs)}", flush=True)
    (a.out / "tags.json").write_text(json.dumps({"data_sha256": sha, "judge": judge,
                                                 "tags": tags}))
    write_tag_dirs(a.feats / "tags", a.feats / "tags-shuf", streams, tags,
                   extra={"data_sha256": sha, "judge": judge})
    counts = {s: {t: tags[s].count(t) for t in TAGS} for s in tags}
    print(json.dumps({"counts": counts, "judge": judge}, indent=1))
    return counts


if __name__ == "__main__":
    main()
