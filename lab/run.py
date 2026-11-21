"""One command per experiment: train (or take a checkpoint), cost it, sample, score, record.

    uv run python -m lab.run --name NAME [train args...]       # v1 recipe + your overrides
    uv run python -m lab.run --name NAME --ckpt runs/v1/best.pt

Unknown arguments go to model/train.py after the v1 recipe (15000 steps, batch 128, lr 5e-3,
warmup 500, p_bos 0.25, --compile, seed 1337, v1 shape), so `--dim 256 --seed 2` changes only
those. Training goes to runs/lab/NAME/train; a finished training there with the same arguments
and data is reused. Then: export (exact flash size), lab.budget, lab.sample (seeds
seed0..seed0+N-1, T 0.8, top-k 20), intref top-1 agreement on val programs, lab.score in the
emulator (runs/lab/NAME/eval), val loss. Writes runs/lab/NAME/result.json and appends one row to
lab/RESULTS.md (the runs table must stay the last thing in that file).

v2 (G1): `--avg-steps N` (training) scores the averaged checkpoint; `--feat-dir DIR` and
`--feat-shuffle` (training) train with per-token tracker features, which the sampler then
computes as it samples (lab/tracking.py); `--masks` samples with the tracker's logit masks
(lab/masks.py; the row's name gets "+masks"). With --ckpt the feature directory is read from the
checkpoint's training arguments. intref has no feature inputs (INTREF.md 11), so feature models
skip the intref agreement check; relative-position models get it too (contexts of up to
AGREE_LEN = 128 tokens, so it passes the window only for windows shorter than that).

Fine-tuning: `--init CKPT` (training) starts from that checkpoint's weights (model/train.py); a
finished training is reused only while CKPT's bytes match the sha256 it recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lab.budget import budget, flash_bytes_from_export
from lab.sample import intref_agreement, sample
from lab.score import score, sliding_val_loss, val_loss
from lab.tracking import make_sampling_tracker
from model.data import bos_windows, feature_meta, load_features, load_split, val_windows
from model.export import export_model
from model.model import ModelConfig, load_checkpoint
from model.tokens import PAD
from model.train import FEATURE_FILES, eval_checkpoint, eval_features
from model.train import main as train_main
from model.train import parse_args as train_parse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tok"
VOCAB = ROOT / "corpus" / "vocab.json"
V1_RECIPE = ["--steps", "15000", "--batch", "128", "--lr", "0.005", "--min-lr", "0.0001",
             "--warmup", "500", "--wd", "0.05", "--grad-clip", "1.0", "--p-bos", "0.25",
             "--eval-every", "250"]
HEADER = ("| name | params (tern+emb) | flash KB | RAM KB | tok/s avg | val loss | complete % | "
          "no-ERR % | strict OK % | SYNTAX % | LABEL % | UNDEFINED % | distinct-4 | overlap % | "
          "run/N |")
AGREE_CONTEXTS, AGREE_LEN = 24, 128


def parse(argv=None) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(prog="python -m lab.run", allow_abbrev=False,
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="run name: runs/lab/NAME, row in RESULTS.md")
    ap.add_argument("--ckpt", type=Path, help="score this checkpoint instead of training")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=1, help="sampling seeds seed0..seed0+N-1")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sample-batch", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24, help="emulator processes")
    ap.add_argument("--budget-ms", type=int, default=30_000, help="emulated ms per program")
    ap.add_argument("--val-windows", type=int, default=0, help="cap val windows (0 = all)")
    ap.add_argument("--masks", action="store_true",
                    help="sample with the tracker's logit masks (lab/masks.py)")
    ap.add_argument("--cap", type=int, default=0,
                    help="sample and judge programs up to CAP tokens (relative positions: "
                         "256-token sliding window); 0 = the model's context")
    ap.add_argument("--train-only", action="store_true",
                    help="train (or find the finished training) and stop; a later run with the "
                         "same arguments reuses it")
    ap.add_argument("--no-compile", action="store_true", help="train without torch.compile")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "lab")
    ap.add_argument("--results", type=Path, default=ROOT / "lab" / "RESULTS.md")
    lab, rest = ap.parse_known_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", lab.name):
        ap.error("--name must be a plain name (letters, digits, . _ -)")
    recipe = V1_RECIPE + ([] if lab.no_compile else ["--compile"])
    train_argv = ["--data", str(DATA), "--vocab", str(VOCAB), *recipe, *rest,
                  "--out", str(lab.out_root / lab.name / "train")]
    if train_parse(train_argv).float and not lab.train_only:   # fail now on a bad argument
        ap.error("--float trains a full-precision model; the rest of lab.run scores the "
                 "quantized model, so use --train-only")
    return lab, train_argv


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _norm_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if k == "out":
            continue
        if k in ("data", "vocab"):
            p = Path(v)
            v = str((p if p.is_absolute() else ROOT / p).resolve())
        out[k] = str(v)
    return out


def _train_defaults() -> dict:
    """model/train.py's argument defaults (for runs saved before an argument existed)."""
    return {k: str(v) for k, v in vars(train_parse(["--data", "", "--vocab", "", "--out", ""])).items()
            if k not in ("data", "vocab", "out")}


def needs_training(train_argv: list[str]) -> bool:
    """False when runs/lab/NAME/train already holds a finished run with these arguments and
    (if recorded) the same data and vocab bytes. An argument missing from the saved run counts
    as its default value; an averaging run also needs its avg.pt."""
    a = train_parse(train_argv)
    meta, done = a.out / "inputs.json", a.out / "result.json"
    ckpts = [a.out / "best.pt"] + ([a.out / "avg.pt"] if a.avg_steps > 0 else [])
    if not all(p.exists() for p in (meta, *ckpts, done)):
        return True
    if done.stat().st_mtime < meta.stat().st_mtime:   # train.py writes result.json last
        return True
    saved = json.loads(meta.read_text())
    saved_args = {**_train_defaults(), **saved.get("args", {})}
    if _norm_args(saved_args) != _norm_args({k: str(v) for k, v in vars(a).items()}):
        return True
    if "sha256" in saved:
        now = {_sha(a.data / "train.bin"), _sha(a.data / "val.bin"), _sha(a.vocab)}
        if set(saved["sha256"].values()) != now:
            return True
    if a.teacher is not None and saved.get("distill", {}).get("teacher_sha256") != _sha(a.teacher):
        return True
    if a.init is not None and saved.get("init", {}).get("sha256") != _sha(a.init):
        return True
    if a.feat_dir is not None:
        now = {str(a.feat_dir / f): _sha(a.feat_dir / f) for f in FEATURE_FILES}
        return saved.get("feat_sha256") != now
    return False


def check_tracker_identity(fmeta: dict) -> None:
    """Features from the C tracker must be sampled with the same tracker build (feats.json
    "identity": sha256 of the tracker sources when the features were computed)."""
    from lab.features import tracker_identity

    if fmeta.get("tracker") != "tracker.bind:Tracker":
        return
    now = tracker_identity()
    if fmeta.get("identity") != now:
        changed = sorted(k for k in now if (fmeta.get("identity") or {}).get(k) != now[k])
        raise ValueError(f"tracker sources changed since the features were computed: {changed}; "
                         "recompute the features (python -m lab.features) and retrain")


def feature_dir(ck: dict) -> Path | None:
    """The --feat-dir a checkpoint was trained with (train.py saves arguments as strings)."""
    d = ck.get("args", {}).get("feat_dir")
    if d in (None, "None"):
        return None
    return Path(d) if Path(d).is_absolute() else ROOT / d


# ---- RESULTS.md ------------------------------------------------------------------------------

def _pct(r: dict) -> str:
    return f"{100 * r['p']:.1f}±{50 * (r['hi'] - r['lo']):.1f}"


def results_row(res: dict) -> str:
    b, s = res["budget"], res["score"]
    cells = [res["name"] + ("+masks" if res.get("masks") else ""),
             f"{b['params']['ternary'] / 1e6:.2f}M+{b['params']['embedding'] / 1e6:.2f}M",
             f"{b['flash']['appvar_file_bytes'] / 1024:.1f}", f"{b['ram']['total'] / 1024:.1f}",
             f"{b['tokens_per_s']['avg']:.2f}", f"{res['val']['val_loss']:.4f}",
             _pct(s["complete"]), _pct(s["no_err"]), _pct(s["strict_ok"]), _pct(s["err_syntax"]),
             _pct(s["err_label"]), _pct(s["err_undefined"]), f"{s['distinct_4']:.3f}",
             f"{100 * s['overlap_32']['mean_seen']:.1f}", f"{s['n_run']}/{s['n_samples']}"]
    return "| " + " | ".join(cells) + " |"


def append_row(path: Path, row: str) -> None:
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    if HEADER not in text:
        sep = "|" + "---|" * (HEADER.count("|") - 1)
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + HEADER + "\n" + sep + "\n"
    elif not text.endswith("\n"):
        text += "\n"
    path.write_text(text + row + "\n")


# ---- the experiment ------------------------------------------------------------------------

def _git() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _agreement_contexts(val, n=AGREE_CONTEXTS, length=AGREE_LEN) -> list[list[int]]:
    """The first n val programs from <bos>, each up to `length` tokens, without padding."""
    xs, _ = bos_windows(val, length)
    out = []
    for row in xs[:n]:
        ids = row.tolist()
        out.append(ids[: ids.index(PAD, 1)] if PAD in ids[1:] else ids)
    return out


def _train_or_reuse(lab, train_argv, targs) -> tuple[Path, str]:
    if lab.ckpt:
        return lab.ckpt, "given"
    if needs_training(train_argv):
        (targs.out / "result.json").unlink(missing_ok=True)   # never pair a new run with an old result
        train_main(train_argv)
        return eval_checkpoint(targs.out), "trained"
    print(f"reusing the finished training in {targs.out}", flush=True)
    return eval_checkpoint(targs.out), "reused"


def _write_samples(path: Path, seeds, samples, lab) -> None:
    path.write_text("".join(json.dumps({"seed": s, "temp": lab.temp, "top_k": lab.top_k, "ids": ids})
                            + "\n" for s, ids in zip(seeds, samples)))


def _timed(times: dict, key: str, fn, *args, **kw):
    t = time.time()
    out = fn(*args, **kw)
    times[key] = round(time.time() - t, 2)
    return out


def main(argv=None) -> dict:
    lab, train_argv = parse(argv)
    targs = train_parse(train_argv)
    run_dir = lab.out_root / lab.name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    ckpt, trained = _timed(times, "train_s", _train_or_reuse, lab, train_argv, targs)
    if lab.train_only:
        return {"name": lab.name, "ckpt": str(ckpt), "training": trained}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, model = ModelConfig(**ck["config"]), load_checkpoint(ckpt)
    val, device = load_split(targs.data / "val.bin"), targs.device
    fdir = feature_dir(ck) if cfg.feat_rows else None
    fmeta = feature_meta(fdir) if fdir else None
    if fmeta:
        check_tracker_identity(fmeta)
    val_feats = load_features(fdir, "val", len(val)) if fdir else None

    calib = val_windows(val, cfg.max_seq_len)[0][:32]   # what model/export.py calibrates on
    calib_feats = eval_features(cfg, val, val_feats, cfg.max_seq_len, 32)[0] if fdir else None
    _timed(times, "export_s", export_model, model, calib, run_dir / "export",
           vocab=json.loads(targs.vocab.read_text()), calib_feats=calib_feats)
    tracker, mask = make_sampling_tracker(fmeta, lab.masks, targs.vocab)
    cap = lab.cap or cfg.max_seq_len
    bag_bits = float(np.mean(np.bitwise_count(val_feats[1]))) if fdir and cfg.feat_bag else None
    bud = budget(cfg, flash=flash_bytes_from_export(run_dir / "export"), bag_bits=bag_bits,
                 mask_ids=mask.n_maskable if mask else 0, cap=cap)
    seeds = list(range(lab.seed0, lab.seed0 + lab.samples))
    samples = _timed(times, "sample_s", sample, model, seeds=seeds, temperature=lab.temp,
                     top_k=lab.top_k, device=device, batch=lab.sample_batch, tracker=tracker,
                     mask=mask, cap=cap)
    _write_samples(run_dir / "samples.jsonl", seeds, samples, lab)
    if cfg.feat_rows:
        agree = {"skipped": "intref has no feature inputs (INTREF.md 11)"}
    else:
        agree = _timed(times, "intref_agreement_s", intref_agreement, model, run_dir / "export",
                       _agreement_contexts(val), device=device,
                       workers=min(lab.workers, AGREE_CONTEXTS))
    sc = _timed(times, "score_s", score, samples, run_dir / "eval", targs.vocab,
                targs.data / "train.bin", lab.workers, lab.budget_ms, cap)
    vl = _timed(times, "val_loss_s", val_loss, ckpt, targs.data, device, lab.val_windows, fdir)
    if cfg.pos != "learned":
        vl |= _timed(times, "sliding_val_loss_s", sliding_val_loss, ckpt, targs.data, device,
                     fdir, lab.val_windows)
    times["total_s"] = round(time.time() - t0, 2)

    res = {"name": lab.name, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git(),
           "command": ["python", "-m", "lab.run", *(sys.argv[1:] if argv is None else argv)],
           "ckpt": str(ckpt), "ckpt_step": ck.get("step"), "training": trained,
           "train_args": ck.get("args", {}),
           "sampler": {"seeds": [seeds[0], seeds[-1]] if seeds else [], "temperature": lab.temp,
                       "temperature_q8": round(lab.temp * 256), "top_k": lab.top_k,
                       "device": device},
           "budget": bud, "intref_agreement": agree, "val": vl, "score": sc, "times": times,
           "masks": lab.masks, "features": fmeta and {"dir": str(fdir), **fmeta}, "cap": cap,
           "mask_stats": mask.stats if mask else None}
    (run_dir / "result.json").write_text(json.dumps(res, indent=1))
    append_row(lab.results, results_row(res))
    print(results_row(res))
    return res


if __name__ == "__main__":
    main()
