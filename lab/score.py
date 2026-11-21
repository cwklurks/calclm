"""Score generated programs: trim to the calculator context, run them on emulated TI-OS,
summarize with 95% confidence intervals.

    uv run python -m lab.score --samples runs/lab/x/samples.jsonl --out runs/lab/x/eval [--ckpt ...]

Definitions (denominators in brackets):
- trimming: eval/truncate.trim_to_line at 256 tokens, as RESULTS.md does; programs left empty
  (no complete line) are not run and are counted in n_empty.
- no_err: OK or TIMEOUT [programs run]; ok, timeout, rejected likewise.
- strict_ok: OK, with visible output (home-screen text or the graph screen) that is not only
  the harness's own input echo (`X=?1`), the red-team definition [programs run].
- err_syntax / err_label / err_undefined / err_other: ERR:<NAME> [programs run].
- complete: the sample ended with <eos> within the 256-token context [all samples].
- length: generated tokens per sample, <bos> excluded [all samples].
- distinct_4: unique 4-grams / all 4-grams over the run programs (<bos> excluded); depends on
  the number and length of programs, so compare at equal N.
- overlap_32: eval/overlap's fraction of each program's 32-token windows found in
  data/tok/train.bin (programs with >= 32 tokens after <bos>).
Rates carry Wilson 95% intervals.
--yields turns on the harness's yield capture (eval/driver.py): each record in results.jsonl
gains `yields`, the home screen at every Input/Pause/Menu/getKey wait; nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval.overlap import ngram_hashes
from eval.truncate import trim_to_line
from model.data import IGNORE, bos_windows, load_features, load_split, val_windows
from model.model import load_checkpoint
from model.tokens import EOS

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "corpus" / "vocab.json"
DATA = ROOT / "data" / "tok"
ECHO = re.compile(r"^.{0,16}\?\s*1$")   # an Input/Prompt line answered by the harness with "1"
ERROR_CLASSES = ("SYNTAX", "LABEL", "UNDEFINED")


# ---- statistics -------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "p": k / n if n else 0.0, "lo": lo, "hi": hi}


# ---- emulator outcomes ----------------------------------------------------------------------

def visible_lines(r: dict) -> list[str]:
    return [x.strip() for x in r.get("home_text") or [] if x.strip()]


def is_strict_ok(r: dict) -> bool:
    if r["outcome"] != "OK":
        return False
    if r.get("graph"):
        return True
    lines = visible_lines(r)
    return bool(lines) and not all(ECHO.match(x) for x in lines)


def outcome_metrics(records: list[dict]) -> dict:
    n = len(records)
    c = Counter(r["outcome"] for r in records)
    errors = Counter(r["error"] for r in records if r["outcome"] == "ERR")
    cls = {e: errors.get(e, 0) for e in ERROR_CLASSES}
    return {
        "n": n,
        "no_err": rate(c["OK"] + c["TIMEOUT"], n),
        "ok": rate(c["OK"], n),
        "strict_ok": rate(sum(is_strict_ok(r) for r in records), n),
        "timeout": rate(c["TIMEOUT"], n),
        **{f"err_{e.lower()}": rate(cls[e], n) for e in ERROR_CLASSES},
        "err_other": rate(c["ERR"] - sum(cls.values()), n),
        "rejected": rate(c["REJECTED"] + c["HARNESS_ERROR"], n),
        "errors": dict(errors.most_common()),
        "timeout_states": dict(Counter(r.get("timeout_state") for r in records
                                       if r["outcome"] == "TIMEOUT").most_common()),
    }


# ---- sample-level metrics ------------------------------------------------------------------

def newline_id(vocab: Path = VOCAB) -> int:
    return json.loads(Path(vocab).read_text()).index("3F")


def trim_samples(samples, newline_id: int, max_len: int = 256) -> tuple[list[list[int]], int]:
    """eval.truncate's trim; returns (non-empty trimmed programs, number left empty)."""
    trimmed = [trim_to_line(list(s), max_len, newline_id, EOS) for s in samples]
    kept = [t for t in trimmed if len(t) > 1]
    return kept, len(trimmed) - len(kept)


def complete_rate(samples, ctx: int = 256) -> dict:
    return rate(sum(EOS in list(s)[:ctx] for s in samples), len(samples))


def length_stats(samples) -> dict:
    lens = np.array([len(s) - 1 for s in samples], dtype=float)
    return {"mean": float(lens.mean()), "median": float(np.median(lens))} if len(lens) else \
        {"mean": 0.0, "median": 0.0}


def distinct_ngrams(seqs, n: int = 4) -> float:
    hs = [ngram_hashes(np.asarray(list(s)[1:], np.uint16), n) for s in seqs]
    total = sum(len(h) for h in hs)
    return len(np.unique(np.concatenate(hs))) / total if total else 0.0


def train_overlap(seqs, train, n: int = 32) -> dict:
    """eval/overlap.py's seen fraction per program against a uint16 stream (or its path)."""
    stream = np.fromfile(train, dtype="<u2") if isinstance(train, (str, Path)) else np.asarray(train)
    ref = ngram_hashes(stream, n)
    ref.sort()
    fr = []
    for s in seqs:
        h = ngram_hashes(np.asarray(list(s)[1:], np.uint16), n)
        if len(h):
            pos = np.searchsorted(ref, h).clip(max=len(ref) - 1)
            fr.append(float((ref[pos] == h).mean()))
    fr = np.array(fr)
    if not len(fr):
        return {"programs": 0, "mean_seen": 0.0, "zero_seen": 0.0, "near_copy": 0.0}
    return {"programs": len(fr), "mean_seen": float(fr.mean()), "zero_seen": float((fr == 0).mean()),
            "near_copy": float((fr >= 0.9).mean())}


# ---- checkpoint val loss ------------------------------------------------------------------

@torch.no_grad()
def _loss(model, xs, ys, device, batch=64, feats=None) -> float:
    from model.train import to_device

    total, count = 0.0, 0
    for i in range(0, len(xs), batch):
        x = torch.from_numpy(xs[i : i + batch]).to(device)
        y = torch.from_numpy(ys[i : i + batch]).to(device)
        f = to_device(model.cfg, *(a[i : i + batch] for a in feats), device) if feats else ()
        logits, _ = model(x, None, *f)
        total += torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                                   ignore_index=IGNORE, reduction="sum").item()
        count += int((y != IGNORE).sum())
    return total / count


def val_loss(ckpt, data: Path = DATA, device: str | None = None, max_windows: int = 0,
             feat_dir: Path | None = None) -> dict:
    """fp32 QAT val loss: contiguous windows (train.py's val_loss) and <bos> windows. A feature
    model reads val features from feat_dir (shuffled for the A2 control, as in training)."""
    from model.train import eval_features

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(ckpt).to(device)
    val = load_split(Path(data) / "val.bin")
    ctx = model.cfg.max_seq_len
    xs, ys = val_windows(val, ctx)
    xb, yb = bos_windows(val, ctx)
    fv = fb = None
    if model.cfg.feat_rows:
        if feat_dir is None:
            raise ValueError("a feature model needs feat_dir for its val features")
        fv, fb = eval_features(model.cfg, val, load_features(feat_dir, "val", len(val)), ctx,
                               max_windows)
    if max_windows:
        xs, ys, xb, yb = xs[:max_windows], ys[:max_windows], xb[:max_windows], yb[:max_windows]
    return {"val_loss": _loss(model, xs, ys, device, feats=fv),
            "val_bos_loss": _loss(model, xb, yb, device, feats=fb),
            "windows": len(xs), "bos_windows": len(xb)}


@torch.no_grad()
def sliding_val_loss(ckpt_or_model, data: Path = DATA, device: str | None = None,
                     feat_dir: Path | None = None, max_programs: int = 0) -> dict:
    """fp32 QAT loss over whole val programs (<bos> to <eos>, any length) for a relative-position
    model: every position sees its last max_seq_len tokens, as the ring-cache sampler does.
    Feature models get each program's features from its start; the A2/S2 control gets the
    stream rows half the stream away instead (another program)."""
    from model.train import to_device

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (load_checkpoint(ckpt_or_model) if isinstance(ckpt_or_model, (str, Path))
             else ckpt_or_model).to(device)
    cfg = model.cfg
    if cfg.pos == "learned":
        raise ValueError("learned positions cannot run past their table; use val_loss")
    val = load_split(Path(data) / "val.bin")
    starts = np.flatnonzero(val == 1)
    ends = np.append(starts[1:], len(val))
    feats = load_features(feat_dir, "val", len(val)) if cfg.feat_rows else None
    shift = len(val) // 2 if cfg.feat_shuffle else 0
    total, count = 0.0, 0
    spans = list(zip(starts, ends))[: max_programs or None]
    for lo, hi in spans:
        x = torch.from_numpy(val[lo : hi - 1].astype(np.int64))[None].to(device)
        y = torch.from_numpy(val[lo + 1 : hi].astype(np.int64))[None].to(device)
        f = ()
        if feats is not None:
            rows = (np.arange(lo, hi - 1) + shift) % len(val)
            c = np.asarray(feats[0])[rows].astype(np.int64)[None]
            b = np.asarray(feats[1])[rows].view(np.int64)[None]
            f = to_device(cfg, c, b, device)
        logits, _ = model(x, None, *f)
        total += torch.nn.functional.cross_entropy(logits.float()[0], y[0], reduction="sum").item()
        count += y.shape[1]
    return {"sliding_val_loss": total / count, "targets": count, "programs": len(spans),
            "window": cfg.max_seq_len}


# ---- everything -----------------------------------------------------------------------------

def run_emulator(programs, out: Path, vocab: Path = VOCAB, workers: int = 24,
                 budget_ms: int = 30_000, yields: bool = False, game: bool = False,
                 input_text: str | None = None,
                 input_values: tuple[str, ...] | None = None, decoy: str | None = None,
                 prefill: tuple[tuple[str, str], ...] | None = None) -> list[dict]:
    from eval import run_programs

    opt = {"game": True} if game else {}            # off: the call it always was
    if input_text is not None:
        opt["input_text"] = input_text
    if input_values is not None:
        opt["input_values"] = tuple(input_values)
    if decoy is not None:                           # task suite v2's hardened harness
        opt["decoy"] = decoy
    if prefill is not None:
        opt["prefill"] = tuple(map(tuple, prefill))
    return run_programs.run_token_ids(programs, Path(vocab), Path(out), workers=workers,
                                      budget_ms=budget_ms, shots=0, title=str(out), yields=yields,
                                      **opt)


def score(samples, out: Path, vocab: Path = VOCAB, train=DATA / "train.bin", workers: int = 24,
          budget_ms: int = 30_000, max_len: int = 256, yields: bool = False,
          game: bool = False, input_text: str | None = None) -> dict:
    """Trim, run, and summarize. Writes the emulator run (results.jsonl etc.) into out."""
    samples = [list(map(int, s)) for s in samples]
    kept, n_empty = trim_samples(samples, newline_id(vocab), max_len)
    opt = {"game": True} if game else {}            # off: the call it always was
    if input_text is not None:
        opt["input_text"] = input_text
    records = run_emulator(kept, out, vocab, workers, budget_ms, yields=yields,
                           **opt) if kept else []
    m = outcome_metrics(records)
    return {
        "n_samples": len(samples), "n_run": m.pop("n"), "n_empty": n_empty, **m,
        "complete": complete_rate(samples, max_len),
        "length": length_stats(samples), "trimmed_length": length_stats(kept),
        "distinct_4": distinct_ngrams(kept, 4),
        "overlap_32": train_overlap(kept, train, 32),
    }


def read_samples(path: Path) -> list[list[int]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r["ids"] if isinstance(r, dict) else r for r in rows]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=True, type=Path, help="jsonl of {ids: [...]} or lists")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ckpt", type=Path, help="also report this checkpoint's val loss")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--vocab", type=Path, default=VOCAB)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--budget-ms", type=int, default=30_000)
    ap.add_argument("--max-len", type=int, default=256,
                    help="cut at a line boundary (and count completion) within this many tokens")
    ap.add_argument("--yields", action="store_true",
                    help="also record the home screen at every Input/Pause/Menu/getKey wait")
    ap.add_argument("--game", action="store_true",
                    help="feed getKey loops the arrows then Clear, with key probes (lab.game)")
    ap.add_argument("--input-text", help="answer typed at every Input/Prompt (lab.mathtier)")
    a = ap.parse_args(argv)
    opt = {"yields": True} if a.yields else {}       # off: the call it always was
    if a.game:
        opt["game"] = True
    if a.input_text is not None:
        opt["input_text"] = a.input_text
    res = score(read_samples(a.samples), a.out, a.vocab, a.data / "train.bin", a.workers, a.budget_ms,
                max_len=a.max_len, **opt)
    res["max_len"] = a.max_len
    if a.game:
        res["game"] = True
    if a.input_text is not None:
        res["input_text"] = a.input_text
    if a.ckpt:
        res["val"] = val_loss(a.ckpt, a.data)
    (a.out / "score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["p"] if isinstance(v, dict) and "p" in v else v) for k, v in res.items()
                      if k not in ("errors", "timeout_states")}, indent=1))


if __name__ == "__main__":
    main()
