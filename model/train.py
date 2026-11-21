"""QAT training for CalcLM (ternary BitLinear, int8 activations, int8 q/k/v, uint8 softmax).

    uv run python -m model.train --data data/tok --vocab corpus/vocab.json --out runs/v0
    uv run python -m model.train --data runs/proxy/tok --vocab runs/proxy/vocab.json --out runs/proxy-v0

Writes runs/<name>/ckpt.pt (latest), best.pt (lowest val loss), log.jsonl, config.json.

Weight averaging (`--avg-steps N`, off by default): avg.pt holds the uniform mean of the model
state (latent float weights and the static q/k/v absmax buffers) after each of the last N
optimizer steps, sampled every `--avg-every` steps and always including the final weights.
Quantization happens as usual when the averaged model runs or is exported. When it is on,
avg.pt is the evaluation checkpoint (result.json "eval_ckpt"), not best.pt.

Why a uniform tail mean and not an EMA: the noise G1 has to remove is the spread between late
checkpoints (250 steps apart they differ by up to 8 points of strict OK, lab/RESULTS.md), and a
uniform mean over a window much longer than that weighs every late checkpoint equally, with one
interpretable parameter (the window) instead of a decay rate that would also set how far back it
reaches; no bias correction or start-up choice; a running float64 sum costs one copy of the
weights. Late in the cosine schedule the learning rate is small (under 11% of peak in the last
20% of steps), so the averaged weights stay in one basin (SWA, Izmailov et al. 2018; LAWA,
Kaddour 2022).

E8 options (PLAN-E8.md, model/distill.py), off by default; a run that uses none of them computes
and writes exactly what it did before they existed (model/tests/test_distill.py):
- `--teacher PATH [--kd-alpha A]`: distill from a checkpoint of this script (any shape with the
  student's vocabulary, context and positions; normally a --float run). Each step the frozen
  teacher runs first, in full precision, on the student's windows; then loss = A * KL(p_T || p_S)
  + (1 - A) * CE (A = 0.9 by default; the KL is summed over the vocabulary and averaged over the
  positions, at temperature 1). log.jsonl has "kd_loss" and "ce_loss"; inputs.json and
  result.json have "distill" (teacher path and sha256, A).
- `--float`: a full-precision run (the teacher's). QAT never runs (no switch, no recalibration),
  every evaluation is the float forward, and best.pt is saved on every improvement of the float
  val loss (ties keep the earlier step). Not with --qat-start, --teacher or --feat-dir.
Either option also records best.pt in result.json "best" (best_checkpoint) and "gpu_s" (the
training seconds of a CUDA run, one GPU).

`--init CKPT` (fine-tuning, off by default; model/tests/test_init.py): the student starts from the
"model" state of a checkpoint of this script, exactly as saved (for avg.pt the averaged weights),
including its static q/k/v absmax buffers, so the start-up calibration is skipped (its batch is
still drawn). The checkpoint's config must equal the run's (shape flags, vocabulary, context,
positions; no features). Everything else is the run's own: a fresh AdamW, the lr schedule, QAT
(a --qat-start switch still recalibrates), averaging, and the batch order of its --seed (loading
moves no RNG state). inputs.json and result.json have "init" (path and sha256). Not with --float
or --feat-dir, and not from a checkpoint of a --float run (its absmax buffers never trained). A
run without --init records nothing of it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model.data import (
    IGNORE, bos_feature_windows, bos_windows, feature_meta, feature_windows, load_features,
    load_split, program_starts, sample_offsets, shuffle_features, val_feature_windows, val_windows,
    windows,
)
from model.distill import distill_loss, float_forward, kd_loss, load_teacher, teacher_logits
from model.model import ModelConfig, Transformer, set_quant_enabled
from model.tokens import load_vocab


def lr_at(step: int, warmup: int, steps: int, lr: float, min_lr: float, flat: int = 0) -> float:
    """Warm up, cosine-decay to min_lr at steps - flat, then hold the floor."""
    decay_end = steps - flat
    if step < warmup:
        return min_lr + (lr - min_lr) * step / warmup
    frac = min(1.0, (step - warmup) / max(1, decay_end - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="CalcLM QAT training")
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin, val.bin")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--p-bos", type=float, default=0.25, help="share of windows starting at <bos>")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=0, help="cap val windows (0 = all)")
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--avg-steps", type=int, default=0,
                    help="save avg.pt, the mean of the weights over the last N steps (0 = off)")
    ap.add_argument("--avg-every", type=int, default=10, help="averaging snapshot interval")
    ap.add_argument("--flat-tail", type=int, default=0,
                    help="hold min_lr for the last N steps after cosine decay")
    ap.add_argument("--qat-start", type=int, default=0,
                    help="full-float training for N steps before QAT (0 = QAT throughout)")
    ap.add_argument("--feat-dir", type=Path, default=None,
                    help="per-token input features aligned with train/val.bin (model/data.py)")
    ap.add_argument("--feat-shuffle", action="store_true",
                    help="A2 control: every window gets the features of another window")
    ap.add_argument("--feat-fold", action="store_true",
                    help="E3: fold the one categorical feature (a per-program tag) into the "
                         "token rows (model.fold_feature)")
    ap.add_argument("--pos", choices=["learned", "alibi", "nope"], default="learned",
                    help="position scheme; alibi/nope run past --ctx with a sliding window")
    ap.add_argument("--teacher", type=Path, default=None,
                    help="distill from this train.py checkpoint (frozen, full precision)")
    ap.add_argument("--kd-alpha", type=float, default=None,
                    help="with --teacher: loss = A*KL + (1-A)*CE (default 0.9; 0 computes the "
                         "teacher for nothing)")
    ap.add_argument("--float", action="store_true",
                    help="full precision throughout: no QAT, float evaluation, best.pt by float "
                         "val loss")
    ap.add_argument("--init", type=Path, default=None,
                    help="fine-tune: start from the weights of this train.py checkpoint (same "
                         "config), with a fresh optimizer and this run's schedule")
    args = ap.parse_args(argv)
    if args.flat_tail < 0 or (args.flat_tail and args.steps - args.flat_tail <= args.warmup):
        ap.error("--flat-tail must be nonnegative and leave a cosine decay after warmup")
    if not 0 <= args.qat_start <= args.steps:
        ap.error("--qat-start must be between 0 and --steps")
    if args.kd_alpha is not None and args.teacher is None:
        ap.error("--kd-alpha needs --teacher")
    if args.teacher is not None:
        if not args.teacher.is_file():
            ap.error(f"--teacher {args.teacher} is not a file")
        args.kd_alpha = 0.9 if args.kd_alpha is None else args.kd_alpha
        if not 0.0 <= args.kd_alpha <= 1.0:
            ap.error("--kd-alpha must be between 0 and 1")
    if args.float and (args.qat_start or args.teacher is not None or args.feat_dir is not None):
        ap.error("--float runs without QAT, a teacher or features (--qat-start, --teacher, "
                 "--feat-dir)")
    if args.init is not None:
        if not args.init.is_file():
            ap.error(f"--init {args.init} is not a file")
        if args.float or args.feat_dir is not None:
            ap.error("--init fine-tunes a QAT model without features (not with --float or "
                     "--feat-dir)")
    return args


E8_DEFAULTS = {"teacher": None, "kd_alpha": None, "float": False}


def args_record(args) -> dict:
    """The training arguments as strings. A run that uses none of the E8 options leaves them out,
    and a run without --init leaves "init" out, so its inputs.json and checkpoints are
    byte-for-byte what they were before the options."""
    d = vars(args)
    default = all(d[k] == v for k, v in E8_DEFAULTS.items())
    return {k: str(v) for k, v in d.items()
            if not (default and k in E8_DEFAULTS) and not (k == "init" and v is None)}


def load_init(path: Path, model: Transformer) -> str:
    """Load the "model" state of a train.py checkpoint, as saved, into model; its config must be
    exactly the model's. Returns the sha256 of the bytes loaded. Moves no global RNG state."""
    blob = path.read_bytes()
    with torch.random.fork_rng(devices=[]):
        ck = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        saved, cfg = ModelConfig(**ck["config"]), model.cfg
        if saved != cfg:
            diff = {k: (v, getattr(cfg, k)) for k, v in vars(saved).items() if getattr(cfg, k) != v}
            raise ValueError(f"--init {path} does not match the run's config (checkpoint, run): "
                             f"{diff}")
        if ck.get("args", {}).get("float") == "True":
            raise ValueError(f"--init {path} is a checkpoint of a --float run: its q/k/v absmax "
                             "buffers were never trained (start-up calibration of random weights), "
                             "so they are no QAT scales")
        model.load_state_dict(ck["model"])
    return hashlib.sha256(blob).hexdigest()


def build_config(args, vocab_size: int) -> ModelConfig:
    feats = {}
    if args.feat_dir is not None:
        meta = feature_meta(args.feat_dir)
        feats = {"feat_cats": tuple(meta["cats"]), "feat_bag": int(meta["bag"])}
    return ModelConfig(vocab_size=vocab_size, dim=args.dim, n_layers=args.n_layers,
                       n_heads=args.n_heads, n_kv_heads=1, hidden_dim=args.hidden_dim,
                       max_seq_len=args.ctx, feat_shuffle=args.feat_shuffle,
                       feat_fold=args.feat_fold, pos=args.pos, **feats)


FEATURE_FILES = ("feats.json", "train.cats.npy", "train.bag.npy", "val.cats.npy", "val.bag.npy")


def to_device(cfg, cats, bag, device):
    """numpy feature windows -> the model's cats / bag tensors (None for an absent part)."""
    c = torch.from_numpy(cats).to(device) if cfg.feat_cats else None
    b = torch.from_numpy(bag).to(device) if cfg.feat_bag else None
    return c, b


def eval_features(cfg, stream, feats, ctx, n_windows=0):
    """Feature windows for val_windows and bos_windows (shuffled for the A2 control)."""
    fv = val_feature_windows(feats, len(stream), ctx)
    fb = bos_feature_windows(stream, feats, ctx)
    if cfg.feat_shuffle:
        fv, fb = shuffle_features(*fv), shuffle_features(*fb)
    if n_windows:
        fv, fb = tuple(a[:n_windows] for a in fv), tuple(a[:n_windows] for a in fb)
    return fv, fb


def make_optimizer(model, args, device):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                             fused=device.startswith("cuda"))


@torch.no_grad()
def evaluate(model, xs, ys, device, amp_ctx, batch=64, feats=None, fp=False) -> float:
    """Mean val loss; feats = (cats, bag) windows aligned with xs for a feature model.
    fp: the full-precision forward (model.distill.float_forward), not the quantized one."""
    if fp:
        with float_forward(model):
            return _mean_loss(model, xs, ys, device, amp_ctx, batch, feats)
    model.eval()
    loss = _mean_loss(model, xs, ys, device, amp_ctx, batch, feats)
    model.train()
    return loss


def _mean_loss(model, xs, ys, device, amp_ctx, batch, feats) -> float:
    total, count = 0.0, 0
    cfg = getattr(model, "cfg", None) or model._orig_mod.cfg
    for i in range(0, len(xs), batch):
        x = torch.from_numpy(xs[i : i + batch]).to(device)
        y = torch.from_numpy(ys[i : i + batch]).to(device)
        f = to_device(cfg, *(a[i : i + batch] for a in feats), device) if feats else ()
        with amp_ctx():
            logits, _ = model(x, None, *f)
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), y.flatten(), ignore_index=IGNORE, reduction="sum")
        total += loss.item()
        count += int((y != IGNORE).sum())
    return total / count


def save(path: Path, raw_model, args, step: int, val_loss, **extra) -> None:
    torch.save({"model": raw_model.state_dict(), "config": raw_model.cfg.to_dict(),
                "step": step, "val_loss": val_loss, "args": args_record(args), **extra}, path)


class TailAverage:
    """Uniform mean of a model's state over the last `window` of `total_steps` optimizer steps:
    a snapshot after every `every`-th step counted back from the last one (which is included)."""

    def __init__(self, total_steps: int, window: int, every: int = 1):
        self.total, self.window, self.every = total_steps, min(window, total_steps), max(1, every)
        self.sum: dict | None = None
        self.count = 0

    def due(self, done_steps: int) -> bool:
        return (self.window > 0 and done_steps > self.total - self.window
                and (self.total - done_steps) % self.every == 0)

    @torch.no_grad()
    def update(self, model) -> None:
        state = model.state_dict()
        if self.sum is None:
            self.sum = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in state.items()
                        if v.is_floating_point()}
        for k, acc in self.sum.items():
            acc.add_(state[k].detach().double())
        self.latest = {k: v.detach().clone() for k, v in state.items() if k not in self.sum}
        self.count += 1

    def averaged_state(self) -> dict:
        return {**self.latest, **{k: (acc / self.count).float() for k, acc in self.sum.items()}}


def eval_checkpoint(out: Path) -> Path:
    """The checkpoint a finished run is evaluated with: avg.pt when it averaged weights."""
    res = json.loads((Path(out) / "result.json").read_text())
    return Path(out) / res.get("eval_ckpt", "best.pt")


def best_checkpoint(out: Path) -> Path:
    """best.pt of a finished --float or --teacher run (result.json "best": its step and val loss).
    Other runs refuse: a --qat-start == --steps run keeps only its final step there."""
    res = json.loads((Path(out) / "result.json").read_text())
    if "best" not in res:
        raise ValueError(f"{out} records no best checkpoint (not a --float or --teacher run)")
    return Path(out) / res["best"]["ckpt"]


def main(argv=None) -> dict:
    try:
        return _train(parse_args(argv))
    finally:
        set_quant_enabled(True)


def _train(args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = args.device
    vocab = load_vocab(args.vocab)
    train, val = load_split(args.data / "train.bin"), load_split(args.data / "val.bin")
    starts = program_starts(train)
    xs_val, ys_val = val_windows(val, args.ctx)
    xs_bos, ys_bos = bos_windows(val, args.ctx)
    if args.eval_windows:
        xs_val, ys_val = xs_val[: args.eval_windows], ys_val[: args.eval_windows]
        xs_bos, ys_bos = xs_bos[: args.eval_windows], ys_bos[: args.eval_windows]
    cfg = build_config(args, len(vocab))
    feats_train = fv = fb = None
    if cfg.feat_rows:
        feats_train = load_features(args.feat_dir, "train", len(train))
        fv, fb = eval_features(cfg, val, load_features(args.feat_dir, "val", len(val)), args.ctx,
                               args.eval_windows)
    raw = Transformer(cfg).to(device)
    init = None
    if args.init is not None:
        init = {"path": str(args.init), "sha256": load_init(args.init, raw)}
    teacher = distill = None
    if args.teacher is not None:
        teacher = load_teacher(args.teacher, cfg, device)
        distill = {"teacher": str(args.teacher),
                   "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                   "kd_alpha": args.kd_alpha}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    sha = lambda files: {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}  # noqa: E731
    record = {"sha256": sha((args.data / "train.bin", args.data / "val.bin", args.vocab)),
              "args": args_record(args)}
    if cfg.feat_rows:
        record["feat_sha256"] = sha(args.feat_dir / f for f in FEATURE_FILES)
    if distill:
        record["distill"] = distill
    if init:
        record["init"] = init
    (args.out / "inputs.json").write_text(json.dumps(record, indent=2))

    def batch():
        offs = sample_offsets(train, args.batch, args.ctx, rng, args.p_bos, starts)
        x, y = windows(train, offs, args.ctx)
        f = ()
        if cfg.feat_rows:
            c, b = feature_windows(feats_train, offs, args.ctx)
            f = to_device(cfg, *(shuffle_features(c, b) if cfg.feat_shuffle else (c, b)), device)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), *f

    x0, _, *f0 = batch()
    if init is None:            # --init keeps the checkpoint's buffers (the batch is still drawn)
        raw.calibrate(x0, *f0)
    set_quant_enabled(args.qat_start == 0 and not args.float)
    model = torch.compile(raw) if args.compile else raw
    opt = make_optimizer(raw, args, device)
    use_amp = args.amp == "bf16"

    def amp_ctx():
        return torch.autocast(device.split(":")[0], dtype=torch.bfloat16) if use_amp else nullcontext()

    log = (args.out / "log.jsonl").open("w")
    avg = TailAverage(args.steps, args.avg_steps, args.avg_every)
    best, best_step, t0, tokens, result = float("inf"), None, time.time(), 0, {}
    for step in range(args.steps + 1):
        if args.qat_start > 0 and step == args.qat_start:
            set_quant_enabled(True)
            xc, _, *fc = batch()
            raw.calibrate(xc, *fc)
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, xs_val, ys_val, device, amp_ctx, feats=fv, fp=args.float)
            vb = evaluate(model, xs_bos, ys_bos, device, amp_ctx, feats=fb, fp=args.float)
            rec = {"step": step, "val_loss": vl, "val_bos_loss": vb,
                   "elapsed_s": time.time() - t0, "tokens": tokens}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if step >= args.qat_start and vl < best:      # --float: qat_start 0, every gain
                best, best_step = vl, step
                save(args.out / "best.pt", raw, args, step, vl)
            result = {"val_loss": vl, "best_val_loss": best, **rec}
        if step == args.steps:
            break
        lr = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr, args.flat_tail)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, *f = batch()
        if teacher is None:
            with amp_ctx():
                _, loss = model(x, y, *f)
        else:       # the teacher first, with QAT off only around its forward
            t_logits = teacher_logits(teacher, x, amp_ctx)
            with amp_ctx():
                logits, ce = model(x, y, *f)
            kd = kd_loss(logits, t_logits)
            loss = distill_loss(ce, kd, args.kd_alpha)
            del logits, t_logits
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
        opt.step()
        if avg.due(step + 1):
            avg.update(raw)
        tokens += x.numel()
        if step % 50 == 0:
            rec = {"step": step, "train_loss": loss.item(), "lr": lr,
                   "grad_norm": float(gnorm), "elapsed_s": time.time() - t0}
            if teacher is not None:
                rec |= {"ce_loss": ce.item(), "kd_loss": kd.item()}
            log.write(json.dumps(rec) + "\n")
    save(args.out / "ckpt.pt", raw, args, args.steps, result.get("val_loss"))
    result["eval_ckpt"] = "best.pt"
    if avg.count:
        averaged = copy.deepcopy(raw)
        averaged.load_state_dict(avg.averaged_state())
        rec = {"averaged": {"window": avg.window, "every": avg.every, "count": avg.count},
               "avg_val_loss": evaluate(averaged, xs_val, ys_val, device, amp_ctx, feats=fv,
                                        fp=args.float),
               "avg_val_bos_loss": evaluate(averaged, xs_bos, ys_bos, device, amp_ctx, feats=fb,
                                            fp=args.float)}
        log.write(json.dumps({"step": args.steps, **rec}) + "\n")
        print(json.dumps(rec), flush=True)
        save(args.out / "avg.pt", averaged, args, args.steps, rec["avg_val_loss"],
             averaged=rec["averaged"])
        result.update(rec, eval_ckpt="avg.pt")
    log.close()
    result["train_time_s"] = time.time() - t0
    if distill or args.float:
        if best_step is None:
            raise RuntimeError(f"no finite val loss in {args.out}; best.pt was never written")
        if distill:
            result["distill"] = distill
        result["best"] = {"ckpt": "best.pt", "step": best_step, "val_loss": best}
        result["gpu_s"] = result["train_time_s"] if device.startswith("cuda") else None
    if init:
        result["init"] = init
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
