"""E8 (PLAN-E8.md): the full-precision forward, the frozen teacher and the distillation loss.

Eval mode always quantizes (model.py), so the float forward of a model is its train-mode forward
with the global QAT switch off. On that path the model has no dropout and updates no buffer (the
q/k/v absmax EMA runs only on the quantized path), so under no_grad it is a pure function of the
weights. `float_forward` switches both and restores both, also when the forward raises.

The teacher (a checkpoint of model/train.py, normally trained with --float) is loaded frozen and
uncompiled, without moving the global RNG, and runs through `float_forward` on the student's
windows; the student's own QAT switch is back on before its forward (train.py).
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import model as model_def
from model.data import load_split
from model.model import ModelConfig, Transformer, load_checkpoint, set_quant_enabled


@contextmanager
def float_forward(model):
    """Train mode with QUANT_ENABLED off; restores the flag and the model's mode in a finally."""
    quant, training = model_def.QUANT_ENABLED, model.training
    try:
        set_quant_enabled(False)
        model.train()
        yield model
    finally:
        set_quant_enabled(quant)
        model.train(training)


def load_teacher(path: Path, cfg: ModelConfig, device: str) -> Transformer:
    """A train.py checkpoint as the frozen teacher of a student with config cfg: eval mode,
    parameters without grad, on device, never compiled. Loading moves no global RNG state."""
    with torch.random.fork_rng(devices=[]):      # Transformer() draws its throwaway init
        teacher = load_checkpoint(path)
    t = teacher.cfg
    mismatch = {k: (getattr(t, k), getattr(cfg, k)) for k in ("vocab_size", "max_seq_len", "pos")
                if getattr(t, k) != getattr(cfg, k)}
    if mismatch:
        raise ValueError(f"teacher {path} does not match the student (teacher, student): {mismatch}")
    if t.feat_rows:
        raise ValueError(f"teacher {path} takes input features; a teacher must be a plain model")
    return teacher.to(device).requires_grad_(False).eval()


@torch.no_grad()
def teacher_logits(teacher, x, amp_ctx=nullcontext):
    """The teacher's full-precision logits on the windows x (under amp_ctx, the student's)."""
    with float_forward(teacher), amp_ctx():
        logits, _ = teacher(x)
    return logits


def kd_loss(student_logits, teacher_logits):
    """Forward KL(p_T || p_S) summed over the vocabulary and averaged over the batch's positions,
    in fp32 at temperature 1. Training windows never hold ignore_index targets (model/data.py
    windows), so this is the cross-entropy's denominator."""
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = F.kl_div(s, t, log_target=True, reduction="sum")
    return kl / (s.numel() // s.shape[-1])


def distill_loss(ce, kd, alpha: float):
    """alpha * KD + (1 - alpha) * CE."""
    return alpha * kd + (1 - alpha) * ce


@torch.no_grad()
def float_sliding_val_loss(ckpt, data: Path, device: str | None = None,
                           max_programs: int = 0, chunk: int = 2048) -> dict:
    """lab.score.sliding_val_loss with the float forward instead of the quantized one: fp32 loss
    over whole val programs (<bos> to <eos>), every position seeing its last max_seq_len tokens,
    for a relative-position model without input features (a --float teacher).

    Programs longer than `chunk` targets are scored `chunk` positions at a time, each chunk fed
    with n_layers * (max_seq_len - 1) earlier tokens: a position's output depends on at most that
    many, so the loss equals one whole-program pass (which runs out of GPU memory for long
    programs at teacher size). chunk=0 always uses one pass."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (load_checkpoint(ckpt) if isinstance(ckpt, (str, Path)) else ckpt).to(device)
    cfg = model.cfg
    if cfg.pos == "learned":
        raise ValueError("learned positions cannot run past their table")
    if cfg.feat_rows:
        raise ValueError("float_sliding_val_loss takes models without input features")
    val = load_split(Path(data) / "val.bin")
    starts = np.flatnonzero(val == 1)
    ends = np.append(starts[1:], len(val))
    spans = list(zip(starts, ends))[: max_programs or None]
    reach = cfg.n_layers * (cfg.max_seq_len - 1)
    total, count = 0.0, 0
    with float_forward(model):
        for lo, hi in spans:
            n = hi - 1 - lo                                  # targets in this program
            step = chunk if chunk and n > chunk else n
            for a in range(0, n, step):                      # score positions a .. b-1
                b = min(a + step, n)
                c = max(0, a - reach)                        # context the outputs can depend on
                x = torch.from_numpy(val[lo + c : lo + b].astype(np.int64))[None].to(device)
                y = torch.from_numpy(val[lo + a + 1 : lo + b + 1].astype(np.int64)).to(device)
                logits, _ = model(x)
                total += F.cross_entropy(logits.float()[0, a - c :], y, reduction="sum").item()
                count += y.shape[0]
    return {"sliding_val_loss": total / count, "targets": count, "programs": len(spans),
            "window": cfg.max_seq_len}
