"""Shared fixtures for the model lab tests.

`tiny_run`: a small model trained on synthetic repeating patterns (peaked distributions, so
top-1 agreement is meaningful), saved as a train.py-style checkpoint plus its integer export.
The v1 paths let tests that need the shipping artifacts skip cleanly when they are absent.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from model.export import export_model
from model.model import ModelConfig, Transformer
from model.tokens import BOS

ROOT = Path(__file__).resolve().parents[2]
V1_RUN = ROOT / "runs" / "v1"
V1_CKPT = V1_RUN / "best.pt"
V1_EXPORT = V1_RUN / "export"
DATA = ROOT / "data" / "tok"
VOCAB = ROOT / "corpus" / "vocab.json"

TINY = ModelConfig(vocab_size=24, dim=40, n_layers=2, n_heads=4, n_kv_heads=1,
                   hidden_dim=96, max_seq_len=32)

have_v1 = pytest.mark.skipif(
    not (V1_CKPT.exists() and (V1_EXPORT / "model.bin").exists()),
    reason="runs/v1 checkpoint and export not present")
have_data = pytest.mark.skipif(
    not ((DATA / "val.bin").exists() and (DATA / "train.bin").exists() and VOCAB.exists()),
    reason="data/tok or corpus/vocab.json not present")
have_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def pattern_batch(rng, batch, ctx, vocab):
    """Each row: <bos> then a random 3-5 token motif (ids >= 4) repeated."""
    out = np.zeros((batch, ctx + 1), dtype=np.int64)
    for b in range(batch):
        motif = rng.integers(4, vocab, size=rng.integers(3, 6))
        out[b, 0] = BOS
        out[b, 1:] = np.resize(motif, ctx)
    return out


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """(checkpoint path, export dir, model) for a tiny pattern-trained model."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Transformer(TINY)
    model.calibrate(torch.from_numpy(pattern_batch(rng, 8, 32, 24)[:, :-1]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(400):
        seq = torch.from_numpy(pattern_batch(rng, 16, 32, 24))
        _, loss = model(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    out = tmp_path_factory.mktemp("tiny")
    ckpt = out / "best.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "step": 400,
                "val_loss": float(loss.detach()), "args": {}}, ckpt)
    calib = pattern_batch(np.random.default_rng(1), 8, 32, 24)[:, :-1]
    export_model(model, calib, out / "export")
    return ckpt, out / "export", model
