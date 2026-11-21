"""Shared fixture: a tiny model trained on synthetic repeating patterns, plus its export.

Training makes the output distributions peaked, so top-1 agreement between the float
fake-quant model and the integer reference is a meaningful check.
"""
import numpy as np
import pytest
import torch

from model.export import export_model
from model.intref import IntModel
from model.model import ModelConfig, Transformer
from model.tokens import BOS

SMALL = ModelConfig(vocab_size=24, dim=40, n_layers=2, n_heads=4, n_kv_heads=1,
                    hidden_dim=96, max_seq_len=32)


def pattern_batch(rng, batch, ctx, vocab):
    """Each row: <bos> then a random 3-5 token motif (ids >= 4) repeated."""
    out = np.zeros((batch, ctx + 1), dtype=np.int64)
    for b in range(batch):
        motif = rng.integers(4, vocab, size=rng.integers(3, 6))
        out[b, 0] = BOS
        out[b, 1:] = np.resize(motif, ctx)
    return out


@pytest.fixture(scope="session")
def trained_small(tmp_path_factory):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Transformer(SMALL)
    model.calibrate(torch.from_numpy(pattern_batch(rng, 8, 32, 24)[:, :-1]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(500):
        seq = torch.from_numpy(pattern_batch(rng, 16, 32, 24))
        _, loss = model(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    out = tmp_path_factory.mktemp("export")
    calib = pattern_batch(np.random.default_rng(1), 8, 32, 24)[:, :-1]
    export_model(model, calib, out)
    return model, IntModel.load(out), float(loss.detach())
