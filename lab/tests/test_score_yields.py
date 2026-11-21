"""lab.score's opt-in yield capture: --yields / score(..., yields=True) reach the harness; off,
every call is the one it was before."""
import json

import numpy as np
import pytest

from lab import score as Sc
from lab.tests.conftest import VOCAB
from model.tokens import BOS, EOS

NL = 4


def test_cli_yields_flag_reaches_score(tmp_path, monkeypatch):
    seen = []

    def fake_score(samples, out, vocab, train, workers, budget_ms, max_len=256, **kw):
        seen.append(kw)
        return {"n_samples": len(samples)}

    monkeypatch.setattr(Sc, "score", fake_score)
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"ids": [BOS, 5, EOS]}) + "\n")
    Sc.main(["--samples", str(path), "--out", str(tmp_path), "--yields"])
    Sc.main(["--samples", str(path), "--out", str(tmp_path)])
    assert seen == [{"yields": True}, {}]
    assert "yields" not in json.loads((tmp_path / "score.json").read_text())


@pytest.mark.parametrize("kw, want", [({}, False), ({"yields": True}, True)])
def test_score_passes_yields_to_the_emulator_run(tmp_path, monkeypatch, kw, want):
    seen = {}

    def fake_run(programs, out, vocab, workers, budget_ms, yields=False):
        seen["yields"] = yields
        return [{"outcome": "OK", "error": None, "home_text": ["5"], "graph": False}
                for _ in programs]

    monkeypatch.setattr(Sc, "run_emulator", fake_run)
    res = Sc.score([[BOS, 5, NL, 6, EOS]], tmp_path, vocab=VOCAB,
                   train=np.arange(4, 100, dtype=np.uint16), **kw)
    assert seen["yields"] is want and res["strict_ok"]["k"] == 1 and "yields" not in res


@pytest.mark.parametrize("kw, want", [({}, False), ({"yields": True}, True)])
def test_run_emulator_threads_yields_to_run_token_ids(tmp_path, monkeypatch, kw, want):
    from eval import run_programs
    seen = {}

    def fake(seqs, vocab, out, workers=1, budget_ms=30_000, **rest):
        seen.update(rest)
        return []

    monkeypatch.setattr(run_programs, "run_token_ids", fake)
    Sc.run_emulator([[BOS, 5, EOS]], tmp_path, VOCAB, 1, 3000, **kw)
    assert seen["yields"] is want and seen["shots"] == 0
