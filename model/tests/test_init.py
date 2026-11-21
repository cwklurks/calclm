"""--init CKPT (fine-tuning): start the student from the weights of a checkpoint of this script.

Neutrality: a run without --init is the code of BASE (the commit before --init) side by side, and
the E8 golden values (model/tests/test_distill.py) still hold.
"""
import json

import numpy as np
import pytest
import torch

from model import train as tr
from model.model import ModelConfig, Transformer, load_checkpoint
from model.tests.test_distill import (
    GOLDEN, _base_main, _data, _golden_run, _log, _outputs, _run, _same_states, _sha, _teacher,
)
from model.tests.test_train import TINY_ARGS
from model.train import main

BASE = "f1ff1bd"                  # the commit before --init (E8's options already in)
TINY = {"vocab_size": 30, "dim": 24, "n_layers": 2, "n_heads": 3, "hidden_dim": 40,
        "max_seq_len": 32}        # the ModelConfig of TINY_ARGS


def _ckpt(tmp_path, name="init.pt", **over):
    """A random checkpoint in this script's format (TINY_ARGS' shape unless overridden)."""
    cfg = ModelConfig(**{**TINY, **over})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        m = Transformer(cfg)
        for a in m.modules():
            if hasattr(a, "amax"):
                a.amax.copy_(torch.rand(3) + 0.5)       # nonzero, unlike a fresh model's
    path = tmp_path / name
    torch.save({"model": m.state_dict(), "config": cfg.to_dict(), "step": 0, "val_loss": None,
                "args": {}}, path)
    return path


def _first_eval(monkeypatch):
    """Record the model's state and the global RNG state at the run's first evaluation (step 0,
    before any optimizer update)."""
    seen = []
    evaluate = tr.evaluate

    def spy(model, *a, **kw):
        if not seen:
            seen.append(({k: v.clone() for k, v in model.state_dict().items()},
                         torch.random.get_rng_state()))
        return evaluate(model, *a, **kw)

    monkeypatch.setattr(tr, "evaluate", spy)
    return seen


def _offsets(monkeypatch):
    """Record every batch's window offsets (the batch order)."""
    seen = []
    sample = tr.sample_offsets

    def spy(*a, **kw):
        offs = sample(*a, **kw)
        seen.append(np.array(offs).copy())
        return offs

    monkeypatch.setattr(tr, "sample_offsets", spy)
    return seen


# ---- neutrality: no --init is the code of BASE --------------------------------------------------

def test_the_e8_golden_values_still_hold(tmp_path):
    assert _golden_run(tmp_path, "run") == GOLDEN


CONFIGS = [(), ("--amp", "bf16"), ("--qat-start", "3", "--avg-steps", "3", "--pos", "alibi"),
           ("--float", "--avg-steps", "2", "--avg-every", "1"), ("TEACHER", "--kd-alpha", "0.5")]


@pytest.mark.parametrize("extra", CONFIGS)
def test_without_init_a_run_writes_exactly_what_the_base_commit_wrote(tmp_path, extra):
    if extra[:1] == ("TEACHER",):
        extra = ("--teacher", str(_teacher(tmp_path)), *extra[1:])
    old, _ = _run(tmp_path, "old", *extra, train=_base_main(tmp_path, BASE))
    new, _ = _run(tmp_path, "new", *extra)
    a, b = _outputs(old), _outputs(new)
    assert (a["log"], a["result"], a["inputs"], a["config"]) == \
        (b["log"], b["result"], b["inputs"], b["config"])
    assert a["ckpts"].keys() == b["ckpts"].keys()
    for n, ca in a["ckpts"].items():
        cb = b["ckpts"][n]
        assert {k: v for k, v in ca.items() if k != "model"} == \
            {k: v for k, v in cb.items() if k != "model"}, n
        assert _same_states(ca["model"], cb["model"]), n


def test_without_init_nothing_records_it(tmp_path):
    out, res = _run(tmp_path, "run")
    assert "init" not in json.loads((out / "inputs.json").read_text())["args"]
    assert "init" not in json.loads((out / "inputs.json").read_text())
    assert "init" not in torch.load(out / "ckpt.pt", weights_only=False)["args"]
    assert "init" not in res


# ---- the loaded weights ----------------------------------------------------------------------

def test_at_step_zero_the_model_holds_the_checkpoints_state_exactly(tmp_path, monkeypatch):
    path = _ckpt(tmp_path)
    seen = _first_eval(monkeypatch)
    _run(tmp_path, "ft", "--init", str(path))
    state, _ = seen[0]
    assert _same_states(state, torch.load(path, weights_only=False)["model"])


def test_the_step_zero_evaluation_is_the_checkpoints_own(tmp_path):
    src, src_res = _run(tmp_path, "src", "--steps", "4")
    out, _ = _run(tmp_path, "ft", "--init", str(src / "ckpt.pt"))
    first = next(r for r in _log(out) if "val_loss" in r)
    assert first["step"] == 0 and first["val_loss"] == src_res["val_loss"]


def test_with_zero_steps_the_saved_checkpoint_is_the_init_checkpoint(tmp_path):
    path = _ckpt(tmp_path)
    out, _ = _run(tmp_path, "ft", "--init", str(path), "--steps", "0")
    assert _same_states(torch.load(out / "ckpt.pt", weights_only=False)["model"],
                        torch.load(path, weights_only=False)["model"])


def test_an_averaged_checkpoint_loads_its_averaged_weights(tmp_path, monkeypatch):
    src, _ = _run(tmp_path, "src", "--avg-steps", "3", "--avg-every", "1")
    avg, last = (torch.load(src / n, weights_only=False)["model"] for n in ("avg.pt", "ckpt.pt"))
    assert not _same_states(avg, last)
    seen = _first_eval(monkeypatch)
    _run(tmp_path, "ft", "--init", str(src / "avg.pt"))
    assert _same_states(seen[0][0], avg)


def test_the_init_run_then_trains(tmp_path):
    path = _ckpt(tmp_path)
    out, res = _run(tmp_path, "ft", "--init", str(path))
    assert np.isfinite(res["val_loss"])
    assert not _same_states(torch.load(out / "ckpt.pt", weights_only=False)["model"],
                            torch.load(path, weights_only=False)["model"])


FT_CONFIGS = [((), ()),
              (("--pos", "alibi"), ("--avg-steps", "3", "--avg-every", "1")),
              ((), ("--qat-start", "3")),
              ((), ("--lr", "0.01", "--warmup", "2", "--min-lr", "0.001"))]


@pytest.mark.parametrize("shape,extra", FT_CONFIGS)
def test_from_a_runs_step_zero_state_init_trains_exactly_like_that_run(tmp_path, shape, extra):
    """Only the starting weights change: a fresh AdamW, the run's own lr schedule, QAT, averaging
    and batch order. A checkpoint of a run's step-0 state (after its start-up calibration) as
    --init therefore reproduces that run bit for bit."""
    snap, _ = _run(tmp_path, "snap", *shape, "--steps", "0")
    plain, _ = _run(tmp_path, "plain", *shape, *extra)
    ft, _ = _run(tmp_path, "ft", *shape, *extra, "--init", str(snap / "ckpt.pt"))
    a, b = _outputs(plain), _outputs(ft)
    assert a["log"] == b["log"] and a["config"] == b["config"]
    assert a["ckpts"].keys() == b["ckpts"].keys()
    for n in a["ckpts"]:
        assert _same_states(a["ckpts"][n]["model"], b["ckpts"][n]["model"]), n
    assert {k: v for k, v in b["result"].items() if k != "init"} == a["result"]


# ---- RNG: loading moves none ----------------------------------------------------------------

def test_loading_moves_no_rng_state_and_keeps_the_batch_order(tmp_path, monkeypatch):
    path = _ckpt(tmp_path)
    seen, offs = _first_eval(monkeypatch), _offsets(monkeypatch)
    _run(tmp_path, "plain", "--seed", "7")
    plain_rng, plain_offs = seen[0][1], list(offs)
    seen.clear()
    offs.clear()
    _run(tmp_path, "ft", "--seed", "7", "--init", str(path))
    assert torch.equal(seen[0][1], plain_rng)
    assert len(offs) == len(plain_offs) and all(np.array_equal(x, y) for x, y in zip(offs, plain_offs))


def test_rng_drawn_while_loading_does_not_leak_into_the_run(tmp_path, monkeypatch):
    """torch.load draws no RNG today; the fork guards the run if loading ever does."""
    path = _ckpt(tmp_path)
    seen = _first_eval(monkeypatch)
    _run(tmp_path, "plain")
    want = seen[0][1]
    seen.clear()
    load = torch.load

    def drawing_load(*a, **kw):
        torch.rand(100)
        return load(*a, **kw)

    monkeypatch.setattr(torch, "load", drawing_load)
    _run(tmp_path, "ft", "--init", str(path))
    assert torch.equal(seen[0][1], want)


def test_the_seed_still_governs_the_batch_order(tmp_path, monkeypatch):
    path = _ckpt(tmp_path)
    offs = _offsets(monkeypatch)
    _run(tmp_path, "a", "--seed", "1", "--init", str(path))
    first = list(offs)
    offs.clear()
    _run(tmp_path, "b", "--seed", "2", "--init", str(path))
    assert not all(np.array_equal(x, y) for x, y in zip(offs, first))


# ---- refusals ----------------------------------------------------------------------------------

@pytest.mark.parametrize("over", [{"dim": 36}, {"n_layers": 1}, {"n_heads": 2}, {"hidden_dim": 48},
                                  {"vocab_size": 31}, {"max_seq_len": 16}, {"pos": "alibi"},
                                  {"feat_cats": (3,)}])
def test_a_checkpoint_with_another_config_is_refused(tmp_path, over):
    with pytest.raises(ValueError, match="does not match"):
        _run(tmp_path, "ft", "--init", str(_ckpt(tmp_path, **over)))
    assert not (tmp_path / "ft" / "inputs.json").exists()          # refused before writing


def test_a_checkpoint_of_a_float_run_is_refused(tmp_path):
    """A --float run never updates its q/k/v absmax buffers (they keep the start-up calibration of
    its random weights), so they are no scales for QAT."""
    src, _ = _run(tmp_path, "fp", "--float")
    with pytest.raises(ValueError, match="--float run"):
        _run(tmp_path, "ft", "--init", str(src / "best.pt"))
    assert not (tmp_path / "ft" / "inputs.json").exists()


@pytest.mark.parametrize("extra", [("--float",), ("--feat-dir", "feats")])
def test_init_refuses_float_and_features(tmp_path, extra):
    with pytest.raises(SystemExit):
        _run(tmp_path, "ft", "--init", str(_ckpt(tmp_path)), *extra)


def test_a_missing_init_file_is_an_argument_error(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "ft", "--init", str(tmp_path / "nope.pt"))


# ---- records -----------------------------------------------------------------------------------

def test_the_run_records_its_init_checkpoint(tmp_path):
    path = _ckpt(tmp_path)
    out, res = _run(tmp_path, "ft", "--init", str(path))
    want = {"path": str(path), "sha256": _sha(path)}
    assert res["init"] == want
    inputs = json.loads((out / "inputs.json").read_text())
    assert inputs["init"] == want and inputs["args"]["init"] == str(path)
    assert torch.load(out / "ckpt.pt", weights_only=False)["args"]["init"] == str(path)
    assert load_checkpoint(out / "ckpt.pt").cfg == ModelConfig(**TINY)


# ---- optional CUDA smoke -------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_smoke_fine_tunes_a_tiny_model(tmp_path):
    path = _ckpt(tmp_path)
    args = [a if a != "cpu" else "cuda" for a in TINY_ARGS]
    res = main([*_data(tmp_path), "--out", str(tmp_path / "cuda"), *args, "--amp", "bf16",
                "--init", str(path)])
    assert np.isfinite(res["val_loss"]) and res["init"]["sha256"] == _sha(path)
