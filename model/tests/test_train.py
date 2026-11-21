import json

import numpy as np
import pytest
import torch

from model.data import write_split
from model.model import load_checkpoint
from model.tokens import BOS, EOS, SPECIALS
from model.train import lr_at, main


def _tiny_dataset(root):
    rng = np.random.default_rng(0)
    progs = [[BOS, *rng.integers(4, 30, size=rng.integers(5, 60)).tolist(), EOS] for _ in range(80)]
    stream = np.array([t for p in progs for t in p], dtype=np.uint16)
    write_split(root / "tok" / "train.bin", stream[: len(stream) * 9 // 10])
    write_split(root / "tok" / "val.bin", stream[len(stream) * 9 // 10 :])
    (root / "vocab.json").write_text(json.dumps(SPECIALS + [f"{i:02X}" for i in range(4, 30)]))


def test_lr_schedule_warmup_then_cosine():
    assert lr_at(0, warmup=10, steps=100, lr=1.0, min_lr=0.1) == 0.1
    assert lr_at(10, warmup=10, steps=100, lr=1.0, min_lr=0.1) == 1.0
    assert abs(lr_at(100, warmup=10, steps=100, lr=1.0, min_lr=0.1) - 0.1) < 1e-9
    mid = lr_at(55, warmup=10, steps=100, lr=1.0, min_lr=0.1)
    assert 0.5 < mid < 0.6


def test_train_writes_checkpoint_and_log(tmp_path):
    _tiny_dataset(tmp_path)
    out = tmp_path / "run"
    main([
        "--data", str(tmp_path / "tok"), "--vocab", str(tmp_path / "vocab.json"),
        "--out", str(out), "--steps", "6", "--batch", "4", "--eval-every", "3",
        "--device", "cpu", "--amp", "none", "--dim", "24", "--n-layers", "2",
        "--n-heads", "3", "--hidden-dim", "40", "--ctx", "32",
    ])
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert any("val_loss" in rec for rec in log)
    model = load_checkpoint(out / "ckpt.pt")
    assert model.cfg.vocab_size == 30 and model.cfg.max_seq_len == 32
    assert (model.layers[0].attn.amax > 0).all()
    with torch.no_grad():
        logits, _ = model(torch.tensor([[BOS, 5, 6]]))
    assert torch.isfinite(logits).all()


# ---- weight averaging (G1: every arm evaluates the mean of its late weights) -------------------

from model.train import TailAverage, eval_checkpoint  # noqa: E402

TINY_ARGS = ["--steps", "6", "--batch", "4", "--eval-every", "3", "--device", "cpu", "--amp", "none",
             "--dim", "24", "--n-layers", "2", "--n-heads", "3", "--hidden-dim", "40", "--ctx", "32"]


def _train(tmp_path, *extra):
    _tiny_dataset(tmp_path)
    out = tmp_path / "run"
    res = main(["--data", str(tmp_path / "tok"), "--vocab", str(tmp_path / "vocab.json"),
                "--out", str(out), *TINY_ARGS, *extra])
    return out, res


def test_tail_average_snapshots_the_last_window_of_steps():
    ta = TailAverage(total_steps=20, window=10, every=5)
    assert [t for t in range(1, 21) if ta.due(t)] == [15, 20]
    assert [t for t in range(1, 21) if TailAverage(20, 3, 1).due(t)] == [18, 19, 20]
    assert [t for t in range(1, 6) if TailAverage(5, 100, 2).due(t)] == [1, 3, 5]  # window > run
    assert not any(TailAverage(20, 0, 1).due(t) for t in range(1, 21))


def test_tail_average_is_the_mean_of_the_snapshots_including_float_buffers():
    lin = torch.nn.Linear(3, 2)
    lin.register_buffer("amax", torch.zeros(3))
    ta = TailAverage(total_steps=3, window=3, every=1)
    for v in (1.0, 2.0, 6.0):
        with torch.no_grad():
            lin.weight.fill_(v)
            lin.bias.fill_(-v)
            lin.amax.fill_(10 * v)
        ta.update(lin)
    avg = ta.averaged_state()
    assert ta.count == 3
    assert torch.equal(avg["weight"], torch.full((2, 3), 3.0))
    assert torch.equal(avg["bias"], torch.full((2,), -3.0))
    assert torch.equal(avg["amax"], torch.full((3,), 30.0))
    assert lin.weight[0, 0] == 6.0  # the live model is untouched


def test_averaging_is_off_by_default(tmp_path):
    out, res = _train(tmp_path)
    assert not (out / "avg.pt").exists()
    assert res["eval_ckpt"] == "best.pt"
    assert eval_checkpoint(out) == out / "best.pt"


def test_a_one_step_window_saves_exactly_the_final_weights(tmp_path):
    out, res = _train(tmp_path, "--avg-steps", "1", "--avg-every", "1")
    avg, last = (torch.load(out / n, weights_only=False) for n in ("avg.pt", "ckpt.pt"))
    assert avg["averaged"] == {"window": 1, "every": 1, "count": 1}
    assert all(torch.equal(avg["model"][k], last["model"][k]) for k in last["model"])


def test_averaging_saves_the_mean_as_the_evaluation_checkpoint(tmp_path):
    out, res = _train(tmp_path, "--avg-steps", "4", "--avg-every", "2")
    ck = torch.load(out / "avg.pt", weights_only=False)
    assert ck["averaged"] == {"window": 4, "every": 2, "count": 2} and ck["step"] == 6
    assert res["eval_ckpt"] == "avg.pt" and eval_checkpoint(out) == out / "avg.pt"
    assert res["avg_val_loss"] == ck["val_loss"] and np.isfinite(ck["val_loss"])
    last = torch.load(out / "ckpt.pt", weights_only=False)["model"]
    assert not torch.equal(ck["model"]["output.weight"], last["output.weight"])
    # the averaged latent weights quantize and export like any checkpoint
    from model.export import export_model
    from model.intref import IntModel
    model = load_checkpoint(out / "avg.pt")
    assert (model.layers[0].attn.amax > 0).all()
    calib = np.random.default_rng(0).integers(4, 30, size=(2, 32))
    export_model(model, calib, tmp_path / "export")
    assert IntModel.load(tmp_path / "export").cfg.vocab_size == 30


# ---- per-token input features ------------------------------------------------------------------

class _Depth:
    """Toy tracker: categorical = tokens since <bos> (capped), bag = parity bits of the last token."""

    def reset(self):
        self.n = 0

    def step(self, tok):
        self.n = 0 if tok == BOS else min(self.n + 1, 9)
        return [self.n, tok % 3], (tok & 7)


def _feature_dir(tmp_path):
    from model.data import load_split, stream_features, write_features
    d = tmp_path / "feats"
    for split in ("train", "val"):
        cats, bag = stream_features(load_split(tmp_path / "tok" / f"{split}.bin"), _Depth())
        write_features(d, split, cats, bag, {"cats": [10, 3], "bag": 3, "source": "toy"})
    return d


def test_training_with_features_builds_the_feature_model(tmp_path):
    _tiny_dataset(tmp_path)
    d = _feature_dir(tmp_path)
    out, res = _train(tmp_path, "--feat-dir", str(d))
    ck = torch.load(out / "ckpt.pt", weights_only=False)
    assert ck["config"]["feat_cats"] == [10, 3] and ck["config"]["feat_bag"] == 3
    assert "feat_shuffle" not in ck["config"] and ck["args"]["feat_dir"] == str(d)
    model = load_checkpoint(out / "ckpt.pt")
    assert model.feat_emb.weight.shape == (10 + 3 + 3, 24)
    assert np.isfinite(res["val_loss"])
    saved = json.loads((out / "inputs.json").read_text())
    assert set(saved["feat_sha256"]) == {str(d / f) for f in (
        "feats.json", "train.cats.npy", "train.bag.npy", "val.cats.npy", "val.bag.npy")}


def test_shuffled_features_are_recorded_in_the_config(tmp_path):
    _tiny_dataset(tmp_path)
    out, _ = _train(tmp_path, "--feat-dir", str(_feature_dir(tmp_path)), "--feat-shuffle")
    assert load_checkpoint(out / "ckpt.pt").cfg.feat_shuffle


def test_feature_arrays_must_match_the_streams(tmp_path):
    from model.data import write_features
    _tiny_dataset(tmp_path)
    d = _feature_dir(tmp_path)
    write_features(d, "val", np.zeros((5, 2)), np.zeros(5), {"cats": [10, 3], "bag": 3})
    with pytest.raises(ValueError):
        _train(tmp_path, "--feat-dir", str(d))


def _tag_dir(tmp_path):
    """E3-style tags: one categorical feature, constant over each program, no bag."""
    from model.data import load_split, write_features
    d = tmp_path / "tags"
    for split in ("train", "val"):
        stream = load_split(tmp_path / "tok" / f"{split}.bin")
        prog = np.cumsum(stream == BOS)
        write_features(d, split, (prog % 4)[:, None], np.zeros(len(stream)),
                       {"cats": [4], "bag": 0, "source": "toy tags"})
    return d


def test_training_with_folded_tags_builds_a_model_that_folds(tmp_path):
    from model.model import fold_feature
    _tiny_dataset(tmp_path)
    out, res = _train(tmp_path, "--feat-dir", str(_tag_dir(tmp_path)), "--feat-fold")
    model = load_checkpoint(out / "ckpt.pt")
    assert model.cfg.feat_fold and model.cfg.feat_cats == (4,) and np.isfinite(res["val_loss"])
    assert fold_feature(model, 3).cfg.feat_rows == 0


def test_fold_needs_a_single_categorical_feature(tmp_path):
    _tiny_dataset(tmp_path)
    with pytest.raises(ValueError):
        _train(tmp_path, "--feat-dir", str(_feature_dir(tmp_path)), "--feat-fold")


def test_shuffle_needs_features(tmp_path):
    with pytest.raises(ValueError):
        _train(tmp_path, "--feat-shuffle")


@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_training_with_a_relative_position_scheme(tmp_path, pos):
    out, res = _train(tmp_path, "--pos", pos, "--avg-steps", "2", "--avg-every", "1")
    model = load_checkpoint(out / "avg.pt")
    assert model.cfg.pos == pos and not hasattr(model, "pos_emb") and np.isfinite(res["val_loss"])
    with torch.no_grad():
        logits, _ = model(torch.randint(4, 30, (1, 70)))      # past the 32-token training window
    assert torch.isfinite(logits).all()


# Captured on fa7c2c3, CPU fp32, one thread, before the E5 flags were ported.
@pytest.mark.parametrize('extra', [(), ('--flat-tail', '0', '--qat-start', '0')])
def test_zero_recipe_flags_match_the_pre_e5_training_and_forward(tmp_path, extra):
    import hashlib
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out, res = _train(tmp_path, *extra)
        state = torch.load(out / 'ckpt.pt', weights_only=False)['model']
        digest = hashlib.sha256(b''.join(v.numpy().tobytes() for k, v in sorted(state.items())))
        assert digest.hexdigest() == '6a38a8077cd4bac19f470bfb9c59e77c979821bdf898f47e62c63300184e5b79'
        logits = load_checkpoint(out / 'ckpt.pt')(torch.tensor([[BOS, 5, 6]]))[0]
        assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == \
            '7218cf133a35ba1b97da0f9f5c57b09240ca6e945a5777d25bd4e8d8a65b14b0'
        assert res['val_loss'] == 3.4030089414209352
    finally:
        torch.set_num_threads(old_threads)


def test_flat_tail_reaches_the_floor_before_the_averaging_window():
    assert [lr_at(s, 10, 100, 1.0, .1, flat=30) for s in (0, 10, 40, 70, 90, 100)] == \
        pytest.approx([.1, 1., .55, .1, .1, .1])
    assert all(lr_at(s, 10, 100, 1., .1, flat=0) == lr_at(s, 10, 100, 1., .1)
               for s in range(110))


def test_delayed_qat_switches_at_the_requested_step_and_restores_export(tmp_path, monkeypatch):
    from model import model as mm
    from model.export import export_model
    phases, calibrations = [], []
    forward, calibrate = mm.Transformer.forward, mm.Transformer.calibrate

    def observe(self, idx, targets=None, *f, **kw):
        if targets is not None:
            phases.append(mm.QUANT_ENABLED)
        return forward(self, idx, targets, *f, **kw)

    def observe_calibration(self, idx, *f):
        calibrations.append(idx.clone())
        return calibrate(self, idx, *f)

    monkeypatch.setattr(mm.Transformer, 'forward', observe)
    monkeypatch.setattr(mm.Transformer, 'calibrate', observe_calibration)
    out, _ = _train(tmp_path, '--qat-start', '3', '--pos', 'alibi')
    assert phases == [False] * 3 + [True] * 3
    assert len(calibrations) == 2 and not torch.equal(*calibrations)
    assert mm.QUANT_ENABLED
    assert torch.load(out / 'best.pt', weights_only=False)['step'] >= 3
    model = load_checkpoint(out / 'ckpt.pt')
    x = torch.tensor([[BOS, 5, 6]])
    expected = model(x)[0]
    try:
        mm.set_quant_enabled(False)
        assert torch.equal(model(x)[0], expected)  # eval always quantizes
        export_model(model, x.numpy(), tmp_path / 'off')
        mm.set_quant_enabled(True)
        export_model(model, x.numpy(), tmp_path / 'on')
        assert (tmp_path / 'off/model.bin').read_bytes() == (tmp_path / 'on/model.bin').read_bytes()
        assert (tmp_path / 'off/model.json').read_bytes() == (tmp_path / 'on/model.json').read_bytes()
        model.train()
        mm.set_quant_enabled(False)
        assert not torch.equal(model(x)[0], expected)
    finally:
        mm.set_quant_enabled(True)


def test_delayed_qat_restores_the_switch_on_failure(tmp_path, monkeypatch):
    from model import model as mm
    from model import train as tr

    def fail(*a, **kw):
        raise RuntimeError('synthetic optimizer failure')

    monkeypatch.setattr(tr, 'make_optimizer', fail)
    with pytest.raises(RuntimeError):
        _train(tmp_path, '--qat-start', '3')
    assert mm.QUANT_ENABLED


@pytest.mark.parametrize('extra', [('--flat-tail', '-1'), ('--flat-tail', '6'),
                                  ('--qat-start', '-1'), ('--qat-start', '7')])
def test_recipe_flags_reject_invalid_ranges(tmp_path, extra):
    with pytest.raises(SystemExit):
        _train(tmp_path, *extra)


def test_compiled_training_observes_the_delayed_qat_transition(tmp_path, monkeypatch):
    # Exercise Dynamo's guards on the global switch without building native kernels.
    compile_ = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda m: compile_(m, backend='eager'))
    out, _ = _train(tmp_path, '--qat-start', '3', '--pos', 'alibi')
    want = torch.load(out / 'ckpt.pt', weights_only=False)['model']
    out, _ = _train(tmp_path, '--qat-start', '3', '--pos', 'alibi', '--compile')
    got = torch.load(out / 'ckpt.pt', weights_only=False)['model']
    assert all(torch.equal(want[k], got[k]) for k in want)
