"""E8 training options (PLAN-E8.md): --teacher / --kd-alpha (distillation, model/distill.py) and
--float (a full-precision run such as the teacher's).

The neutrality tests pin the default path (no teacher, no --float) to the training code of
commit 7b0546c, before these options existed: same losses, same parameters, same files.
"""
import copy
import hashlib
import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from model import model as mm
from model.distill import (
    distill_loss, float_forward, float_sliding_val_loss, kd_loss, load_teacher, teacher_logits,
)
from model.model import ModelConfig, Transformer, load_checkpoint
from model.tests.test_train import TINY_ARGS, _tiny_dataset
from model.tokens import BOS
from model.train import best_checkpoint, eval_checkpoint, evaluate, main

ROOT = Path(__file__).resolve().parents[2]
BASE = "7b0546caa9ce1bbbcd586ecd5bda7ec4c7f42e0f"     # the branch's base, before the E8 options
TIMING = {"elapsed_s", "train_time_s"}
GOLDEN_ARGS = ("--pos", "alibi", "--avg-steps", "4", "--avg-every", "1")


def _data(tmp_path):
    if not (tmp_path / "tok" / "train.bin").exists():
        _tiny_dataset(tmp_path)
    return ["--data", str(tmp_path / "tok"), "--vocab", str(tmp_path / "vocab.json")]


def _run(tmp_path, name, *extra, train=main):
    out = tmp_path / name
    res = train([*_data(tmp_path), "--out", str(out), *TINY_ARGS, *extra])
    return out, res


def _log(out):
    return [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]


def _teacher(tmp_path, name="teacher.pt", **over):
    """A random teacher checkpoint of another shape than TINY_ARGS' student (dim 24)."""
    cfg = ModelConfig(**{"vocab_size": 30, "dim": 36, "n_layers": 1, "n_heads": 3,
                         "hidden_dim": 48, "max_seq_len": 32, **over})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        t = Transformer(cfg)
    path = tmp_path / name
    torch.save({"model": t.state_dict(), "config": cfg.to_dict(), "step": 0, "val_loss": None,
                "args": {}}, path)
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_sha(path):
    st = torch.load(path, weights_only=False)["model"]
    return hashlib.sha256(b"".join(v.numpy().tobytes() for k, v in sorted(st.items()))).hexdigest()


def _base_main(tmp_path, rev=BASE):
    """model.train.main as of rev (loaded from git), sharing today's model/ and data code."""
    src = subprocess.run(["git", "show", f"{rev}:model/train.py"], cwd=ROOT, capture_output=True,
                         text=True)
    if src.returncode:
        pytest.skip(f"base commit {rev[:7]} is not in this checkout")
    path = tmp_path / f"train_{rev[:7]}.py"
    path.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location(f"train_{rev[:7]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _outputs(out):
    """Everything a run writes, with timing fields and the output directory left out."""
    ckpts = {}
    for n in ("ckpt.pt", "best.pt", "avg.pt"):
        if (out / n).exists():
            ck = torch.load(out / n, weights_only=False)
            ck["args"] = {k: v for k, v in ck["args"].items() if k != "out"}
            ckpts[n] = ck
    log = [{k: v for k, v in r.items() if k not in TIMING} for r in _log(out)]
    res = {k: v for k, v in json.loads((out / "result.json").read_text()).items() if k not in TIMING}
    inputs = json.loads((out / "inputs.json").read_text())
    inputs["args"] = {k: v for k, v in inputs["args"].items() if k != "out"}
    return {"ckpts": ckpts, "log": log, "result": res, "inputs": inputs,
            "config": (out / "config.json").read_text()}


def _same_states(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---- training-path neutrality ------------------------------------------------------------------

# (a) Captured on BASE: CPU fp32, one thread, torch 2.14.1, the tiny config + GOLDEN_ARGS.
GOLDEN = {"ckpt": "149792e2c590284cf3b947177fa70a9fcf9ba9a6e3a57d370eac63514dd216dc",
          "avg": "fbfb4413423c5453498da79737add832325859c7a86cb9e55a4a1bb188f8af76",
          "log": "a947e86248d9f71cfbefaf33142bb854410aaa320811701dd7fc05f8fa7a2ec4",
          "val_loss": 3.400828655501057}


def _golden_run(tmp_path, name, *extra):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out, res = _run(tmp_path, name, *GOLDEN_ARGS, *extra)
    finally:
        torch.set_num_threads(old_threads)
    log = [{k: v for k, v in r.items() if k != "elapsed_s"} for r in _log(out)]
    return {"ckpt": _state_sha(out / "ckpt.pt"), "avg": _state_sha(out / "avg.pt"),
            "log": hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest(),
            "val_loss": res["val_loss"]}


def test_the_default_path_matches_the_golden_values_of_the_base_commit(tmp_path):
    assert _golden_run(tmp_path, "run") == GOLDEN


def test_a_teacher_at_alpha_zero_leaves_the_golden_states_unchanged(tmp_path):
    """(b) teacher loading and its forward move no RNG state and leave the QAT switch alone."""
    got = _golden_run(tmp_path, "kd", "--teacher", str(_teacher(tmp_path, pos="alibi")),
                      "--kd-alpha", "0")
    assert (got["ckpt"], got["avg"], got["val_loss"]) == \
        (GOLDEN["ckpt"], GOLDEN["avg"], GOLDEN["val_loss"])


# (c) The model code is unchanged: sha256 of these files at commit 20d990f.
MODEL_FILES = {"model/model.py": "46e60ada346051f473b3541e6a9c33d7295db9e7364125ffe5312bb8d4b39418",
               "model/intref.py": "132c9a3fa4907042975f90026c840ab712ef50a66519bc0bc6cf9d96426d510d",
               "model/export.py": "8e26da61ea8e038d3678481835d38eeb97725b7f5e16085752df13bd1fd6563b"}


@pytest.mark.parametrize("path", sorted(MODEL_FILES))
def test_the_model_code_is_byte_identical_to_20d990f(path):
    assert _sha(ROOT / path) == MODEL_FILES[path]


CONFIGS = [(), ("--amp", "bf16"), ("--qat-start", "3", "--avg-steps", "3", "--pos", "alibi"),
           ("--qat-start", "6")]


@pytest.mark.parametrize("extra", CONFIGS)
def test_the_default_path_writes_exactly_what_the_base_commit_wrote(tmp_path, extra):
    old, _ = _run(tmp_path, "old", *extra, train=_base_main(tmp_path))
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


@pytest.mark.parametrize("extra", [(), ("--amp", "bf16"), ("--qat-start", "3")])
def test_a_teacher_at_alpha_zero_trains_exactly_like_cross_entropy(tmp_path, extra):
    plain, _ = _run(tmp_path, "plain", *extra)
    kd, _ = _run(tmp_path, "kd", *extra, "--teacher", str(_teacher(tmp_path)), "--kd-alpha", "0")
    a, b = _outputs(plain), _outputs(kd)
    for n in ("ckpt.pt", "best.pt"):
        assert _same_states(a["ckpts"][n]["model"], b["ckpts"][n]["model"]), n
    strip = lambda log: [{k: v for k, v in r.items() if k not in ("ce_loss", "kd_loss")} for r in log]  # noqa: E731
    assert strip(a["log"]) == strip(b["log"])
    assert all(r["ce_loss"] == r["train_loss"] and r["kd_loss"] > 0
               for r in b["log"] if "train_loss" in r)


# ---- the distillation loss ---------------------------------------------------------------------

def _logits(seed, shape=(2, 5, 7)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * 3


def test_kd_loss_is_zero_for_identical_logits():
    s = _logits(0)
    assert kd_loss(s, s.clone()) == 0


def test_kd_loss_is_the_forward_kl_summed_over_the_vocabulary_and_averaged_over_positions():
    s, t = _logits(1), _logits(2)
    want = 0.0
    for b in range(2):
        for i in range(5):
            p, q = torch.softmax(t[b, i].double(), -1), torch.softmax(s[b, i].double(), -1)
            want += float((p * (p.log() - q.log())).sum())
    got = kd_loss(s.to(torch.bfloat16), t.to(torch.bfloat16))
    assert got.dtype == torch.float32
    assert float(kd_loss(s, t)) == pytest.approx(want / 10, rel=1e-5)
    assert float(kd_loss(t, s)) != pytest.approx(want / 10, rel=1e-3)      # forward, not reverse


def test_the_mixed_loss_is_exactly_ce_at_alpha_zero_and_exactly_kd_at_alpha_one():
    s = _logits(5).requires_grad_()
    t = _logits(6)
    y = torch.randint(0, 7, (2, 5))
    ce = torch.nn.functional.cross_entropy(s.flatten(0, 1), y.flatten(), ignore_index=-100)
    kd = kd_loss(s, t)
    assert torch.equal(distill_loss(ce, kd, 0.0), ce)
    assert torch.equal(distill_loss(ce, kd, 1.0), kd)
    assert torch.equal(distill_loss(ce, kd, 0.25), 0.25 * kd + 0.75 * ce)
    g_ce, = torch.autograd.grad(ce, s, retain_graph=True)
    g_mix, = torch.autograd.grad(distill_loss(ce, kd, 0.0), s)
    assert torch.equal(g_ce, g_mix)


def test_kd_is_zero_when_the_teacher_has_the_student_weights():
    cfg = ModelConfig(vocab_size=30, dim=24, n_layers=2, n_heads=3, hidden_dim=40, max_seq_len=32)
    torch.manual_seed(0)
    student = Transformer(cfg)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    x, y = torch.randint(4, 30, (2, 16)), torch.randint(4, 30, (2, 16))
    try:
        mm.set_quant_enabled(False)                   # a student training in full precision
        s_logits, _ = student(x, y)
    finally:
        mm.set_quant_enabled(True)
    assert kd_loss(s_logits, teacher_logits(teacher, x)) == 0


# ---- the teacher: full precision, frozen, the student's QAT untouched --------------------------

def _student_cfg(**over):
    return ModelConfig(**{"vocab_size": 30, "max_seq_len": 32, **over})


@pytest.mark.parametrize("quant", [True, False])
def test_the_teacher_runs_in_full_precision_and_restores_flag_and_mode(tmp_path, quant):
    teacher = load_teacher(_teacher(tmp_path), _student_cfg(), "cpu")
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    state = {k: v.clone() for k, v in teacher.state_dict().items()}
    x = torch.tensor([[BOS, 5, 6, 7, 8]])
    try:
        mm.set_quant_enabled(quant)
        got = teacher_logits(teacher, x)
        assert mm.QUANT_ENABLED is quant and not teacher.training and not got.requires_grad
        mm.set_quant_enabled(False)
        teacher.train()
        with torch.no_grad():
            want = teacher(x)[0]                       # the float training forward
        teacher.eval()
        with torch.no_grad():
            quantized = teacher(x)[0]                  # eval mode always quantizes
    finally:
        mm.set_quant_enabled(True)
        teacher.eval()
    assert torch.equal(got, want) and not torch.equal(got, quantized)
    assert _same_states(teacher.state_dict(), state)


def test_the_flag_and_mode_are_restored_when_the_teacher_forward_raises(tmp_path, monkeypatch):
    teacher = load_teacher(_teacher(tmp_path), _student_cfg(), "cpu")

    def fail(*a, **kw):
        assert not mm.QUANT_ENABLED and teacher.training
        raise RuntimeError("synthetic teacher failure")

    monkeypatch.setattr(teacher, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        teacher_logits(teacher, torch.tensor([[BOS, 5]]))
    assert mm.QUANT_ENABLED and not teacher.training


def test_the_students_qat_schedule_is_unaffected_by_the_teacher(tmp_path, monkeypatch):
    calls = []
    forward = mm.Transformer.forward

    def observe(self, idx, targets=None, *f, **kw):
        if self.training:
            calls.append((self.cfg.dim, mm.QUANT_ENABLED, targets is not None))
        return forward(self, idx, targets, *f, **kw)

    monkeypatch.setattr(mm.Transformer, "forward", observe)
    _run(tmp_path, "kd", "--qat-start", "3", "--teacher", str(_teacher(tmp_path)))
    assert [q for d, q, tgt in calls if d == 24 and tgt] == [False] * 3 + [True] * 3
    assert [(q, tgt) for d, q, tgt in calls if d == 36] == [(False, False)] * 6
    assert mm.QUANT_ENABLED


@pytest.mark.parametrize("over", [{"vocab_size": 31}, {"max_seq_len": 16}, {"pos": "alibi"}])
def test_the_teacher_must_share_the_students_vocabulary_context_and_positions(tmp_path, over):
    with pytest.raises(ValueError, match="does not match"):
        _run(tmp_path, "kd", "--teacher", str(_teacher(tmp_path, **over)))


def test_a_teacher_with_input_features_is_refused(tmp_path):
    with pytest.raises(ValueError, match="features"):
        load_teacher(_teacher(tmp_path, feat_cats=(3,)), _student_cfg(), "cpu")


def test_loading_the_teacher_moves_no_global_rng_state(tmp_path):
    path = _teacher(tmp_path)
    torch.manual_seed(0)
    load_teacher(path, _student_cfg(), "cpu")
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(torch.rand(3), after)


@pytest.mark.parametrize("alpha", [None, "1.0", "0.5"])
def test_the_logged_loss_is_the_mix_of_kd_and_ce(tmp_path, alpha):
    t = str(_teacher(tmp_path))
    out, res = _run(tmp_path, "kd", "--teacher", t, *(("--kd-alpha", alpha) if alpha else ()))
    a = float(alpha or 0.9)                          # 0.9 by default
    assert res["distill"]["kd_alpha"] == a
    rows = [r for r in _log(out) if "train_loss" in r]
    assert rows and all(r["train_loss"] == pytest.approx(a * r["kd_loss"] + (1 - a) * r["ce_loss"],
                                                         rel=1e-6) for r in rows)
    if a == 1.0:
        assert all(r["train_loss"] == r["kd_loss"] for r in rows)


# ---- flags and records -------------------------------------------------------------------------

def test_kd_alpha_needs_a_teacher(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "x", "--kd-alpha", "0.5")


@pytest.mark.parametrize("alpha", ["-0.1", "1.5", "nan"])
def test_kd_alpha_must_be_between_zero_and_one(tmp_path, alpha):
    with pytest.raises(SystemExit):
        _run(tmp_path, "x", "--teacher", str(_teacher(tmp_path)), "--kd-alpha", alpha)


def test_a_missing_teacher_file_is_an_argument_error(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "x", "--teacher", str(tmp_path / "nope.pt"))


def test_a_distillation_run_records_its_teacher(tmp_path):
    t = _teacher(tmp_path)
    out, res = _run(tmp_path, "kd", "--teacher", str(t), "--kd-alpha", "0.9")
    want = {"teacher": str(t), "teacher_sha256": _sha(t), "kd_alpha": 0.9}
    assert res["distill"] == want and json.loads((out / "inputs.json").read_text())["distill"] == want
    args = json.loads((out / "inputs.json").read_text())["args"]
    assert (args["teacher"], args["kd_alpha"], args["float"]) == (str(t), "0.9", "False")
    assert res["gpu_s"] is None                      # a CPU run
    ck = torch.load(out / "best.pt", weights_only=False)
    assert res["best"] == {"ckpt": "best.pt", "step": ck["step"], "val_loss": ck["val_loss"]}


def test_the_default_path_records_none_of_the_new_options(tmp_path):
    out, res = _run(tmp_path, "run")
    new = {"teacher", "kd_alpha", "float"}
    assert not new & set(json.loads((out / "inputs.json").read_text())["args"])
    assert not new & set(torch.load(out / "ckpt.pt", weights_only=False)["args"])
    assert not {"distill", "best", "gpu_s"} & set(res)


# ---- --float -----------------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [("--qat-start", "3"), ("--feat-dir", "feats")])
def test_float_refuses_qat_and_features(tmp_path, extra):
    with pytest.raises(SystemExit):
        _run(tmp_path, "x", "--float", *extra)


def test_float_refuses_a_teacher(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "x", "--float", "--teacher", str(_teacher(tmp_path)))


def test_a_float_run_never_quantizes_or_recalibrates(tmp_path, monkeypatch):
    phases, calibrations = [], []
    forward, calibrate = mm.Transformer.forward, mm.Transformer.calibrate

    def observe(self, idx, targets=None, *f, **kw):
        phases.append((self.training, mm.QUANT_ENABLED))
        return forward(self, idx, targets, *f, **kw)

    def observe_calibration(self, idx, *f):
        calibrations.append(1)
        return calibrate(self, idx, *f)

    monkeypatch.setattr(mm.Transformer, "forward", observe)
    monkeypatch.setattr(mm.Transformer, "calibrate", observe_calibration)
    _run(tmp_path, "fp", "--float", "--avg-steps", "2", "--avg-every", "1")
    assert len(calibrations) == 1                    # the start-up one only, as in every run
    after = phases[1:]                               # phases[0] is that calibration's forward
    assert after and all(p == (True, False) for p in after)
    assert mm.QUANT_ENABLED


def test_float_best_is_the_checkpoint_with_the_lowest_float_val_loss(tmp_path):
    out, res = _run(tmp_path, "fp", "--float", "--eval-every", "2")
    evals = [r for r in _log(out) if "val_loss" in r]
    assert [r["step"] for r in evals] == [0, 2, 4, 6]
    best = min(evals, key=lambda r: r["val_loss"])               # the first of any tie
    ck = torch.load(out / "best.pt", weights_only=False)
    assert (ck["step"], ck["val_loss"]) == (best["step"], best["val_loss"])
    assert res["best"] == {"ckpt": "best.pt", "step": best["step"], "val_loss": best["val_loss"]}
    assert best_checkpoint(out) == out / "best.pt" == eval_checkpoint(out)
    from model.data import load_split, val_windows
    xs, ys = val_windows(load_split(tmp_path / "tok" / "val.bin"), 32)
    model = load_checkpoint(out / "best.pt")
    assert evaluate(model, xs, ys, "cpu", nullcontext, fp=True) == best["val_loss"]
    assert not model.training and mm.QUANT_ENABLED     # the float forward restored both
    assert evaluate(model, xs, ys, "cpu", nullcontext) != best["val_loss"]   # not the ternary loss


def test_float_ties_go_to_the_earlier_step(tmp_path, monkeypatch):
    from model import train as tr
    monkeypatch.setattr(tr, "evaluate", lambda *a, **kw: 1.0)
    out, res = _run(tmp_path, "fp", "--float")
    assert torch.load(out / "best.pt", weights_only=False)["step"] == 0 == res["best"]["step"]


def test_a_float_run_that_never_improves_fails_instead_of_recording_no_best(tmp_path, monkeypatch):
    from model import train as tr
    monkeypatch.setattr(tr, "evaluate", lambda *a, **kw: float("nan"))
    with pytest.raises(RuntimeError, match="best.pt was never written"):
        _run(tmp_path, "fp", "--float")


def test_best_checkpoint_needs_a_run_that_records_its_best(tmp_path):
    out, _ = _run(tmp_path, "run", "--qat-start", "6")
    with pytest.raises(ValueError):
        best_checkpoint(out)


def test_a_float_run_with_averaging_still_evaluates_the_average(tmp_path):
    out, res = _run(tmp_path, "fp", "--float", "--avg-steps", "2", "--avg-every", "1")
    assert eval_checkpoint(out) == out / "avg.pt" and best_checkpoint(out) == out / "best.pt"
    from model.data import load_split, val_windows
    xs, ys = val_windows(load_split(tmp_path / "tok" / "val.bin"), 32)
    avg = load_checkpoint(out / "avg.pt")
    assert evaluate(avg, xs, ys, "cpu", nullcontext, fp=True) == res["avg_val_loss"]


def test_a_float_run_serves_as_a_teacher(tmp_path):
    t_out, _ = _run(tmp_path, "teacher", "--float", "--dim", "36", "--n-layers", "1")
    _, res = _run(tmp_path, "student", "--teacher", str(best_checkpoint(t_out)))
    assert np.isfinite(res["val_loss"]) and res["distill"]["kd_alpha"] == 0.9


# ---- float sliding val loss --------------------------------------------------------------------

def test_float_sliding_val_loss_is_lab_scores_computation_with_the_float_forward(tmp_path):
    from lab.score import sliding_val_loss
    out, _ = _run(tmp_path, "fp", "--float", "--pos", "alibi")
    data = tmp_path / "tok"
    got = float_sliding_val_loss(out / "best.pt", data, "cpu")
    model = load_checkpoint(out / "best.pt")
    with float_forward(model):
        want = sliding_val_loss(model, data, "cpu")
    assert got == want
    assert sliding_val_loss(out / "best.pt", data, "cpu") != got          # the quantized loss
    assert mm.QUANT_ENABLED


def test_float_sliding_val_loss_needs_relative_positions(tmp_path):
    with pytest.raises(ValueError):
        float_sliding_val_loss(_teacher(tmp_path), tmp_path / "tok", "cpu")


# ---- the E8 teacher shapes ---------------------------------------------------------------------

@pytest.mark.parametrize("dim,heads,hidden", [(384, 6, 1152), (512, 8, 1536)])
def test_the_e8_teacher_shapes_build_and_run_a_float_forward(dim, heads, hidden):
    cfg = ModelConfig(vocab_size=781, dim=dim, n_layers=8, n_heads=heads, n_kv_heads=1,
                      hidden_dim=hidden, max_seq_len=256, pos="alibi")
    assert cfg.head_dim == 64
    with torch.random.fork_rng(devices=[]):
        teacher = Transformer(cfg).eval().requires_grad_(False)
    n = sum(p.numel() for p in teacher.parameters())
    assert {384: 13e6, 512: 24e6}[dim] * 0.8 < n < {384: 13e6, 512: 24e6}[dim] * 1.2
    logits = teacher_logits(teacher, torch.randint(4, 781, (1, 40)))
    assert logits.shape == (1, 40, 781) and torch.isfinite(logits).all()


# ---- optional CUDA smoke -------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_smoke_distills_a_tiny_student(tmp_path):
    t = _teacher(tmp_path)
    out = tmp_path / "cuda"
    res = main([*_data(tmp_path), "--out", str(out), "--steps", "8", "--batch", "8",
                "--eval-every", "4", "--device", "cuda", "--amp", "bf16", "--dim", "24",
                "--n-layers", "2", "--n-heads", "3", "--hidden-dim", "40", "--ctx", "32",
                "--teacher", str(t)])
    assert np.isfinite(res["val_loss"]) and res["gpu_s"] > 0


def test_float_sliding_val_loss_in_chunks_equals_the_whole_program(tmp_path):
    """E8 gate: whole long val programs ran out of GPU memory for the 24M teacher. Chunks fed with
    n_layers * (window - 1) tokens of context give the same loss as one pass."""
    from model.data import write_split
    cfg = ModelConfig(vocab_size=30, dim=24, n_layers=2, n_heads=2, hidden_dim=32, max_seq_len=8,
                      pos="alibi")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model = Transformer(cfg)
    rng = np.random.default_rng(1)
    progs = [[BOS, *rng.integers(4, 30, size=n).tolist(), 2] for n in (200, 7, 45)]
    data = tmp_path / "tok"
    write_split(data / "val.bin", np.array([t for p in progs for t in p], dtype=np.uint16))
    whole = float_sliding_val_loss(model, data, "cpu", chunk=0)
    for chunk in (5, 16, 64):
        part = float_sliding_val_loss(model, data, "cpu", chunk=chunk)
        assert part["targets"] == whole["targets"]
        assert abs(part["sliding_val_loss"] - whole["sliding_val_loss"]) < 1e-6
    assert mm.QUANT_ENABLED
