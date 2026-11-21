"""lab.run: one command per experiment (train or --ckpt, budget, sample, score, record)."""
import json
from pathlib import Path

import pytest

from lab import run as R
from lab.tests.conftest import DATA, have_data
from model.train import parse_args as train_parse


def test_train_args_default_to_the_v1_recipe(tmp_path):
    lab, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path)])
    assert lab.name == "x" and lab.ckpt is None
    a = train_parse(train_argv)
    assert (a.steps, a.batch, a.lr, a.min_lr, a.warmup, a.wd, a.p_bos, a.compile, a.seed) == \
        (15000, 128, 0.005, 0.0001, 500, 0.05, 0.25, True, 1337)
    assert (a.dim, a.n_layers, a.n_heads, a.hidden_dim, a.ctx) == (192, 6, 6, 512, 256)
    assert a.out == tmp_path / "x" / "train"


def test_unknown_args_go_to_training_and_override_the_recipe(tmp_path):
    lab, train_argv = R.parse(["--name", "wide", "--dim", "256", "--seed", "7", "--steps", "100",
                               "--samples", "50", "--out-root", str(tmp_path)])
    a = train_parse(train_argv)
    assert (a.dim, a.seed, a.steps, a.lr) == (256, 7, 100, 0.005)
    assert lab.samples == 50


def test_lab_defaults_and_no_compile(tmp_path):
    lab, train_argv = R.parse(["--name", "x", "--no-compile", "--out-root", str(tmp_path)])
    assert (lab.samples, lab.seed0, lab.temp, lab.top_k, lab.workers) == (1000, 1, 0.8, 20, 24)
    assert not train_parse(train_argv).compile


def test_name_is_required_and_must_be_a_plain_name(tmp_path):
    with pytest.raises(SystemExit):
        R.parse([])
    with pytest.raises(SystemExit):
        R.parse(["--name", "../escape", "--out-root", str(tmp_path)])


def test_completed_training_with_the_same_args_is_reused(tmp_path):
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10"])
    out = tmp_path / "x" / "train"
    assert R.needs_training(train_argv)
    out.mkdir(parents=True)
    a = train_parse(train_argv)
    (out / "inputs.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}}))
    (out / "best.pt").write_bytes(b"x")
    (out / "result.json").write_text("{}")
    assert not R.needs_training(train_argv)
    _, other = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "11"])
    assert R.needs_training(other)


def test_a_restarted_training_is_not_mistaken_for_a_finished_one(tmp_path):
    """train.py writes inputs.json first and result.json last: a result.json older than
    inputs.json belongs to an earlier run in the same directory."""
    import os
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10"])
    out = tmp_path / "x" / "train"
    out.mkdir(parents=True)
    a = train_parse(train_argv)
    (out / "result.json").write_text("{}")
    (out / "best.pt").write_bytes(b"x")
    (out / "inputs.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}}))
    later = (out / "result.json").stat().st_mtime + 10
    os.utime(out / "inputs.json", (later, later))
    assert R.needs_training(train_argv)


def test_agreement_contexts_stop_at_the_end_of_each_program():
    import numpy as np
    stream = np.array([1, 5, 6, 2, 1, 7, 2, 1] + [8] * 300 + [2], dtype=np.uint16)
    ctxs = R._agreement_contexts(stream, n=3, length=16)
    assert ctxs == [[1, 5, 6], [1, 7], [1] + [8] * 15]


FAKE = {
    "name": "demo",
    "budget": {"params": {"ternary": 2_435_520, "embedding": 199_104},
               "flash": {"appvar_file_bytes": 727_923}, "ram": {"total": 177_761},
               "tokens_per_s": {"avg": 0.947}},
    "val": {"val_loss": 1.2287},
    "score": {"n_samples": 1000, "n_run": 960,
              **{k: {"p": p, "lo": p - 0.03, "hi": p + 0.03} for k, p in
                 [("complete", 0.083), ("no_err", 0.588), ("strict_ok", 0.225),
                  ("err_syntax", 0.066), ("err_label", 0.143), ("err_undefined", 0.149)]},
              "distinct_4": 0.412, "overlap_32": {"mean_seen": 0.041}},
}


def test_results_row_has_one_cell_per_header_column():
    row = R.results_row(FAKE)
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    header = [c.strip() for c in R.HEADER.strip().strip("|").split("|")]
    assert len(cells) == len(header)
    assert cells[0] == "demo"
    assert "710.9" in row          # flash KB
    assert "58.8" in row and "22.5" in row and "8.3" in row
    assert "1.2287" in row and "0.95" in row


def test_append_row_creates_the_table_then_appends(tmp_path):
    md = tmp_path / "RESULTS.md"
    R.append_row(md, R.results_row(FAKE))
    R.append_row(md, R.results_row({**FAKE, "name": "second"}))
    text = md.read_text()
    assert text.count(R.HEADER.strip()) == 1
    lines = text.strip().splitlines()
    assert lines[-1].startswith("| second |") and lines[-2].startswith("| demo |")


@have_data
def test_end_to_end_smoke(tmp_path):
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    md = tmp_path / "RESULTS.md"
    res = R.main(["--name", "smoke", "--out-root", str(tmp_path), "--results", str(md),
                  "--samples", "4", "--workers", "2", "--val-windows", "16", "--no-compile",
                  "--steps", "20", "--eval-every", "20", "--eval-windows", "4", "--batch", "8",
                  "--dim", "40", "--n-layers", "1", "--n-heads", "2", "--hidden-dim", "64",
                  "--data", str(DATA)])
    run = tmp_path / "smoke"
    assert (run / "train" / "best.pt").exists()
    saved = json.loads((run / "result.json").read_text())
    assert saved["name"] == "smoke" and saved["score"]["n_samples"] == 4
    assert saved["budget"]["config"]["dim"] == 40
    assert saved["budget"]["flash"]["exact"]
    assert len((run / "samples.jsonl").read_text().splitlines()) == 4
    assert md.read_text().strip().splitlines()[-1].startswith("| smoke |")
    assert res["score"]["n_samples"] == 4


def _finish(train_argv, *names, result=None):
    a = train_parse(train_argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "inputs.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}}))
    for n in names:
        (a.out / n).write_bytes(b"x")
    (a.out / "result.json").write_text(json.dumps(result or {}))
    return a


def test_an_averaging_run_is_finished_only_with_its_averaged_checkpoint(tmp_path):
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--avg-steps", "3000"])
    a = _finish(train_argv, "best.pt", result={"eval_ckpt": "avg.pt"})
    assert R.needs_training(train_argv)
    (a.out / "avg.pt").write_bytes(b"x")
    assert not R.needs_training(train_argv)


def test_arguments_added_later_at_their_defaults_keep_a_finished_run(tmp_path):
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10"])
    a = _finish(train_argv, "best.pt")
    saved = json.loads((a.out / "inputs.json").read_text())
    del saved["args"]["avg_steps"], saved["args"]["avg_every"]      # a run from before the flags
    (a.out / "inputs.json").write_text(json.dumps(saved))
    (a.out / "result.json").write_text("{}")                          # written last, as train.py does
    assert not R.needs_training(train_argv)
    _, averaged = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10",
                           "--avg-steps", "5"])
    assert R.needs_training(averaged)


def test_the_evaluation_checkpoint_follows_the_training_result(tmp_path):
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--avg-steps", "3000"])
    lab, _ = R.parse(["--name", "x", "--out-root", str(tmp_path)])
    a = _finish(train_argv, "best.pt", "avg.pt", result={"eval_ckpt": "avg.pt"})
    ckpt, how = R._train_or_reuse(lab, train_argv, a)
    assert (ckpt, how) == (a.out / "avg.pt", "reused")


# ---- v2: features, masks ---------------------------------------------------------------------

def test_feature_and_averaging_flags_pass_through_to_training(tmp_path):
    lab, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--feat-dir", "feats",
                               "--feat-shuffle", "--avg-steps", "3000", "--masks"])
    a = train_parse(train_argv)
    assert a.feat_dir == Path("feats") and a.feat_shuffle and a.avg_steps == 3000
    assert lab.masks


def test_masks_are_off_by_default(tmp_path):
    lab, _ = R.parse(["--name", "x", "--out-root", str(tmp_path)])
    assert not lab.masks


def test_the_feature_directory_comes_from_the_checkpoint_when_scoring_one():
    assert R.feature_dir({"args": {"feat_dir": "/x/feats"}}) == Path("/x/feats")
    assert R.feature_dir({"args": {"feat_dir": "None"}}) is None
    assert R.feature_dir({"args": {}}) is None
    assert R.feature_dir({"args": {"feat_dir": "data/feats"}}) == R.ROOT / "data" / "feats"


def test_changed_feature_files_force_retraining(tmp_path):
    from model.data import write_features
    from model.train import FEATURE_FILES
    import numpy as np
    d = tmp_path / "feats"
    for split in ("train", "val"):
        write_features(d, split, np.zeros((3, 1)), np.zeros(3), {"cats": [2], "bag": 0})
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--feat-dir", str(d)])
    a = _finish(train_argv, "best.pt")
    sha = {str(d / f): R._sha(d / f) for f in FEATURE_FILES}
    saved = json.loads((a.out / "inputs.json").read_text())
    (a.out / "inputs.json").write_text(json.dumps({**saved, "feat_sha256": sha}))
    (a.out / "result.json").write_text("{}")
    assert not R.needs_training(train_argv)
    write_features(d, "val", np.ones((3, 1)), np.zeros(3), {"cats": [2], "bag": 0})
    assert R.needs_training(train_argv)


def test_results_row_marks_masked_runs():
    assert "| demo+masks |" in R.results_row({**FAKE, "name": "demo", "masks": True})
    assert "| demo |" in R.results_row({**FAKE, "name": "demo", "masks": False})


@have_data
def test_end_to_end_smoke_with_features(tmp_path):
    """Toy step-function features on a slice of the real data: train, export with feature
    calibration, sample with the tracker running alongside, score, val loss with features."""
    import numpy as np
    from eval import emu as E
    from model.data import load_split, program_starts, stream_features, write_features, write_split
    from lab.tests.test_sample import Toy
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    tok = tmp_path / "tok"
    for split, n in (("train", 60), ("val", 12)):
        s = load_split(DATA / f"{split}.bin")
        s = s[: program_starts(s)[n]]
        write_split(tok / f"{split}.bin", s)
        write_features(tmp_path / "feats", split, *stream_features(s, Toy()),
                       {"cats": [8, 3], "bag": 10, "tracker": "lab.tests.test_sample:Toy"})
    md = tmp_path / "RESULTS.md"
    res = R.main(["--name", "fsmoke", "--out-root", str(tmp_path), "--results", str(md),
                  "--samples", "4", "--workers", "2", "--no-compile", "--steps", "20",
                  "--eval-every", "20", "--batch", "8", "--dim", "40", "--n-layers", "1",
                  "--n-heads", "2", "--hidden-dim", "64", "--data", str(tok),
                  "--feat-dir", str(tmp_path / "feats"), "--avg-steps", "10", "--avg-every", "5"])
    assert res["ckpt"].endswith("avg.pt")
    assert res["features"]["cats"] == [8, 3] and res["intref_agreement"].get("skipped")
    assert res["budget"]["extras"]["bag_bits"]["source"] == "given"
    assert res["budget"]["flash"]["parts"]["embeddings"] == (781 + 256 + 8 + 3 + 10) * 40
    assert np.isfinite(res["val"]["val_loss"]) and res["score"]["n_samples"] == 4


def test_sampling_refuses_a_tracker_that_differs_from_the_one_behind_the_features():
    from lab.features import tracker_identity
    try:
        now = tracker_identity()
    except FileNotFoundError:
        pytest.skip("tracker sources not present")
    meta = {"tracker": "tracker.bind:Tracker", "identity": now}
    R.check_tracker_identity(meta)                                   # same build: fine
    R.check_tracker_identity({"tracker": "lab.tests.test_sample:Toy"})   # not the C tracker
    with pytest.raises(ValueError):
        R.check_tracker_identity({**meta, "identity": {**now, "tracker.c": "0" * 64}})


def test_train_only_stops_after_training_and_is_reused_later(tmp_path):
    argv = ["--name", "x", "--out-root", str(tmp_path), "--avg-steps", "3000", "--pos", "alibi"]
    lab, train_argv = R.parse(argv + ["--train-only"])
    assert lab.train_only and train_parse(train_argv).pos == "alibi"
    assert R.parse(argv)[1] == train_argv                  # same training arguments either way
    a = _finish(train_argv, "best.pt", "avg.pt", result={"eval_ckpt": "avg.pt"})
    res = R.main(argv + ["--train-only"])
    assert res == {"name": "x", "ckpt": str(a.out / "avg.pt"), "training": "reused"}
    assert not (tmp_path / "x" / "result.json").exists()


def test_cap_defaults_to_the_context(tmp_path):
    assert R.parse(["--name", "x", "--out-root", str(tmp_path)])[0].cap == 0
    assert R.parse(["--name", "x", "--out-root", str(tmp_path), "--cap", "1024"])[0].cap == 1024


@have_data
def test_end_to_end_smoke_past_the_window(tmp_path):
    """ALiBi model: samples and programs judged up to the cap, sliding-window val loss."""
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    res = R.main(["--name", "capsmoke", "--out-root", str(tmp_path), "--results",
                  str(tmp_path / "RESULTS.md"), "--samples", "4", "--workers", "2", "--no-compile",
                  "--steps", "20", "--eval-every", "20", "--eval-windows", "4", "--batch", "8",
                  "--dim", "40", "--n-layers", "1", "--n-heads", "2", "--hidden-dim", "64",
                  "--ctx", "32", "--pos", "alibi", "--cap", "80", "--val-windows", "8",
                  "--data", str(DATA)])
    assert res["cap"] == 80 and res["budget"]["cap"]["tokens"] == 80
    assert res["val"]["sliding_val_loss"] > 0
    agree = res["intref_agreement"]            # ALiBi without features: intref v2 runs it,
    assert "skipped" not in agree and agree["top1_agree"] > 0.8         # here 96.5%
    assert agree["n_positions"] > 32 * agree["n_contexts"]   # contexts longer than the window
    samples = [json.loads(x)["ids"] for x in (tmp_path / "capsmoke" / "samples.jsonl").read_text().splitlines()]
    assert max(len(s) for s in samples) <= 80


# ---- E8: distillation and full-precision runs --------------------------------------------------

def test_distillation_and_float_flags_pass_through_to_training(tmp_path):
    t = tmp_path / "teacher.pt"
    t.write_bytes(b"x")
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--teacher", str(t),
                             "--kd-alpha", "0.5"])
    a = train_parse(train_argv)
    assert (a.teacher, a.kd_alpha, a.float) == (t, 0.5, False)
    a = train_parse(R.parse(["--name", "x", "--out-root", str(tmp_path), "--float",
                             "--train-only"])[1])
    assert (a.teacher, a.kd_alpha, a.float) == (None, None, True)
    with pytest.raises(SystemExit):        # the scoring pipeline would quantize a float model
        R.parse(["--name", "x", "--out-root", str(tmp_path), "--float"])
    a = train_parse(R.parse(["--name", "x", "--out-root", str(tmp_path)])[1])
    assert (a.teacher, a.kd_alpha, a.float) == (None, None, False)


def test_a_changed_teacher_forces_retraining(tmp_path):
    t = tmp_path / "teacher.pt"
    t.write_bytes(b"x")
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--teacher", str(t),
                             "--kd-alpha", "0.9"])
    a = _finish(train_argv, "best.pt")
    saved = json.loads((a.out / "inputs.json").read_text())
    (a.out / "inputs.json").write_text(json.dumps({**saved, "distill": {"teacher_sha256": R._sha(t)}}))
    (a.out / "result.json").write_text("{}")
    assert not R.needs_training(train_argv)
    t.write_bytes(b"y")
    assert R.needs_training(train_argv)


# ---- --init: fine-tuning -----------------------------------------------------------------------

def test_init_passes_through_to_training(tmp_path):
    ck = tmp_path / "init.pt"
    ck.write_bytes(b"x")
    a = train_parse(R.parse(["--name", "x", "--out-root", str(tmp_path), "--init", str(ck)])[1])
    assert a.init == ck
    assert train_parse(R.parse(["--name", "x", "--out-root", str(tmp_path)])[1]).init is None


def test_a_changed_init_checkpoint_forces_retraining(tmp_path):
    ck = tmp_path / "init.pt"
    ck.write_bytes(b"x")
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--init", str(ck)])
    a = _finish(train_argv, "best.pt")
    saved = json.loads((a.out / "inputs.json").read_text())
    assert R.needs_training(train_argv)                 # no init sha recorded
    init = {"path": str(ck), "sha256": R._sha(ck)}
    (a.out / "inputs.json").write_text(json.dumps({**saved, "init": init}))
    (a.out / "result.json").write_text("{}")
    assert not R.needs_training(train_argv)
    ck.write_bytes(b"y")
    assert R.needs_training(train_argv)


def test_a_run_saved_before_init_existed_is_still_reused(tmp_path):
    _, train_argv = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10"])
    a = _finish(train_argv, "best.pt")
    saved = json.loads((a.out / "inputs.json").read_text())
    del saved["args"]["init"]                           # what train.py records without --init
    (a.out / "inputs.json").write_text(json.dumps(saved))
    (a.out / "result.json").write_text("{}")
    assert not R.needs_training(train_argv)
    ck = tmp_path / "init.pt"
    ck.write_bytes(b"x")
    _, with_init = R.parse(["--name", "x", "--out-root", str(tmp_path), "--steps", "10",
                            "--init", str(ck)])
    assert R.needs_training(with_init)
