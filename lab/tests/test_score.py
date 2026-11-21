"""lab.score: trim, run in the emulator, and summarize generated programs."""
import json

import numpy as np
import pytest

from lab import score as Sc
from lab.tests.conftest import DATA, ROOT, V1_CKPT, VOCAB, have_data, have_v1
from model.tokens import BOS, EOS

NL = 4  # "3F" (newline) in corpus/vocab.json


def rec(outcome, error=None, home=(), graph=False, timeout_state=None):
    return {"outcome": outcome, "error": error, "home_text": list(home), "graph": graph,
            "timeout_state": timeout_state}


# ---- statistics -------------------------------------------------------------------------------

def test_wilson_interval_known_values():
    lo, hi = Sc.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = Sc.wilson(0, 20)
    assert lo == 0 and hi == pytest.approx(0.1611, abs=1e-3)
    assert Sc.wilson(0, 0) == (0.0, 1.0)


def test_rate_record():
    r = Sc.rate(3, 12)
    assert r["k"] == 3 and r["n"] == 12 and r["p"] == 0.25
    assert r["lo"] < 0.25 < r["hi"]


# ---- strict OK (red-team definition: OK with visible output that is not only the input echo) --

@pytest.mark.parametrize("r, strict", [
    (rec("OK", home=["", "HELLO", ""]), True),
    (rec("OK", graph=True), True),
    (rec("OK", home=["X=?1", "          1"]), True),       # real output besides the echo
    (rec("OK", home=["X=?1"]), False),                   # only the harness's own input echo
    (rec("OK", home=["?1", "  ?  1"]), False),
    (rec("OK", home=["", ""]), False),                   # nothing visible
    (rec("TIMEOUT", home=["HELLO"], timeout_state="getkey"), False),
    (rec("ERR", "SYNTAX", home=["HELLO"]), False),
])
def test_strict_ok(r, strict):
    assert Sc.is_strict_ok(r) is strict


def test_outcome_metrics_rates_and_error_classes():
    recs = ([rec("OK", home=["HI"])] * 4 + [rec("OK")] * 2 + [rec("TIMEOUT", timeout_state="input")] * 2
            + [rec("ERR", "SYNTAX")] * 3 + [rec("ERR", "LABEL")] * 2 + [rec("ERR", "UNDEFINED")]
            + [rec("ERR", "DOMAIN"), rec("ERR", "INVALID DIM")]
            + [{**rec("REJECTED"), "reject_reason": "tokens"}])
    m = Sc.outcome_metrics(recs)
    n = len(recs)
    assert m["n"] == n == 17
    assert m["no_err"]["k"] == 8 and m["no_err"]["n"] == n
    assert m["ok"]["k"] == 6
    assert m["strict_ok"]["k"] == 4
    assert m["timeout"]["k"] == 2
    assert m["err_syntax"]["k"] == 3 and m["err_label"]["k"] == 2 and m["err_undefined"]["k"] == 1
    assert m["err_other"]["k"] == 2
    assert m["rejected"]["k"] == 1
    assert m["errors"] == {"SYNTAX": 3, "LABEL": 2, "UNDEFINED": 1, "DOMAIN": 1, "INVALID DIM": 1}


# ---- sample-level metrics ------------------------------------------------------------------

def test_trim_uses_eval_truncate_and_counts_empties():
    long_line = [BOS] + [10] * 300                       # no newline in 256 tokens: empty
    two_lines = [BOS] + [10] * 100 + [NL] + [11] * 200    # cut after the first line
    done = [BOS, 10, NL, 11, EOS]
    kept, n_empty = Sc.trim_samples([long_line, two_lines, done], newline_id=NL)
    assert n_empty == 1
    assert kept == [[BOS] + [10] * 100, done]


def test_complete_rate_and_lengths():
    samples = [[BOS, 5, EOS], [BOS] + [5] * 255, [BOS, 5, 6, 7, EOS], [BOS] + [5] * 254 + [EOS]]
    c = Sc.complete_rate(samples)
    assert c["k"] == 3 and c["n"] == 4
    lens = Sc.length_stats(samples)
    assert lens["mean"] == pytest.approx((2 + 255 + 4 + 255) / 4)


def test_distinct_ngrams():
    assert Sc.distinct_ngrams([[BOS, 1, 2, 3, 4, 5]], 4) == 1.0          # 2 grams, both new
    assert Sc.distinct_ngrams([[BOS, 7, 7, 7, 7, 7, 7]], 4) == pytest.approx(1 / 3)
    assert Sc.distinct_ngrams([[BOS, 1, 2, 3, 4]] * 4, 4) == pytest.approx(1 / 4)
    assert Sc.distinct_ngrams([[BOS, 1]], 4) == 0.0


def test_train_overlap_against_a_small_stream():
    rng = np.random.default_rng(0)
    train = rng.integers(4, 700, size=5000).astype(np.uint16)
    copied = [BOS] + train[100:180].tolist()                    # every 32-gram seen
    fresh = [BOS] + rng.integers(4, 700, size=80).tolist()      # none seen
    short = [BOS, 5, 6]                                         # no 32-gram at all: ignored
    ov = Sc.train_overlap([copied, fresh, short], train, n=32)
    assert ov["programs"] == 2
    assert ov["mean_seen"] == pytest.approx(0.5)
    assert ov["zero_seen"] == pytest.approx(0.5) and ov["near_copy"] == pytest.approx(0.5)


# ---- against the existing v1 evaluation --------------------------------------------------

def test_strict_ok_reproduces_the_red_team_number_on_v1_trim():
    path = ROOT / "eval" / "runs" / "v1-trim" / "results.jsonl"
    if not path.exists():
        pytest.skip("eval/runs/v1-trim not present")
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    m = Sc.outcome_metrics(recs)
    assert m["n"] == 960
    assert m["strict_ok"]["p"] == pytest.approx(0.225, abs=0.0015)   # ANGLES.md: 22.5%
    assert m["no_err"]["p"] == pytest.approx(0.588, abs=0.0015)      # RESULTS.md: 58.8%


# ---- end to end through the emulator ------------------------------------------------------

@have_data
def test_score_runs_programs_in_the_emulator(tmp_path):
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    from eval.truncate import split_programs
    progs = split_programs(np.fromfile(DATA / "val.bin", dtype="<u2")[:20000])[:6]
    train = np.fromfile(DATA / "val.bin", dtype="<u2")[:50000]   # small stand-in for train.bin
    res = Sc.score(progs, tmp_path, vocab=VOCAB, train=train, workers=2)
    assert res["n_samples"] == 6
    assert res["n_run"] + res["n_empty"] == 6
    assert (tmp_path / "results.jsonl").exists()
    for key in ("no_err", "strict_ok", "err_syntax", "err_label", "err_undefined", "err_other",
                "timeout", "complete"):
        assert 0 <= res[key]["p"] <= 1 and res[key]["lo"] <= res[key]["p"] <= res[key]["hi"]
    assert res["complete"]["k"] == sum(EOS in p[:256] for p in progs)
    assert 0 < res["distinct_4"] <= 1
    assert res["overlap_32"]["mean_seen"] > 0.9                    # they are in the stand-in


@have_v1
@have_data
def test_val_loss_of_v1_checkpoint():
    vl = Sc.val_loss(V1_CKPT, DATA, max_windows=64)
    assert 0.9 < vl["val_loss"] < 1.6 and 0.9 < vl["val_bos_loss"] < 1.6
    assert vl["windows"] == 64


# ---- sliding-window val loss over whole programs (G2a) ----------------------------------------

def _relative_ckpt(tmp_path, feat=None, shuffle=False, pos="alibi"):
    import torch
    from model.model import ModelConfig, Transformer
    cfg = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, hidden_dim=40, max_seq_len=16,
                      pos=pos, feat_cats=(4,) if feat else (), feat_bag=0, feat_shuffle=shuffle)
    torch.manual_seed(0)
    m = Transformer(cfg)
    x = torch.randint(4, 40, (2, 16))
    m.calibrate(x, torch.zeros(2, 16, 1, dtype=torch.long) if feat else None)
    m.eval()
    ck = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "config": cfg.to_dict(), "args": {}}, ck)
    return ck, m


def _programs(lengths):
    out, rng = [], np.random.default_rng(0)
    for n in lengths:
        out += [BOS, *rng.integers(4, 40, size=n).tolist(), EOS]
    return np.array(out, dtype=np.uint16)


def test_sliding_val_loss_covers_every_target_of_whole_programs(tmp_path):
    from model.data import bos_windows, write_split
    stream = _programs([5, 40, 9, 70])                    # two programs longer than the window
    write_split(tmp_path / "val.bin", stream)
    ck, m = _relative_ckpt(tmp_path)
    res = Sc.sliding_val_loss(ck, tmp_path, "cpu")
    assert res["targets"] == len(stream) - 4 and res["programs"] == 4 and res["window"] == 16
    short = _programs([5, 9])                            # fits the window: same as <bos> windows
    write_split(tmp_path / "val.bin", short)
    xs, ys = bos_windows(short, 16)
    assert Sc.sliding_val_loss(ck, tmp_path, "cpu")["sliding_val_loss"] == pytest.approx(
        Sc._loss(m, xs, ys, "cpu"), rel=1e-6)


def test_sliding_val_loss_feeds_program_aligned_or_far_shifted_features(tmp_path):
    from model.data import write_features, write_split
    stream = _programs([30, 20, 50])
    write_split(tmp_path / "val.bin", stream)
    cats = (np.arange(len(stream)) % 4)[:, None].astype(np.uint8)
    write_features(tmp_path / "f", "val", cats, np.zeros(len(stream)), {"cats": [4], "bag": 0})
    seen = []
    for shuffle in (False, True):
        _, m = _relative_ckpt(tmp_path, feat=True, shuffle=shuffle)
        seen.append([])
        Sc.sliding_val_loss(_wrap(m, seen[-1]), tmp_path, "cpu", tmp_path / "f")
    n = len(stream)
    lo = 0
    for k, prog_len in enumerate((32, 22, 52)):
        want = np.arange(lo, lo + prog_len - 1) % 4
        far = (np.arange(lo, lo + prog_len - 1) + n // 2) % n % 4
        assert np.array_equal(seen[0][k], want) and np.array_equal(seen[1][k], far)
        lo += prog_len


def _wrap(model, seen):
    fwd = model.forward

    def forward(idx, targets=None, cats=None, bag=None):
        seen.append(cats[0, :, 0].cpu().numpy())
        return fwd(idx, targets, cats, bag)
    model.forward = forward
    return model


def test_cli_passes_the_cut_length_to_score(tmp_path, monkeypatch):
    """--max-len cuts (and counts completion) at the given cap: 1024 for v2's regime (ii)."""
    seen = {}

    def fake_score(samples, out, vocab, train, workers, budget_ms, max_len=256):
        seen["max_len"] = max_len
        return {"n_samples": len(samples)}

    monkeypatch.setattr(Sc, "score", fake_score)
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"ids": [BOS, 5, EOS]}) + "\n")
    out = tmp_path / "eval"
    out.mkdir()
    Sc.main(["--samples", str(path), "--out", str(out), "--max-len", "1024"])
    assert seen["max_len"] == 1024 and json.loads((out / "score.json").read_text())["max_len"] == 1024
    Sc.main(["--samples", str(path), "--out", str(out)])
    assert seen["max_len"] == 256
