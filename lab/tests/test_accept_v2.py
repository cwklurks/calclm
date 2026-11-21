"""lab/accept_v2.py: the v2 acceptance criteria of PLAN-v2.md's shipping rules, from score files."""
import pytest

from lab import accept_v2 as A


def _rate(p, lo=None, hi=None):
    return {"p": p, "lo": p - 0.03 if lo is None else lo, "hi": p + 0.03 if hi is None else hi}


def _score(strict, syn, lab, und, overlap, complete=0.1):
    return {"strict_ok": _rate(strict), "err_syntax": _rate(syn), "err_label": _rate(lab),
            "err_undefined": _rate(und), "overlap_32": {"mean_seen": overlap},
            "complete": _rate(complete), "no_err": _rate(0.5), "distinct_4": 0.2,
            "length": {"mean": 200.0}, "n_run": 990, "n_samples": 1000}


def test_criteria_pass_when_every_rule_holds():
    v1 = _score(0.22, 0.07, 0.14, 0.15, 0.034)
    v2 = _score(0.30, 0.03, 0.05, 0.10, 0.040)            # L+U+S 0.18 <= 0.5 * 0.36
    long = _score(0.18, 0.07, 0.2, 0.1, 0.08, complete=0.50)
    c = A.criteria(v1, v2, long, val_complete=392 / 666)
    assert c["M2"]["pass"] and c["M3"]["pass"] and c["M4"]["pass"] and c["M5"]["pass"]
    assert c["M3"]["v2"] == pytest.approx(0.18) and c["M3"]["v1"] == pytest.approx(0.36)
    assert c["M4"]["threshold"] == pytest.approx(0.8 * 392 / 666)
    assert c["M5"]["difference_points"] == pytest.approx(0.6)


def test_criteria_fail_as_measured():
    v1 = _score(0.22, 0.07, 0.14, 0.15, 0.034)
    v2 = _score(0.24, 0.05, 0.10, 0.10, 0.050)            # intervals overlap; 0.25 > 0.18
    long = _score(0.18, 0.07, 0.2, 0.1, 0.08, complete=0.40)
    c = A.criteria(v1, v2, long, val_complete=392 / 666)
    assert not c["M2"]["pass"] and not c["M3"]["pass"] and not c["M4"]["pass"]
    assert not c["M5"]["pass"]                             # 1.6 points above v1


def test_m2_needs_non_overlapping_intervals_not_just_a_higher_rate():
    v1 = {**_score(0.22, 0.07, 0.14, 0.15, 0.03), "strict_ok": _rate(0.22, 0.195, 0.247)}
    v2 = {**_score(0.27, 0.07, 0.14, 0.15, 0.03), "strict_ok": _rate(0.27, 0.247, 0.298)}
    assert not A.criteria(v1, v2, v2, 0.5)["M2"]["pass"]   # touching intervals do not pass
    v2["strict_ok"] = _rate(0.27, 0.248, 0.298)
    assert A.criteria(v1, v2, v2, 0.5)["M2"]["pass"]


def test_table_rows_have_one_cell_per_column():
    row = A.table_row("v2", _score(0.3, 0.03, 0.05, 0.1, 0.04))
    assert row.count("|") == len(A.COLUMNS) + 2


def test_criteria_boundaries_count_as_pass():
    v1 = _score(0.22, 0.10, 0.10, 0.10, 0.030)
    v2 = _score(0.30, 0.05, 0.05, 0.05, 0.040)            # L+U+S exactly half; +1.0 point
    long = _score(0.18, 0.07, 0.2, 0.1, 0.08, complete=0.8 * 0.5)
    c = A.criteria(v1, v2, long, val_complete=0.5)
    assert c["M3"]["pass"] and c["M4"]["pass"] and c["M5"]["pass"]
    v2 = _score(0.30, 0.05, 0.05, 0.0501, 0.0401)
    long = _score(0.18, 0.07, 0.2, 0.1, 0.08, complete=0.399)
    c = A.criteria(v1, v2, long, val_complete=0.5)
    assert not (c["M3"]["pass"] or c["M4"]["pass"] or c["M5"]["pass"])


def _write_run(d, name, cap, masks, max_len, seeds=range(1, 1001), temp=0.8, top_k=20):
    import json
    (d / f"{name}.jsonl").write_text("".join(
        json.dumps({"seed": s, "temp": temp, "top_k": top_k, "cap": cap, "masks": masks,
                    "ids": [1, 2]}) + "\n" for s in seeds))
    (d / name).mkdir(exist_ok=True)
    sc = _score(0.3, 0.05, 0.05, 0.05, 0.04, complete=0.5)
    if max_len is not None:
        sc["max_len"] = max_len
    (d / name / "score.json").write_text(json.dumps(sc))


def _write_all(tmp_path, **override):
    import json
    runs = {"v1-256": (256, False, 256), "v2-256": (256, True, 256), "v2-1024": (1024, True, 1024)}
    for name, (cap, masks, max_len) in runs.items():
        _write_run(tmp_path, name, *override.get(name, (cap, masks, max_len)))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"cap": 1024, "score": {**_score(0.27, 0.02, 0.07, 0.31, 0.016),
                                                      "trimmed_length": {"mean": 653.3}}}))
    return ref


def test_main_reads_the_runs_and_writes_criteria(tmp_path):
    ref = _write_all(tmp_path)
    crit = A.main(["--dir", str(tmp_path), "--ref", str(ref)])
    assert set(crit) == {"M2", "M3", "M4", "M5"} and (tmp_path / "criteria.json").exists()


@pytest.mark.parametrize("override", [
    {"v2-256": (256, False, 256)},              # v2 without masks
    {"v1-256": (256, True, 256)},               # v1 with masks
    {"v2-1024": (1024, True, 256)},             # cut at the wrong length
    {"v2-1024": (1024, True, None)},            # no recorded cut length
    {"v2-256": (300, True, 256)},               # another cap
])
def test_main_refuses_runs_that_are_not_the_shipping_setting(tmp_path, override):
    ref = _write_all(tmp_path, **override)
    with pytest.raises(ValueError):
        A.main(["--dir", str(tmp_path), "--ref", str(ref)])


def test_main_refuses_other_seeds_or_sampler(tmp_path):
    ref = _write_all(tmp_path)
    _write_run(tmp_path, "v2-256", 256, True, 256, seeds=range(2, 1002))
    with pytest.raises(ValueError):
        A.main(["--dir", str(tmp_path), "--ref", str(ref)])
    _write_run(tmp_path, "v2-256", 256, True, 256, top_k=40)
    with pytest.raises(ValueError):
        A.main(["--dir", str(tmp_path), "--ref", str(ref)])
