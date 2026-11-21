"""lab.tags: E3 behaviour tags (PLAN-v3-E3.md, "Tags") from yield-run records of human programs."""
import json

import numpy as np
import pytest

from lab import tags as T
from lab.tests.test_function import Q, rec, tok, write_run
from model.data import feature_meta, load_features
from model.tokens import BOS, EOS


# ---- one program ----------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", ["ERR", "REJECTED", "HARNESS_ERROR"])
def test_a_run_that_is_neither_ok_nor_timeout_is_err(outcome):
    assert T.classify(tok("DE", "A"), rec(outcome, home=["5"])) == "ERR"


@pytest.mark.parametrize("outcome", ["OK", "TIMEOUT"])
def test_a_computed_value_is_computed(outcome):
    assert T.classify(tok("DE", "A"), rec(outcome, home=["5"])) == "COMPUTED"


def test_a_drawn_graph_is_computed_and_bare_axes_are_blank():
    drawn = rec(graph=True, lcd=["   x    ", "xxxxxxxx", "   x  x "])
    axes = rec(graph=True, lcd=["   x    ", "xxxxxxxx", "   x    "])
    assert T.classify(tok("DE", "A"), drawn) == "COMPUTED"
    assert T.classify(tok("DE", "A"), axes) == "BLANK"


def test_only_literals_or_zero_and_one_is_literal():
    assert T.classify(tok("DE", Q, "H", "I", Q), rec(home=["HI"])) == "LITERAL"
    assert T.classify(tok("DE", "A"), rec(home=["1"])) == "LITERAL"


def test_nothing_visible_is_blank():
    assert T.classify(tok("DE", "A"), rec(home=["", "  "])) == "BLANK"


# ---- a run of many programs -----------------------------------------------------------------

def test_program_tags_follow_the_programs_and_unrun_ones_are_err(tmp_path):
    progs = [tok("DE", "A") + [EOS],
             tok("DE", *["A"] * 1100) + [EOS],          # one line past the cap: nothing is run
             tok("DE", Q, "H", "I", Q) + [EOS]]
    d = write_run(tmp_path, [rec(home=["5"], index=0), rec(home=["HI"], index=1)])
    assert T.program_tags(progs, d) == ["COMPUTED", "ERR", "LITERAL"]


def test_program_tags_refuse_a_run_without_yields(tmp_path):
    d = write_run(tmp_path, [rec(home=["5"])], yields=False)
    with pytest.raises(ValueError):
        T.program_tags([tok("DE", "A") + [EOS]], d)


# ---- per-token tags and the feature directories ---------------------------------------------

def test_every_stream_position_gets_its_programs_tag():
    stream = np.array([BOS, 5, 6, EOS, BOS, 7, EOS, BOS, EOS], dtype=np.uint16)
    got = T.token_tags(stream, ["COMPUTED", "ERR", "BLANK"])
    assert got.shape == (9, 1) and got.dtype == np.uint8
    v = T.VALUE
    assert got[:, 0].tolist() == [v["COMPUTED"]] * 4 + [v["ERR"]] * 3 + [v["BLANK"]] * 2


def test_token_tags_need_one_tag_per_program_and_a_leading_bos():
    with pytest.raises(ValueError):
        T.token_tags(np.array([BOS, 5, EOS]), ["ERR", "ERR"])
    with pytest.raises(ValueError):
        T.token_tags(np.array([5, BOS, EOS]), ["ERR"])


def test_tag_dirs_hold_the_true_tags_and_a_seeded_permutation(tmp_path):
    streams = {"train": np.array([BOS, 5, EOS, BOS, 6, EOS, BOS, EOS], dtype=np.uint16),
               "val": np.array([BOS, 7, EOS, BOS, EOS], dtype=np.uint16)}
    tags = {"train": ["COMPUTED", "ERR", "BLANK"], "val": ["LITERAL", "ERR"]}
    T.write_tag_dirs(tmp_path / "tags", tmp_path / "shuf", streams, tags)
    rng = np.random.default_rng(0)                       # train first, then val
    perm = {"train": rng.permutation(3), "val": rng.permutation(2)}
    for d, shuffled in (("tags", False), ("shuf", True)):
        meta = feature_meta(tmp_path / d)
        assert meta["cats"] == [4] and meta["bag"] == 0 and meta["tags"] == list(T.TAGS)
        assert meta["shuffled"] is shuffled
        for split, stream in streams.items():
            cats, bag = load_features(tmp_path / d, split, len(stream))
            want = [tags[split][i] for i in perm[split]] if shuffled else tags[split]
            assert cats[:, 0].tolist() == T.token_tags(stream, want)[:, 0].tolist()
            assert not np.asarray(bag).any()
    counts = json.loads((tmp_path / "tags" / "feats.json").read_text())["counts"]
    assert counts["train"] == {"ERR": 1, "BLANK": 1, "LITERAL": 0, "COMPUTED": 1}


def test_judge_counts_report_unrun_programs_harness_errors_and_wall_limit_hits(tmp_path):
    progs = [tok("DE", "A") + [EOS], tok("DE", *["A"] * 1100) + [EOS], tok("DE", "B") + [EOS]]
    d = write_run(tmp_path, [rec(home=["5"], index=0, wall_s=1.0),
                             rec("HARNESS_ERROR", index=1, wall_s=200.0)])
    assert T.judge_counts(progs, d) == {"programs": 3, "not_run": 1, "harness_errors": 1,
                                        "wall_limit_hits": 1}


def test_tag_dirs_carry_extra_provenance(tmp_path):
    streams = {s: np.array([BOS, 5, EOS], dtype=np.uint16) for s in ("train", "val")}
    tags = {s: ["ERR"] for s in streams}
    T.write_tag_dirs(tmp_path / "t", tmp_path / "s", streams, tags, extra={"judge": {"x": 1}})
    assert feature_meta(tmp_path / "t")["judge"] == {"x": 1}
    assert feature_meta(tmp_path / "s")["judge"] == {"x": 1}
