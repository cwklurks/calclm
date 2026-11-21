"""Summary table over per-program results."""
from eval import summary as S


def rec(label, **kw):
    outcome = label.split(":")[0]
    r = {"outcome": outcome, "label": label, "error": label[4:] if outcome == "ERR" else None,
         "timeout_state": None, "wall_s": 1.0, "emu_ms": 1000, "graph": False}
    r.update(kw)
    return r


def test_counts_and_percentages():
    rs = [rec("OK"), rec("OK"), rec("ERR:SYNTAX"), rec("ERR:SYNTAX"), rec("ERR:DOMAIN"),
          rec("TIMEOUT", timeout_state="computing"), rec("REJECTED"), rec("OK", graph=True)]
    s = S.summarize(rs, wall_s=60.0, workers=4)
    assert s["n"] == 8
    assert s["outcomes"]["OK"] == {"n": 3, "pct": 37.5}
    assert s["outcomes"]["ERR"]["n"] == 3
    assert s["outcomes"]["TIMEOUT"]["n"] == 1
    assert s["outcomes"]["REJECTED"]["n"] == 1
    assert s["errors"] == {"SYNTAX": 2, "DOMAIN": 1}
    assert list(s["errors"]) == ["SYNTAX", "DOMAIN"]   # most common first
    assert s["timeout_states"] == {"computing": 1}
    assert s["tokenize_pct"] == 87.5                     # not REJECTED
    assert s["no_err_pct"] == 50.0                       # OK + TIMEOUT
    assert s["graph"] == 1
    assert s["programs_per_min"] == 8.0


def test_empty():
    s = S.summarize([], wall_s=0.0, workers=1)
    assert s["n"] == 0 and s["programs_per_min"] == 0.0


def test_markdown_table_mentions_every_class():
    rs = [rec("OK"), rec("ERR:SYNTAX"), rec("TIMEOUT", timeout_state="input"), rec("REJECTED")]
    md = S.to_markdown(S.summarize(rs, wall_s=30.0, workers=2), title="t")
    for word in ("OK", "ERR", "TIMEOUT", "REJECTED", "SYNTAX", "input", "programs/min"):
        assert word in md


def test_harness_errors_are_counted_separately():
    s = S.summarize([rec("OK"), {**rec("OK"), "outcome": "HARNESS_ERROR",
                                 "label": "HARNESS_ERROR"}], wall_s=1.0, workers=1)
    assert s["outcomes"]["HARNESS_ERROR"]["n"] == 1 and s["outcomes"]["OK"]["n"] == 1
    assert "HARNESS_ERROR" not in S.summarize([rec("OK")], wall_s=1.0, workers=1)["outcomes"]


def test_subsets_by_static_features():
    def r(label, **static):
        x = rec(label)
        x["static"] = {"prgm_call": False, "asm": False, "getkey": False, "input": False,
                       "menu": False, "draw": False, **static}
        x["meta"] = {"tags": []}
        return x
    rs = [r("OK"), r("ERR:UNDEFINED", prgm_call=True), r("ERR:INVALID", asm=True),
          r("TIMEOUT", getkey=True), r("OK", input=True)]
    rs[0]["meta"]["tags"] = ["lib_calls"]
    sub = S.subsets(rs)
    assert sub["self_contained"]["n"] == 2               # no prgm/asm/lib_calls
    assert sub["self_contained"]["pct"]["OK"] == 50.0
    assert sub["calls_prgm"]["n"] == 1 and sub["uses_asm"]["n"] == 1
    assert sub["all"]["n"] == 5
    md = S.to_markdown(S.summarize(rs, wall_s=1.0, workers=1), title="t")
    assert "self_contained" in md


def test_errors_right_after_a_driven_input_are_counted():
    rs = [rec("ERR:DATA TYPE", last_action="input", ms_since_action=110),
          rec("ERR:DOMAIN", last_action="input", ms_since_action=9000),
          rec("ERR:SYNTAX", last_action=None, ms_since_action=None),
          rec("OK", last_action="input", ms_since_action=50)]
    s = S.summarize(rs, wall_s=1.0, workers=1)
    assert s["err_after_input"] == 1


def test_subset_programs_that_fit_the_context():
    rs = [{**rec("OK"), "n_tokens": 10}, {**rec("ERR:SYNTAX"), "n_tokens": 300}]
    assert S.subsets(rs)["le_256_tokens"]["n"] == 1
