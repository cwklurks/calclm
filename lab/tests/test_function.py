"""lab.function: the v2.1 metrics (PLAN-v2.1.md, Metrics), moved from the v3-angles scratch code
with the plan's amendment (string literals and TRIV follow TI-OS).

In the public export, function_cases.json is a hand-written stand-in
(lab/tests/make_function_cases.py): made-up programs and screens, with lab.function's
verdicts frozen beside them. The private fixture of model samples is not published."""
import json
from pathlib import Path

import pytest

from lab import function as Fn
from lab.score import is_strict_ok
from lab.tests.conftest import VOCAB

FIX = json.loads((Path(__file__).parent / "function_cases.json").read_text())
CASES = FIX["cases"]
V = json.loads(VOCAB.read_text())
BOS, EOS = 1, 2


Q, SP, EQ, NL, COLON, STORE = "2A", "29", "6A", "3F", "3E", "04"   # " space = newline : →


def tok(*keys):
    """<bos> + vocab ids: token hex keys, or one letter or digit (same code as its ASCII)."""
    return [BOS] + [V.index(f"{ord(k):02X}" if len(k) == 1 else k) for k in keys]


def rec_of(case):
    return FIX["records"][CASES.index(case)]


def ids_of(case, cap=1024):
    from eval.truncate import trim_to_line
    return trim_to_line(FIX["samples"][case["sample"]], cap, V.index("3F"), EOS)


def write_run(d, records, cap=1024, distinct=0.42, seen=0.075, yields=True):
    """A lab.score --out dir: results.jsonl, score.json (max_len = cap unless cap is None) and
    run_programs' config.json (with "yields": true when yields)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                             for r in records))
    score = {"distinct_4": distinct, "overlap_32": {"mean_seen": seen}}
    if cap is not None:
        score["max_len"] = cap
    (d / "score.json").write_text(json.dumps(score))
    config = {"n": len(records), "config": {"wall_limit_s": 180.0}}
    if yields:
        config["yields"] = True
    (d / "config.json").write_text(json.dumps(config))
    return d


# ---- the frozen scratch verdicts, case by case -------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case_matches_the_scratch_code(case):
    t, r, e = ids_of(case), rec_of(case), case["expect"]
    assert is_strict_ok(r) is e["strict"]
    assert Fn.strict2(r) is e["strict2"]
    assert Fn.computed2(t, r) is e["computed2"]
    assert Fn.is_lus(r) is e["lus"]
    assert list(Fn.computed(t, r)) == e["computed"]
    assert Fn.axes_only(r) is e["axes_only"]
    assert Fn.outcome_cls(r) == e["class"]
    assert Fn.string_literals(t) == e["literals"]
    assert [list(x) for x in Fn.positioned_literals(t)] == e["tail_literals"]
    assert sorted(Fn.vis_lines(r)) == e["vis_lines"]
    assert Fn.dead_share(t) == pytest.approx(e["dead"], abs=1e-12)
    if e["tail_only"] is not None:                                      # strict-OK programs
        assert Fn.tail_only_share({0: (FIX["samples"][case["sample"]], t, r)}) == e["tail_only"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_both_literal_readers_find_the_same_strings(case):
    t = ids_of(case)
    assert [s for _, s in Fn.positioned_literals(t)] == [
        x for x in Fn.string_literals(t) if len(x) >= 3]


def test_fixture_covers_the_cases_the_plan_names():
    by = {c["name"]: c["expect"] for c in CASES}
    assert by["axes_only_graph"]["strict"] and not by["axes_only_graph"]["strict2"]
    assert not by["axes_only_graph"]["computed2"] and by["drawn_graph"]["computed2"]
    assert not by["literal_substring"]["computed2"] and not by["literal_superstring"]["computed2"]
    assert not by["trivial_only"]["computed2"]
    assert by["yield_only"]["computed2"] and not by["yield_only"]["strict"]
    assert by["yield_only_timeout"]["computed2"]
    assert by["dead_code"]["dead"] > 0.2
    assert not by["minus_one"]["computed2"] and not by["colon_in_string"]["computed2"]
    assert any(":" in x for x in by["colon_in_string"]["literals"])
    assert by["unclosed_at_cut"]["tail_only"] is True


def test_yield_screens_are_what_makes_yield_only_computed():
    for name in ("yield_only", "yield_only_timeout"):
        case = next(c for c in CASES if c["name"] == name)
        r = rec_of(case)
        assert Fn.computed2(ids_of(case), r)
        assert not Fn.computed2(ids_of(case), {**r, "yields": []})


def test_a_screen_of_minus_one_is_not_computed():
    """TI-OS shows a negative number with '⁻' (U+207B), which TRIV now accepts like '-'."""
    case = next(c for c in CASES if c["name"] == "minus_one")
    assert "⁻1" in Fn.vis_lines(rec_of(case)) and not Fn.computed2(ids_of(case), rec_of(case))


# ---- whole-run API -----------------------------------------------------------------------------

def test_per_sample_booleans_follow_the_line_cut_alignment(tmp_path):
    d = write_run(tmp_path / "eval", FIX["records"])
    ps = Fn.per_sample(FIX["samples"], d, FIX["cap"])
    assert sorted(ps) == [c["sample"] for c in CASES]
    assert 3 not in ps                                          # trimmed to <bos>: never run
    for c in CASES:
        e = c["expect"]
        assert ps[c["sample"]] == {k: e[k] for k in ("strict", "computed2", "strict2", "lus")}


def test_run_metrics_reproduces_the_scratch_numbers(tmp_path):
    d = write_run(tmp_path / "eval", FIX["records"], distinct=0.4321, seen=0.0756)
    m = Fn.run_metrics(FIX["samples"], d, FIX["cap"])
    s = FIX["metrics"]                                          # the scratch code rounds to 0.1
    for k in ("strict", "strict2", "computed2", "noerr", "complete", "fin_len_med", "fin_le40",
              "dead"):
        assert round(m[k], 1) == s[k], k
    assert tuple(round(m[k], 1) for k in ("label", "undefined", "syntax", "timeout")) == (
        s["ERR:LABEL"], s["ERR:UNDEFINED"], s["ERR:SYNTAX"], s["TIMEOUT"])
    n_run, n = len(CASES), len(FIX["samples"])
    share = lambda k: 100 * sum(c["expect"][k] for c in CASES) / n_run   # noqa: E731
    assert m["strict"] == pytest.approx(share("strict"), abs=1e-9)     # unrounded
    assert m["computed2"] == pytest.approx(share("computed2"), abs=1e-9)
    assert m["lus"] == pytest.approx(share("lus"), abs=1e-9)
    assert m["tail_only"] == pytest.approx(100 * FIX["tail_only_share"], abs=1e-9)
    assert m["distinct_4"] == 0.4321 and m["overlap_32"] == 0.0756   # score.json's own units
    assert (m["n_samples"], m["n_run"]) == (n, n_run) == (len(CASES) + 1, len(CASES))
    finished = sum(EOS in s[:1024] for s in FIX["samples"])
    assert m["complete"] == pytest.approx(100 * finished / n, abs=1e-9)  # over every sample
    assert (m["harness_errors"], m["wall_limit_hits"]) == (0, 0)
    assert set(m) == {"strict", "strict2", "computed2", "lus", "label", "undefined", "syntax",
                      "timeout", "noerr", "complete", "fin_le40", "fin_len_med", "tail_only",
                      "dead", "distinct_4", "overlap_32", "n_samples", "n_run", "harness_errors",
                      "wall_limit_hits"}
    json.dumps(m)


def rec(outcome="OK", home=(), graph=False, lcd=(), error=None, index=0, **kw):
    return {"index": index, "outcome": outcome, "error": error, "home_text": list(home),
            "graph": graph, "lcd_text": list(lcd), "yields": [], **kw}


def test_rates_are_none_when_their_denominator_is_zero(tmp_path):
    unfinished = tok("DE", "1", NL, "DE", "2")                       # no <eos>: not finished
    err = rec("ERR", error="SYNTAX")
    m = Fn.run_metrics([unfinished], write_run(tmp_path / "a", [err]), 1024)
    assert m["fin_le40"] is None and m["fin_len_med"] is None       # no finished program
    assert m["tail_only"] is None                                   # no strict-OK program
    assert m["strict"] == 0.0 and m["lus"] == 100.0 and m["complete"] == 0.0
    m = Fn.run_metrics([tok("DE", "1")], write_run(tmp_path / "b", []), 1024)   # never run
    assert m["n_run"] == 0 and m["n_samples"] == 1 and m["complete"] == 0.0
    for k in ("strict", "strict2", "computed2", "lus", "label", "undefined", "syntax", "timeout",
              "noerr", "dead", "tail_only", "fin_le40", "fin_len_med"):
        assert m[k] is None, k
    m = Fn.run_metrics([], write_run(tmp_path / "c", []), 1024)
    assert m["complete"] is None and m["n_samples"] == 0


def test_harness_errors_and_wall_limit_hits_are_counted(tmp_path):
    progs = [tok("DE", "1") + [EOS]] * 3
    recs = [rec("HARNESS_ERROR", index=0, reject_reason="worker lost or no result in 240 s"),
            rec("TIMEOUT", index=1, timeout_state="computing", wall_s=181.2),
            rec("OK", home=["1"], index=2, wall_s=0.4)]
    m = Fn.run_metrics(progs, write_run(tmp_path / "e", recs), 1024)
    assert (m["harness_errors"], m["wall_limit_hits"]) == (1, 1)


def test_run_metrics_needs_max_len_and_a_yield_run(tmp_path):
    with pytest.raises(ValueError, match="max_len"):
        Fn.run_metrics(FIX["samples"], write_run(tmp_path / "a", FIX["records"], cap=None), 1024)
    with pytest.raises(ValueError, match="yields"):
        Fn.run_metrics(FIX["samples"], write_run(tmp_path / "b", FIX["records"], yields=False),
                       1024)
    d = write_run(tmp_path / "c", FIX["records"])
    (d / "config.json").unlink()
    with pytest.raises(ValueError, match="yields"):
        Fn.per_sample(FIX["samples"], d, 1024)


def test_alignment_refuses_a_results_file_of_the_wrong_length(tmp_path):
    d = write_run(tmp_path / "eval", FIX["records"][:-1])
    with pytest.raises(ValueError, match=f"{len(CASES) - 1} records"):
        Fn.per_sample(FIX["samples"], d, FIX["cap"])


def test_run_metrics_refuses_a_run_cut_at_another_cap(tmp_path):
    d = write_run(tmp_path / "eval", FIX["records"], cap=256)
    with pytest.raises(ValueError, match="256"):
        Fn.run_metrics(FIX["samples"], d, 1024)


# ---- the pieces, on hand-made programs ---------------------------------------------------------

# TI-OS (checked in the emulator): "HELLO→Str1 stores HELLO; Disp "AB<newline> shows AB; Disp "XY
# at the end shows XY; Disp "A:B" shows A:B. A string closes at a quote, →, a newline or the end.

def test_string_literals_close_on_quote_store_newline_and_end_not_colon():
    ids = tok("DE", Q, "A", "B", Q, NL,                 # Disp "AB"
              Q, "C", "D", STORE, "A", NL,              # "CD→A
              Q, "E", COLON, "F", NL,                   # "E:F   (the colon is part of it)
              Q, SP, SP, Q, NL,                         # "  "   (blank: dropped)
              Q, "G", "H")                              # "GH    (open at the end: kept)
    assert Fn.string_literals(ids) == ["AB", "CD", "E:F", "GH"]
    assert Fn.string_literals(tok("DE", Q, "A", "B") + [EOS]) == ["AB"]


def test_positioned_literals_keep_3_plus_chars_with_the_quote_position():
    ids = tok("DE", Q, "A", "B", "C", Q, NL, "DE", Q, "X", "Y", Q, NL,
              Q, "P", COLON, "Q", STORE, "B", NL, "DE", Q, "L", "M", "N")
    assert Fn.positioned_literals(ids) == [(2, "ABC"), (14, "P:Q"), (22, "LMN")]


def test_colon_inside_a_string_is_not_computed_output():
    ids = tok("DE", Q, "A", "B", COLON, "C", "D", Q)     # Disp "AB:CD", shown wrapped
    r = {"outcome": "OK", "graph": False, "home_text": ["AB", ":CD"]}
    assert Fn.string_literals(ids) == ["AB:CD"] and not Fn.computed2(ids, r)


def test_store_ends_a_string_and_what_follows_is_code():
    ids = tok(Q, "H", "I", STORE, "A", NL, "DE", "A")    # "HI→A, Disp A: no literal "A
    assert Fn.string_literals(ids) == ["HI"]
    assert Fn.computed2(ids, {"outcome": "OK", "graph": False, "home_text": ["A"]})


def test_a_string_open_at_the_cut_is_a_literal():
    ids = tok("DE", "1", NL, "DE", Q, "H", "E", "L", "L", "O")    # cut inside the last line
    r = {"outcome": "OK", "graph": False, "home_text": ["1", "HELLO"]}
    assert not Fn.computed2(ids, {**r, "home_text": ["HELLO"]})
    assert Fn.tail_only_share({0: (ids, ids, r)}) == 1.0


@pytest.mark.parametrize("line, trivial", [
    ("0", True), ("1", True), ("-1", True), ("1.0", True), ("000", True), ("0.", True),
    ("⁻1", True), ("⁻0", True), ("⁻1.00", True), ("⁻⁻1", False), ("⁻2", False), ("-⁻1", False),
    ("2", False), ("10", False), ("0    0", False), ("A", False)])
def test_computed2_drops_zero_and_one(line, trivial):
    r = {"outcome": "OK", "graph": False, "home_text": [line], "yields": []}
    assert Fn.computed2(tok("DE", "A"), r) is not trivial


def test_computed2_drops_substrings_and_superstrings_of_literals():
    ids = tok("DE", Q, "S", "U", "M", EQ, Q)              # Disp "SUM="
    for line, comp in [("SUM=", False), ("SUM", False), ("SUM=12", False), ("12", True)]:
        r = {"outcome": "OK", "graph": False, "home_text": [line]}
        assert Fn.computed2(ids, r) is comp, line


def test_computed2_reads_yield_screens_and_skips_echo_and_prompt_lines():
    ids = tok("DE", "A")
    r = {"outcome": "TIMEOUT", "graph": False, "home_text": ["X=?1"],
         "yields": [["input", 50, False, ["X=?█"]], ["pause", 90, False, ["  42  "]]]}
    assert Fn.computed2(ids, r)
    assert not Fn.computed2(ids, {**r, "yields": r["yields"][:1]})
    assert not Fn.computed2(ids, {**r, "outcome": "ERR"})


def screen(*points, rows=10, cols=26, cells=None):
    g = [[" "] * cols for _ in range(rows)]
    for i, j in points:
        g[i][j] = "."
    for (i, j), ch in (cells or {}).items():
        g[i][j] = ch
    return ["".join(x) for x in g]


DONE = {(0, 22): "�", (0, 23): "�", (0, 24): "�", (0, 25): "�", (1, 0): "_"}


def test_home_screen_done_and_cursor_are_not_graph_output():
    """After a program returns, TI-OS shows "Done" right-aligned on row 0 (read as four U+FFFD)
    and the entry cursor (read as "_") at row 1: the screen of a program that showed a graph
    earlier and then quit. Checked against emulator screenshots of s10 G2b programs."""
    assert Fn.HOME_DONE_CELLS == DONE
    done = {"outcome": "OK", "graph": True, "home_text": [], "lcd_text": screen(cells=DONE)}
    assert Fn.axes_only(done) and is_strict_ok(done)
    assert not Fn.strict2(done) and not Fn.computed2(tok("DE", "1"), done)
    drawn = {**done, "lcd_text": screen((5, 3), (7, 9), cells=DONE)}       # more than Done
    assert not Fn.axes_only(drawn) and Fn.strict2(drawn)
    axes = screen(*[(5, j) for j in range(26)], *[(i, 13) for i in range(10)], cells=DONE)
    assert Fn.axes_only({**done, "lcd_text": axes})                         # Done over axes


NO_CURSOR = {k: v for k, v in DONE.items() if k != (1, 0)}


@pytest.mark.parametrize("cells", [
    {**NO_CURSOR, (5, 5): "�"},                                      # Done, no cursor, a dot
    {**NO_CURSOR, (1, 0): "�"},                                      # another glyph there
    {(2, 22): "�", (2, 23): "�", (2, 24): "�", (2, 25): "�", (3, 0): "_"}])   # on row 2
def test_only_the_exact_done_screen_is_masked(cells):
    """Each screen is more than axes as it is, and would read as axes-only if masked."""
    r = {"outcome": "OK", "graph": True, "home_text": [], "lcd_text": screen(cells=cells)}
    assert not Fn.axes_only(r) and Fn.strict2(r)


def test_axes_only_and_strict2():
    axes = screen(*[(5, j) for j in range(26)], *[(i, 13) for i in range(10)])
    drawn = screen(*[(5, j) for j in range(26)], *[(i, 13) for i in range(10)], (2, 3))
    assert Fn.axes_only({"lcd_text": screen()}) and Fn.axes_only({"lcd_text": axes})
    assert not Fn.axes_only({"lcd_text": drawn})
    g = {"outcome": "OK", "graph": True, "home_text": []}
    assert is_strict_ok({**g, "lcd_text": axes}) and not Fn.strict2({**g, "lcd_text": axes})
    assert Fn.strict2({**g, "lcd_text": drawn})
    assert not Fn.strict2({"outcome": "OK", "graph": False, "home_text": ["A=?1"]})
    assert Fn.strict2({"outcome": "OK", "graph": False, "home_text": ["A=?1", "5"]})


def test_dead_share_counts_code_after_a_top_level_goto():
    ids = tok("D7", "A", NL, "DE", "1", NL, "D6", "A", NL, "DE", "2") + [EOS]   # Goto A ... Lbl A
    info, regions, reach, dead, _ = Fn.analyze(ids)
    assert dead == [False, True, False, False]
    assert Fn.dead_share(ids) == pytest.approx(3 / 13)       # statement spans 3, [3], 3, 4
    assert Fn.dead_share(tok("DE", "1")) == 0


def test_outcome_cls_and_lus():
    assert Fn.outcome_cls({"outcome": "ERR", "error": "LABEL"}) == "ERR:LABEL"
    assert Fn.outcome_cls({"outcome": "OK", "graph": False, "home_text": []}) == "blank_ok"
    assert Fn.outcome_cls({"outcome": "TIMEOUT"}) == "TIMEOUT"
    assert Fn.is_lus({"outcome": "ERR", "error": "SYNTAX"})
    assert not Fn.is_lus({"outcome": "ERR", "error": "DOMAIN"})
    assert not Fn.is_lus({"outcome": "TIMEOUT", "error": None})


def test_signp_and_pair():
    assert Fn.signp(0, 0) == 1.0 and Fn.signp(5, 0) == pytest.approx(2 / 32)
    assert Fn.signp(3, 9) == Fn.signp(9, 3)
    base = {0: {"strict": True}, 1: {"strict": False}, 2: {"strict": False}, 5: {"strict": True}}
    arm = {0: {"strict": False}, 1: {"strict": True}, 2: {"strict": True}, 3: {"strict": True}}
    assert Fn.pair(base, arm, lambda v: v["strict"]) == (2, 1, "%.2g" % Fn.signp(2, 1))


def test_names_are_the_tivars_display_names():
    nm = Fn.names()
    assert len(nm) == len(V) and nm[:4] == ("<pad>", "<bos>", "<eos>", "<unk>")
    assert nm[V.index("3F")] == "\n" and nm[V.index("2A")] == '"' and nm[V.index("04")] == "→"
