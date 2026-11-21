"""lab.masks: logit masks from the tracker state (G1 arms A3/A4), on TI-BASIC token sequences."""
import json

import numpy as np
import pytest

from lab.tests.conftest import VOCAB

masks = pytest.importorskip("lab.masks")
try:
    from tracker import bind
    bind.lib()
except Exception as exc:  # noqa: BLE001 - the C tracker is optional for the rest of the suite
    pytest.skip(f"C tracker not available: {exc}", allow_module_level=True)

V = json.loads(VOCAB.read_text())
ID = {h: i for i, h in enumerate(V)}
BOS, EOS = 1, 2
T = {"If": "CE", "Then": "CF", "Else": "D0", "End": "D4", "While": "D1", "Disp": "DE",
     "Input": "DC", "->": "04", '"': "2A", ",": "2B", "(": "10", ")": "11", "{": "08", "}": "09",
     "NL": "3F", "L": "EB", "Str1": "AA00", "[A]": "5C00", "+": "70", "Goto": "D7", "Lbl": "D6",
     "DelVar": "BB54", "ClrAllLists": "BB52", "Archive": "BB68", "UnArchive": "BB69",
     "ClrList": "FA"}


def ids(*words):
    out = []
    for w in words:
        if w in T:
            out.append(ID[T[w]])
        else:                                  # letters and digits of names and numbers
            out += [ID["%02X" % ord(ch)] for ch in w]
    return out


@pytest.fixture(scope="module")
def mask():
    return masks.TrackerMask(VOCAB)


def banned_after(mask, *words):
    """The mask for the next token after <bos> + words (one row)."""
    rows = mask.rows
    rows.reset(1)
    rows.step(np.array([BOS]))
    for t in ids(*words):
        rows.step(np.array([t]))
    return mask.banned(rows)[0]


def allowed(b, word):
    return not b[ids(word)[0]]


def test_data_destroying_tokens_are_always_banned(mask):
    b = banned_after(mask, "Disp", "1", "NL")
    for w in ("DelVar", "ClrAllLists", "Archive", "UnArchive", "ClrList"):
        assert not allowed(b, w)
    assert allowed(b, "Disp") and allowed(b, "If") and b[EOS] == 0


def test_end_needs_an_open_block(mask):
    assert not allowed(banned_after(mask), "End")
    assert allowed(banned_after(mask, "While", "1", "NL"), "End")
    assert not allowed(banned_after(mask, "While", "1", "NL", "End", "NL"), "End")


def test_else_needs_an_open_if_then(mask):
    assert not allowed(banned_after(mask), "Else")
    assert not allowed(banned_after(mask, "While", "1", "NL"), "Else")
    assert allowed(banned_after(mask, "If", "A", "NL", "Then", "NL"), "Else")
    assert not allowed(banned_after(mask, "If", "A", "NL", "Then", "NL", "Else", "NL"), "Else")


def test_strings_and_matrices_are_read_only_after_an_assignment(mask):
    assert not allowed(banned_after(mask, "Disp"), "Str1")
    assert not allowed(banned_after(mask, "Disp"), "[A]")
    assert allowed(banned_after(mask, '"', "A", '"', "->"), "Str1")          # store target
    assert allowed(banned_after(mask, '"', "A", '"', "->", "Str1", "NL", "Disp"), "Str1")
    assert allowed(banned_after(mask, "Disp", '"'), "Str1")                  # inside a string


def test_a_pending_read_is_banned_where_it_resolves(mask):
    b = banned_after(mask, "Input", "Str1")           # Input Str1,A uses Str1 as the prompt
    assert not allowed(b, ",") and allowed(b, "NL")
    b = banned_after(mask, "1", "->", "[A]")          # ->[A](1,1) stores into an element of [A]
    assert not allowed(b, "(") and allowed(b, "NL")


def test_a_list_read_needs_some_assigned_list(mask):
    assert not allowed(banned_after(mask, "Disp"), "L")
    assert allowed(banned_after(mask, "{", "1", "}", "->"), "L")              # a definition
    assert allowed(banned_after(mask, "{", "1", "}", "->", "L", "AB", "NL", "Disp"), "L")


def test_a_list_read_follows_the_names_of_assigned_lists(mask):
    pre = ("{", "1", "}", "->", "L", "AB", "NL", "Disp", "L")
    b = banned_after(mask, *pre)
    assert allowed(b, "A") and not allowed(b, "B") and not allowed(b, "1")
    assert not allowed(b, "NL") and b[EOS]                                   # empty name
    b = banned_after(mask, *pre, "A")
    assert allowed(b, "B") and not allowed(b, "C") and not allowed(b, "NL") and b[EOS]
    b = banned_after(mask, *pre, "AB")
    assert allowed(b, "NL") and allowed(b, "+") and not b[EOS] and not allowed(b, "C")


def test_list_definitions_are_not_restricted(mask):
    b = banned_after(mask, "{", "1", "}", "->", "L")
    assert allowed(b, "Z") and allowed(b, "Q")
    assert allowed(banned_after(mask, "{", "1", "}", "->", "L", "ZQ"), "NL")


def test_every_row_keeps_an_allowed_token_on_real_programs(mask):
    val = VOCAB.parents[1] / "data" / "tok" / "val.bin"
    if not val.exists():
        pytest.skip("data/tok/val.bin not present")
    stream = np.fromfile(val, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)[:64]
    rows = mask.rows
    rows.reset(64)
    rows.step(np.full(64, BOS))
    for p in range(1, 120):
        rows.step(stream[starts + p])
        b = mask.banned(rows)
        b[:, [0, 1, 3]] = True
        assert (~b).any(axis=1).all()


def test_goto_targets_are_not_masked(mask):
    """A label can still be defined later, so a Goto to an undefined label is allowed."""
    b = banned_after(mask, "Goto")
    assert allowed(b, "A") and allowed(b, "Z")


def test_mask_rows_give_the_tracker_features_training_uses(mask):
    stream = np.fromfile(VOCAB.parents[1] / "data" / "tok" / "val.bin", dtype="<u2")[:3000] \
        if (VOCAB.parents[1] / "data" / "tok" / "val.bin").exists() else None
    if stream is None:
        pytest.skip("data/tok/val.bin not present")
    want_c, want_b = bind.batch(stream)
    rows = mask.rows
    rows.reset(2)
    got_c, got_b = [], []
    for t in stream:
        c, b = rows.step(np.array([t, t]))
        got_c.append(c[1])
        got_b.append(b[1])
    assert np.array_equal(np.array(got_c), want_c.astype(np.int64))
    assert np.array_equal(np.array(got_b), want_b.view(np.int64))


def test_stats_count_what_the_rules_banned(mask):
    mask.stats.clear()
    banned_after(mask, "Disp")                  # mid-statement: End is not a statement here
    assert mask.stats["positions"] == 1 and mask.stats["var"] >= 2 and mask.stats["end"] == 0
    banned_after(mask)
    assert mask.stats["positions"] == 2 and mask.stats["end"] == 1
