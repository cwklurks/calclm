"""tracker/gen_tables.py: the generated token-class table matches corpus/vocab.json."""
from __future__ import annotations

from pathlib import Path

from tracker import gen_tables
from tracker.tests.conftest import INDEX, VOCAB

HERE = Path(__file__).resolve().parents[1]


def test_generated_header_is_current():
    want = gen_tables.render(VOCAB)
    assert (HERE / "trk_tables.h").read_text() == want, \
        "run: uv run python -m tracker.gen_tables"


def test_classes_of_key_tokens():
    cls = gen_tables.classify(VOCAB)
    C = gen_tables.CLASSES
    assert cls[INDEX["41"]] == 0 and cls[INDEX["5A"]] == 25          # A, Z
    assert cls[INDEX["30"]] == 26 and cls[INDEX["39"]] == 35         # 0, 9
    assert cls[INDEX["5B"]] == 36                                     # theta
    assert cls[INDEX["AA00"]] == C["STR0"] and cls[INDEX["AA09"]] == C["STR0"] + 9
    assert cls[INDEX["5C00"]] == C["MAT0"]
    assert cls[INDEX["6000"]] == C["PIC0"] + 1 and cls[INDEX["6009"]] == C["PIC0"]  # Pic1, Pic0
    assert cls[INDEX["6100"]] == C["GDB0"] + 1 and cls[INDEX["6109"]] == C["GDB0"]  # GDB1, GDB0
    assert cls[INDEX["5D00"]] == C["SYSLIST"]
    for hx, name in [("3F", "NEWLINE"), ("3E", "COLON"), ("2A", "QUOTE"), ("04", "STORE"),
                     ("CE", "IF"), ("CF", "THEN"), ("D0", "ELSE"), ("D4", "END"),
                     ("D3", "FOR"), ("D1", "WHILE"), ("D2", "REPEAT"), ("D6", "LBL"),
                     ("D7", "GOTO"), ("E6", "MENU"), ("EB", "LISTL"), ("5F", "PRGM"),
                     ("BB54", "DELVAR"), ("DC", "INPUT"), ("DD", "PROMPT"),
                     ("BB4A", "SETUPED"), ("98", "STOREPIC"), ("99", "RECALLPIC"),
                     ("9A", "STOREGDB"), ("9B", "RECALLGDB"), ("B5", "DIM"),
                     ("BB55", "EQUSTR"), ("BB3A", "LISTMATR"), ("BB39", "MATRLIST"),
                     ("10", "LPAREN"), ("11", "RPAREN"), ("08", "LBRACE"), ("09", "RBRACE"),
                     ("06", "LBRACKET"), ("07", "RBRACKET"), ("2B", "COMMA"),
                     ("DA", "ISDS"), ("DB", "ISDS"), ("BB0C", "FUNC")]:
        assert cls[INDEX[hx]] == C[name], (hx, name)
    assert cls[0] == C["PAD"] and cls[1] == C["BOS"] and cls[2] == C["EOS"]
    assert cls[3] == C["OTHER"]
    assert cls[INDEX["DE"]] == C["OTHER"]                             # Disp
    assert cls[INDEX["6212"]] == C["STATVAR"] and cls[INDEX["6201"]] == C["STATVAR"]
    assert cls[INDEX["6221"]] == C["OTHER"]                           # sequence n exists
    assert cls[INDEX["F2"]] == C["STATCMD"] and cls[INDEX["BB3B"]] == C["STATOPEN"]


def test_openers_are_contiguous():
    C = gen_tables.CLASSES
    openers = [v for k, v in C.items() if k in gen_tables.OPENERS]
    assert sorted(openers) == list(range(C["LPAREN"], C["LPAREN"] + len(openers)))
    assert max(C.values()) < 256


def test_every_paren_token_is_an_opener():
    cls = gen_tables.classify(VOCAB)
    names = gen_tables.display_names()
    C = gen_tables.CLASSES
    lo, hi = C["LPAREN"], C["LPAREN"] + len(gen_tables.OPENERS)
    for i, hx in enumerate(VOCAB[4:], start=4):
        assert (lo <= cls[i] < hi) == names[hx].endswith("("), hx
        if cls[i] == C["STATOPEN"]:
            assert names[hx].endswith("(")
