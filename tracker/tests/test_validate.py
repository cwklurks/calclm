"""tracker/validate.py: tracker events are mapped to the program lines the OS reports."""
from __future__ import annotations

from tracker import spec, validate
from tracker.tests.conftest import ids_of

VOCAB = list(spec.default_vocab())


def test_read_before_assign_on_the_error_line():
    # "Disp 1" newline = bytes 0..2, so offset 3 is line 2; the list read finishes at <eos>
    ids = ids_of("Disp 1\nDisp ʟAB")
    rec = {"index": 0, "label": "ERR:UNDEFINED", "outcome": "ERR", "error_offset": 3,
           "error_line": "Disp ʟAB"}
    a = validate.analyze(ids, rec, VOCAB)
    assert a["err_line_no"] == 2 and a["undef_lines"] == [2]


def test_unresolved_labels_and_prgm_lines():
    ids = ids_of("Goto A\nprgmX\nDisp 1")
    rec = {"index": 1, "label": "ERR:LABEL", "outcome": "ERR", "error_offset": None}
    a = validate.analyze(ids, rec, VOCAB)
    assert a["unres"] == ["A"] and a["prgm_lines"] == [2] and a["err_line_no"] is None


def test_decoding_stops_at_eos_like_the_harness():
    ids = ids_of("Disp 1") + [2] + ids_of("Goto Q", bos=False)
    rec = {"index": 2, "label": "OK", "outcome": "OK"}
    assert validate.analyze(ids, rec, VOCAB)["sum"]["n_unres"] == 0
