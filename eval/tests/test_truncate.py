from eval.truncate import trim_to_line

NL, EOS, BOS = 9, 2, 1


def test_complete_program_is_unchanged():
    ids = [BOS, 5, 6, NL, 7, EOS]
    assert trim_to_line(ids, max_len=256, newline_id=NL, eos_id=EOS) == ids


def test_unfinished_program_drops_partial_last_line():
    assert trim_to_line([BOS, 5, NL, 6, 7], 256, NL, EOS) == [BOS, 5]


def test_long_program_is_cut_at_last_newline_within_limit():
    ids = [BOS, 5, NL, 6, NL, 7, 8, NL, 9, EOS]
    assert trim_to_line(ids, max_len=7, newline_id=NL, eos_id=EOS) == [BOS, 5, NL, 6]


def test_single_unfinished_line_trims_to_empty_body():
    assert trim_to_line([BOS, 5, 6, 7], 256, NL, EOS) == [BOS]
