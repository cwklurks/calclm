"""Pure token helpers: splitting, static features, program names, token ids -> bytes."""
import pytest

from eval import tokens as T


def test_split_two_byte_prefix():
    # BB 6A = Asm(, 5F = prgm, 41 = A, 3F = newline
    assert T.split_tokens(bytes.fromhex("BB6A5F413F")) == [b"\xbb\x6a", b"\x5f", b"A", b"\x3f"]


def test_split_truncated_prefix_is_one_byte():
    assert T.split_tokens(b"\x41\xbb") == [b"A", b"\xbb"]


def test_features_detect_tokens_not_bytes():
    # AD = getKey; but in BB AD the AD is the second byte of a 2-byte token, not getKey
    f = T.features(bytes.fromhex("BBAD3F"))
    assert not f["getkey"]
    f = T.features(bytes.fromhex("AD044B3F"))
    assert f["getkey"]


def test_features_all_flags():
    body = bytes.fromhex("DC41" "3F" "5F4142" "3F" "BB6A5F43" "3F" "9C" "3F" "E6" "3F" "D8" "3F"
                         "DD42")
    f = T.features(body)
    assert f["input"] and f["prgm_call"] and f["asm"] and f["draw"] and f["menu"] and f["pause"]
    assert f["n_tokens"] == len(T.split_tokens(body))


def test_asm_header_counts_as_asm():
    assert T.features(bytes.fromhex("EF7B0000"))["asm"]


@pytest.mark.parametrize("name,ok", [("A", True), ("ABCDEFGH", True), ("A1θ", True),
                                     ("1A", False), ("", False), ("ABCDEFGHI", False),
                                     ("abc", False), ("A B", False), ("#", False)])
def test_valid_name(name, ok):
    assert T.valid_name(name) is ok


def test_ids_to_body_skips_specials_and_stops_at_eos():
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE", "31", "3F", "BB6A"]
    assert T.ids_to_body([1, 4, 5, 6, 7, 2, 4, 4], vocab) == bytes.fromhex("DE313FBB6A")


def test_ids_to_body_rejects_unk_and_out_of_range():
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE"]
    with pytest.raises(T.TokenError):
        T.ids_to_body([1, 4, 3, 2], vocab)
    with pytest.raises(T.TokenError):
        T.ids_to_body([1, 99, 2], vocab)


def test_unknown_tokens_listed():
    # 0xBB 0xFF is not in the CE sheet
    assert T.features(bytes.fromhex("BBFF3F"))["unknown_tokens"] == ["BBFF"]


def test_line_at_uses_newline_tokens():
    # Disp 1 / Disp 1/0 / Disp 3
    body = bytes.fromhex("DE313F" "DE318330" "3F" "DE33")
    assert T.line_at(body, 7) == (2, "Disp 1/0")    # cursor at the end of line 2
    assert T.line_at(body, 0) == (1, "Disp 1")
    assert T.line_at(body, 9) == (3, "Disp 3")


def test_line_at_ignores_3f_inside_two_byte_tokens():
    body = bytes.fromhex("BB3F" "3F" "DE31")         # BB3F is one token, then newline
    assert T.line_at(body, 4)[0] == 2


def test_display_unknown_token_as_hex():
    assert T.display(bytes.fromhex("BBFF")) == "[BBFF]"
