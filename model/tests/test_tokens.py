import pytest

from model.tokens import (
    BOS, EOS, PAD, SPECIALS, UNK,
    decode_ids, encode_body, prefix_bytes, render_text, sheet_vocab, split_tokens,
)


def test_specials_are_first_four_ids():
    assert SPECIALS == ["<pad>", "<bos>", "<eos>", "<unk>"]
    assert (PAD, BOS, EOS, UNK) == (0, 1, 2, 3)


def test_prefix_set_matches_token_sheet():
    assert {0x5C, 0x5D, 0x5E, 0x60, 0x61, 0x62, 0x63, 0x7E, 0xAA, 0xBB, 0xEF} == prefix_bytes()


def test_split_uses_two_byte_prefix_rule():
    # Disp "H" newline, then a two-byte token BB 00, then a one-byte token
    body = bytes([0xDE, 0x2A, 0x48, 0x2A, 0x3F, 0xBB, 0x00, 0x3F])
    assert split_tokens(body) == [b"\xde", b"\x2a", b"\x48", b"\x2a", b"\x3f", b"\xbb\x00", b"\x3f"]


def test_split_dangling_prefix_is_single_byte():
    assert split_tokens(bytes([0x3F, 0xBB])) == [b"\x3f", b"\xbb"]


def test_sheet_vocab_layout():
    vocab = sheet_vocab()
    assert vocab[:4] == SPECIALS
    assert len(vocab) == len(set(vocab)) <= 1024
    assert "3F" in vocab and "BB00" in vocab
    assert all(v == v.upper() for v in vocab[4:])


def test_encode_wraps_with_bos_eos_and_maps_unknown():
    vocab = SPECIALS + ["3F", "DE"]
    index = {tok: i for i, tok in enumerate(vocab)}
    assert encode_body(bytes([0xDE, 0x3F, 0x99]), index) == [BOS, 5, 4, UNK, EOS]


def test_decode_roundtrip_skips_specials():
    vocab = sheet_vocab()
    index = {tok: i for i, tok in enumerate(vocab)}
    body = bytes([0xDE, 0x2A, 0x48, 0x2A, 0x3F, 0xBB, 0x00])
    ids = encode_body(body, index)
    assert decode_ids([PAD] + ids, vocab) == body


def test_render_text_uses_tivars():
    assert render_text(bytes([0xDE, 0x2A, 0x48, 0x2A])) == 'Disp "H"'


@pytest.mark.parametrize("bad", ["<unk>"])
def test_decode_drops_unknown(bad):
    vocab = SPECIALS + ["3F"]
    assert decode_ids([3, 4], vocab) == b"\x3f"
