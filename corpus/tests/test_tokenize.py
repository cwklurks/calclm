import json
import random
from pathlib import Path

import numpy as np
import pytest
from tivars.models import TI_84PCE
from tivars.types import TIProgram

from corpus.extract import AUTHOR_DIRS  # the author's own programs (CALCLM_EVAL_DIRS)
from corpus.tokenize import (BOS, EOS, PAD, PREFIXES, SPECIALS, UNK, VOCAB_PATH, Vocab,
                             body_to_8xp, ids_to_8xp, is_known, read_8xp_bodies, render,
                             render_ids, split_tokens)

ROOT = Path(__file__).resolve().parents[2]
TOK = ROOT / "data" / "tok"

GUESS_GAME = """ClrHome
randInt(1,100)->N
0->T
Repeat G=N
Input "GUESS? ",G
T+1->T
If G<N
Disp "HIGHER"
If G>N
Disp "LOWER"
End
Disp "GOT IT IN",T
{1,2,3}->L1
[[1,2][3,4]]->[A]
"HELLO"->Str1
Output(1,1,sub(Str1,1,2))
"""


def encode_text(text: str) -> bytes:
    return TIProgram.encode(text, model=TI_84PCE)


@pytest.fixture(scope="module")
def full_vocab() -> Vocab:
    """Vocab containing every token in the TI-84 Plus CE sheet (independent of the corpus)."""
    toks = sorted(TI_84PCE.tokens.bytes)
    return Vocab(list(SPECIALS) + [t.hex().upper() for t in toks])


def test_prefix_set_is_the_two_byte_lead_bytes():
    assert PREFIXES == {0x5C, 0x5D, 0x5E, 0x60, 0x61, 0x62, 0x63, 0x7E, 0xAA, 0xBB, 0xEF}


def test_split_uses_prefix_rule():
    body = bytes.fromhex("5C00 3F BB6D EF7A 29 41 AA00")
    assert split_tokens(body) == [b"\x5C\x00", b"\x3F", b"\xBB\x6D", b"\xEF\x7A", b"\x29",
                                  b"\x41", b"\xAA\x00"]


def test_prefix_byte_consumes_next_byte_even_if_it_looks_like_a_prefix():
    assert split_tokens(b"\xBB\xBB\x41") == [b"\xBB\xBB", b"\x41"]


def test_trailing_prefix_byte_is_a_single_token():
    assert split_tokens(b"\x41\xBB") == [b"\x41", b"\xBB"]


@pytest.mark.parametrize("seed", range(5))
def test_split_is_lossless_on_random_bytes(seed):
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(2000))
    assert b"".join(split_tokens(data)) == data


def test_encode_decode_roundtrip_on_program(full_vocab):
    body = encode_text(GUESS_GAME)
    ids = full_vocab.encode(body)
    assert UNK not in ids
    assert full_vocab.decode(ids) == body
    assert len(ids) == len(split_tokens(body))


def test_special_tokens(full_vocab):
    body = encode_text("Disp 1")
    ids = full_vocab.encode(body, add_special=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert full_vocab.decode([PAD, *ids, 50, 60]) == body  # stops at <eos>


def test_unknown_and_rare_tokens_map_to_unk():
    vocab = Vocab(list(SPECIALS) + ["3F", "41"])
    ids = vocab.encode(b"\x41\x3F\x42\x00\xBB\xFF")
    assert ids == [4 + 1, 4, UNK, UNK, UNK]
    with pytest.raises(ValueError):
        vocab.decode(ids)
    assert vocab.decode(ids, strict=False) == b"\x41\x3F"


def test_unknown_sheet_tokens_are_not_known():
    assert not is_known(b"\x00")
    assert not is_known(b"\xBB\x6D")  # compiled-asm marker is not a displayable token
    assert is_known(b"\xDE")  # Disp
    assert is_known(b"\xEF\x7A")


def test_vocab_rejects_missing_specials():
    with pytest.raises(ValueError):
        Vocab(["3F", "41"])


def test_render_shows_readable_text(full_vocab):
    body = encode_text('Disp "HI"\nFor(I,1,3)\nEnd')
    assert render(body) == 'Disp "HI"\nFor(I,1,3)\nEnd'
    ids = full_vocab.encode(body)
    assert render_ids(ids[:2] + [UNK] + ids[2:], full_vocab).startswith('Disp "<unk>HI"')


def test_ids_to_8xp_roundtrip(tmp_path, full_vocab):
    body = encode_text(GUESS_GAME)
    path = ids_to_8xp(full_vocab.encode(body, add_special=True), tmp_path / "G.8xp",
                      name="GUESS", vocab=full_vocab)
    progs = read_8xp_bodies(path.read_bytes())
    assert progs == [("GUESS", 5, body)]


def test_body_to_8xp_is_byte_identical_for_real_file(tmp_path):
    files = sorted(f for d in AUTHOR_DIRS for f in d.glob("*.8xp"))[:5]
    if not files:
        pytest.skip("no local sample 8xp files")
    for f in files:
        raw = f.read_bytes()
        (name, _tid, body), = read_8xp_bodies(raw)
        out = body_to_8xp(body, tmp_path / f.name, name=name)
        assert read_8xp_bodies(out.read_bytes())[0][2] == body


# ------------------------------------------------------------------ built artifacts


def test_vocab_file_contract():
    if not VOCAB_PATH.exists():
        pytest.skip("corpus/vocab.json not built")
    entries = json.loads(VOCAB_PATH.read_text())
    assert entries[:4] == list(SPECIALS)
    assert len(entries) <= 1024
    assert len(set(entries)) == len(entries)
    for e in entries[4:]:
        assert e == e.upper()
        tok = bytes.fromhex(e)
        assert split_tokens(tok) == [tok], e
        assert is_known(tok), e


def test_real_corpus_programs_roundtrip():
    if not (VOCAB_PATH.exists() and (TOK / "programs.jsonl").exists()):
        pytest.skip("corpus not built")
    vocab = Vocab.load()
    ids = np.fromfile(TOK / "val.bin", dtype="<u2")
    rows = [json.loads(ln) for ln in (TOK / "programs.jsonl").open()
            if '"split": "val"' in ln][:200]
    checked = 0
    for r in rows:
        seq = ids[r["offset"]:r["offset"] + r["length"]].tolist()
        assert seq[0] == BOS and seq[-1] == EOS and EOS not in seq[1:-1]
        if UNK in seq:
            continue
        body = vocab.decode(seq)
        assert vocab.encode(body, add_special=True) == seq
        checked += 1
    assert checked > 0


def test_token_streams_are_well_formed():
    meta_path = TOK / "meta.json"
    if not meta_path.exists():
        pytest.skip("corpus not built")
    meta = json.loads(meta_path.read_text())
    for split in ("train", "val"):
        arr = np.fromfile(TOK / f"{split}.bin", dtype="<u2")
        assert arr.size == meta[split]["ids"]
        assert arr[0] == BOS and arr[-1] == EOS
        assert (arr == BOS).sum() == (arr == EOS).sum() == meta[split]["programs"]
        assert int(arr.max()) < meta["vocab_size"]
