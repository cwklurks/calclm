"""TI-BASIC token helpers: split program bodies, map tokens to ids, render text.

The corpus workstream (M1) owns the real vocabulary (corpus/vocab.json). These helpers follow
the same contract (SPEC.md "Tokens") so the model code can build a proxy dataset and decode
generated ids back into program bytes.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)


@lru_cache(maxsize=1)
def _sheet() -> dict:
    from tivars.models import TI_84PCE

    return TI_84PCE.tokens.bytes


@lru_cache(maxsize=1)
def prefix_bytes() -> frozenset[int]:
    """First bytes of the two-byte tokens in the TI-84 Plus CE token sheet."""
    return frozenset(k[0] for k in _sheet() if len(k) == 2)


def split_tokens(body: bytes) -> list[bytes]:
    """Decompose a program body: a prefix byte starts a 2-byte token, others are 1 byte."""
    prefixes = prefix_bytes()
    out, i = [], 0
    while i < len(body):
        width = 2 if body[i] in prefixes and i + 1 < len(body) else 1
        out.append(body[i : i + width])
        i += width
    return out


def sheet_vocab() -> list[str]:
    """Specials followed by every token in the sheet, sorted by bytes (proxy vocabulary)."""
    return SPECIALS + [k.hex().upper() for k in sorted(_sheet())]


def load_vocab(path: str | Path) -> list[str]:
    vocab = json.loads(Path(path).read_text())
    if vocab[:4] != SPECIALS:
        raise ValueError(f"{path}: ids 0-3 must be {SPECIALS}")
    return vocab


def encode_body(body: bytes, index: dict[str, int]) -> list[int]:
    """<bos> + token ids (unknown tokens -> <unk>) + <eos>."""
    ids = [index.get(tok.hex().upper(), UNK) for tok in split_tokens(body)]
    return [BOS, *ids, EOS]


def decode_ids(ids, vocab: list[str]) -> bytes:
    """Concatenate token bytes, dropping special ids."""
    return b"".join(bytes.fromhex(vocab[i]) for i in ids if i >= len(SPECIALS))


def render_text(body: bytes) -> str:
    """Program bytes -> TI-BASIC source text via tivars."""
    from tivars import TIProgram

    prog = TIProgram(name="CALCLM")
    prog.data = body
    return prog.string().rstrip("\n")
