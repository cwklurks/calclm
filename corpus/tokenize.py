"""TI-BASIC token handling for CalcLM (SPEC.md "Tokens").

A program body (raw token bytes of an 8xp program variable, no header) splits uniquely into
tokens: a byte in the two-byte prefix set of the TI-84 Plus CE token sheet (tivars
TI_84PCE.tokens) starts a 2-byte token, any other byte is a 1-byte token.

corpus/vocab.json is a JSON list; index = token id; entry = uppercase hex of the token bytes.
Ids 0-3 are "<pad>", "<bos>", "<eos>", "<unk>".

Import as `corpus.tokenize` (never put the corpus/ directory itself on sys.path: this module's
name would shadow the standard library's `tokenize`).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

from tivars.models import TI_84PCE

SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

_SHEET = TI_84PCE.tokens
PREFIXES: frozenset[int] = frozenset(b[0] for b in _SHEET.bytes if len(b) == 2)
KNOWN: frozenset[bytes] = frozenset(_SHEET.bytes)


def split_tokens(body: bytes) -> list[bytes]:
    """Split a program body into token byte strings using the prefix rule.

    A prefix byte at the very end (truncated body) becomes a 1-byte token.
    """
    out = []
    i, n = 0, len(body)
    while i < n:
        step = 2 if body[i] in PREFIXES and i + 1 < n else 1
        out.append(body[i:i + step])
        i += step
    return out


def is_known(tok: bytes) -> bool:
    """True if the token is defined in the TI-84 Plus CE sheet."""
    return tok in KNOWN


class Vocab:
    """Token-bytes <-> id mapping backed by a vocab.json list."""

    def __init__(self, entries: Sequence[str]):
        if tuple(entries[:4]) != SPECIALS:
            raise ValueError(f"vocab must start with {SPECIALS}")
        self.entries = list(entries)
        self.tok_to_id: dict[bytes, int] = {}
        for i, e in enumerate(self.entries[4:], start=4):
            self.tok_to_id[bytes.fromhex(e)] = i
        self.id_to_tok: list[bytes | None] = [None] * 4 + [bytes.fromhex(e) for e in self.entries[4:]]

    @classmethod
    def load(cls, path: str | Path = VOCAB_PATH) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))

    def __len__(self) -> int:
        return len(self.entries)

    def encode(self, body: bytes, *, add_special: bool = False) -> list[int]:
        """Program body bytes -> token ids (rare/unknown tokens -> <unk>)."""
        ids = [self.tok_to_id.get(t, UNK) for t in split_tokens(body)]
        return [BOS, *ids, EOS] if add_special else ids

    def decode(self, ids: Iterable[int], *, strict: bool = True) -> bytes:
        """Token ids -> program body bytes.

        <pad>/<bos> are skipped and decoding stops at the first <eos>. <unk> raises ValueError
        when strict, otherwise it is dropped.
        """
        out = bytearray()
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            if i == UNK or not 0 <= i < len(self.id_to_tok):
                if strict:
                    raise ValueError(f"cannot decode token id {i}")
                continue
            out += self.id_to_tok[i]
        return bytes(out)

    def name(self, i: int) -> str:
        """Human-readable name of one token id (tivars accessible name)."""
        if i < 4:
            return SPECIALS[i]
        tok = _SHEET.bytes.get(self.id_to_tok[i])
        return tok.accessible if tok else f"<0x{self.entries[i]}>"


def render(body: bytes) -> str:
    """Program body -> readable TI-BASIC text (tivars decoding, TI-84 Plus CE sheet)."""
    from tivars.types import TIProgram

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TIProgram.decode(body, model=TI_84PCE, mode="accessible")


def render_ids(ids: Iterable[int], vocab: Vocab | None = None) -> str:
    """Token ids -> readable text; <unk> is shown as '<unk>'."""
    vocab = vocab or Vocab.load()
    parts = []
    chunk: list[int] = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i == UNK:
            parts.append(render(vocab.decode(chunk)))
            parts.append("<unk>")
            chunk = []
        elif i >= 4:
            chunk.append(i)
    parts.append(render(vocab.decode(chunk)))
    return "".join(parts)


def body_to_8xp(body: bytes, path: str | Path, name: str = "CALCLM") -> Path:
    """Write a program body as an 8xp file via tivars."""
    from tivars.types import TIProgram

    prog = TIProgram(name=name)
    prog.data = body
    path = Path(path)
    prog.export(name=name, model=TI_84PCE).save(str(path))
    return path


def ids_to_8xp(ids: Iterable[int], path: str | Path, name: str = "CALCLM",
               vocab: Vocab | None = None) -> Path:
    """Token ids -> 8xp file (<unk> ids are dropped)."""
    vocab = vocab or Vocab.load()
    return body_to_8xp(vocab.decode(ids, strict=False), path, name)


def read_8xp_bodies(data: bytes) -> list[tuple[str, int, bytes]]:
    """Parse a TI var file's bytes and return (name, type_id, body) for each program entry."""
    from tivars.var import TIVarFile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = TIVarFile()
        var.load_bytes(data)
    return [(e.name, e.type_id, bytes(e.data)) for e in var.entries if e.type_id in (5, 6)]
