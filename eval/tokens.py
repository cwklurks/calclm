"""TI-BASIC token helpers for the evaluator (SPEC.md "Tokens").

A program body splits uniquely into tokens: a byte in the two-byte prefix set of the TI-84
Plus CE token sheet (tivars TI_84PCE.tokens) starts a 2-byte token, any other byte is a
1-byte token. Same rule as corpus/tokenize.py, kept local so eval/ has no import-time
dependency on a workstream that is still changing.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from tivars.models import TI_84PCE

_SHEET = TI_84PCE.tokens
PREFIXES = frozenset(b[0] for b in _SHEET.bytes if len(b) == 2)
KNOWN = frozenset(_SHEET.bytes)
SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
BOS, EOS, UNK = 1, 2, 3

GETKEY = b"\xad"
PRGM = b"\x5f"
INPUT, PROMPT, PAUSE, MENU = b"\xdc", b"\xdd", b"\xd8", b"\xe6"
ASM_TOKENS = frozenset({b"\xbb\x6a", b"\xbb\x6b", b"\xbb\x6c"})   # Asm( AsmComp( AsmPrgm
ASM_HEADERS = (b"\xef\x7b", b"\xbb\x6d", b"\xef\x69", b"\xef\x68")
_DRAW_NAMES = ("ClrDraw", "DispGraph", "Line(", "Pt-On(", "Pt-Off(", "Pt-Change(", "Pxl-On(",
               "Pxl-Off(", "Pxl-Change(", "Circle(", "Text(", "Horizontal ", "Vertical ",
               "Shade(", "DrawF ", "DrawInv ", "Tangent(", "RecallPic ", "BackgroundOn ")
DRAW_TOKENS = frozenset(b for b, t in _SHEET.bytes.items()
                        if t.langs["en"].display in _DRAW_NAMES)
MAX_PROGRAM = 65_535
NAME_RE = re.compile(r"[A-Zθ][A-Z0-9θ]{0,7}")


class TokenError(ValueError):
    pass


def split_tokens(body: bytes) -> list[bytes]:
    out, i, n = [], 0, len(body)
    while i < n:
        step = 2 if body[i] in PREFIXES and i + 1 < n else 1
        out.append(body[i:i + step])
        i += step
    return out


def token_offsets(body: bytes) -> list[tuple[int, bytes]]:
    """(byte offset, token) pairs."""
    out, i = [], 0
    for tok in split_tokens(body):
        out.append((i, tok))
        i += len(tok)
    return out


def features(body: bytes) -> dict:
    """Static facts about a program body used to explain outcomes."""
    toks = split_tokens(body)
    s = set(toks)
    return {
        "n_tokens": len(toks),
        "getkey": GETKEY in s,
        "input": INPUT in s or PROMPT in s,
        "pause": PAUSE in s,
        "menu": MENU in s,
        "prgm_call": PRGM in s,
        "asm": bool(s & ASM_TOKENS) or body[:2] in ASM_HEADERS,
        "draw": bool(s & DRAW_TOKENS),
        "unknown_tokens": sorted({t.hex().upper() for t in s if t not in KNOWN}),
    }


def valid_name(name: str) -> bool:
    """A program name the home screen can type: letter or θ, then up to 7 of A-Z 0-9 θ."""
    return bool(NAME_RE.fullmatch(name))


def ids_to_body(ids: Sequence[int], vocab: Sequence[str], strict: bool = True) -> bytes:
    """Token ids (SPEC.md vocab.json order) to program bytes. Skips <pad>/<bos>, stops at
    <eos>; <unk> or an id outside the vocab raises TokenError (strict) or is dropped."""
    out = bytearray()
    for i in ids:
        bad = None if 0 <= i < len(vocab) else f"id {i} outside vocab of {len(vocab)}"
        entry = vocab[i] if bad is None else None
        if entry == "<eos>":
            break
        if entry in ("<pad>", "<bos>"):
            continue
        if entry == "<unk>":
            bad = "<unk> token"
        if bad:
            if strict:
                raise TokenError(bad)
            continue
        out += bytes.fromhex(entry)
    return bytes(out)


def display(tok: bytes) -> str:
    """Display text of one token ([HEX] if it is not in the CE token sheet)."""
    t = _SHEET.bytes.get(tok)
    return t.langs["en"].display if t is not None else f"[{tok.hex().upper()}]"


def line_at(body: bytes, offset: int) -> tuple[int, str]:
    """(1-based line number, display text) of the source line holding byte `offset`.
    Lines end at the newline token 0x3F; an offset on a newline belongs to the line it ends."""
    lines: list[list[bytes]] = [[]]
    line_no = 1
    for off, tok in token_offsets(body):
        if tok == b"\x3f":
            if off < offset:
                line_no += 1
            lines.append([])
        else:
            lines[-1].append(tok)
    line_no = min(line_no, len(lines))
    return line_no, "".join(display(t) for t in lines[line_no - 1])
