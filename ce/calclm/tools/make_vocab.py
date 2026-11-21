#!/usr/bin/env python3
"""Build CLMVOC.8xv: the token table the calculator program needs, from an export's own
model.json["vocab"] (never corpus/vocab.json: ids differ between exports).

Layout (little-endian):
    0   "CV"                    magic
    2   u8   format version (1)
    3   u8   glyph count G (<= 128): custom 8x8 glyphs for character codes 0x80 + i
    4   u16  vocab size (must equal the blob header's vocab)
    6   u24  blob size (must equal the end of the last CLMnn AppVar slice)
    9   6 B  first bytes of the export's sha256 (shown on screen, not checked)
    15  u8   reserved (0)
    16  G * 8 glyph rows (MSB = leftmost pixel)
    ..  u16  offsets[vocab + 1] into the record area (relative to its start)
    ..  records: u8 n (token byte count, 0 for <pad> <bos> <eos> <unk>), n token bytes,
        then the display string (ASCII 0x20-0x7E, 0x0A = newline, 0x80+ = glyph)

    uv run python ce/calclm/tools/make_vocab.py runs/v1/export -o ce/calclm/bin/CLMVOC.8xv
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

FORMAT = 1
NEWLINE_TOKEN = "3F"

# 6-wide glyphs in an 8x8 cell (rows top to bottom), for TI characters not in ASCII.
_G = {
    "→": ["......", "...#..", "....#.", "######", "....#.", "...#..", "......", "......"],
    "ʟ": ["......", "......", "#.....", "#.....", "#.....", "#.....", "####..", "......"],
    "⁻": ["......", "####..", "......", "......", "......", "......", "......", "......"],
    "θ": [".###..", "#...#.", "#...#.", "#####.", "#...#.", "#...#.", ".###..", "......"],
    "≠": ["......", "....#.", "#####.", "..#...", "#####.", ".#....", "......", "......"],
    "²": [".##...", "#..#..", "..#...", ".#....", "####..", "......", "......", "......"],
    "³": ["###...", "...#..", ".##...", "...#..", "###...", "......", "......", "......"],
    "¹": [".#....", "##....", ".#....", ".#....", "###...", "......", "......", "......"],
    "≥": ["#.....", ".##...", "...##.", ".##...", "#.....", "......", "#####.", "......"],
    "≤": ["...##.", ".##...", "#.....", ".##...", "...##.", "......", "#####.", "......"],
    "π": ["......", "......", "######", ".#..#.", ".#..#.", ".#..#.", ".#..#.", "......"],
    "√": ["...###", "...#..", "...#..", "#..#..", ".#.#..", "..##..", "...#..", "......"],
    "°": [".##...", "#..#..", "#..#..", ".##...", "......", "......", "......", "......"],
    "ᴇ": ["......", "......", "####..", "#.....", "###...", "#.....", "####..", "......"],
    "►": ["#.....", "##....", "###...", "####..", "###...", "##....", "#.....", "......"],
    "◄": ["...#..", "..##..", ".###..", "####..", ".###..", "..##..", "...#..", "......"],
    "·": ["......", "......", "......", "..##..", "..##..", "......", "......", "......"],
    "Σ": ["#####.", "#.....", ".#....", "..#...", ".#....", "#.....", "#####.", "......"],
    "Δ": ["......", "..#...", ".#.#..", ".#.#..", "#...#.", "#...#.", "#####.", "......"],
    "Ω": [".###..", "#...#.", "#...#.", "#...#.", ".#.#..", ".#.#..", "##.##.", "......"],
    "Φ": ["..#...", ".###..", "#.#.#.", "#.#.#.", "#.#.#.", ".###..", "..#...", "......"],
    "α": ["......", "......", ".##.#.", "#..#..", "#..#..", "#..#..", ".##.#.", "......"],
    "β": [".##...", "#..#..", "#.#...", "#..#..", "#..#..", "###...", "#.....", "#....."],
    "γ": ["......", "......", "#...#.", ".#.#..", "..#...", "..#...", "..#...", "......"],
    "δ": [".###..", "#.....", ".##...", "#..#..", "#..#..", ".##...", "......", "......"],
    "ε": ["......", "......", ".###..", "#.....", ".##...", "#.....", ".###..", "......"],
    "λ": ["#.....", ".#....", ".#....", "..#...", ".#.#..", "#...#.", "#...#.", "......"],
    "μ": ["......", "......", "#..#..", "#..#..", "#..#..", "###.#.", "#.....", "#....."],
    "ρ": ["......", "......", ".##...", "#..#..", "#..#..", "###...", "#.....", "#....."],
    "σ": ["......", "......", ".####.", "#..#..", "#..#..", "#..#..", ".##...", "......"],
    "τ": ["......", "......", "#####.", "..#...", "..#...", "..#...", "...#..", "......"],
    "χ": ["......", "......", "#...#.", ".#.#..", "..#...", ".#.#..", "#...#.", "......"],
    "∫": ["...##.", "..#...", "..#...", "..#...", "..#...", "..#...", "##....", "......"],
    "↑": ["..#...", ".###..", "#.#.#.", "..#...", "..#...", "..#...", "..#...", "......"],
    "↓": ["..#...", "..#...", "..#...", "..#...", "#.#.#.", ".###..", "..#...", "......"],
    "×": ["......", "#...#.", ".#.#..", "..#...", ".#.#..", "#...#.", "......", "......"],
    "ʳ": ["#.##..", "##....", "#.....", "#.....", "......", "......", "......", "......"],
    "ᵀ": ["###...", ".#....", ".#....", ".#....", "......", "......", "......", "......"],
    "ˣ": ["#.#...", ".#....", "#.#...", "......", "......", "......", "......", "......"],
    "…": ["......", "......", "......", "......", "......", "#.#.#.", "......", "......"],
    "□": ["......", "#####.", "#...#.", "#...#.", "#...#.", "#####.", "......", "......"],
    "∠": ["......", "....#.", "...#..", "..#...", ".#....", "#####.", "......", "......"],
    "x̄": ["#####.", "......", "#...#.", ".#.#..", "..#...", ".#.#..", "#...#.", "......"],
    "ȳ": ["#####.", "......", "#...#.", "#...#.", ".####.", "....#.", ".###..", "......"],
    "p̂": ["..#...", ".#.#..", "......", "####..", "#...#.", "####..", "#.....", "#....."],
}
_DIGITS = ["###,#.#,#.#,#.#,###", ".#.,##.,.#.,.#.,###", "###,..#,###,#..,###",
           "###,..#,.##,..#,###", "#.#,#.#,###,..#,..#", "###,#..,###,..#,###",
           "###,#..,###,#.#,###", "###,..#,..#,.#.,.#.", "###,#.#,###,#.#,###",
           "###,#.#,###,..#,###"]
for _i, _d in enumerate(_DIGITS):       # subscript digits: 3x5, rows 3..7
    _G[chr(0x2080 + _i)] = ["......"] * 3 + [r + "..." for r in _d.split(",")]
_ASCII_FALLBACK = {"﹢": "+", "⁄": "/", "⌸": "=", "⬚": "?", "¡": "!", "¿": "?", "´": "'",
                   "¨": '"', "ß": "ss", "🡁": "^", "🠿": "v"}


def glyph_bytes(rows: list[str]) -> bytes:
    return bytes(int(r.ljust(8, ".").replace("#", "1").replace(".", "0"), 2) for r in rows)


def display_of(hexbytes: str) -> str:
    from tivars.models import TI_84PCE
    tok = TI_84PCE.tokens.bytes[bytes.fromhex(hexbytes)]
    return tok.langs["en"].display


class Encoder:
    """Display strings -> bytes; collects the glyphs actually used."""

    def __init__(self):
        self.glyphs: list[str] = []

    def code(self, key: str) -> int:
        if key not in self.glyphs:
            if len(self.glyphs) >= 128:
                raise ValueError("more than 128 glyphs")
            self.glyphs.append(key)
        return 0x80 + self.glyphs.index(key)

    def encode(self, text: str) -> bytes:
        out = bytearray()
        i = 0
        while i < len(text):
            pair = text[i:i + 2]
            if pair in _G:                           # base letter + combining mark
                out.append(self.code(pair))
                i += 2
                continue
            ch = text[i]
            i += 1
            if ch == "\n":
                out.append(0x0A)
            elif 0x20 <= ord(ch) < 0x7F:
                out.append(ord(ch))
            elif ch in _G:
                out.append(self.code(ch))
            elif ch in _ASCII_FALLBACK:
                out += _ASCII_FALLBACK[ch].encode()
            else:                                    # accents, math italics: NFKD to ASCII
                plain = "".join(c for c in unicodedata.normalize("NFKD", ch)
                                if 0x20 <= ord(c) < 0x7F)
                out += (plain or "?").encode()
        return bytes(out)


def build(export_dir: Path) -> tuple[bytes, list[str], list[bytes]]:
    meta = json.loads((export_dir / "model.json").read_text())
    vocab = meta["vocab"]
    if vocab[:4] != ["<pad>", "<bos>", "<eos>", "<unk>"]:
        raise ValueError("ids 0-3 must be the specials")
    enc = Encoder()
    records = []
    for i, h in enumerate(vocab):
        if i < 4:
            records.append(b"\x00")
            continue
        raw = bytes.fromhex(h)
        text = "\n" if h == NEWLINE_TOKEN else display_of(h)
        records.append(bytes([len(raw)]) + raw + enc.encode(text))
    offs, pos = [], 0
    for r in records:
        offs.append(pos)
        pos += len(r)
    offs.append(pos)
    if pos >= 1 << 16:
        raise ValueError("record area too large")
    sha = bytes.fromhex(meta["sha256"][:12])
    head = (b"CV" + bytes([FORMAT, len(enc.glyphs)]) + len(vocab).to_bytes(2, "little")
            + meta["blob_bytes"].to_bytes(3, "little") + sha + b"\0")
    glyphs = b"".join(glyph_bytes(_G[g]) for g in enc.glyphs)
    data = head + glyphs + b"".join(o.to_bytes(2, "little") for o in offs) + b"".join(records)
    return data, enc.glyphs, records


def decode(data: bytes) -> list[tuple[bytes, bytes]]:
    """(token bytes, display bytes) per id, read back like the calculator does."""
    if data[:2] != b"CV" or data[2] != FORMAT:
        raise ValueError("not a CLMVOC payload")
    g, v = data[3], int.from_bytes(data[4:6], "little")
    base = 16 + 8 * g
    offs = [int.from_bytes(data[base + 2 * i: base + 2 * i + 2], "little") for i in range(v + 1)]
    rec0 = base + 2 * (v + 1)
    out = []
    for i in range(v):
        r = data[rec0 + offs[i]: rec0 + offs[i + 1]]
        n = r[0]
        out.append((r[1:1 + n], r[1 + n:]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args(argv)
    data, glyphs, _ = build(args.export)
    if len(data) > 65505:
        raise SystemExit("CLMVOC larger than an AppVar")
    from tivars.types import TIAppVar
    var = TIAppVar(name="CLMVOC")
    var.data = data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    var.save(str(args.output))
    print(f"{args.output}: {len(data)} bytes, {len(glyphs)} glyphs: {''.join(glyphs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
