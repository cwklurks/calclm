"""Input adapters: turn .8xp files, token-id sequences or a corpus sample into ProgramSpecs,
each with a .8xp file ready to send to the emulated calculator (or a reject reason)."""
from __future__ import annotations

import json
import random
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tivars import TIEntry, TIProgram

from . import tokens as T

FALLBACK_NAME = "ZZEVAL"          # used when a program's own name cannot be typed
MAX_BODY = 65_505                 # largest TI variable body
TI_MAGIC = b"**TI83F*"
PROGRAM_TYPES = (5, 6)            # program, protected program


@dataclass
class ProgramSpec:
    index: int
    name: str                     # name on the calculator (what the home screen types)
    orig_name: str
    source: str
    body: bytes
    path: Path | None             # .8xp to send; None when rejected
    reject_reason: str | None = None
    meta: dict = field(default_factory=dict)


def _save(name: str, body: bytes, path: Path) -> None:
    p = TIProgram(name=name)
    p.data = body
    path.parent.mkdir(parents=True, exist_ok=True)
    p.save(str(path))


def _spec(index: int, orig: str, body: bytes, source: str, workdir: Path,
          meta: dict | None = None, reason: str | None = None) -> ProgramSpec:
    name = orig if T.valid_name(orig) else FALLBACK_NAME
    if reason is None and not body:
        reason = "empty program"
    if reason is None and len(body) > MAX_BODY:
        reason = f"program body {len(body)} bytes > {MAX_BODY}"
    path = None
    if reason is None:
        path = Path(workdir) / f"{index:06d}-{name}.8xp"
        try:
            _save(name, body, path)
        except Exception as exc:  # noqa: BLE001 - any tivars refusal is a REJECTED outcome
            reason, path = f"tivars: {type(exc).__name__}: {exc}", None
    return ProgramSpec(index, name, orig, source, body, path, reason, meta or {})


def from_bodies(items: Iterable[tuple[str, bytes]], workdir: Path, start: int = 0,
                source: str = "bodies") -> list[ProgramSpec]:
    return [_spec(start + i, name, body, f"{source}[{i}]", workdir)
            for i, (name, body) in enumerate(items)]


def read_8xp(path: Path) -> tuple[str, bytes]:
    """(name, body) of the first program in a .8xp file; ValueError if it is not one."""
    raw = Path(path).read_bytes()
    if raw[:8] != TI_MAGIC:
        raise ValueError("not a TI-83 Plus/84 Plus family variable file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entry = TIEntry.open(str(path))
    if entry.type_id not in PROGRAM_TYPES:
        raise ValueError(f"variable type {entry.type_id} is not a program")
    return entry.name, bytes(entry.data)


def from_dir(directory: Path, workdir: Path) -> list[ProgramSpec]:
    """Every *.8xp in `directory` (sorted by file name)."""
    specs = []
    for i, f in enumerate(sorted(Path(directory).glob("*.8xp"), key=lambda p: p.name)):
        try:
            name, body = read_8xp(f)
        except Exception as exc:  # noqa: BLE001 - unreadable file = REJECTED
            specs.append(ProgramSpec(i, f.stem, f.stem, str(f), b"", None,
                                     f"tivars: {type(exc).__name__}: {exc}"))
            continue
        spec = _spec(i, name, body, str(f), workdir)
        if spec.path is not None and spec.name == name:
            spec.path = f          # send the original file untouched
        specs.append(spec)
    return specs


def load_vocab(path: Path) -> list[str]:
    vocab = json.loads(Path(path).read_text())
    if tuple(vocab[:4]) != T.SPECIALS:
        raise ValueError(f"{path}: vocab must start with {T.SPECIALS}")
    return vocab


def read_bin(path: Path) -> list[list[int]]:
    """Token-id sequences from a corpus .bin (little-endian uint16, <bos> ... <eos> each)."""
    import numpy as np  # noqa: PLC0415
    ids = np.fromfile(path, dtype="<u2").tolist()
    seqs, cur = [], []
    for i in ids:
        if i == T.BOS and cur:
            seqs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        seqs.append(cur)
    return seqs


def gen_name(i: int) -> str:
    return f"G{i:07d}"


def from_token_ids(seqs: Sequence[Sequence[int]], vocab_path: Path, workdir: Path,
                   start: int = 0) -> list[ProgramSpec]:
    """Token-id sequences (SPEC.md vocab order) -> programs named G0000000, G0000001, ..."""
    vocab = load_vocab(vocab_path)
    specs = []
    for i, ids in enumerate(seqs):
        idx = start + i
        try:
            body = T.ids_to_body(ids, vocab)
        except T.TokenError as exc:   # keep the decodable tokens for the statistics
            specs.append(ProgramSpec(idx, gen_name(idx), gen_name(idx), f"ids[{i}]",
                                     T.ids_to_body(ids, vocab, strict=False), None,
                                     f"tokens: {exc}"))
            continue
        specs.append(_spec(idx, gen_name(idx), body, f"ids[{i}]", workdir))
    return specs


def from_corpus(path: Path, n: int, seed: int, workdir: Path, statuses: tuple = ("kept",),
                exclude_tags: tuple = ("ice_source", "axe_source"),
                exclude_sources: tuple = ("author_eval",)) -> list[ProgramSpec]:
    """Random sample (fixed seed) of programs from corpus data/extract/programs.jsonl."""
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("status") in statuses and r.get("source") not in exclude_sources
                    and not set(r.get("tags", ())) & set(exclude_tags) and r.get("body")):
                rows.append(r)
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    specs = []
    for i, r in enumerate(picked):
        meta = {k: r.get(k) for k in ("source", "file", "member", "sha1", "status", "tags",
                                      "type_id")}
        specs.append(_spec(i, r["name"], bytes.fromhex(r["body"]), r.get("member", "?"),
                           workdir, meta))
    return specs
