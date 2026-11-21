"""Extract TI-BASIC program variables from downloaded raw files.

Everything is processed in memory: archives are never unpacked to disk, so member names cannot
cause path traversal; symlink/encrypted members are skipped; per-member, per-file and nesting
limits stop zip bombs. Downloaded content is only parsed, never executed.

For every program variable found (in .8xp/.83p files, .8xg/.83g groups, inside zip/tar/gzip,
nested up to MAX_DEPTH) one JSON line is written to data/extract/programs.jsonl with a status:
  kept | asm | asm_hex83 | undecodable | trivial
plus tags for kept programs (ice_source, axe_source, dcs_header, lib_calls, asm_call, protected).
Per-file outcomes go to data/extract/files.jsonl.

Usage:  uv run python -I corpus/extract.py [--workers N]

The author's own programs (eval only, never trained on) are read from CALCLM_EVAL_DIRS: folders
of .8xp files, separated like PATH.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import stat
import sys
import tarfile
import warnings
import zipfile
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.tokenize import is_known, split_tokens  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "extract"
SOURCES = {  # source name -> directory of downloaded files
    "ticalc_warc2014": RAW / "ticalc_warc2014" / "files",
    "ticalc": RAW / "ticalc" / "files",
    "tiplanet": RAW / "tiplanet" / "files",
    "github": RAW / "github" / "files",
    "author_eval": None,  # read from AUTHOR_DIRS (eval only, never trained on)
}
# The author's own programs: CALCLM_EVAL_DIRS (folders of .8xp files, os.pathsep-separated);
# none by default.
SETUP_EVAL_DIRS: list[Path] = []
AUTHOR_DIRS = ([Path(d) for d in os.environ["CALCLM_EVAL_DIRS"].split(os.pathsep) if d]
               if os.environ.get("CALCLM_EVAL_DIRS") else SETUP_EVAL_DIRS)

MAX_FILE = 64 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024
MAX_TOTAL_PER_FILE = 512 * 1024 * 1024
MAX_DEPTH = 4
MAX_RATIO = 2000

TI_MAGICS = (b"**TI83F*", b"**TI83**")
OTHER_TI_MAGICS = (b"**TI82**", b"**TI73**", b"**TI85**", b"**TI86**", b"**TI89**",
                   b"**TI92**", b"**TI92P*")
ASM_HEADERS = (b"\xBB\x6D", b"\xBB\x6C", b"\xEF\x7B", b"\xEF\x7A", b"\xEF\x6D",
               b"\xEF\x69", b"\xEF\x68")
MIN_TOKENS = 8
UNK_MAX_FRAC = 0.005
UNK_MAX_ABS = 2
LIB_FUNCS = {b"\xBB\x26": "real(", b"\xB3": "det(", b"\xB6": "sum(", b"\xB4": "identity("}


class Budget:
    def __init__(self) -> None:
        self.left = MAX_TOTAL_PER_FILE

    def take(self, n: int) -> bool:
        self.left -= n
        return self.left >= 0


# ---------------------------------------------------------------- TI var parsing

def parse_var_manual(data: bytes) -> list[tuple[str, int, bytes]]:
    """Minimal fallback parser for 8xp/83p/8xg files that tivars rejects."""
    out = []
    if len(data) < 57:
        return out
    pos, end = 55, min(len(data), 55 + int.from_bytes(data[53:55], "little"))
    while pos + 4 <= end:
        meta_len = int.from_bytes(data[pos:pos + 2], "little")
        if meta_len not in (0x0B, 0x0D):
            break
        dlen = int.from_bytes(data[pos + 2:pos + 4], "little")
        type_id = data[pos + 4]
        name = data[pos + 5:pos + 13].rstrip(b"\x00").decode("latin-1")
        dpos = pos + 2 + meta_len + 2
        blob = data[dpos:dpos + dlen]
        if type_id in (5, 6) and len(blob) >= 2:
            n = int.from_bytes(blob[:2], "little")
            out.append((name, type_id, blob[2:2 + n]))
        elif type_id == 0x17:
            out.extend(parse_group_manual(blob[2:]))
        pos = dpos + dlen
    return out


def parse_group_manual(data: bytes) -> list[tuple[str, int, bytes]]:
    out = []
    pos = 0
    while pos + 7 <= len(data):
        t = data[pos] & 0x3F
        pos += 3
        if t in (5, 6, 0x15, 0x17):
            page, nlen = data[pos + 2], data[pos + 3]
            pos += 4
            name = data[pos:pos + nlen].decode("latin-1")
            pos += nlen
        elif t in (1, 0x0D):
            nlen = data[pos + 3]
            pos += 4
            name = data[pos:pos + nlen - 1].decode("latin-1")
            pos += nlen
        else:
            pos += 3
            name = data[pos:pos + 3].decode("latin-1")
            pos += 3
        if pos + 2 > len(data):
            break
        dlen = int.from_bytes(data[pos:pos + 2], "little")
        if t in (5, 6):
            out.append((name, t, data[pos + 2:pos + 2 + dlen]))
        pos += 2 + dlen
    return out


def parse_var(data: bytes) -> tuple[list[tuple[str, int, bytes]], str]:
    """Return program entries (name, type_id, body) and which parser succeeded."""
    from tivars.types.group import TIGroup
    from tivars.var import TIVarFile

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var = TIVarFile()
            var.load_bytes(data)
            progs = []
            for e in var.entries:
                if e.type_id in (5, 6):
                    progs.append((e.name, e.type_id, bytes(e.data)))
                elif e.type_id == TIGroup.type_id:
                    for sub in e.ungroup():
                        if sub.type_id in (5, 6):
                            progs.append((sub.name, sub.type_id, bytes(sub.data)))
        return progs, "tivars"
    except Exception:  # noqa: BLE001 - malformed untrusted input; fall back
        try:
            return parse_var_manual(data), "manual"
        except Exception:  # noqa: BLE001
            return [], "failed"


# ---------------------------------------------------------------- classification

def _lib_call(toks: list[bytes], i: int) -> bool:
    j = i + 1
    while j < len(toks) and len(toks[j]) == 1 and 0x30 <= toks[j][0] <= 0x39:
        j += 1
    return j > i + 1 and j < len(toks) and toks[j] == b"\x2B"


def classify(body: bytes) -> dict:
    stripped = body.lstrip(b"\x3E\x3F")  # leading ':' / empty lines are cosmetic
    if body[:2] in ASM_HEADERS or stripped[:2] in ASM_HEADERS:
        return {"status": "asm"}
    toks = split_tokens(body)
    n = len(toks)
    n_unk = sum(1 for t in toks if not is_known(t))
    tags = []
    # TI-83 (non-plus) hex assembly loaded via Send(9prgm...): ends with End / 0000 / End
    if body.rstrip(b"\x3F").endswith(b"\xD4\x3F\x30\x30\x30\x30\x3F\xD4"):
        return {"status": "asm_hex83", "n_tokens": n}
    if n_unk > max(UNK_MAX_ABS, UNK_MAX_FRAC * n):
        return {"status": "undecodable", "n_tokens": n, "n_unk": n_unk}
    if n < MIN_TOKENS:
        return {"status": "trivial", "n_tokens": n, "n_unk": n_unk}
    head = stripped[:4]
    # ICE source starts with "[i]": '[' imaginary-i ']' as typed on calc, or the single 0x2C
    # token that text tokenizers produce for "[i]".
    if head.startswith((b"\x06\x2C\x07", b"\x06\xBB\xB8\x07")) or (
            head[:1] == b"\x2C" and len(head) > 1 and 0x41 <= head[1] <= 0x5B):
        tags.append("ice_source")
    if len(stripped) > 1 and stripped[0] == 0x3A and 0x41 <= stripped[1] <= 0x5B:
        tags.append("axe_source")
    if stripped.startswith(b"DCS"):
        tags.append("dcs_header")
    if any(t in LIB_FUNCS and _lib_call(toks, i) for i, t in enumerate(toks)) \
            or b"\xEF\x11" in toks:
        tags.append("lib_calls")
    if b"\xBB\x6A" in toks:
        tags.append("asm_call")
    return {"status": "kept", "n_tokens": n, "n_unk": n_unk, "tags": tags}


# ---------------------------------------------------------------- container walking

def is_symlink_zipinfo(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def walk_blob(data: bytes, chain: str, depth: int, budget: Budget, found: list, events: Counter,
              hint: str = "") -> None:
    if depth > MAX_DEPTH:
        events["too_deep"] += 1
        return
    if data[:8] in TI_MAGICS:
        progs, how = parse_var(data)
        events[f"var_{how}"] += 1
        for name, tid, body in progs:
            found.append((chain, name, tid, body))
        if not progs:
            events["var_without_program"] += 1
        return
    if data[:8] in OTHER_TI_MAGICS:
        events["other_model_var"] += 1
        return
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        walk_zip(data, chain, depth, budget, found, events)
        return
    if data[:2] == b"\x1f\x8b":
        if walk_tar_stream(data, chain, depth, budget, found, events):
            return
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
                inner = g.read(MAX_MEMBER + 1)
        except (OSError, EOFError, zlib.error):
            events["bad_gzip"] += 1
            return
        if len(inner) > MAX_MEMBER or not budget.take(len(inner)):
            events["oversize"] += 1
            return
        walk_blob(inner, chain + "!gz", depth + 1, budget, found, events)
        return
    if len(data) > 262 and data[257:262] == b"ustar":
        walk_tar(data, chain, depth, budget, found, events)
        return
    if data[:4] == b"Rar!" or data[:6] == b"7z\xbc\xaf\x27\x1c" or data[:4] == b"SIT!" \
            or data[:8] == b"StuffIt ":
        events["unsupported_archive"] += 1
        return
    events["other_file"] += 1


def walk_zip(data: bytes, chain: str, depth: int, budget: Budget, found: list,
             events: Counter) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        events["bad_zip"] += 1
        return
    for info in infos:
        if info.is_dir() or is_symlink_zipinfo(info):
            continue
        if info.flag_bits & 0x1:
            events["zip_encrypted_member"] += 1
            continue
        if info.file_size > MAX_MEMBER or (
                info.compress_size and info.file_size / info.compress_size > MAX_RATIO
                and info.file_size > 1_000_000):
            events["oversize"] += 1
            continue
        try:
            with zf.open(info) as fh:
                member = fh.read(MAX_MEMBER + 1)
        except NotImplementedError:
            events["zip_unsupported_method"] += 1
            continue
        except (zipfile.BadZipFile, OSError, EOFError, zlib.error, ValueError):
            events["zip_member_error"] += 1
            continue
        if len(member) > MAX_MEMBER or not budget.take(len(member)):
            events["oversize"] += 1
            continue
        walk_blob(member, f"{chain}/{info.filename}", depth + 1, budget, found, events)


def walk_tar(data: bytes, chain: str, depth: int, budget: Budget, found: list,
             events: Counter) -> None:
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        members = tf.getmembers()
    except (tarfile.TarError, OSError, EOFError):
        events["bad_tar"] += 1
        return
    for m in members:
        if not m.isfile() or m.size > MAX_MEMBER:
            continue
        fh = tf.extractfile(m)
        if fh is None:
            continue
        member = fh.read(MAX_MEMBER + 1)
        if not budget.take(len(member)):
            events["oversize"] += 1
            continue
        walk_blob(member, f"{chain}/{m.name}", depth + 1, budget, found, events)


TAR_INTERESTING = (".8xp", ".83p", ".8xg", ".83g", ".zip", ".gz", ".tgz", ".tar")
TAR_SMALL = 256 * 1024


def walk_tar_stream(data: bytes, chain: str, depth: int, budget: Budget, found: list,
                    events: Counter) -> bool:
    """Stream a .tar.gz (e.g. a GitHub repo tarball) member by member with bounded memory.

    Only members with a TI/archive extension or <= 256 KB are read. Returns False if the gzip
    payload is not a tar archive (caller then treats it as a single gzipped file).
    """
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r|gz")
    except (tarfile.ReadError, OSError, EOFError, zlib.error):
        return False
    try:
        for m in tf:
            if not m.isfile():
                continue
            if m.size > MAX_MEMBER:
                events["oversize"] += 1
                continue
            if not (m.name.lower().endswith(TAR_INTERESTING) or m.size <= TAR_SMALL):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            member = fh.read(MAX_MEMBER + 1)
            if not budget.take(len(member)):
                events["oversize"] += 1
                break
            walk_blob(member, f"{chain}/{m.name}", depth + 1, budget, found, events)
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        events["bad_tar"] += 1
    return True


def process_file(args: tuple[str, str, str]) -> tuple[list[dict], dict]:
    source, path, rel = args
    events: Counter = Counter()
    found: list = []
    try:
        size = Path(path).stat().st_size
        if size > MAX_FILE:
            events["oversize"] += 1
        else:
            walk_blob(Path(path).read_bytes(), rel, 0, Budget(), found, events)
    except Exception as e:  # noqa: BLE001 - one bad file must not kill the run
        events[f"crash:{type(e).__name__}"] += 1
    progs = []
    for chain, name, tid, body in found:
        rec = {"source": source, "file": rel, "member": chain, "name": name, "type_id": tid,
               "n_bytes": len(body), "sha1": hashlib.sha1(body).hexdigest()}
        rec.update(classify(body))
        if tid == 6 and rec["status"] == "kept":
            rec["tags"] = rec["tags"] + ["protected"]
        if rec["status"] in ("kept", "undecodable", "trivial"):
            rec["body"] = body.hex()
        progs.append(rec)
    return progs, {"source": source, "file": rel, "n_programs": len(found), "events": dict(events)}


def list_inputs() -> list[tuple[str, str, str]]:
    jobs = []
    for source, d in SOURCES.items():
        if source == "author_eval":
            for cd in AUTHOR_DIRS:
                for p in sorted(cd.glob("*.8xp")):
                    jobs.append((source, str(p), f"{cd.name}/{p.name}"))
            continue
        if d is None or not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.is_symlink():
                jobs.append((source, str(p), str(p.relative_to(d))))
    return jobs


def main() -> None:
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = list_inputs()
    print(f"{len(jobs)} input files", flush=True)
    n_prog = 0
    with open(OUT / "programs.jsonl", "w") as fp, open(OUT / "files.jsonl", "w") as ff, \
            mp.get_context("spawn").Pool(workers) as pool:
        for i, (progs, finfo) in enumerate(pool.imap_unordered(process_file, jobs, chunksize=16)):
            for r in progs:
                fp.write(json.dumps(r) + "\n")
            ff.write(json.dumps(finfo) + "\n")
            n_prog += len(progs)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)} files, {n_prog} program vars", flush=True)
    print(f"done: {len(jobs)} files, {n_prog} program vars", flush=True)


if __name__ == "__main__":
    main()
