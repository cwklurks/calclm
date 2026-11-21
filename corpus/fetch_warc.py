"""Fetch ticalc.org /pub BASIC files from the Internet Archive's Aug 2014 ticalc.org crawl.

Item: https://archive.org/details/ticalc-2014-08 (one 2.68 GB WARC + CDX index).
Instead of downloading the whole WARC we read the CDX index, select the response records under
the wanted /pub/ prefixes, merge nearby records into ~200 byte ranges, and fetch only those ranges
(HTTP Range requests, 1 req/s). Each record is then cut out of its range, gunzipped, and its HTTP
body written to data/raw/ticalc_warc2014/files/<url path>.

Resumable: finished ranges are kept on disk and skipped on rerun.

Usage:  uv run python -I corpus/fetch_warc.py [--extract-only]
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import polite  # noqa: E402
from corpus.safepath import safe_relpath  # noqa: E402

ITEM = "ticalc-2014-08"
WARC_NAME = "ticalc.org-20140813.warc.gz"
CDX_NAME = "ticalc.org-20140813.cdx.gz"
BASE = f"https://archive.org/download/{ITEM}/"
OUT = ROOT / "data" / "raw" / "ticalc_warc2014"
PREFIXES = ("/pub/83plus/basic/", "/pub/83/basic/", "/pub/84pluscse/basic/")
MERGE_GAP = 1_000_000


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "crawl.log", "a") as f:
        f.write(line + "\n")


def load_records() -> list[dict]:
    cdx = OUT / CDX_NAME
    if not cdx.exists():
        body, _ = polite.fetch(BASE + CDX_NAME)
        cdx.write_bytes(body)
    recs = []
    with gzip.open(cdx, "rt", encoding="utf-8", errors="replace") as f:
        next(f)  # " CDX N b a m s k r M S V g"
        for line in f:
            p = line.split()
            if len(p) < 11 or p[4] != "200" or p[3].startswith("text/html"):
                continue
            path = "/" + p[0].split(")/", 1)[1] if ")/" in p[0] else ""
            if not path.startswith(PREFIXES):
                continue
            recs.append({"path": path, "url": p[2], "mime": p[3],
                         "length": int(p[8]), "offset": int(p[9])})
    recs.sort(key=lambda r: r["offset"])
    return recs


def merge_ranges(recs: list[dict]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for r in recs:
        s, e = r["offset"], r["offset"] + r["length"]
        if ranges and s - ranges[-1][1] <= MERGE_GAP:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], e))
        else:
            ranges.append((s, e))
    return ranges


def fetch_ranges(ranges: list[tuple[int, int]]) -> None:
    chunks = OUT / "chunks"
    chunks.mkdir(exist_ok=True)
    total = sum(e - s for s, e in ranges)
    done = 0
    for i, (s, e) in enumerate(ranges):
        dest = chunks / f"{s:012d}-{e:012d}.bin"
        if dest.exists() and dest.stat().st_size == e - s:
            done += e - s
            continue
        body, info = polite.fetch(BASE + WARC_NAME, headers={"Range": f"bytes={s}-{e - 1}"},
                                  timeout=600, max_bytes=e - s)
        if info["status"] != 206 or len(body) != e - s:
            raise polite.FetchError(f"bad range response {info['status']} len={len(body)}")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.rename(dest)
        done += e - s
        log(f"range {i + 1}/{len(ranges)} ok ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")


def http_body(raw: bytes) -> bytes | None:
    """Split a WARC response record payload into the HTTP body (handles chunked encoding)."""
    # WARC header block
    i = raw.find(b"\r\n\r\n")
    if i < 0 or not raw.startswith(b"WARC/"):
        return None
    payload = raw[i + 4:]
    j = payload.find(b"\r\n\r\n")
    if j < 0:
        return None
    head, body = payload[:j].decode("latin-1"), payload[j + 4:]
    lines = head.split("\r\n")
    if " 200 " not in lines[0] + " ":
        return None
    hdrs = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
    if "chunked" in hdrs.get("transfer-encoding", "").lower():
        out, pos = bytearray(), 0
        while True:
            nl = body.find(b"\r\n", pos)
            if nl < 0:
                return None
            size = int(body[pos:nl].split(b";")[0] or b"0", 16)
            if size == 0:
                break
            out += body[nl + 2:nl + 2 + size]
            pos = nl + 2 + size + 2
        body = bytes(out)
    if hdrs.get("content-encoding", "").lower() == "gzip":
        body = zlib.decompress(body, 16 + zlib.MAX_WBITS)
    if body.endswith(b"\r\n\r\n"):  # WARC record trailer when not stripped by gunzip
        body = body[:-4]
    return body


def extract(recs: list[dict], ranges: list[tuple[int, int]]) -> None:
    files = OUT / "files"
    chunks = OUT / "chunks"
    manifest = []
    ri = 0
    data = b""
    cur = (-1, -1)
    bad = 0
    for r in recs:
        while not (ranges[ri][0] <= r["offset"] < ranges[ri][1]):
            ri += 1
        if ranges[ri] != cur:
            cur = ranges[ri]
            data = (chunks / f"{cur[0]:012d}-{cur[1]:012d}.bin").read_bytes()
        rel = r["offset"] - cur[0]
        try:
            raw = gzip.decompress(data[rel:rel + r["length"]])
            body = http_body(raw)
        except (OSError, EOFError, ValueError, zlib.error):
            body = None
        rp = safe_relpath(r["path"].removeprefix("/pub/"))
        if body is None or rp is None:
            bad += 1
            continue
        dest = files / rp
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        manifest.append({"path": r["path"], "url": r["url"], "bytes": len(body),
                         "local": str(dest.relative_to(OUT))})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=0))
    log(f"extracted {len(manifest)} files, {bad} unreadable records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    ranges = merge_ranges(recs)
    log(f"{len(recs)} records in {len(ranges)} ranges, "
        f"{sum(e - s for s, e in ranges) / 1e6:.1f} MB to fetch")
    if "--extract-only" not in sys.argv:
        fetch_ranges(ranges)
    extract(recs, ranges)


if __name__ == "__main__":
    main()
