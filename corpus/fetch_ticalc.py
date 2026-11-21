"""Fetch ticalc.org BASIC files that are NOT already in the 2014 archive.org WARC.

ticalc.org publishes /pub/master.index, a plain-text index of every file in the archive, so
no directory crawling is needed: one request for the index, then one request per missing file
(1 req/s, robots.txt checked). Files go to data/raw/ticalc/files/<path under /pub/>.

Resumable: files already on disk are skipped; failures are logged and retried on rerun.

Usage:  uv run python -I corpus/fetch_ticalc.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import polite  # noqa: E402
from corpus.safepath import safe_relpath  # noqa: E402

OUT = ROOT / "data" / "raw" / "ticalc"
WARC_CDX = ROOT / "data" / "raw" / "ticalc_warc2014" / "ticalc.org-20140813.cdx.gz"
BASE = "https://www.ticalc.org"
# TI-83 (.83p), TI-83 Plus/84 Plus (.8xp), TI-84 Plus C SE / CE (ticalc files CE BASIC here),
# TI-84 Evo (CE-family hardware).
WANT = re.compile(r"/pub/(83plus|83|84pluscse|84plusce|84evo)/basic/")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "crawl.log", "a") as f:
        f.write(line + "\n")


def parse_master_index(text: str) -> list[str]:
    cur = None
    files = []
    for line in text.splitlines():
        m = re.match(r"Index of (/pub/\S*)", line)
        if m:
            cur = m.group(1).rstrip("/") + "/"
            continue
        if cur and line.strip() and not line[0].isspace():
            files.append(cur + line.split()[0])
    return files


def warc_paths() -> set[str]:
    out = set()
    if not WARC_CDX.exists():
        return out
    with gzip.open(WARC_CDX, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) > 4 and p[4] == "200" and ")/" in p[0]:
                out.add(("/" + p[0].split(")/", 1)[1]).lower())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = OUT / "master.index"
    if not idx.exists():
        body, _ = polite.fetch(BASE + "/pub/master.index")
        idx.write_bytes(body)
    files = [f for f in parse_master_index(idx.read_text("latin-1")) if WANT.match(f)]
    in_warc = warc_paths()
    todo = [f for f in files if f.lower() not in in_warc]
    (OUT / "wanted.json").write_text(json.dumps({"listed": files, "fetch_live": todo}, indent=0))
    log(f"{len(files)} BASIC files listed, {len(todo)} not in 2014 WARC -> fetch live")
    ok = fail = skip = 0
    for i, path in enumerate(todo):
        rp = safe_relpath(path.removeprefix("/pub/"))
        if rp is None:
            fail += 1
            continue
        dest = OUT / "files" / rp
        if dest.exists():
            skip += 1
            continue
        try:
            body, _ = polite.fetch(BASE + path, timeout=120, max_bytes=32 * 1024 * 1024)
        except polite.FetchError as e:
            fail += 1
            log(f"FAIL {path}: {e}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        ok += 1
        if (i + 1) % 50 == 0:
            log(f"{i + 1}/{len(todo)} ok={ok} skip={skip} fail={fail}")
    log(f"DONE ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
