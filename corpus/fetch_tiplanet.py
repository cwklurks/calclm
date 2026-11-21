"""Fetch TI-BASIC archives from TI-Planet (tiplanet.org/forum/archives_*.php).

Phase 1 (listing): for each top category (ce = TI-83 Premium CE / 84 Plus CE, z80 = 82+/83+/84+,
84c = 84 Plus C SE, 82a = 82 Advanced) read its subcategories, page through
archives_list.php (25 rows per page) and keep rows whose type label starts with "Basic".
Phase 2 (download): GET /modules/archives/download.php?id=N without following redirects.
Many TI-Planet entries are pointers to ticalc.org files; those are recorded and skipped when the
ticalc path is already covered by the ticalc sources, otherwise fetched from ticalc.org.

All requests go through corpus.polite (robots.txt, 1 req/s). Resumable: listing.jsonl and
downloads.jsonl are append-only logs; finished work is skipped on rerun.

Usage:  uv run python -I corpus/fetch_tiplanet.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import polite  # noqa: E402
from corpus.safepath import safe_relpath  # noqa: E402

OUT = ROOT / "data" / "raw" / "tiplanet"
BASE = "https://tiplanet.org"
TOP_CATS = ("ce", "z80", "84c", "82a")
TICALC_WANTED = ROOT / "data" / "raw" / "ticalc" / "wanted.json"

ROW_RE = re.compile(
    r'topictitle"><a href="/forum/archives_voir\.php\?id=(\d+)"[^>]*><b>(.*?)</b></a>'
    r'.*?<span class="postdetails">(.*?)</span>.*?<span class="postdetails">(.*?)</span>',
    re.S)
SUBCAT_RE = re.compile(r'archives_list\.php\?cat=(\w+)&amp;[^"\']*?&amp;cat=([^"\'&#]+)')


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "crawl.log", "a") as f:
        f.write(line + "\n")


def text(fragment: str) -> str:
    return html.unescape(re.sub(r"<br\s*/?>", " | ", re.sub(r"<(?!br)[^>]+>", "", fragment))).strip()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def subcategories(top: str) -> list[str]:
    body, _ = polite.fetch(f"{BASE}/forum/archives_cat.php?cat={top}")
    subs = []
    for t, sub in SUBCAT_RE.findall(body.decode("utf-8", "replace")):
        sub = urllib.parse.unquote_plus(html.unescape(sub))
        if t == top and sub not in subs:
            subs.append(sub)
    return subs


def list_subcat(sub: str, listing: Path) -> int:
    seen: set[str] = set()
    start = 0
    kept = 0
    while True:
        q = urllib.parse.urlencode({"start": start, "cat": sub})
        body, _ = polite.fetch(f"{BASE}/forum/archives_list.php?{q}")
        rows = ROW_RE.findall(body.decode("utf-8", "replace"))
        new = [r for r in rows if r[0] not in seen]
        if not new:
            break
        for aid, title, cats, kind in new:
            seen.add(aid)
            kind_t = text(kind)
            if kind_t.lower().startswith("basic"):
                append_jsonl(listing, {"id": int(aid), "title": text(title), "cats": text(cats),
                                       "kind": kind_t, "subcat": sub})
                kept += 1
        start += 25
    return kept


def phase_listing() -> list[dict]:
    listing = OUT / "listing.jsonl"
    done_path = OUT / "listing_done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    for top in TOP_CATS:
        try:
            subs = subcategories(top)
        except polite.FetchError as e:
            log(f"FAIL subcategories {top}: {e}")
            continue
        for sub in subs:
            if sub in done:
                continue
            n = list_subcat(sub, listing)
            done.add(sub)
            done_path.write_text(json.dumps(sorted(done)))
            log(f"listed {sub!r}: {n} Basic rows")
    uniq: dict[int, dict] = {}
    for r in read_jsonl(listing):
        uniq.setdefault(r["id"], r)
    return list(uniq.values())


def ticalc_covered() -> set[str]:
    if not TICALC_WANTED.exists():
        return set()
    return {p.lower() for p in json.loads(TICALC_WANTED.read_text())["listed"]}


def filename_from(info: dict, fallback: str) -> str:
    cd = info.get("headers", {}).get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
    name = urllib.parse.unquote(m.group(1)) if m else fallback
    return safe_relpath(name.rsplit("/", 1)[-1]) or "download.bin"


def phase_download(rows: list[dict]) -> None:
    dl_log = OUT / "downloads.jsonl"
    done = {r["id"] for r in read_jsonl(dl_log)}
    covered = ticalc_covered()
    todo = [r for r in rows if r["id"] not in done]
    log(f"{len(rows)} Basic archives listed, {len(todo)} left to resolve")
    for i, r in enumerate(todo):
        rec = {"id": r["id"]}
        try:
            body, info = polite.fetch(f"{BASE}/modules/archives/download.php?id={r['id']}",
                                      follow_redirects=False, max_bytes=32 * 1024 * 1024)
            url = info.get("location") or info["final_url"]
            if info["status"] in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(BASE + "/", url)
                parts = urllib.parse.urlsplit(url)
                # Location headers sometimes carry raw spaces/non-ASCII: percent-encode the path
                url = urllib.parse.urlunsplit(parts._replace(
                    path=urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=-._~")))
                rec["redirect"] = url
                parts = urllib.parse.urlsplit(url)
                if parts.scheme not in ("http", "https"):
                    raise polite.FetchError(f"refusing non-http redirect {url[:100]}")
                if parts.netloc.endswith("ticalc.org") and parts.path.lower() in covered:
                    rec["status"] = "ticalc-covered"
                    append_jsonl(dl_log, rec)
                    continue
                body, info = polite.fetch(url, max_bytes=32 * 1024 * 1024)
                if parts.netloc.endswith("ticalc.org"):
                    rec["status"] = "ticalc-extra"
            name = filename_from(info, urllib.parse.urlsplit(url).path or "download.bin")
            dest = OUT / "files" / str(r["id"]) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            rec.update(status=rec.get("status", "downloaded"), file=str(dest.relative_to(OUT)),
                       bytes=len(body))
        except polite.FetchError as e:
            rec.update(status="fail", error=str(e)[:200])
        append_jsonl(dl_log, rec)
        if (i + 1) % 50 == 0:
            log(f"download {i + 1}/{len(todo)}")
    log("DONE downloads")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = phase_listing()
    phase_download(rows)


if __name__ == "__main__":
    main()
