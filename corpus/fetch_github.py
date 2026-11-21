"""Fetch TI-BASIC program files (.8xp/.83p/.8xg/.83g and small .zip) from public GitHub repos.

1. Candidate repos from the search API (topics, linguist language "TI Program", keywords).
   Forks are skipped (their files duplicate the parent). Max 1000 results per query (API cap).
2. One git-tree request per repo (recursive) to find matching files.
3. Repos with >= 8 matching files and <= 30 MB: one tarball request (API /tarball, stored under
   files/_tarballs/); other repos: one git-blob request per file (base64 via the API).

All calls go through the authenticated `gh` CLI at <= 1 request/s. Resumable via append-only
JSONL logs in data/raw/github/.

Usage:  uv run python -I corpus/fetch_github.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.safepath import safe_relpath  # noqa: E402

OUT = ROOT / "data" / "raw" / "github"
QUERIES = [
    'language:"TI Program"',
    "topic:ti-basic", "topic:tibasic", "topic:ti-84", "topic:ti84", "topic:ti-84-plus",
    "topic:ti-84-plus-ce", "topic:ti84plusce", "topic:ti-84-ce", "topic:ti-83", "topic:ti83",
    "topic:ti-83-plus", "topic:ti-calculator", "topic:ti-84plus", "topic:8xp",
    "ti-basic", "tibasic", "ti-84", "ti84", "ti-83", "ti83", "8xp", "ti-84 plus ce",
    "ti 84 programs", "ti-84 games", "ti84 calculator", "TI-83 Plus",
]
EXTS = (".8xp", ".83p", ".8xg", ".83g")
ZIP_MAX = 5 * 1024 * 1024
MAX_FILES_PER_REPO = 800
TARBALL_MIN_FILES = 8  # repos with this many matching files are fetched as one tarball
TARBALL_MAX_KB = 30_000  # ...if the repo (GitHub "size", KB) is at most this big
TARBALL_MAX_BYTES = 64 * 1024 * 1024
BUDGET_BYTES = 1_000_000_000
MIN_INTERVAL_S = 1.05
_last = 0.0


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "crawl.log", "a") as f:
        f.write(line + "\n")


def gh_bytes(endpoint: str) -> bytes | None:
    """Raw (binary) API response, e.g. a tarball; None on failure."""
    global _last
    wait = _last + MIN_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()
    p = subprocess.run(["gh", "api", endpoint], capture_output=True, timeout=900)
    if p.returncode != 0 or len(p.stdout) > TARBALL_MAX_BYTES:
        log(f"FAIL {endpoint}: rc={p.returncode} bytes={len(p.stdout)}")
        return None
    return p.stdout


def gh(endpoint: str, *fields: str, retries: int = 4) -> dict | None:
    global _last
    delay = 10.0
    for _ in range(retries):
        wait = _last + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        args = ["gh", "api", "-X", "GET", endpoint]
        for f in fields:
            args += ["-f", f]
        p = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if p.returncode == 0:
            return json.loads(p.stdout)
        err = (p.stderr + p.stdout)[:300]
        if "Not Found" in err or "HTTP 404" in err or "HTTP 409" in err or "HTTP 451" in err:
            return None
        if "rate limit" in err.lower() or "HTTP 403" in err or "HTTP 429" in err:
            time.sleep(60)
        time.sleep(delay)
        delay *= 2
    log(f"FAIL {endpoint}: {err}")
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def phase_search() -> dict[str, dict]:
    repos_path = OUT / "repos.jsonl"
    done_path = OUT / "queries_done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    for q in QUERIES:
        if q in done:
            continue
        for page in range(1, 11):
            res = gh("search/repositories", f"q={q}", "per_page=100", f"page={page}")
            items = (res or {}).get("items", [])
            for it in items:
                lic = (it.get("license") or {}).get("spdx_id")
                append_jsonl(repos_path, {
                    "full_name": it["full_name"], "fork": it["fork"], "license": lic,
                    "branch": it.get("default_branch"), "size_kb": it.get("size"),
                    "stars": it.get("stargazers_count"), "query": q})
            if len(items) < 100:
                break
            time.sleep(2.0)  # search API: 30 req/min
        done.add(q)
        done_path.write_text(json.dumps(sorted(done)))
        log(f"search {q!r} done")
    repos: dict[str, dict] = {}
    for r in read_jsonl(repos_path):
        if not r["fork"]:
            repos.setdefault(r["full_name"], r)
    return repos


def phase_trees(repos: dict[str, dict]) -> list[dict]:
    trees_path = OUT / "trees.jsonl"
    done = {r["repo"] for r in read_jsonl(trees_path)}
    todo = [r for name, r in repos.items() if name not in done]
    log(f"{len(repos)} candidate repos, {len(todo)} trees to fetch")
    for i, r in enumerate(todo):
        res = gh(f"repos/{r['full_name']}/git/trees/{r['branch']}", "recursive=1")
        files = []
        for e in (res or {}).get("tree", []):
            p = e.get("path", "")
            low = p.lower()
            if e.get("type") != "blob":
                continue
            if low.endswith(EXTS) or (low.endswith(".zip") and e.get("size", 0) <= ZIP_MAX):
                files.append({"path": p, "sha": e["sha"], "size": e.get("size", 0)})
        append_jsonl(trees_path, {"repo": r["full_name"], "license": r.get("license"),
                                  "truncated": bool((res or {}).get("truncated")),
                                  "files": files[:MAX_FILES_PER_REPO], "n_match": len(files)})
        if (i + 1) % 100 == 0:
            log(f"trees {i + 1}/{len(todo)}")
    return read_jsonl(trees_path)


def phase_blobs(trees: list[dict], repos: dict[str, dict]) -> None:
    blobs_path = OUT / "blobs.jsonl"
    prev = read_jsonl(blobs_path)
    done = {(r["repo"], r["sha"]) for r in prev}
    seen_sha = {r["sha"] for r in prev if r.get("status") in ("ok", "tarball")}
    total = sum(r.get("bytes", 0) for r in prev)
    todo_repos = [t for t in trees if any((t["repo"], f["sha"]) not in done for f in t["files"])]
    log(f"{sum(len(t['files']) for t in todo_repos)} blobs in {len(todo_repos)} repos to fetch")
    for i, t in enumerate(todo_repos):
        if total > BUDGET_BYTES:
            log("byte budget reached; stopping")
            break
        repo, meta = t["repo"], repos.get(t["repo"], {})
        files = [f for f in t["files"] if (repo, f["sha"]) not in done]
        if len(t["files"]) >= TARBALL_MIN_FILES and (meta.get("size_kb") or 0) <= TARBALL_MAX_KB:
            data = gh_bytes(f"repos/{repo}/tarball/{meta.get('branch') or 'HEAD'}")
            if data is not None and data[:2] == b"\x1f\x8b":
                dest = OUT / "files" / "_tarballs" / (safe_relpath(repo.replace("/", "__"))
                                                       + ".tar.gz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                total += len(data)
                for f in files:
                    append_jsonl(blobs_path, {"repo": repo, "path": f["path"], "sha": f["sha"],
                                              "status": "tarball",
                                              "file": str(dest.relative_to(OUT))})
                    seen_sha.add(f["sha"])
                append_jsonl(OUT / "tarballs.jsonl", {"repo": repo, "bytes": len(data),
                                                      "n_files": len(files)})
                continue
        for f in files:
            rec = {"repo": repo, "path": f["path"], "sha": f["sha"]}
            if f["sha"] in seen_sha:  # identical content already fetched from another repo
                rec["status"] = "dup-sha"
                append_jsonl(blobs_path, rec)
                continue
            res = gh(f"repos/{repo}/git/blobs/{f['sha']}")
            if not res or res.get("encoding") != "base64":
                rec["status"] = "fail"
                append_jsonl(blobs_path, rec)
                continue
            data = base64.b64decode(res["content"])
            rp = safe_relpath(repo.replace("/", "__") + "/" + f["path"])
            if rp is None:
                rec["status"] = "badpath"
                append_jsonl(blobs_path, rec)
                continue
            dest = OUT / "files" / rp
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            seen_sha.add(f["sha"])
            total += len(data)
            rec.update(status="ok", file=str(dest.relative_to(OUT)), bytes=len(data))
            append_jsonl(blobs_path, rec)
        if (i + 1) % 25 == 0:
            log(f"blob repos {i + 1}/{len(todo_repos)} ({total / 1e6:.1f} MB)")
    log("DONE blobs")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repos = phase_search()
    trees = phase_trees(repos)
    phase_blobs(trees, repos)


if __name__ == "__main__":
    main()
