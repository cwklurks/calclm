"""Dedup, split, build the vocabulary and write the token streams.

Input:  data/extract/programs.jsonl (from corpus/extract.py)
Output: corpus/vocab.json
        data/tok/train.bin, data/tok/val.bin   little-endian uint16 ids, each program <bos> .. <eos>
        data/tok/eval_author.bin               the author's own programs (eval only, never trained on)
        data/tok/programs.jsonl                per-program index (split, offset, length, tags, ...)
        data/tok/meta.json                     vocab size, counts, split seed
        data/stats.json                        numbers used by corpus/report.py

Split (by program, never by token): exact duplicates are removed first; programs are then
grouped by normalized body, MinHash similarity and shared top-level upload, a stable hash picks
~3% of groups (<= 60 programs) for val, and a containment pass moves any val group whose
program sits mostly inside a single train program back to train. Group choice depends only on
group content, so val membership stays stable as the corpus grows.

Usage:  uv run python -I corpus/build.py [--freeze-vocab]
        --freeze-vocab keeps the ids of the existing corpus/vocab.json and only appends tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.tokenize import BOS, EOS, SPECIALS, UNK, Vocab, is_known, split_tokens  # noqa: E402

EXTRACT = ROOT / "data" / "extract" / "programs.jsonl"
TOK = ROOT / "data" / "tok"
VOCAB_OUT = ROOT / "corpus" / "vocab.json"
STATS_OUT = ROOT / "data" / "stats.json"
SOURCE_ORDER = ["ticalc_warc2014", "ticalc", "tiplanet", "github"]
EVAL_SOURCE = "author_eval"
SEED = 1234
MIN_COUNT = 3
MAX_VOCAB = 1024
TARGET_COVERAGE = 0.9995
SHINGLE = 5
NUM_PERM = 64
BANDS = 16
NEAR_DUP_J = 0.7  # reported near-duplicate clusters
SPLIT_J = 0.5  # looser similarity used for grouping the split
VAL_PERMILLE = 30  # a split group goes to val iff hash(seed, min sha1) % 1000 < this
MAX_VAL_GROUP = 60  # bigger split groups always go to train
CONTAIN_GRAM = 16
CONTAIN_MAX = 0.3


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file + rename so concurrent readers never see a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def normalized_key(body: bytes) -> str:
    """Hash of the body modulo cosmetic edits (spaces, ':' separators, closing ')'/'"' at EOL)."""
    toks = [t for t in split_tokens(body) if t not in (b"\x29", b"\x3E")]
    out: list[bytes] = []
    for t in toks + [b"\x3F"]:
        if t == b"\x3F":
            while out and out[-1] in (b"\x11", b"\x2A"):
                out.pop()
            if out and out[-1] == b"\x3F":
                continue
        out.append(t)
    return hashlib.sha1(b"".join(out)).hexdigest()


def load_programs() -> tuple[list[dict], list[dict], Counter]:
    corpus, evals = [], []
    status = Counter()
    for line in EXTRACT.open():
        p = json.loads(line)
        status[(p["source"], p["status"])] += 1
        if p["status"] != "kept":
            continue
        (evals if p["source"] == EVAL_SOURCE else corpus).append(p)
    return corpus, evals, status


def archive_key(p: dict) -> str:
    """Top-level upload a program came from: a ticalc file, a TI-Planet entry, a GitHub repo."""
    f = p["file"]
    if p["source"] == "github":
        if f.startswith("_tarballs/"):
            return "gh:" + f.split("/", 1)[1].removesuffix(".tar.gz")
        return "gh:" + f.split("/", 1)[0]
    if p["source"] == "tiplanet":
        return "tp:" + f.split("/", 1)[0]
    return "tc:" + f


def dedup(corpus: list[dict]) -> list[dict]:
    rank = {s: i for i, s in enumerate(SOURCE_ORDER)}
    by_sha: dict[str, dict] = {}
    for p in sorted(corpus, key=lambda p: (rank.get(p["source"], 99), p["file"], p["member"])):
        u = by_sha.get(p["sha1"])
        if u is None:
            by_sha[p["sha1"]] = u = {**p, "sources": set(), "names": set(), "archives": set(),
                                     "copies": 0}
        u["sources"].add(p["source"])
        u["archives"].add(archive_key(p))
        u["names"].add(p["name"])
        u["copies"] += 1
    return list(by_sha.values())


def near_dup_stats(uniq: list[dict]) -> dict:
    by_name_len = defaultdict(set)
    by_norm = defaultdict(set)
    for u in uniq:
        by_name_len[(u["name"], u["n_bytes"])].add(u["sha1"])
        by_norm[u["norm"]].add(u["sha1"])
    nl = [len(v) for v in by_name_len.values() if len(v) > 1]
    nn = [len(v) for v in by_norm.values() if len(v) > 1]
    return {"same_name_same_len_groups": len(nl), "same_name_same_len_programs": sum(nl),
            "normalized_groups": len(nn), "normalized_programs": sum(nn),
            "normalized_redundant": sum(nn) - len(nn)}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def minhash_signatures(uniq: list[dict]) -> np.ndarray:
    """64 MinHash values per program over 5-token shingles (token codes, uint64 arithmetic)."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 2**63, size=NUM_PERM, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=NUM_PERM, dtype=np.uint64)
    sigs = np.zeros((len(uniq), NUM_PERM), dtype=np.uint64)
    mult = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for i, u in enumerate(uniq):
            codes = np.array([int.from_bytes(t, "big") + 65536 * (len(t) - 1)
                              for t in split_tokens(u["bytes"])], dtype=np.uint64)
            k = min(SHINGLE, len(codes))
            h = np.zeros(len(codes) - k + 1, dtype=np.uint64)
            for j in range(k):
                h = h * mult + codes[j:len(codes) - k + 1 + j]
            sigs[i] = (a[:, None] * np.unique(h)[None, :] + b[:, None]).min(axis=1)
    return sigs


def similar_pairs(uniq: list[dict], min_j: float) -> list[tuple[int, int, float]]:
    """Candidate pairs from MinHash LSH with estimated Jaccard >= min_j."""
    sigs = minhash_signatures(uniq)
    rows = NUM_PERM // BANDS
    seen: set[tuple[int, int]] = set()
    out = []
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = np.ascontiguousarray(sigs[:, band * rows:(band + 1) * rows])
        for i in range(len(uniq)):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            pairs = ((x, y) for xi, x in enumerate(members) for y in members[xi + 1:]) \
                if len(members) <= 50 else ((members[0], y) for y in members[1:])
            for x, y in pairs:
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                j = float((sigs[x] == sigs[y]).mean())
                if j >= min_j:
                    out.append((x, y, j))
    return out


def union_by(uf: UnionFind, keys: list) -> None:
    first: dict = {}
    for i, ks in enumerate(keys):
        for k in ks:
            uf.union(i, first.setdefault(k, i))


def grams(u: dict, w: int = CONTAIN_GRAM) -> np.ndarray:
    codes = np.array([int.from_bytes(t, "big") + 65536 * (len(t) - 1)
                      for t in split_tokens(u["bytes"])], dtype=np.uint64)
    if len(codes) < w:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(codes) - w + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(w):
            h = h * np.uint64(1_000_003) + codes[j:len(codes) - w + 1 + j]
    return np.unique(h)


def contained(queries: list[dict], refs: list[dict]) -> list[dict]:
    """Queries with >= CONTAIN_MAX of their 16-token windows inside a single ref program."""
    if not refs or not queries:
        return []
    rg = [grams(r) for r in refs]
    allg = np.concatenate(rg)
    owner = np.concatenate([np.full(g.size, i, dtype=np.int32) for i, g in enumerate(rg)])
    order = np.argsort(allg, kind="stable")
    allg, owner = allg[order], owner[order]
    hits = []
    for q in queries:
        g = grams(q)
        if g.size == 0:
            continue
        lo = np.searchsorted(allg, g, "left")
        hi = np.searchsorted(allg, g, "right")
        cnt: Counter = Counter()
        for a, b in zip(lo.tolist(), hi.tolist()):
            if 0 < b - a <= 200:
                cnt.update(set(owner[a:b].tolist()))
        if cnt and cnt.most_common(1)[0][1] / g.size >= CONTAIN_MAX:
            hits.append(q)
    return hits


def split_programs(uniq: list[dict]) -> dict:
    """Assign train/val by split group; returns near-duplicate and split statistics."""
    pairs = similar_pairs(uniq, SPLIT_J)
    # content clusters (reported as near-duplicates): normalized hash or Jaccard >= NEAR_DUP_J
    content = UnionFind(len(uniq))
    union_by(content, [[u["norm"]] for u in uniq])
    for x, y, j in pairs:
        if j >= NEAR_DUP_J:
            content.union(x, y)
    # split groups: content clusters + Jaccard >= SPLIT_J + same top-level upload
    grp = UnionFind(len(uniq))
    union_by(grp, [[u["norm"]] + sorted(u["archives"]) for u in uniq])
    for x, y, _ in pairs:
        grp.union(x, y)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, u in enumerate(uniq):
        u["group"] = content.find(i)
        u["split_group"] = grp.find(i)
        groups[u["split_group"]].append(u)
    for members in groups.values():
        key = min(u["sha1"] for u in members)
        bucket = int(hashlib.sha1(f"{SEED}:{key}".encode()).hexdigest()[:8], 16) % 1000
        is_val = bucket < VAL_PERMILLE and len(members) <= MAX_VAL_GROUP
        for u in members:
            u["split"] = "val" if is_val else "train"
    # containment pass: a val program mostly inside one train program moves (with its group)
    moved_groups = 0
    refs = [u for u in uniq if u["split"] == "train"]
    while True:
        hits = contained([u for u in uniq if u["split"] == "val"], refs)
        if not hits:
            break
        refs = []
        for g in {u["split_group"] for u in hits}:
            moved_groups += 1
            for u in groups[g]:
                u["split"] = "train"
                refs.append(u)
    by_content: dict[int, list[dict]] = defaultdict(list)
    for u in uniq:
        by_content[u["group"]].append(u)
    multi = [g for g in by_content.values() if len(g) > 1]
    sizes = sorted((len(g) for g in multi), reverse=True)
    gsizes = sorted((len(g) for g in groups.values()), reverse=True)
    return {"clusters": len(multi), "programs_in_clusters": sum(sizes),
            "redundant_programs": sum(sizes) - len(sizes),
            "redundant_tokens": sum(sum(u["n_tokens"] for u in g) - max(u["n_tokens"] for u in g)
                                    for g in multi),
            "largest_clusters": sizes[:10], "jaccard_threshold": NEAR_DUP_J,
            "shingle_tokens": SHINGLE, "num_perm": NUM_PERM, "bands": BANDS,
            "split_groups": len(gsizes), "largest_split_groups": gsizes[:5],
            "split_jaccard": SPLIT_J, "val_permille": VAL_PERMILLE,
            "max_val_group": MAX_VAL_GROUP, "containment_gram": CONTAIN_GRAM,
            "containment_max": CONTAIN_MAX, "containment_moved_groups": moved_groups}


MODEL_ORDER = ["TI-82", "TI-83", "TI-83+", "TI-84+", "TI-84+CSE", "TI-84+CE"]


def _token_model_rank() -> dict[bytes, int]:
    """Earliest model per token byte value from the tivars sheet XML.

    tivars' TIToken.since is the since of the token's *latest* name, so renamed tokens (e.g. 7E08
    AxesOn, renamed for the 84+CSE) would look CSE-only; the XML keeps every version.
    """
    import xml.etree.ElementTree as ET

    import tivars

    root = ET.parse(Path(tivars.__file__).parent / "tokens" / "8X.xml").getroot()
    ranks: dict[bytes, int] = {}

    def earliest(tok: ET.Element) -> int:
        models = [v.findtext("since/model") for v in tok.findall("version")]
        return min(MODEL_ORDER.index(m) for m in models if m in MODEL_ORDER)

    for el in root:
        hi = int(el.get("value")[1:], 16)
        if el.tag == "token":
            ranks[bytes([hi])] = earliest(el)
        elif el.tag == "two-byte":
            for sub in el.findall("token"):
                ranks[bytes([hi, int(sub.get("value")[1:], 16)])] = earliest(sub)
    return ranks


_MODEL_RANK = _token_model_rank()


def min_model(body: bytes) -> str:
    """Oldest model family whose token set covers every (known) token in the body."""
    return MODEL_ORDER[max((_MODEL_RANK.get(t, 0) for t in split_tokens(body)), default=0)]


def string_tokens(body: bytes) -> int:
    """Tokens inside string literals (after an opening '"' until '"', newline or store)."""
    n = 0
    inside = False
    for t in split_tokens(body):
        if t == b"\x2A":
            inside = not inside
        elif t in (b"\x3F", b"\x04"):
            inside = False
        elif inside:
            n += 1
    return n


def build_vocab(train: list[dict], frozen: list[str] | None = None) -> tuple[list[str], dict]:
    """Frequency-ordered vocab from train counts.

    With `frozen` (an existing vocab.json), its ids are kept and qualifying new tokens are only
    appended, so models trained on the old vocab stay valid.
    """
    counts = Counter()
    for u in train:
        counts.update(split_tokens(u["bytes"]))
    total = sum(counts.values())
    known = [(t, c) for t, c in counts.items() if is_known(t)]
    known.sort(key=lambda tc: (-tc[1], tc[0].hex()))
    base = [bytes.fromhex(e) for e in frozen[len(SPECIALS):]] if frozen else []
    min_count = MIN_COUNT
    while True:
        extra = [t for t, c in known if c >= min_count and t not in set(base)]
        chosen = (base + extra)[:MAX_VOCAB - len(SPECIALS)]
        cov = sum(counts[t] for t in chosen) / total
        if cov >= TARGET_COVERAGE or min_count == 1:
            break
        min_count -= 1
    entries = list(SPECIALS) + [t.hex().upper() for t in chosen]
    info = {"train_token_total": total, "distinct_tokens_seen": len(counts),
            "distinct_known_seen": len(known),
            "distinct_unknown_seen": len(counts) - len(known),
            "unknown_token_occurrences": sum(c for t, c in counts.items() if not is_known(t)),
            "min_count": min_count, "coverage_train": cov,
            "dropped_known_rare": [(t.hex().upper(), c) for t, c in known if t not in set(chosen)],
            "token_counts": {t.hex().upper(): c for t, c in counts.items()}}
    return entries, info


def encode_stream(progs: list[dict], vocab: Vocab, split_name: str,
                  index: list[dict]) -> tuple[bytes, dict]:
    ids: list[int] = []
    n_unk = 0
    for u in progs:
        enc = vocab.encode(u["bytes"], add_special=True)
        n_unk += enc.count(UNK)
        index.append({"split": split_name, "offset": len(ids), "length": len(enc),
                      "sha1": u["sha1"], "source": u["source"],
                      "sources": sorted(u.get("sources", {u["source"]})), "name": u["name"],
                      "file": u["file"], "member": u["member"], "tags": u.get("tags", []),
                      "n_tokens": len(enc) - 2, "n_string_tokens": u.get("n_string_tokens"),
                      "near_dup_group": u.get("group"), "split_group": u.get("split_group"),
                      "min_model": u.get("min_model")})
        ids.extend(enc)
    arr = np.asarray(ids, dtype="<u2")
    return arr.tobytes(), {"programs": len(progs), "ids": int(arr.size),
                           "program_tokens": int(arr.size) - 2 * len(progs), "unk": n_unk}


def main() -> None:
    TOK.mkdir(parents=True, exist_ok=True)
    corpus, evals, status = load_programs()
    uniq = dedup(corpus)
    for u in uniq:
        u["bytes"] = bytes.fromhex(u.pop("body"))
        u["norm"] = normalized_key(u["bytes"])
    eval_shas = {p["sha1"] for p in evals}
    overlap = [u for u in uniq if u["sha1"] in eval_shas]
    uniq = [u for u in uniq if u["sha1"] not in eval_shas]
    uniq.sort(key=lambda u: u["sha1"])
    for u in uniq:
        u["n_string_tokens"] = string_tokens(u["bytes"])
        u["min_model"] = min_model(u["bytes"])
    cluster_stats = split_programs(uniq)
    train = [u for u in uniq if u["split"] == "train"]
    val = [u for u in uniq if u["split"] == "val"]
    rng = random.Random(SEED)
    train.sort(key=lambda u: u["sha1"])
    val.sort(key=lambda u: u["sha1"])
    rng.shuffle(train)
    rng.shuffle(val)

    frozen = None
    if "--freeze-vocab" in sys.argv and VOCAB_OUT.exists():
        frozen = json.loads(VOCAB_OUT.read_text())
    entries, vinfo = build_vocab(train, frozen)
    assert len(entries) <= MAX_VOCAB
    vocab = Vocab(entries)

    index: list[dict] = []
    b_train, s_train = encode_stream(train, vocab, "train", index)
    b_val, s_val = encode_stream(val, vocab, "val", index)
    eval_uniq = list({p["sha1"]: {**p, "bytes": bytes.fromhex(p["body"])} for p in evals}.values())
    b_eval, s_eval = encode_stream(eval_uniq, vocab, "eval_author", index)

    # leakage checks before anything is written
    for key in ("sha1", "norm", "group", "split_group"):
        val_keys = {u[key] for u in val}
        assert not any(u[key] in val_keys for u in train), f"{key} straddles train/val"

    atomic_write(VOCAB_OUT, (json.dumps(entries, indent=0) + "\n").encode())
    atomic_write(TOK / "train.bin", b_train)
    atomic_write(TOK / "val.bin", b_val)
    atomic_write(TOK / "eval_author.bin", b_eval)
    atomic_write(TOK / "programs.jsonl", "".join(json.dumps(r) + "\n" for r in index).encode())

    meta = {
        "format": "little-endian uint16 token ids; each program is <bos> tokens <eos>; "
                  "programs.jsonl gives per-program offsets (in ids) and lengths",
        "vocab_file": "corpus/vocab.json", "vocab_size": len(entries),
        "vocab_frozen_from_previous": frozen is not None,
        "specials": {"<pad>": 0, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK},
        "split_seed": SEED, "val_fraction_actual": len(val) / max(len(uniq), 1),
        "split_unit": "split group: exact-deduped programs unioned by normalized body, MinHash "
                      f"Jaccard >= {SPLIT_J} (5-token shingles) and same top-level upload "
                      "(ticalc file / TI-Planet entry / GitHub repo); a group is val iff "
                      f"sha1('{SEED}:'+min member sha1) % 1000 < {VAL_PERMILLE} and it has <= "
                      f"{MAX_VAL_GROUP} programs; then val groups with a program whose "
                      f"{CONTAIN_GRAM}-token windows are >= {CONTAIN_MAX:.0%} inside one train "
                      "program move to train",
        "vocab_min_count": vinfo["min_count"], "vocab_coverage_train": vinfo["coverage_train"],
        "train": s_train, "val": s_val, "eval_author": s_eval,
    }
    atomic_write(TOK / "meta.json", (json.dumps(meta, indent=2) + "\n").encode())

    copies_total = sum(u["copies"] for u in uniq) + sum(u["copies"] for u in overlap)
    stats = {
        "status_counts": {f"{s}|{st}": n for (s, st), n in status.items()},
        "kept_program_vars": len(corpus), "unique_programs": len(uniq) + len(overlap),
        "exact_duplicates_removed": copies_total - len(uniq) - len(overlap),
        "eval_overlap_removed": len(overlap),
        "near_dups": near_dup_stats(uniq),
        "near_dup_clusters": cluster_stats,
        "string_tokens": sum(u["n_string_tokens"] for u in uniq),
        "programs_mostly_strings": sum(1 for u in uniq
                                       if u["n_string_tokens"] > 0.8 * u["n_tokens"]),
        "tokens_in_mostly_string_programs": sum(u["n_tokens"] for u in uniq
                                                if u["n_string_tokens"] > 0.8 * u["n_tokens"]),
        "unique_by_first_source": Counter(u["source"] for u in uniq),
        "unique_only_in_source": Counter(next(iter(u["sources"])) for u in uniq
                                         if len(u["sources"]) == 1),
        "unique_tokens_by_first_source": Counter(),
        "lengths_train_val": sorted(u["n_tokens"] for u in uniq),
        "tags_unique": Counter(t for u in uniq for t in u["tags"]),
        "min_model_programs": Counter(u["min_model"] for u in uniq),
        "min_model_tokens": Counter(),
        "tag_tokens_unique": Counter(),
        "vocab": {k: v for k, v in vinfo.items() if k != "token_counts"},
        "token_counts_train": vinfo["token_counts"],
        "meta": meta,
    }
    for u in uniq:
        stats["min_model_tokens"][u["min_model"]] += u["n_tokens"]
        stats["unique_tokens_by_first_source"][u["source"]] += u["n_tokens"]
        for t in u["tags"]:
            stats["tag_tokens_unique"][t] += u["n_tokens"]
    STATS_OUT.write_text(json.dumps(stats, indent=1, default=list) + "\n")
    print(json.dumps({"unique": len(uniq), "train": s_train, "val": s_val, "eval": s_eval,
                      "vocab": len(entries), "coverage": vinfo["coverage_train"]}, indent=1))


if __name__ == "__main__":
    main()
