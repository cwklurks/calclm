"""E22's cleaner (PLAN-E22.md Sections 3.1, 3.3 and 3.5): the scrubber S (rules R1-R5 on string
literals only), the lexicon builder, the dedup steps X0-X3 and the direct-link leakage check.

Stateless: every function takes its inputs and returns new values; nothing here reads or writes a
file (lab/e22_data.py does). Programs are token ids (corpus/vocab.json); the string text a rule
sees is the uppercase display of each token (lab.phase1's full display, lowercase letters folded),
and a rule's deletion removes every token one of whose characters it matched.

Privacy (decision D6): nothing here prints, logs or returns string text to a caller other than as
values the build keeps under runs/e22; counts only leave this module in build.json.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np

from corpus.build import (BANDS, CONTAIN_GRAM, CONTAIN_MAX, NEAR_DUP_J, NUM_PERM, SEED, SHINGLE,
                          SPLIT_J, UnionFind)
from eval.overlap import ngram_hashes

QUOTE, STORE, NEWLINE, SPACE, COLON, PRGM = '2A', '04', '3F', '29', '3E', '5F'
LIST_L = 'EB'                                  # the custom-list marker ʟ
RULES = ('R1', 'R2', 'R3', 'R4', 'R5')
# R1's BY exemptions: the word before BY in math and procedure text (PLAN-E22 3.1; the list is
# fixed at the resolution commit from the dev labels).
BY_EXEMPT = frozenset({
    'DIVIDED', 'DIVIDE', 'DIVIDING', 'DIVISIBLE', 'MULTIPLIED', 'MULTIPLY', 'MULTIPLYING',
    'TIMES', 'INCREASED', 'INCREASE', 'DECREASED', 'DECREASE', 'SORTED', 'SORT', 'STEP',
    'GROUPED', 'REDUCED', 'RAISED', 'LOWERED', 'ROUNDED', 'ROTATED', 'SCALED', 'SHIFTED',
    'MOVED', 'MOVE', 'REPLACED', 'FOLLOWED', 'SEPARATED', 'SPLIT', 'DEFINED', 'GIVEN', 'GO',
    'GOES', 'ADDED', 'SUBTRACTED', 'OFFSET', 'BOUNDED', 'GENERATED', 'DETERMINED', 'MEASURED',
    'SOLVE', 'SOLVED', 'FACTOR', 'FACTORED', 'ORDER', 'ORDERED', 'MULTIPLES', 'COUNT',
    'COUNTING', 'SKIP', 'ONE', 'SIDE', 'PAGE', 'LINE', 'BIT', 'LITTLE', 'CASE', 'ROW', 'COLUMN',
    'DROP', 'DROPS', 'PASS', 'PASSED', 'STAND', 'STOOD', 'CLOSE', 'NEAR', 'HIT', 'KILLED',
    'BEATEN', 'CAUGHT', 'EATEN', 'PRESS', 'PRESSING', 'PUSHED', 'ENTERED', 'REPRESENTED',
    'DENOTED', 'CALLED', 'PARAMETERIZED', 'SURROUNDED', 'COVERED', 'FILLED', 'MULTIPLIER'})
SUFFIXES = ('COM', 'ORG', 'NET', 'FR', 'DE', 'INFO', 'EDU')
MIN_NAME = 3

L = r'[^\W\d_]'                                # one letter (any script)
NOT_L = rf'(?<!{L})'
WORD = re.compile(rf'{L}+')
_MARK = re.compile(
    rf'{NOT_L}(?:AUTHORS?|CREDITS?|THANKS\s+TO|THANX|SPECIAL\s+THANKS|GREETZ|COPYRIGHT)(?!{L})'
    rf'|{NOT_L}COPR\.|(?<![^\W_])\(C\)|©')
_BY = re.compile(rf'{NOT_L}BY[ :\-]+(?={L})')
_PREV_WORD = re.compile(rf'({L}+)[^\w]*$')
_EMAIL = re.compile(r'[A-Z0-9._%+\-]+@[A-Z0-9\-]+(?:\.[A-Z0-9\-]+)+')
_URL = re.compile(r'(?:HTTPS?:|WWW\.)\S*')
_DOMAIN = re.compile(r'(?<![A-Z0-9@.\-])[A-Z0-9\-]*[A-Z][A-Z0-9\-]*(?:\.[A-Z0-9\-]+)*\.(?:'
                     + '|'.join(SUFFIXES) + r')(?![A-Z0-9])(?:/\S*)?')
_HANDLE = re.compile(r'(?<![A-Z0-9])@[A-Z0-9_]+')
_SEP = re.compile(r'[\s,.:&\-]*')


# ---------------------------------------------------------------- display and lexing

@lru_cache(maxsize=4)
def _display(vocab: tuple[str, ...]) -> tuple[str, ...]:
    from lab.phase1 import _full_display
    return tuple(d.upper() for d in _full_display(vocab))


def display_table(vocab: Sequence[str]) -> tuple[str, ...]:
    """Per id: the uppercase full display ('' for <pad>, <bos>, <eos>, <unk>)."""
    return _display(tuple(vocab))


def _ids_of(vocab, *hexes) -> tuple[int, ...]:
    return tuple(vocab.index(h) for h in hexes)


def string_ranges(ids: Sequence[int], vocab: Sequence[str]) -> list[tuple[int, int]]:
    """[a, b) positions of each string literal's content: from an opening quote to the closing
    quote, a store, a newline, a special token or the program's end (lab.phase1's lexing)."""
    quote, store, newline = _ids_of(vocab, QUOTE, STORE, NEWLINE)
    out, start = [], None
    for i, t in enumerate(ids):
        t = int(t)
        if start is None:
            if t == quote:
                start = i + 1
            continue
        if t in (quote, store, newline) or t < 4:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(ids)))
    return out


def line_of(ids: Sequence[int], vocab: Sequence[str]) -> list[int]:
    """The line number of every position (newline tokens belong to the line they end)."""
    (newline,) = _ids_of(vocab, NEWLINE)
    out, n = [], 0
    for t in ids:
        out.append(n)
        if int(t) == newline:
            n += 1
    return out


# ---------------------------------------------------------------- the rules

def _r1(text: str, exempt: frozenset) -> int | None:
    """The start of the first credit marker (R1), or None."""
    best = None
    m = _MARK.search(text)
    if m:
        best = m.start()
    for m in _BY.finditer(text):
        if best is not None and m.start() >= best:
            break
        prev = _PREV_WORD.search(text[:m.start()])
        if prev is None or prev.group(1) not in exempt:
            best = m.start()
            break
    return best


def _rule_spans(rule: str, text: str, lexicon: frozenset, exempt: frozenset):
    if rule == 'R1':
        s = _r1(text, exempt)
        return [] if s is None else [(s, len(text))]
    if rule == 'R2':
        return [m.span() for m in _EMAIL.finditer(text)]
    if rule == 'R3':
        return [m.span() for m in _URL.finditer(text)] + [m.span() for m in _DOMAIN.finditer(text)]
    if rule == 'R4':
        return [m.span() for m in _HANDLE.finditer(text)]
    return [m.span() for m in WORD.finditer(text) if m.group() in lexicon]


def text_spans(text: str, lexicon: frozenset = frozenset(),
               exempt: frozenset = BY_EXEMPT) -> list[tuple[str, int, int]]:
    """(rule, start, end) of every deletion, in the original text's coordinates. Rules apply in
    order R1-R5, each to the text the earlier rules left."""
    alive = list(range(len(text)))
    out = []
    for rule in RULES:
        cur = ''.join(text[i] for i in alive)
        dead = set()
        for s, e in _rule_spans(rule, cur, lexicon, exempt):
            if e > s:
                dead.update(range(s, e))
                out.append((rule, alive[s], alive[e - 1] + 1))
        if dead:
            alive = [i for k, i in enumerate(alive) if k not in dead]
    return out


def deleted_chars(spans) -> set[int]:
    return {i for _, s, e in spans for i in range(s, e)}


class Scrubber:
    """S: rules R1-R5 on the string literals of a program; code outside strings untouched."""

    def __init__(self, vocab: Sequence[str], lexicon: Iterable[str] = frozenset(),
                 exempt: Iterable[str] = BY_EXEMPT):
        self.vocab = list(vocab)
        self.disp = display_table(self.vocab)
        self.lexicon = frozenset(lexicon)
        self.exempt = frozenset(exempt)
        self.prgm = self.vocab.index(PRGM) if PRGM in self.vocab else -1
        self.list_l = self.vocab.index(LIST_L) if LIST_L in self.vocab else -1

    def text(self, toks: Sequence[int]) -> tuple[str, list[int]]:
        """A string's uppercase display and, per character, the index of its token."""
        chars, owner = [], []
        for k, t in enumerate(toks):
            d = self.disp[int(t)]
            chars.append(d)
            owner += [k] * len(d)
        return ''.join(chars), owner

    def spans(self, toks: Sequence[int]):
        text, owner = self.text(toks)
        return text, owner, text_spans(text, self.lexicon, self.exempt)

    def scrub_string(self, toks: Sequence[int]) -> tuple[list[int], set[str]]:
        text, owner, spans = self.spans(toks)
        if not spans:
            return list(toks), set()
        drop = {owner[i] for i in deleted_chars(spans)}
        return [int(t) for k, t in enumerate(toks) if k not in drop], {r for r, _, _ in spans}

    def code_names(self, ids: Sequence[int], ranges) -> int:
        """Program names after prgm and list names after ʟ (outside strings) that match the
        lexicon: counted, never renamed."""
        inside = set()
        for a, b in ranges:
            inside.update(range(a, b))
        n = 0
        for i, t in enumerate(ids):
            if int(t) in (self.prgm, self.list_l) and i not in inside:
                word = []
                for u in ids[i + 1:]:
                    d = self.disp[int(u)]
                    if len(d) != 1 or not d.isalnum():
                        break
                    word.append(d)
                if ''.join(word) in self.lexicon:
                    n += 1
        return n

    def scrub(self, ids: Sequence[int]) -> tuple[list[int], dict]:
        ids = [int(t) for t in ids]
        ranges = string_ranges(ids, self.vocab)
        out, last = [], 0
        rule_strings = dict.fromkeys(RULES, 0)
        edited = 0
        for a, b in ranges:
            out += ids[last:a]
            kept, rules = self.scrub_string(ids[a:b])
            out += kept
            last = b
            if rules:
                edited += 1
                for r in rules:
                    rule_strings[r] += 1
        out += ids[last:]
        return out, {'strings': len(ranges), 'edited_strings': edited,
                     'rule_strings': rule_strings, 'tokens_removed': len(ids) - len(out),
                     'code_names': self.code_names(ids, ranges) if self.lexicon else 0}


# ---------------------------------------------------------------- the lexicon

def marker_words(text: str, exempt: frozenset = BY_EXEMPT) -> list[str]:
    """The one to three words right after each R1 marker (separated by spaces, commas, dots,
    hyphens or &)."""
    starts = []
    m = _MARK.search(text)
    while m:
        starts.append(m.end())
        m = _MARK.search(text, m.end())
    for m in _BY.finditer(text):
        prev = _PREV_WORD.search(text[:m.start()])
        if prev is None or prev.group(1) not in exempt:
            starts.append(m.end())
    out = []
    for pos in starts:
        for _ in range(3):
            pos = _SEP.match(text, pos).end()
            w = WORD.match(text, pos)
            if not w:
                break
            out.append(w.group())
            pos = w.end()
    return out


def has_marker(text: str, exempt: frozenset = BY_EXEMPT) -> bool:
    return _r1(text, exempt) is not None


def github_owner(source: str, file: str) -> str | None:
    if source != 'github':
        return None
    head = file.split('/', 1)[1] if file.startswith('_tarballs/') else file.split('/', 1)[0]
    return head.split('__', 1)[0]


def owner_words(owner: str) -> list[str]:
    return WORD.findall(owner.upper())


def frequent_words(programs: Iterable[Iterable[str]], threshold: int,
                   exempt: frozenset = BY_EXEMPT) -> set[str]:
    """Words of 3+ letters found in strings with no credit marker in at least `threshold`
    programs (the stop list's corpus part)."""
    count = Counter()
    for strings in programs:
        words = set()
        for text in strings:
            if not has_marker(text, exempt):
                words.update(w for w in WORD.findall(text) if len(w) >= MIN_NAME)
        count.update(words)
    return {w for w, n in count.items() if n >= threshold}


def build_lexicon(texts: Iterable[str], owners: Iterable[str], stop: Iterable[str],
                  exempt: frozenset = BY_EXEMPT) -> set[str]:
    stop = set(stop)
    words = {w for t in texts for w in marker_words(t, exempt)}
    words |= {w for o in owners for w in owner_words(o)}
    return {w for w in words if len(w) >= MIN_NAME and w not in stop}


def token_words(vocab_display: Iterable[str]) -> set[str]:
    """Every word of every TI token's display name (stop list)."""
    return {w for d in vocab_display for w in WORD.findall(d.upper())}


# ---------------------------------------------------------------- program records

def codes(ids: np.ndarray, vocab: Sequence[str]) -> np.ndarray:
    """build.py's token codes (int.from_bytes(tok) + 65536 * (len - 1)); <unk> gets 2**40."""
    table = _code_table(tuple(vocab))
    return table[np.asarray(ids, dtype=np.int64)]


@lru_cache(maxsize=4)
def _code_table(vocab: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.uint64)
    for i, h in enumerate(vocab):
        if h.startswith('<'):
            out[i] = np.uint64(2 ** 40 + i)
        else:
            b = bytes.fromhex(h)
            out[i] = int.from_bytes(b, 'big') + 65536 * (len(b) - 1)
    return out


def normalized_key(ids: np.ndarray, vocab: Sequence[str]) -> str:
    """build.py's normalized_key on ids (<unk> kept as a 4-byte marker)."""
    space, colon, newline, quote = _ids_of(vocab, SPACE, COLON, NEWLINE, QUOTE)
    toks = [int(t) for t in ids if int(t) not in (space, colon)]
    out: list[int] = []
    for t in toks + [newline]:
        if t == newline:
            while out and out[-1] in (colon, quote):
                out.pop()
            if out and out[-1] == newline:
                continue
        out.append(t)
    raw = b''.join(bytes.fromhex(vocab[t]) if not vocab[t].startswith('<') else b'\xff\xfe\xfd'
                   + bytes([t]) for t in out)
    return hashlib.sha1(raw).hexdigest()


def signatures(code_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """build.py's minhash_signatures on code arrays (same seed, permutations and shingles)."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 2**63, size=NUM_PERM, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=NUM_PERM, dtype=np.uint64)
    sigs = np.zeros((len(code_arrays), NUM_PERM), dtype=np.uint64)
    mult = np.uint64(1_000_003)
    with np.errstate(over='ignore'):
        for i, c in enumerate(code_arrays):
            c = np.asarray(c, dtype=np.uint64)
            k = min(SHINGLE, len(c))
            if k == 0:
                continue
            h = np.zeros(len(c) - k + 1, dtype=np.uint64)
            for j in range(k):
                h = h * mult + c[j:len(c) - k + 1 + j]
            sigs[i] = (a[:, None] * np.unique(h)[None, :] + b[:, None]).min(axis=1)
    return sigs


def similar_pairs(sigs: np.ndarray, min_j: float) -> list[tuple[int, int, float]]:
    """build.py's LSH candidate pairs with estimated Jaccard >= min_j."""
    rows = NUM_PERM // BANDS
    seen: set[tuple[int, int]] = set()
    out = []
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = np.ascontiguousarray(sigs[:, band * rows:(band + 1) * rows])
        for i in range(len(sigs)):
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


def large_buckets(sigs: np.ndarray, is_train) -> int:
    """Band buckets with more than 50 members holding both a train and a valc program (inside
    those buckets build.py's LSH pairs only the first member, so links can be missed)."""
    rows = NUM_PERM // BANDS
    n = 0
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = np.ascontiguousarray(sigs[:, band * rows:(band + 1) * rows])
        for i in range(len(sigs)):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) > 50:
                kinds = {bool(is_train(i)) for i in members}
                n += kinds == {True, False}
    return n


def _rank_key(p: dict):
    """Seeds first, then the longest, then the smallest sha1."""
    return (not p['seed'], -len(p['ids']), p['sha1'])


# ---------------------------------------------------------------- X0-X2

def x0(progs: list[dict]) -> tuple[list[dict], list]:
    """Exact duplicates of the scrubbed body: keep a seed if one is present, else the first."""
    best: dict[str, int] = {}
    for i, p in enumerate(progs):
        j = best.get(p['sha1'])
        if j is None or (p['seed'] and not progs[j]['seed']):
            best[p['sha1']] = i
    keep = set(best.values())
    return ([p for i, p in enumerate(progs) if i in keep],
            [p['key'] for i, p in enumerate(progs) if i not in keep])


def x1(progs: list[dict], vocab: Sequence[str], sigs: np.ndarray | None = None
       ) -> tuple[list[dict], list]:
    """build.py's content clusters (normalized body, MinHash Jaccard >= 0.7): keep the longest
    seed when the cluster holds one, else the longest member (ties: the smallest sha1)."""
    if sigs is None:
        sigs = signatures([codes(p['ids'], vocab) for p in progs])
    uf = UnionFind(len(progs))
    first: dict[str, int] = {}
    for i, p in enumerate(progs):
        uf.union(i, first.setdefault(normalized_key(p['ids'], vocab), i))
    for x, y, j in similar_pairs(sigs, NEAR_DUP_J):
        uf.union(x, y)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(progs)):
        groups[uf.find(i)].append(i)
    keep = {min(g, key=lambda i: _rank_key(progs[i])) for g in groups.values()}
    return ([p for i, p in enumerate(progs) if i in keep],
            [p['key'] for i, p in enumerate(progs) if i not in keep])


def x2(progs: list[dict], frac: float = 0.7, w: int = 32) -> tuple[list[dict], list]:
    """Containment: in descending length (ties: smallest sha1), drop a program when at least
    `frac` of its distinct w-token windows already occur in kept programs. Seeds are never
    dropped."""
    order = sorted(range(len(progs)), key=lambda i: (-len(progs[i]['ids']), progs[i]['sha1']))
    per = [np.unique(ngram_hashes(progs[i]['ids'], w)) for i in order]
    allh = np.concatenate(per) if per else np.zeros(0, np.uint64)
    uniq, inv = np.unique(allh, return_inverse=True)
    covered = np.zeros(len(uniq), dtype=bool)
    keep, pos = set(), 0
    for i, h in zip(order, per):
        idx = inv[pos:pos + len(h)]
        pos += len(h)
        share = covered[idx].mean() if len(idx) else 0.0
        if progs[i]['seed'] or share < frac:
            keep.add(i)
            covered[idx] = True
    return ([p for i, p in enumerate(progs) if i in keep],
            [p['key'] for i, p in enumerate(progs) if i not in keep])


# ---------------------------------------------------------------- X3

def informative(ids: np.ndarray, w: int = 32, min_distinct: int = 5) -> np.ndarray:
    """Per w-token window start: does the window hold at least `min_distinct` distinct ids?"""
    x = np.asarray(ids)
    n = len(x) - w + 1
    if n <= 0:
        return np.zeros(0, dtype=bool)
    prev = np.full(len(x), -1, dtype=np.int64)
    order = np.lexsort((np.arange(len(x)), x))
    same = x[order][1:] == x[order][:-1]
    prev[order[1:][same]] = order[:-1][same]
    starts = np.arange(n)
    distinct = np.zeros(n, dtype=np.int64)
    for d in range(w):
        distinct += prev[starts + d] < starts
    return distinct >= min_distinct


def _lines(ids: np.ndarray, newline: int) -> list[tuple[int, int]]:
    ends = np.flatnonzero(np.asarray(ids) == newline).tolist()
    out, a = [], 0
    for e in ends:
        out.append((a, e + 1))
        a = e + 1
    if a < len(ids):
        out.append((a, len(ids)))
    return out


def x3(progs: list[dict], K: int = 4, w: int = 32, min_distinct: int = 5, gut: float = 0.5,
       newline: int | None = None) -> tuple[list[dict], dict]:
    """The span cap: an informative w-token window in more than K programs stays in the first K
    (seeds first, then descending length, then the smallest sha1) and is excised from the others,
    whole lines only. A program losing more than `gut` of its tokens is dropped whole. Seeds are
    never excised."""
    if newline is None:
        from corpus.tokenize import Vocab
        newline = Vocab.load().tok_to_id[bytes.fromhex(NEWLINE)]
    order = sorted(range(len(progs)), key=lambda i: _rank_key(progs[i]))
    rank = np.empty(len(progs), dtype=np.int64)
    rank[order] = np.arange(len(progs))
    hs, owners, starts = [], [], []
    for i, p in enumerate(progs):
        h = ngram_hashes(p['ids'], w)
        keep = informative(p['ids'], w, min_distinct)
        hs.append(h[keep])
        owners.append(np.full(int(keep.sum()), i, dtype=np.int64))
        starts.append(np.flatnonzero(keep))
    H = np.concatenate(hs) if hs else np.zeros(0, np.uint64)
    O = np.concatenate(owners) if owners else np.zeros(0, np.int64)
    S = np.concatenate(starts) if starts else np.zeros(0, np.int64)
    # distinct (hash, owner) pairs in owner-rank order; the K-th and later owners are capped
    pairs = np.unique(np.stack([H, rank[O].astype(np.uint64)], axis=1), axis=0)
    first = np.ones(len(pairs), dtype=bool)
    first[1:] = pairs[1:, 0] != pairs[:-1, 0]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(pairs)), 0))
    slot = np.arange(len(pairs)) - group_start
    capped = pairs[slot >= K]
    capped_set = set(zip(capped[:, 0].tolist(), capped[:, 1].tolist()))
    cover: dict[int, np.ndarray] = {}
    if capped_set:
        rk = rank[O]
        hit = np.fromiter(((h, r) in capped_set for h, r in zip(H.tolist(), rk.tolist())),
                          dtype=bool, count=len(H))
        for i, s in zip(O[hit].tolist(), S[hit].tolist()):
            if progs[i]['seed']:
                continue
            c = cover.setdefault(i, np.zeros(len(progs[i]['ids']) + 1, dtype=np.int64))
            c[s] += 1
            c[s + w] -= 1
    out, shares, gutted, touched = [], [], 0, 0
    for i, p in enumerate(progs):
        c = cover.get(i)
        if c is None:
            out.append(p)
            continue
        covered = np.cumsum(c)[:-1] > 0
        drop = np.zeros(len(p['ids']), dtype=bool)
        for a, b in _lines(p['ids'], newline):
            if covered[a:b].all():
                drop[a:b] = True
        if not drop.any():
            out.append(p)
            continue
        touched += 1
        share = float(drop.mean())
        shares.append(share)
        if share > gut:
            gutted += 1
            continue
        ids = p['ids'][~drop]
        out.append(p | {'ids': ids, 'sha1': hashlib.sha1(ids.tobytes()).hexdigest(),
                        'x3_removed': int(drop.sum())})
    q = (lambda v: float(np.quantile(shares, v))) if shares else (lambda v: 0.0)
    stats = {'K': K, 'touched': touched, 'gutted': gutted,
             'excised_tokens': int(sum(p.get('x3_removed', 0) for p in out)),
             'capped_windows': int(len(capped)),
             'excised_share_q50': q(.5), 'excised_share_q90': q(.9),
             'excised_share_q99': q(.99), 'excised_share_max': max(shares, default=0.0)}
    return out, stats


# ---------------------------------------------------------------- leakage (Section 3.5)

class LeakageStop(RuntimeError):
    pass


MAX_LEAK_DROPS = 100


def check_leakage_count(n: int) -> None:
    if n > MAX_LEAK_DROPS:
        raise LeakageStop(f'the leakage re-check would drop {n} train programs (> '
                          f'{MAX_LEAK_DROPS}): the build stops for review')


def _grams(c: np.ndarray, w: int = CONTAIN_GRAM) -> np.ndarray:
    """build.py's grams() on a code array."""
    c = np.asarray(c, dtype=np.uint64)
    if len(c) < w:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(c) - w + 1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(w):
            h = h * np.uint64(1_000_003) + c[j:len(c) - w + 1 + j]
    return np.unique(h)


def containing(queries: list[np.ndarray], refs: list[np.ndarray]) -> dict[int, set[int]]:
    """build.py's containment rule, keeping the owner: query q -> the refs holding at least
    CONTAIN_MAX of q's 16-token windows (windows in more than 200 refs ignored)."""
    if not queries or not refs:
        return {}
    rg = [_grams(r) for r in refs]
    allg = np.concatenate(rg)
    owner = np.concatenate([np.full(g.size, i, dtype=np.int64) for i, g in enumerate(rg)])
    o = np.argsort(allg, kind='stable')
    allg, owner = allg[o], owner[o]
    out = {}
    for qi, q in enumerate(queries):
        g = _grams(q)
        if g.size == 0:
            continue
        lo, hi = np.searchsorted(allg, g, 'left'), np.searchsorted(allg, g, 'right')
        cnt: Counter = Counter()
        for a, b in zip(lo.tolist(), hi.tolist()):
            if 0 < b - a <= 200:
                cnt.update(set(owner[a:b].tolist()))
        hits = {r for r, n in cnt.items() if n / g.size >= CONTAIN_MAX}
        if hits:
            out[qi] = hits
    return out


def leakage_links(train: list[dict], valc: list[dict], vocab: Sequence[str],
                  train_sigs: np.ndarray | None = None) -> dict:
    """Train programs directly linked to a valc program: key -> sorted link types (sha1,
    normalized, minhash_0.7, minhash_0.5, contains). Transitive groups are never followed."""
    links: dict = defaultdict(set)
    vs = {p['sha1'] for p in valc}
    vn = {normalized_key(p['ids'], vocab) for p in valc}
    for p in train:
        if p['sha1'] in vs:
            links[p['key']].add('sha1')
        if normalized_key(p['ids'], vocab) in vn:
            links[p['key']].add('normalized')
    if train_sigs is None:
        train_sigs = signatures([codes(p['ids'], vocab) for p in train])
    vsig = signatures([codes(p['ids'], vocab) for p in valc])
    n = len(train)
    for x, y, j in similar_pairs(np.concatenate([train_sigs, vsig]), SPLIT_J):
        if (x < n) == (y < n):
            continue
        t = x if x < n else y
        links[train[t]['key']].add('minhash_0.7' if j >= NEAR_DUP_J else 'minhash_0.5')
    tc = [codes(p['ids'], vocab) for p in train]
    vc = [codes(p['ids'], vocab) for p in valc]
    for _, refs in containing(vc, tc).items():
        for r in refs:
            links[train[r]['key']].add('contains')
    return {k: sorted(v) for k, v in links.items()}
