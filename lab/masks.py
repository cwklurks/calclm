"""Logit masks from the TI-BASIC tracker state (G1 arms A3 and A4), applied at sampling time.

Before every draw, each row bans (lab.sample.choose removes banned ids from the ranking):
- End when the tracker would count it as unmatched (no open block);
- Else when the tracker would count it as outside an If-Then;
- a token that would make the tracker count a read of a list, string or matrix that was never
  assigned: a Str/matrix token read before any assignment, the token that resolves a pending
  read (`,` after `Input Str1`, `(` after `->[A]`); for custom lists, a prefix mask: on a read
  (not a definition), `ʟ` only when some list is assigned, then only name characters that
  extend an assigned list's name (the tracker's hint bits) and a token that ends the name only
  once it names an assigned list;
- always: the data-destroying tokens DelVar, ClrAllLists, Archive, UnArchive, ClrList.
<eos> is banned only where it would end a list name being read; a row left with no allowed
token gets <eos> back (stats["forced_eos"], not expected: the prefix mask has no dead ends).
Not masked: Goto / Menu( targets. Whether a label "can never be defined" is not decidable
while the program is being written (a later Lbl can define any name), so no mask applies.

The decisions are a one-token look-ahead on the C tracker (tracker/tracker.h through
tracker.bind.lib()): step the candidate on a copy of the row's state and compare the tracker's
own counters (n_extra_end, n_bad, n_undef) before and after, so the masks follow exactly the
tracker's rules for statement starts, string literals, Input/Prompt, store targets and pending
reads. A calculator port would test the same conditions on the state directly (lab/budget.py
costs MASK_ID_CYCLES per maskable id).
"""
from __future__ import annotations

import ctypes
from collections import Counter
from pathlib import Path

import numpy as np

from model.tokens import BOS, EOS, PAD, UNK, load_vocab

SAFETY_HEX = ("BB54", "BB52", "BB68", "BB69", "FA")   # DelVar ClrAllLists Archive UnArchive ClrList
END_HEX, ELSE_HEX = "D4", "D0"
VAR_HEX = tuple(f"AA{k:02X}" for k in range(10)) + tuple(f"5C{k:02X}" for k in range(10))
COMMA_HEX, LPAREN_HEX, NEWLINE_HEX, LISTL_HEX = "2B", "10", "3F", "EB"
NAME_HEX = tuple("%02X" % c for c in [*range(0x41, 0x5B), *range(0x30, 0x3A), 0x5B])
N_UNDEF, N_BAD, N_EXTRA_END = 2, 4, 6        # tracker.bind.SUMMARY_KEYS positions
CTX_LIST = 6                                  # tracker/feats.py CTX_LIST
HINT_BITS, HINT_END, BAG_LIST = 37, 37, 58     # tracker/feats.py bag layout


class CTrackerRows:
    """One C tracker state per sample row: reset(batch), step(tokens) -> (cats, bag) as int64
    (the layout of tracker.bind.batch / model.data), trial(row, tok) -> summary counters after
    tok without changing the row."""

    def __init__(self):
        from tracker import bind
        from tracker import feats as F

        self.lib, self.ncat = bind.lib(), F.NCAT
        self.size = bind.state_size()
        self.feat = ctypes.create_string_buffer(bind.feat_size())
        self.scratch = ctypes.create_string_buffer(self.size)
        self.sum = (ctypes.c_uint16 * 8)()
        self.batch = 0

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.states = [ctypes.create_string_buffer(self.size) for _ in range(batch)]
        for s in self.states:
            self.lib.tracker_reset(s)
        self.prev = np.full(batch, -1, dtype=np.int64)
        self.ctx = np.zeros(batch, dtype=np.int64)
        self.bag = [0] * batch
        self.list_read: list[bool | None] = [None] * batch   # role of the list name being read
        self.base = [self._counters(s) for s in self.states]

    def _counters(self, state) -> tuple[int, ...]:
        self.lib.tracker_summary(state, self.sum)
        return tuple(self.sum)

    def _features(self) -> tuple[list[int], int]:
        raw = self.feat.raw              # trk_feat: cat[NCAT], hint[5] (38 bits), var[4] (26 bits)
        hint = int.from_bytes(raw[self.ncat : self.ncat + 5], "little") & ((1 << 38) - 1)
        var = int.from_bytes(raw[self.ncat + 5 : self.ncat + 9], "little")
        return list(raw[: self.ncat]), hint | (var << 38)

    def step(self, tokens) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens).astype(np.int64)
        cats = np.zeros((self.batch, self.ncat), dtype=np.int64)
        bag = np.zeros(self.batch, dtype=np.uint64)
        for r, (s, t) in enumerate(zip(self.states, tokens.tolist())):
            self.lib.tracker_step(s, t, self.feat)
            cats[r], self.bag[r] = self._features()
            bag[r] = self.bag[r]
            self.base[r] = self._counters(s)
            if cats[r, 0] != CTX_LIST:
                self.list_read[r] = None
        self.prev, self.ctx = tokens, cats[:, 0].copy()
        return cats, bag.view(np.int64)

    def trial(self, row: int, *toks: int) -> tuple[int, ...]:
        """Summary counters after toks, stepped on a copy of the row's state."""
        ctypes.memmove(self.scratch, self.states[row], self.size)
        for t in toks:
            self.lib.tracker_step(self.scratch, int(t), None)
        return self._counters(self.scratch)


class TrackerMask:
    """banned(rows) -> (B, V) bool for lab.sample; rows is self.rows (CTrackerRows)."""

    tracker_name = "tracker.bind:Tracker"

    def __init__(self, vocab: Path):
        hexes = load_vocab(vocab)
        idx = {h: i for i, h in enumerate(hexes)}
        self.v = len(hexes)
        self.safety = [idx[h] for h in SAFETY_HEX if h in idx]
        self.end, self.else_ = idx[END_HEX], idx[ELSE_HEX]
        self.vars = [idx[h] for h in VAR_HEX if h in idx]
        self.pending = [idx[COMMA_HEX], idx[LPAREN_HEX]]
        self.newline, self.listl = idx[NEWLINE_HEX], idx[LISTL_HEX]
        self.name_bits = [idx[h] for h in NAME_HEX]   # hint bit k = name character k (A-Z 0-9 θ)
        self.not_names = np.setdiff1d(np.arange(4, self.v), self.name_bits)
        # ids whose ban is decided per token (the cost model: lab/budget.py mask_term)
        self.n_maskable = len(self.safety) + 2 + len(self.vars) + len(self.pending) + 2
        self.rows = CTrackerRows()
        self.stats: Counter = Counter()

    def _grows(self, row: int, key: int, *toks: int) -> bool:
        return self.rows.trial(row, *toks)[key] > self.rows.base[row][key]

    def _reads_list(self, row: int, *prefix: int) -> bool:
        """Is the list name after `prefix` (at most ʟ) a read? Probe: end it as a one-letter
        name that no assigned list starts with and see whether the tracker counts a read."""
        hint = self.rows.bag[row] & ((1 << HINT_BITS) - 1) if not prefix else 0
        free = next((self.name_bits[k] for k in range(26) if not (hint >> k) & 1), None)
        return free is None or self._grows(row, N_UNDEF, *prefix, free, self.newline)

    def banned(self, rows: CTrackerRows) -> np.ndarray:
        b = np.zeros((rows.batch, self.v), dtype=bool)
        b[:, self.safety] = True
        var_set = set(self.vars)
        st = self.stats
        for r in range(rows.batch):
            st["positions"] += 1
            if self._grows(r, N_EXTRA_END, self.end):
                b[r, self.end] = True
                st["end"] += 1
            if self._grows(r, N_BAD, self.else_):
                b[r, self.else_] = True
                st["else"] += 1
            for t in self.vars:
                if self._grows(r, N_UNDEF, t):
                    b[r, t] = True
                    st["var"] += 1
            if int(rows.prev[r]) in var_set:
                for t in self.pending:
                    if self._grows(r, N_UNDEF, t):
                        b[r, t] = True
                        st["pending"] += 1
            self._list_rules(r, b[r])
            allowed = ~b[r]
            allowed[[PAD, BOS, UNK]] = False
            if not allowed.any():
                b[r, EOS] = False
                st["forced_eos"] += 1
        return b

    def _list_rules(self, r: int, br: np.ndarray) -> None:
        rows, st = self.rows, self.stats
        bag = rows.bag[r]
        if rows.ctx[r] != CTX_LIST:
            if not (bag >> BAG_LIST) & 0x3F and self._reads_list(r, self.listl):
                br[self.listl] = True               # a read, and no list is assigned
                st["list_start"] += 1
            return
        if rows.list_read[r] is None:               # first position inside this name
            rows.list_read[r] = self._reads_list(r)
        if not rows.list_read[r]:
            return
        for k, t in enumerate(self.name_bits):
            if not (bag >> k) & 1:
                br[t] = True
        if not (bag >> HINT_END) & 1:               # the name so far is not an assigned list
            br[self.not_names] = True
            br[EOS] = True
        st["list_name"] += 1
