"""Logit masks at sampling (INTREF.md 13): lab/masks.py's rules for one sequence, written so the
calculator (ce/calclm/core/clm_mask.c) can follow them line by line.

The state is the C tracker's own (tracker/tracker.c, the file the calculator compiles), driven
through ctypes (tracker.bind). A rule asks a one-token look-ahead question: copy the state, step
the candidate token(s) on the copy, and compare one of the tracker's summary counters before
and after. Before every draw `banned()` returns the ids to exclude from the ranking:

1. always: DelVar, ClrAllLists, Archive, UnArchive, ClrList;
2. End when it would count as unmatched (`n_extra_end` grows);
3. Else when it would count as bad (`n_bad` grows);
4. each Str0-9 / [A]-[J] token whose read would be undefined (`n_undef` grows);
5. if the last token fed was one of those: `,` and `(` where they resolve a pending read into an
   undefined read (`Input Str1,` uses Str1 as a prompt; `->[A](` stores into an element);
6. custom lists (`ʟ` and the name after it):
   - outside a list name: `ʟ` when no list is assigned (bag bits 58-63 all clear) and a list
     name after it would be a read (probe: `ʟ`, `A`, newline grows `n_undef`);
   - inside a list name: decide once, at the first banned() in the name, whether the name is
     read (probe: the first letter A-Z with its hint bit clear, then newline, grows `n_undef`;
     no such letter counts as a read). For a read: ban every name character whose hint bit is
     clear; when the END hint bit is clear (the name so far is no assigned list) also ban
     every id from 4 up that is not a name character, and <eos>;
7. if no id is left (ids 0, 1, 3 never count), <eos> is allowed again.

Level 2 (v2.4, model/safety.py; `IntMask(vocab, level=2)`, clm_reset's masks = 2) adds, after
rule 6 and before rule 7: calls to other code and setting commands always; window variables
where they would be assigned (after ->, in an Input/Prompt statement, after For( IS>( DS<();
inside a string, the class the string automaton bans (credit lines; the state carries across
"..."+"..."). Level 1 is v2.3's rules. Level 3 (clm_reset's masks = 3, the calculator's Masks
row off) is level 2's additions and v2.3's rule 1, without v2.3's grammar rules 2-6: the
destructive and safety bans are always on.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

from model import safety as S
from model.tokens import BOS, EOS, PAD, UNK

SAFETY_HEX = ("BB54", "BB52", "BB68", "BB69", "FA")   # DelVar ClrAllLists Archive UnArchive ClrList
END_HEX, ELSE_HEX = "D4", "D0"
VAR_HEX = tuple(f"AA{k:02X}" for k in range(10)) + tuple(f"5C{k:02X}" for k in range(10))
COMMA_HEX, LPAREN_HEX, NEWLINE_HEX, LISTL_HEX = "2B", "10", "3F", "EB"
NAME_HEX = tuple("%02X" % c for c in [*range(0x41, 0x5B), *range(0x30, 0x3A), 0x5B])
N_UNDEF, N_BAD, N_EXTRA_END = 2, 4, 6          # trk_summary fields (tracker.bind.SUMMARY_KEYS)
CTX_START, CTX_STRING, CTX_STORE, CTX_LIST = 0, 2, 3, 6   # trk_feat.cat[0] (tracker.h TRK_CTX_*)
HINT_BITS, HINT_END, BAG_LIST = 37, 37, 58     # bag = hint (bits 0-37) | var << 38
N_LETTERS = 26                                 # A-Z: the probe letters

TRACKER_TABLES = Path(__file__).resolve().parents[1] / "tracker" / "trk_tables.h"


@lru_cache(maxsize=4)
def _tables_match(vocab: tuple[str, ...]) -> bool:
    from tracker import gen_tables
    return TRACKER_TABLES.read_text() == gen_tables.render(list(vocab))


def mask_ids(vocab: list[str]) -> dict:
    """Token ids the rules use (ce/calclm/tools/gen_mask_ids.py writes them for the C core)."""
    idx = {h: i for i, h in enumerate(vocab)}
    cls = S.str_classes(vocab)
    return {"safety": [idx[h] for h in SAFETY_HEX if h in idx], "end": idx[END_HEX],
            "else": idx[ELSE_HEX], "vars": [idx[h] for h in VAR_HEX if h in idx],
            "pending": [idx[COMMA_HEX], idx[LPAREN_HEX]], "newline": idx[NEWLINE_HEX],
            "listl": idx[LISTL_HEX], "names": [idx[h] for h in NAME_HEX],
            # level 2 (model/safety.py)
            "always": [idx[h] for h in S.ALWAYS_HEX if h in idx], "window": S.window_ids(vocab),
            "loopvar": [idx[h] for h in S.LOOPVAR_HEX], "io": [idx[h] for h in S.IO_HEX],
            "str_ids": [i for i, c in enumerate(cls) if c != S.C_W],
            "str_cls": [int(c) for c in cls if c != S.C_W],
            "quote": idx[S.QUOTE_HEX], "plus": idx[S.PLUS_HEX]}


class IntMask:
    """reset(); step(tok) for every token fed (<bos> included); banned() -> (V,) bool.
    level: 1 the v2.3 rules, 2 v2.4's (model/safety.py), 3 v2.4's additions and v2.3's
    destructive bans (rule 1) without v2.3's grammar rules."""

    def __init__(self, vocab: list[str], level: int = S.V23):
        if level not in (S.V23, S.V24, S.SAFETY_ONLY):
            raise ValueError(f"mask level {level}: 1 (v2.3), 2 (v2.4) or 3 (safety rules only)")
        if not _tables_match(tuple(vocab)):
            raise ValueError("the C tracker's token-class table (tracker/trk_tables.h) was not "
                             "generated from this vocab")
        from tracker import bind
        self.lib = bind.lib()
        self.v = len(vocab)
        ids = mask_ids(vocab)
        self.safety, self.end, self.else_ = ids["safety"], ids["end"], ids["else"]
        self.vars, self.pending = ids["vars"], ids["pending"]
        self.newline, self.listl, self.names = ids["newline"], ids["listl"], ids["names"]
        self.not_names = np.setdiff1d(np.arange(4, self.v), self.names)
        self.level = level
        self.always, self.window = ids["always"], ids["window"]
        self.loopvar = set(ids["loopvar"])
        classes = self.lib.tracker_classes()
        self.io_cls = {classes[t] for t in ids["io"]}
        self.stmt_off = self.lib.trk_offsetof_stmt()
        self.cls = S.str_classes(vocab)
        self.quote, self.plus = ids["quote"], ids["plus"]
        self.str_ban = np.array(S.STR_BAN)
        self.state = ctypes.create_string_buffer(bind.state_size())
        self.trial_state = ctypes.create_string_buffer(bind.state_size())
        self.feat = ctypes.create_string_buffer(bind.feat_size())
        self.sum = (ctypes.c_uint16 * 8)()
        self.ncat = 9
        self.reset()

    # ---- tracker state ----------------------------------------------------------------------

    def reset(self) -> None:
        self.lib.tracker_reset(self.state)
        self.prev, self.ctx, self.bag, self.list_read = -1, 0, 0, None
        self.str_state, self.str_carry = S.STR_START, 0
        self.base = self._counters(self.state)

    def step(self, tok: int) -> None:
        self.lib.tracker_step(self.state, int(tok), self.feat)
        raw = self.feat.raw          # trk_feat: cat[9], hint[5] (bits 0-37), var[4] (bits 0-25)
        hint = int.from_bytes(raw[self.ncat:self.ncat + 5], "little") & ((1 << (HINT_BITS + 1)) - 1)
        self.bag = hint | int.from_bytes(raw[self.ncat + 5:self.ncat + 9], "little") << (HINT_BITS + 1)
        was_str, self.ctx = self.ctx == CTX_STRING, raw[0]
        self._str_step(int(tok), was_str)
        self.base = self._counters(self.state)
        if self.ctx != CTX_LIST:
            self.list_read = None
        self.prev = int(tok)

    def _str_step(self, tok: int, was_str: bool) -> None:
        """The string automaton: on inside a string; kept across a closing quote, + and a
        reopening quote (str_carry 1, then 2); START otherwise."""
        if self.ctx == CTX_STRING:
            if was_str:
                self.str_state = S.STR_NEXT[self.str_state][self.cls[tok]]
            elif self.str_carry != 2:                     # a string just opened
                self.str_state = S.STR_START
            self.str_carry = 0
        elif was_str and tok == self.quote:
            self.str_carry = 1
        elif self.str_carry == 1 and tok == self.plus:
            self.str_carry = 2
        else:
            self.str_state, self.str_carry = S.STR_START, 0

    def _counters(self, state) -> tuple[int, ...]:
        self.lib.tracker_summary(state, self.sum)
        return tuple(self.sum)

    def _grows(self, key: int, *toks: int) -> bool:
        """Does counter `key` grow when toks are stepped on a copy of the state?"""
        ctypes.memmove(self.trial_state, self.state, len(self.state))
        for t in toks:
            self.lib.tracker_step(self.trial_state, int(t), None)
        return self._counters(self.trial_state)[key] > self.base[key]

    def _reads_list(self, *prefix: int) -> bool:
        hint = 0 if prefix else self.bag & ((1 << HINT_BITS) - 1)
        free = next((self.names[k] for k in range(N_LETTERS) if not (hint >> k) & 1), None)
        return free is None or self._grows(N_UNDEF, *prefix, free, self.newline)

    # ---- the rules ----------------------------------------------------------------------------

    def banned(self) -> np.ndarray:
        b = np.zeros(self.v, dtype=bool)
        b[self.safety] = True                             # rule 1 at every level
        if self.level != S.SAFETY_ONLY:
            self._v23_rules(b)
        if self.level != S.V23:
            self._safety_rules(b)
        allowed = ~b
        allowed[[PAD, BOS, UNK]] = False
        if not allowed.any():
            b[EOS] = False
        return b

    def _v23_rules(self, b: np.ndarray) -> None:
        """v2.3's grammar rules 2-6 (rule 1 is in banned())."""
        if self._grows(N_EXTRA_END, self.end):
            b[self.end] = True
        if self._grows(N_BAD, self.else_):
            b[self.else_] = True
        for t in self.vars:
            if self._grows(N_UNDEF, t):
                b[t] = True
        if self.prev in self.vars:
            for t in self.pending:
                if self._grows(N_UNDEF, t):
                    b[t] = True
        self._list_rules(b)

    def _list_rules(self, b: np.ndarray) -> None:
        if self.ctx != CTX_LIST:
            if not (self.bag >> BAG_LIST) & 0x3F and self._reads_list(self.listl):
                b[self.listl] = True
            return
        if self.list_read is None:
            self.list_read = self._reads_list()
        if not self.list_read:
            return
        for k, t in enumerate(self.names):
            if not (self.bag >> k) & 1:
                b[t] = True
        if not (self.bag >> HINT_END) & 1:
            b[self.not_names] = True
            b[EOS] = True

    def _safety_rules(self, b: np.ndarray) -> None:
        b[self.always] = True
        stmt = self.state.raw[self.stmt_off]
        if (self.ctx == CTX_STORE or (stmt in self.io_cls and self.ctx != CTX_START)
                or self.prev in self.loopvar):
            b[self.window] = True
        if self.ctx == CTX_STRING and self.str_ban[self.str_state]:
            bits = self.str_ban[self.str_state]
            b[2:] |= ((bits >> self.cls[2:]) & 1).astype(bool)
