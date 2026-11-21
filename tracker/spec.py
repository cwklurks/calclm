"""Python spec of the TI-BASIC program-state tracker: a straightforward, readable statement
of the semantics that tracker/tracker.c implements in fixed-size C. The two must produce
identical features (tests/test_diff.py, python -m tracker.difftest).

The token roles come from the token hex strings and tivars display names directly (not from
the generated C table), so a table bug shows up as a C/spec difference.

    t = Tracker(); t.reset()
    cats, bag = t.step(token_id)      # state AFTER the token: NCAT ids + 64-bit bag
    t.summary()                       # end-of-program diagnostics (for T4)
    cats, bags = batch(uint16_stream) # <bos> resets
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import feats as F

VOCAB_PATH = Path(__file__).resolve().parent.parent / "corpus" / "vocab.json"
U16 = 0xFFFF

FIXED_ROLES = {
    "3F": "newline", "3E": "colon", "2A": "quote", "04": "store", "2B": "comma",
    "10": "lparen", "11": "rparen", "08": "lbrace", "09": "rbrace", "06": "lbracket",
    "07": "rbracket", "CE": "if", "CF": "then", "D0": "else", "D4": "end", "D3": "for",
    "D1": "while", "D2": "repeat", "D6": "lbl", "D7": "goto", "E6": "menu", "DA": "isds",
    "DB": "isds", "EB": "listl", "5F": "prgm", "BB54": "delvar", "DC": "input",
    "DD": "prompt", "BB4A": "setuped", "98": "storepic", "99": "recallpic",
    "9A": "storegdb", "9B": "recallgdb", "B5": "dim", "BB55": "equstr", "BB3A": "listmatr",
    "BB39": "matrlist",
}
OPENER_ROLES = {"lparen", "for", "menu", "isds", "dim", "equstr", "listmatr", "matrlist",
                "statopen", "func"}
# statistics result variables, UNDEFINED until a statistics command runs (6221 is sequence n)
STAT_VARS = {f"62{k:02X}" for k in range(0x01, 0x3D)} - {"6221"}
STAT_CMDS = {"F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "FF", "2E", "2F", "BB32", "BB33",
             "BB34", "BB3C", "BB41", "BB46", "BB47", "BB48", "BB49", "EF15", "BB3B", "BB3D",
             "BB3E", "BB3F", "BB40", "BB42", "BB43", "BB44", "BB59", "EF14"}
STATEMENT_ROLES = {"if", "then", "else", "end", "for", "while", "repeat", "lbl", "goto",
                   "menu", "isds", "delvar", "input", "prompt", "setuped", "storepic",
                   "recallpic", "storegdb", "recallgdb"}
SEPARATORS = {"newline", "colon", "eos"}
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZθ")


@lru_cache(maxsize=4)
def roles(vocab_key: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    """(role, argument) per vocab id."""
    from tivars.models import TI_84PCE

    names = {b.hex().upper(): t.langs["en"].display for b, t in TI_84PCE.tokens.bytes.items()}
    out = []
    for hx in vocab_key:
        if hx in ("<pad>", "<bos>", "<eos>"):
            out.append((hx[1:-1], None))
        elif hx in FIXED_ROLES:
            out.append((FIXED_ROLES[hx], None))
        elif hx in STAT_VARS:
            out.append(("statvar", hx))
        elif hx in STAT_CMDS:
            out.append(("statopen" if names.get(hx, "").endswith("(") else "statcmd", None))
        elif len(hx) == 2 and ("41" <= hx <= "5A" or "30" <= hx <= "39" or hx == "5B"):
            out.append(("name", "θ" if hx == "5B" else chr(int(hx, 16))))
        elif hx[:2] == "AA" and len(hx) == 4 and int(hx[2:], 16) <= 9:
            out.append(("str", int(hx[2:], 16)))
        elif hx[:2] == "5C" and len(hx) == 4 and int(hx[2:], 16) <= 9:
            out.append(("mat", int(hx[2:], 16)))
        elif hx[:2] == "60" and len(hx) == 4 and int(hx[2:], 16) <= 9:
            out.append(("pic", (int(hx[2:], 16) + 1) % 10))
        elif hx[:2] == "61" and len(hx) == 4 and int(hx[2:], 16) <= 9:
            out.append(("gdb", (int(hx[2:], 16) + 1) % 10))
        elif hx[:2] == "5D" and len(hx) == 4 and int(hx[2:], 16) <= 5:
            out.append(("syslist", None))
        elif names.get(hx, "").endswith("("):
            out.append(("func", None))
        else:
            out.append(("other", None))
    return tuple(out)


@lru_cache(maxsize=1)
def default_vocab() -> tuple[str, ...]:
    return tuple(json.loads(VOCAB_PATH.read_text()))


class _Name:
    """A name being read: kind is lbl, goto, menu, list, prgm, store_run or setup_run;
    role (lists only) is def, undef or read."""

    def __init__(self, kind: str, role: str | None = None, chars: list[str] | None = None):
        self.kind, self.role, self.chars = kind, role, chars or []


class Tracker:
    def __init__(self, vocab: tuple[str, ...] | None = None):
        self.roles = roles(tuple(vocab) if vocab is not None else default_vocab())
        self.reset()

    # ---- state ------------------------------------------------------------------------------
    def reset(self) -> None:
        self.pos = 0
        self.in_str = False
        self.at_start = True
        self.mode = None            # one-token contexts: store, store_dim, delvar, picop
        self.picop = None           # (pic|gdb, define|read) for the picop mode
        self.name: _Name | None = None
        self.pending = None         # (mat_store|str_input, index, pos) decided by next token
        self.stmt = None            # role of the statement's first token
        self.first_lbrace = False   # the statement starts with {
        self.arg_start = False      # next token starts a SetUpEditor argument
        self.paren = self.brace = self.bracket = 0
        self.menu = None            # [paren depth inside Menu(, argument index]
        self.fn = None              # (equstr|listmatr|matrlist, paren depth inside it)
        self.blocks: list[int] = []
        self.if_state = F.IF_NONE
        self.defined: set[str] = set()
        self.unres: set[str] = set()
        self.lists: list[list] = []  # [name, assigned], first MAX_LISTS names defined
        self.list_overflow = False
        self.stats = False          # a statistics command has appeared
        self.strs: set[int] = set()
        self.mats: set[int] = set()
        self.pics: set[int] = set()
        self.gdbs: set[int] = set()
        self.n_undef = self.first_undef_pos = 0
        self.n_bad = self.first_bad_pos = 0
        self.n_extra_end = self.n_prgm = 0
        self.events: list[tuple] = []   # spec-only detail for validation reports

    # ---- counters ---------------------------------------------------------------------------
    def _bad(self, why: str) -> None:
        self.n_bad = min(self.n_bad + 1, U16)
        if not self.first_bad_pos:
            self.first_bad_pos = self.pos
        self.events.append(("bad", self.pos, why))

    def _undef(self, kind: str, what, pos: int | None = None) -> None:
        pos = self.pos if pos is None else pos
        self.n_undef = min(self.n_undef + 1, U16)
        if not self.first_undef_pos:
            self.first_undef_pos = pos
        self.events.append(("undef", pos, kind, what))

    # ---- variables --------------------------------------------------------------------------
    def _list_define(self, name: str) -> None:
        for entry in self.lists:
            if entry[0] == name:
                entry[1] = True
                return
        if len(self.lists) < F.MAX_LISTS:
            self.lists.append([name, True])
        else:
            self.list_overflow = True

    def _list_undefine(self, name: str) -> None:
        for entry in self.lists:
            if entry[0] == name:
                entry[1] = False

    def _list_read(self, name: str) -> None:
        for entry in self.lists:
            if entry[0] == name:
                if not entry[1]:
                    self._undef("list", name)
                return
        if not self.list_overflow:
            self._undef("list", name)

    def _read(self, kind: str, k: int, pos: int | None = None) -> None:
        if k not in getattr(self, kind + "s"):
            self._undef(kind, k, pos)

    def _define(self, kind: str, k: int) -> None:
        getattr(self, kind + "s").add(k)

    def _undefine(self, kind: str, k: int) -> None:
        getattr(self, kind + "s").discard(k)

    # ---- blocks -----------------------------------------------------------------------------
    def _push(self, kind: int) -> None:
        if len(self.blocks) < U16:
            self.blocks.append(kind)

    # ---- statements -------------------------------------------------------------------------
    def _end_statement(self) -> None:
        if self.pending:
            self._resolve_pending("eos")
        if self.at_start:                       # empty statement: it is a single-line If body
            if self.if_state == F.IF_AFTER:
                self.if_state = F.IF_NONE
        elif self.if_state == F.IF_COND:
            self.if_state = F.IF_AFTER
        elif self.if_state == F.IF_BODY:
            self.if_state = F.IF_NONE
        self.at_start = True
        self.stmt = None
        self.first_lbrace = self.arg_start = self.in_str = False
        self.paren = self.brace = self.bracket = 0
        self.menu = self.fn = self.mode = self.picop = None

    def _resolve_pending(self, role: str) -> None:
        kind, k, pos = self.pending
        self.pending = None
        if kind == "mat_store":                 # →[A]( is an element store: needs [A]
            if role == "lparen":
                self._read("mat", k, pos)
            else:
                self._define("mat", k)
        else:                                    # Input Str1,... uses Str1 as the prompt
            if role == "comma":
                self._read("str", k, pos)
            else:
                self._define("str", k)

    def _store(self) -> None:
        self.paren = self.brace = self.bracket = 0
        self.menu = self.fn = None
        self.mode = "store"

    def _open(self) -> None:
        self.paren = min(self.paren + 1, 255)

    # ---- names ------------------------------------------------------------------------------
    def _register_label(self, kind: str, name: str) -> None:
        if kind == "lbl":
            self.defined.add(name)
            self.unres.discard(name)
        elif name not in self.defined:
            self.unres.add(name)

    def _finish_list(self) -> None:
        nm, self.name = self.name, None
        if not nm.chars:
            self._bad("empty list name")
            return
        name = "".join(nm.chars)
        if nm.role == "def":
            self._list_define(name)
        elif nm.role == "undef":
            self._list_undefine(name)
        else:
            self._list_read(name)

    def _finish_run(self) -> None:
        nm, self.name = self.name, None
        name = "".join(nm.chars)
        if nm.kind == "setup_run" or len(nm.chars) >= 2 or self.first_lbrace:
            self._list_define(name)

    def _name_step(self, role: str, arg) -> bool:
        """Feed one token to the name in progress. True if the token was consumed."""
        nm = self.name
        is_char = role == "name"
        if nm.kind in ("lbl", "goto", "menu"):
            if is_char:
                if len(nm.chars) < 2:
                    nm.chars.append(arg)
                else:
                    self._bad("label name longer than 2")
                    self.name = None
                return True
            term = role in SEPARATORS or (nm.kind == "menu" and role in ("comma", "rparen"))
            if nm.chars and term:
                self._register_label(nm.kind, "".join(nm.chars))
            else:
                self._bad("malformed label name")
            self.name = None
            return False
        if nm.kind == "prgm":
            if is_char and len(nm.chars) < 8:
                nm.chars.append(arg)
                return True
            self.name = None
            if nm.role == "call":           # a call statement: 1-8 characters, then a separator
                if is_char or not nm.chars or role not in SEPARATORS:
                    self._bad("malformed program call")
                return is_char                # a 9th character goes with the bad name
            return False
        if nm.kind == "list":
            if is_char and (nm.chars or arg in LETTERS):
                if len(nm.chars) < 5:
                    nm.chars.append(arg)
                    return True
                self._finish_list()
                self._bad("list name longer than 5")
                return True
            self._finish_list()
            if nm.role == "undef" and role not in SEPARATORS:
                self._end_statement()       # DelVar ʟNAME chains: this token starts a statement
            return False
        # store_run / setup_run: list names written without ʟ
        if is_char:
            if len(nm.chars) < 5:
                nm.chars.append(arg)
                return True
            self._finish_run()
            self._bad("list name longer than 5")
            return True
        self._finish_run()
        return False

    # ---- tokens -----------------------------------------------------------------------------
    def step(self, tok: int):
        role, arg = self.roles[tok] if 0 <= tok < len(self.roles) else ("other", None)
        if role == "bos":
            self.reset()
            return self.features()
        if role == "pad":
            return self.features()
        self.pos = min(self.pos + 1, U16)
        if role == "eos":
            if self.name is not None:
                self._name_step("eos", None)
            if self.if_state in (F.IF_COND, F.IF_AFTER):   # If with no statement after it
                self._bad("single-line If at the end of the program")
            self._end_statement()
            return self.features()
        if self.in_str:
            if role == "quote":
                self.in_str = False
            elif role == "store":
                self.in_str = False
                self._store()
            elif role == "newline":
                self._end_statement()
            return self.features()
        if self.pending:
            self._resolve_pending(role)
        if self.name is not None and self._name_step(role, arg):
            return self.features()
        if role in ("newline", "colon"):
            self._end_statement()
            return self.features()
        self._token(role, arg)
        return self.features()

    def _token(self, role: str, arg) -> None:
        mode, self.mode = self.mode, None
        arg_start, self.arg_start = self.arg_start, False
        if self.at_start:
            self.at_start = False
            if self.if_state == F.IF_AFTER:
                if role == "then":
                    self.stmt = "then"
                    self._push(F.BLK_IF)
                    self.if_state = F.IF_NONE
                    return
                self.if_state = F.IF_BODY
            self.stmt = role
            self.first_lbrace = role == "lbrace"
            if role in STATEMENT_ROLES:
                self._statement(role)
                return
            if role == "prgm":
                self._prgm("call")
                return
        if mode == "store" and self._store_target(role, arg):
            return
        if mode == "store_dim":
            if role == "listl":
                self.name = _Name("list", "def")
                return
            if role == "mat":
                self._define("mat", arg)
                return
        if mode == "delvar":
            self._delvar_operand(role, arg)
            return
        if mode == "picop" and self._pic_operand(role, arg):
            return
        if (role == "name" and arg in LETTERS and arg_start and self.stmt == "setuped"
                and self.paren == 0):
            self.name = _Name("setup_run", None, [arg])
            return
        self._general(role, arg)

    def _prgm(self, role: str | None) -> None:
        self.n_prgm = min(self.n_prgm + 1, U16)
        self.events.append(("prgm", self.pos))
        self.name = _Name("prgm", role)

    def _statement(self, role: str) -> None:
        if role in ("if", "isds"):
            self.if_state = F.IF_COND
            if role == "isds":
                self._open()
        elif role == "then":                     # not right after an If: ERR:SYNTAX
            self._bad("Then without If")
            self._push(F.BLK_IF)
        elif role == "else":
            depth = len(self.blocks)
            if depth > F.MAX_BLOCKS:
                pass
            elif depth and self.blocks[-1] == F.BLK_IF:
                self.blocks[-1] = F.BLK_ELSE
            else:
                self._bad("Else without If-Then")
        elif role == "end":
            if self.blocks:
                self.blocks.pop()
            else:
                self.n_extra_end = min(self.n_extra_end + 1, U16)
                self.events.append(("extra_end", self.pos))
        elif role == "for":
            self._push(F.BLK_FOR)
            self._open()
        elif role == "while":
            self._push(F.BLK_WHILE)
        elif role == "repeat":
            self._push(F.BLK_REPEAT)
        elif role == "lbl":
            self.name = _Name("lbl")
        elif role == "goto":
            self.name = _Name("goto")
        elif role == "menu":
            self._open()
            self.menu = [self.paren, 0]
        elif role == "delvar":
            self.mode = "delvar"
        elif role == "setuped":
            self.arg_start = True
        elif role in ("storepic", "recallpic", "storegdb", "recallgdb"):
            self.mode = "picop"
            self.picop = ("pic" if role.endswith("pic") else "gdb",
                          "define" if role.startswith("store") else "read")

    def _store_target(self, role: str, arg) -> bool:
        if role == "str":
            self._define("str", arg)
        elif role == "mat":
            self.pending = ("mat_store", arg, self.pos)
        elif role == "listl":
            self.name = _Name("list", "def")
        elif role == "dim":
            self._open()
            self.mode = "store_dim"
        elif role == "name" and arg in LETTERS:
            self.name = _Name("store_run", None, [arg])
        else:
            return False
        return True

    def _delvar_operand(self, role: str, arg) -> None:
        if role == "listl":
            self.name = _Name("list", "undef")
            return
        if role in ("str", "mat", "pic", "gdb"):
            self._undefine(role, arg)
        self._end_statement()                   # DelVar X chains: next token starts a statement

    def _pic_operand(self, role: str, arg) -> bool:
        kind, op = self.picop
        self.picop = None
        if role == "name" and arg.isdigit():
            n = int(arg)
        elif role == kind:
            n = arg
        else:
            return False
        if op == "define":
            self._define(kind, n)
        else:
            self._read(kind, n)
        return True

    def _general(self, role: str, arg) -> None:
        if role == "quote":
            self.in_str = True
        elif role == "store":
            self._store()
        elif role in OPENER_ROLES:
            self._open()
            if role in ("equstr", "listmatr", "matrlist") and self.fn is None:
                self.fn = (role, self.paren)
            if role == "statopen":
                self.stats = True
        elif role == "rparen":
            if self.paren:
                self.paren -= 1
            if self.menu and self.paren < self.menu[0]:
                self.menu = None
            if self.fn and self.paren < self.fn[1]:
                self.fn = None
        elif role == "lbrace":
            self.brace = min(self.brace + 1, 255)
        elif role == "rbrace":
            self.brace = max(self.brace - 1, 0)
        elif role == "lbracket":
            self.bracket = min(self.bracket + 1, 255)
        elif role == "rbracket":
            self.bracket = max(self.bracket - 1, 0)
        elif role == "comma":
            if self.menu and self.paren == self.menu[0]:
                self.menu[1] = min(self.menu[1] + 1, 255)
                if self.menu[1] >= 2 and self.menu[1] % 2 == 0:
                    self.name = _Name("menu")
            if self.stmt == "setuped" and self.paren == 0:
                self.arg_start = True
        elif role == "listl":
            defining = (self.stmt in ("input", "prompt", "setuped")
                        or (self.fn is not None and self.fn[0] == "matrlist"))
            self.name = _Name("list", "def" if defining else "read")
        elif role == "str":
            if self.stmt == "input":
                self.pending = ("str_input", arg, self.pos)
            elif self.stmt == "prompt" or (self.fn is not None and self.fn[0] == "equstr"):
                self._define("str", arg)
            else:
                self._read("str", arg)
        elif role == "mat":
            if self.stmt in ("input", "prompt") or (self.fn is not None
                                                     and self.fn[0] == "listmatr"):
                self._define("mat", arg)
            else:
                self._read("mat", arg)
        elif role == "prgm":                         # inside an expression (Asm(prgmX), ...)
            self._prgm(None)
        elif role in ("lbl", "goto"):
            self._bad(role + " inside a statement")
        elif role in ("pic", "gdb"):                # Archive Pic1, UnArchive GDB2, ...
            self._read(role, arg)
        elif role == "statvar":
            if not self.stats:
                self._undef("stat", arg)
        elif role == "statcmd":
            self.stats = True

    # ---- features ---------------------------------------------------------------------------
    def _ctx(self) -> int:
        if self.in_str:
            return F.CTX_STRING
        nm = self.name
        if nm is not None:
            if nm.kind == "lbl":
                return F.CTX_LBL
            if nm.kind in ("goto", "menu"):
                return F.CTX_REF
            if nm.kind == "prgm":
                return F.CTX_PRGM
            if nm.kind == "store_run" and len(nm.chars) < 2:
                return F.CTX_EXPR
            return F.CTX_LIST
        if self.mode == "store":
            return F.CTX_STORE
        if self.mode == "delvar":
            return F.CTX_DELVAR
        return F.CTX_START if self.at_start else F.CTX_EXPR

    def _hint(self, ctx: int) -> int:
        if ctx == F.CTX_LBL:
            cands = self.unres
        elif ctx == F.CTX_REF:
            cands = self.defined
        elif ctx == F.CTX_LIST:
            cands = {name for name, assigned in self.lists if assigned}
        else:
            return 0
        prefix = "".join(self.name.chars)
        bits = 0
        for cand in cands:
            if not cand.startswith(prefix):
                continue
            if len(cand) > len(prefix):
                bits |= 1 << F.NAME_CHARS.index(cand[len(prefix)])
            elif prefix:
                bits |= 1 << F.HINT_END
        return bits

    def features(self) -> tuple[tuple[int, ...], int]:
        ctx = self._ctx()
        name_len = (min(len(self.name.chars), 2)
                    if ctx in (F.CTX_LBL, F.CTX_REF, F.CTX_LIST, F.CTX_PRGM) else 0)
        lit = F.LIT_BRACKET if self.bracket else (F.LIT_BRACE if self.brace else F.LIT_NONE)
        if self.menu is None:
            menu = F.MENU_NONE
        else:
            a = self.menu[1]
            menu = F.MENU_TITLE if a == 0 else (F.MENU_TEXT if a % 2 else F.MENU_LABEL)
        depth = len(self.blocks)
        if depth == 0:
            top = F.BLK_NONE
        elif depth > F.MAX_BLOCKS:
            top = F.BLK_DEEP
        else:
            top = self.blocks[-1]
        cats = (ctx, name_len, min(self.paren, 3), lit, menu, top, min(depth, 4),
                self.if_state, min(len(self.unres), 2))
        bag = self._hint(ctx)
        for k in self.strs:
            bag |= 1 << (F.BAG_STR + k)
        for k in self.mats:
            bag |= 1 << (F.BAG_MAT + k)
        for name, assigned in self.lists:
            if assigned:
                bag |= 1 << (F.BAG_LIST + F.list_bucket(name))
        return cats, bag

    def summary(self) -> dict:
        return {
            "n_unres": len(self.unres), "depth": min(len(self.blocks), U16),
            "n_undef": self.n_undef, "first_undef_pos": self.first_undef_pos,
            "n_bad": self.n_bad, "first_bad_pos": self.first_bad_pos,
            "n_extra_end": self.n_extra_end, "n_prgm": self.n_prgm,
        }

    def details(self) -> dict:
        """Spec-only detail for reports: unresolved names and the event log."""
        return {"unres": sorted(self.unres), "defined": sorted(self.defined),
                "events": list(self.events)}


def batch(stream) -> tuple[np.ndarray, np.ndarray]:
    """Features for a uint16 token stream (programs separated by <bos>):
    cats (T, NCAT) uint8 and bags (T,) uint64; row i is the state after token i."""
    stream = np.asarray(stream)
    t = Tracker()
    cats = np.zeros((len(stream), F.NCAT), dtype=np.uint8)
    bags = np.zeros(len(stream), dtype=np.uint64)
    for i, tok in enumerate(stream.tolist()):
        c, b = t.step(tok)
        cats[i] = c
        bags[i] = b
    return cats, bags
