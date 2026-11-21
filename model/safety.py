"""The v2.4 safety masks' tokens and string automaton (docs/agents/phase1-safety-masks.md),
shared by model/intmask.py (IntMask level 2) and ce/calclm/tools/gen_mask_ids.py (the C core's
tables). Three rules, before every draw:

1. always: calls to other code (prgm, OpenLib(, ExecLib) and the commands that change a setting
   that outlives the program (ALWAYS_HEX);
2. the window, table and finance variables (every vocab token on the 63xx page) where they would
   be assigned: right after ->, inside an Input or Prompt statement, or as the variable of For(,
   IS>( or DS<(;
3. inside a string, the letter that would complete a credit line: Y of a word starting with BY,
   R of COPYR, ) of (C), in either case. The string's tokens are folded into N_CLASSES classes
   (str_classes) and a DFA built from PATTERNS by subset construction follows them; STR_BAN[s]
   is the set of classes banned in state s (bit c = class c). A string closed by " and followed
   by + and a reopening " continues the state ("B"+"Y is BY); anything else resets it.
"""
from __future__ import annotations

import numpy as np

V23, V24 = 1, 2           # mask levels: clm_reset's `masks` (clm.h CLM_MASKS_V23 / V24)
SAFETY_ONLY = 3           # these rules and v2.3's destructive bans, without v2.3's grammar
                          # rules (CLM_MASKS_SAFETY: the Masks row off)

CALL_HEX = ("5F", "EF11", "EF12")                       # prgm OpenLib( ExecLib
MODE_HEX = ("66", "67", "68", "69", "73", "64", "65",   # Normal Sci Eng Float Fix Radian Degree
            "76", "77", "78", "79", "7E00", "7E01",     # Func Param Polar Seq Sequential Simul
            "BB4D", "BB4F", "BB4E", "75", "74", "BB64",  # Real a+bi re^θi Full Horiz G-T
            "EF37", "EF38", "EF39", "EF3A", "EF3B", "EF3C", "EF3D",  # MATHPRINT CLASSIC n/d Un/d
            "EF3F", "EF40", "BB4B", "BB4C",             # STATWIZARD ON/OFF Pmt_End Pmt_Bgn
            "BB66", "BB67", "EF10", "EF0F",             # DiagnosticOn/Off ClockOn ClockOff
            "EF00", "EF01", "EF03", "EF04")             # setDate( setTime( setDtFmt( setTmFmt(
FORMAT_HEX = ("7E03", "7E02", "7E04", "7E05", "7E0B", "7E0A", "EF5A",  # RectGC PolarGC Coord Grid
              "7E08", "7E09", "7E0C", "7E0D", "BB50", "BB51",          # Axes Label Expr On/Off
              "EF6A", "EF6B", "EF5B", "EF64", "EF6C",   # DetectAsym Background On/Off BorderColor
              "7E0E", "7E0F", "7E10", "7E11", "7E12",   # Web Time uvAxes vwAxes uwAxes
              "7E06", "7E07", "EF74", "EF75",           # Thick Dot-Thick Thin Dot-Thin
              "96", "97", "E9", "EA", "EC", "ED", "EE",  # FnOn FnOff PlotsOn PlotsOff Plot1-3(
              "BB45", "EF65",                           # GraphStyle( GraphColor(
              "7A", "7B", "7C", "7D")                   # IndpntAuto/Ask DependAuto/Ask
ZOOM_HEX = ("86", "87", "88", "89", "8A", "8B", "8C", "8D", "8E", "8F", "90", "92", "BB65",
            "EF17", "EF18", "EF19", "EF1A", "EF1B", "EF1C", "EF1D")
OTHER_HEX = ("9B", "BB4A")                              # RecallGDB SetUpEditor
ALWAYS_HEX = CALL_HEX + MODE_HEX + FORMAT_HEX + ZOOM_HEX + OTHER_HEX
WINDOW_PAGE = "63"                                      # Xmin ... TraceStep, TblStart, N, PV ...
LOOPVAR_HEX = ("D3", "DA", "DB")                        # For( IS>( DS<(
IO_HEX = ("DC", "DD")                                   # Input Prompt
QUOTE_HEX, PLUS_HEX = "2A", "70"     # "..."+"...": the automaton runs on across the + (rule 3)

# ---- string classes ---------------------------------------------------------------------------

C_W, C_L, C_LP, C_RP = 0, 1, 2, 3                        # non-letter, other letter, ( )
LETTER_CLASS = {ch: 4 + k for k, ch in enumerate("bycopr")}
N_CLASSES = 4 + len(LETTER_CLASS)


def letter_of(h: str) -> str | None:
    """The letter (lowercase) a token reads as inside a string: its TI display text's only ASCII
    letter, when that letter comes first (A, b, the statistics/sequence tokens shown as a-z, and
    letter-plus-subscript tokens such as y₁, Y₁, Y₁ᴛ, r²), else None."""
    if h.startswith("<"):
        return None
    d = _display(h)
    letters = [c for c in d if c.isascii() and c.isalpha()]
    return d[0].lower() if len(letters) == 1 and d[0] == letters[0] else None


def _display(h: str) -> str:
    from tivars.models import TI_84PCE
    return TI_84PCE.tokens.bytes[bytes.fromhex(h)].langs["en"].display


def str_classes(vocab) -> np.ndarray:
    """(V,) uint8: each id's class inside a string."""
    out = np.full(len(vocab), C_W, dtype=np.uint8)
    for i, h in enumerate(vocab):
        ch = letter_of(h)
        if ch is not None:
            out[i] = LETTER_CLASS.get(ch, C_L)
        elif h == "10":
            out[i] = C_LP
        elif h == "11":
            out[i] = C_RP
    return out


# ---- the automaton ----------------------------------------------------------------------------

_NONLETTER = frozenset({C_W, C_LP, C_RP})
_b, _y, _c, _o, _p, _r = (LETTER_CLASS[ch] for ch in "bycopr")
# each pattern: a sequence of class sets; its last set is banned one symbol before completion
PATTERNS = ((_NONLETTER, {_b}, {_y}),                     # a word starting with BY
            ({_c}, {_o}, {_p}, {_y}, {_r}),               # COPYR
            ({C_LP}, {_c}, {C_RP}))                       # (C)


def _step(state: frozenset, c: int) -> frozenset:
    """Partial matches (pattern, symbols matched) after class c; completed ones are dropped
    (they occur only in text the masks did not choose, such as a typed prompt)."""
    nxt = {(i, k + 1) for i, k in state if c in PATTERNS[i][k]}
    nxt |= {(i, 1) for i, p in enumerate(PATTERNS) if c in p[0]}
    return frozenset((i, k) for i, k in nxt if k < len(PATTERNS[i]))


def _ban(state: frozenset) -> int:
    bits = 0
    for i, k in state:
        if k == len(PATTERNS[i]) - 1:
            for c in PATTERNS[i][k]:
                bits |= 1 << c
    return bits


def build_automaton() -> tuple[list[list[int]], list[int]]:
    """(next[state][class], ban[state]); state 0 is the string start (as after a non-letter).
    States are numbered in breadth-first order over the classes, so the table is
    deterministic."""
    start = _step(frozenset(), C_W)
    order, num = [start], {start: 0}
    nxt = []
    for s in order:
        row = []
        for c in range(N_CLASSES):
            t = _step(s, c)
            if t not in num:
                num[t] = len(order)
                order.append(t)
            row.append(num[t])
        nxt.append(row)
    return nxt, [_ban(s) for s in order]


STR_NEXT, STR_BAN = build_automaton()
STR_START = 0


def window_ids(vocab) -> list[int]:
    return [i for i, h in enumerate(vocab) if h.startswith(WINDOW_PAGE)]
