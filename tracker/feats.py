"""Feature vocabulary of the TI-BASIC program-state tracker (shared by spec.py and bind.py).

Every token yields the tracker state AFTER that token has been read: NCAT categorical ids
(one per group, each in 0..size-1) and a 64-bit bag (multi-hot set). See tracker/README.md.
These numbers are the interface; tracker/tracker.h defines the same values for C.
"""
from __future__ import annotations

# categorical groups, in output column order: (name, number of values)
GROUPS: tuple[tuple[str, int], ...] = (
    ("ctx", 9),        # lexical context for the next token
    ("name_len", 3),   # characters read of the name in progress: 0, 1, 2+
    ("paren", 4),      # open ( and NAME( depth: 0, 1, 2, 3+
    ("lit", 3),        # innermost literal: none, { list, [ matrix
    ("menu", 4),       # Menu( argument kind
    ("blk_top", 7),    # innermost open block
    ("blk_depth", 5),  # open blocks: 0, 1, 2, 3, 4+
    ("if_state", 4),   # single-line If tracking
    ("unres", 3),      # referenced-but-undefined labels: 0, 1, 2+
)
NCAT = len(GROUPS)
GROUP_INDEX = {name: i for i, (name, _) in enumerate(GROUPS)}

# ctx values
(CTX_START, CTX_EXPR, CTX_STRING, CTX_STORE, CTX_LBL, CTX_REF, CTX_LIST, CTX_PRGM,
 CTX_DELVAR) = range(9)
# lit values
LIT_NONE, LIT_BRACE, LIT_BRACKET = range(3)
# menu values
MENU_NONE, MENU_TITLE, MENU_TEXT, MENU_LABEL = range(4)
# blk_top values
BLK_NONE, BLK_IF, BLK_ELSE, BLK_FOR, BLK_WHILE, BLK_REPEAT, BLK_DEEP = range(7)
# if_state values
IF_NONE, IF_COND, IF_AFTER, IF_BODY = range(4)

# name characters (labels, list names, program names), in name-index order
NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789θ"
N_NAME = len(NAME_CHARS)          # 37

# bag bits
HINT_BITS = 37                    # bits 0..36: next-name-character hint (name index)
HINT_END = 37                     # bit 37: the name read so far is itself a candidate
BAG_STR = 38                      # bits 38..47: Str token AA00+k assigned (k = 0..9)
BAG_MAT = 48                      # bits 48..57: matrix token 5C00+k ([A]+k) assigned
BAG_LIST = 58                     # bits 58..63: a custom list in this hash bucket assigned
N_LIST_BUCKETS = 6
BAG_BITS = 64

MAX_BLOCKS = 32                   # block types kept; deeper blocks report BLK_DEEP
MAX_LISTS = 16                    # custom list names kept
LABEL_SPACE = N_NAME + N_NAME * N_NAME   # 1406 label names (1 or 2 characters)

CAT_OFFSETS: tuple[int, ...] = tuple(sum(n for _, n in GROUPS[:i]) for i in range(NCAT))
N_CAT_ROWS = sum(n for _, n in GROUPS)   # 42
N_ROWS = N_CAT_ROWS + BAG_BITS           # 106 embedding rows in all


def row_ids(cats, bag: int) -> list[int]:
    """Global embedding-row ids of one token's features: one row per categorical group,
    then N_CAT_ROWS + b for every set bag bit b."""
    rows = [CAT_OFFSETS[g] + int(v) for g, v in enumerate(cats)]
    rows += [N_CAT_ROWS + b for b in range(BAG_BITS) if (int(bag) >> b) & 1]
    return rows


def label_id(name: str) -> int:
    """Label name (1-2 name characters) -> 0..1405, the tracker's bitset index."""
    idx = [NAME_CHARS.index(ch) for ch in name]
    return idx[0] if len(idx) == 1 else N_NAME + idx[0] * N_NAME + idx[1]


def list_bucket(name: str) -> int:
    """Bucket (0..5) of a custom list name for the BAG_LIST bits: its first character's name
    index plus one, mod 6 (one table lookup on the calculator)."""
    return (NAME_CHARS.index(name[0]) + 1) % N_LIST_BUCKETS
