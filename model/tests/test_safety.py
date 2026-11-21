"""v2.4 safety masks (docs/agents/phase1-safety-masks.md): model/safety.py's token lists and
string automaton, and IntMask(level=2)'s three rules (credit lines in strings, calls to other
programs, persistent settings). Level 1 stays v2.3's rules."""
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

from model.tokens import BOS

ROOT = Path(__file__).resolve().parents[2]
VOCAB = json.loads((ROOT / "corpus" / "vocab.json").read_text())
IDX = {h: i for i, h in enumerate(VOCAB)}
VAL = ROOT / "data" / "tok" / "val.bin"


def ids(*hexes) -> list[int]:
    return [IDX[h] for h in hexes]


def text_ids(s: str) -> list[int]:
    """Uppercase letters, digits, space and simple punctuation as their one-byte tokens;
    lowercase letters as the BB lowercase tokens."""
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            k = ord(ch) - ord("a")
            out.append(IDX["BB%02X" % (0xB0 + k + (k >= 11))])
        else:
            out.append(IDX[{" ": "29", "(": "10", ")": "11", ":": "3E", ",": "2B", ".": "3A",
                            "?": "AF", "-": "71"}.get(ch, "%02X" % ord(ch))])
    return out


QUOTE, STORE, NL = IDX["2A"], IDX["04"], IDX["3F"]
DISP, INPUT, PROMPT, FOR = IDX["DE"], IDX["DC"], IDX["DD"], IDX["D3"]


def mask_after(prefix, level=2):
    from model.intmask import IntMask
    m = IntMask(VOCAB, level=level)
    m.reset()
    for t in [BOS, *prefix]:
        m.step(t)
    return m.banned()


# ---- lists ------------------------------------------------------------------------------------

def test_lists_resolve_in_the_vocab():
    from model import safety as S
    assert all(h in IDX for h in S.ALWAYS_HEX), [h for h in S.ALWAYS_HEX if h not in IDX]
    assert len(set(S.ALWAYS_HEX)) == len(S.ALWAYS_HEX)
    assert {"5F", "EF11", "EF12"} <= set(S.ALWAYS_HEX)                 # prgm OpenLib( ExecLib
    assert {"73", "69", "67", "68", "64", "65", "74", "75", "BB64"} <= set(S.ALWAYS_HEX)
    assert {"7E09", "7E08", "7E0B", "7E0D", "7E0C", "7E05", "7E04", "97", "96", "EA", "E9",
            "86", "9B"} <= set(S.ALWAYS_HEX)
    w = S.window_ids(VOCAB)
    assert len(w) == 57 and all(VOCAB[i].startswith("63") for i in w)
    for h in ("630A", "630B", "630C", "630D", "6302", "6303", "6336", "6326", "6327", "6338"):
        assert IDX[h] in w                    # Xmin Xmax Ymin Ymax Xscl Yscl Xres dX dY TraceStep
    # nothing that reads data or draws is on the always list
    for h in ("DE", "93", "A1", "9C", "85", "E1", "72", "AD", "5D00", "AA00", "D3", "EF67"):
        assert h not in S.ALWAYS_HEX


def test_letter_classes_follow_the_tokens_display_text():
    """Every vocab token whose display text has one ASCII letter, first (A, b, y₁, Y₁ᴛ), gets
    that letter's class (the model wrote B then y₁ for "BY" before this); ( and ) their own;
    everything else is W."""
    from tivars.models import TI_84PCE

    from model import safety as S
    sheet = TI_84PCE.tokens.bytes
    cls = S.str_classes(VOCAB)
    for i, h in enumerate(VOCAB):
        if h.startswith("<"):
            assert cls[i] == S.C_W
            continue
        d = sheet[bytes.fromhex(h)].langs["en"].display
        letters = [c for c in d if c.isascii() and c.isalpha()]
        if len(letters) == 1 and d[0] == letters[0]:          # A, b, y₁, Y₁ᴛ, r²: that letter
            assert cls[i] == S.LETTER_CLASS.get(d[0].lower(), S.C_L), (h, d)
        elif h == "10":
            assert cls[i] == S.C_LP
        elif h == "11":
            assert cls[i] == S.C_RP
        else:
            assert cls[i] == S.C_W, (h, d)


# ---- the string automaton against a regex oracle ------------------------------------------------

def _rep():
    from model import safety as S
    return {S.C_W: " ", S.C_L: "a", S.C_LP: "(", S.C_RP: ")", **{c: ch for ch, c in
                                                                 S.LETTER_CLASS.items()}}


ORACLE = [re.compile(r"(^|[^a-z])by$"), re.compile(r"copyr$"), re.compile(r"\(c\)$")]


def _oracle_bans(s: str) -> set[str]:
    return {ch for ch in _rep().values() if any(p.search(s + ch) for p in ORACLE)}


def _dfa_bans(s: str) -> set[str]:
    from model import safety as S
    rep = _rep()
    inv = {ch: c for c, ch in rep.items()}
    st = S.STR_START
    for ch in s:
        st = S.STR_NEXT[st][inv[ch]]
    return {rep[c] for c in rep if (S.STR_BAN[st] >> c) & 1}


def _norm(s: str) -> str:
    """Letters outside the pattern letters as 'a', other non-letters but ( ) as ' '."""
    return "".join(ch if ch in "bycopr()" else "a" if ch.isalpha() else " " for ch in s)


@pytest.mark.parametrize("s", ["", "b", " b", "ab", "(b", ")b", "by", "made b", "copy", "cop",
                               "ccopy", "copcopy", "(", "(c", "((c", "x(c", "(c)", "(c) b",
                               "copy b", "bcopy", "abby", "(cop", "copy(c"])
def test_automaton_bans_what_the_oracle_bans(s):
    assert _dfa_bans(_norm(s)) == _oracle_bans(_norm(s)), s


def test_automaton_matches_the_oracle_on_random_strings():
    alphabet = list(_rep().values())
    rng = random.Random(7)
    for _ in range(20000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))
        assert _dfa_bans(s) == _oracle_bans(s), s


def test_automaton_table_is_small():
    from model import safety as S
    assert len(S.STR_NEXT) <= 32 and all(len(r) == S.N_CLASSES for r in S.STR_NEXT)
    assert S.N_CLASSES <= 16 and len(S.STR_BAN) == len(S.STR_NEXT)


# ---- IntMask level 2 ------------------------------------------------------------------------------

def test_credit_lines_in_strings():
    y, ly, r, rp = IDX["59"], IDX["BBC9"], IDX["52"], IDX["11"]
    assert mask_after([DISP, QUOTE, *text_ids("B")])[[y, ly]].all()        # "BY at the start
    assert mask_after([DISP, QUOTE, *text_ids("MADE b")])[[y, ly]].all()   # "MADE by
    assert mask_after([DISP, QUOTE, *text_ids("v1.0 B")])[y]
    assert not mask_after([DISP, QUOTE, *text_ids("AB")])[y]               # ABY: not a word start
    assert mask_after([DISP, QUOTE, *text_ids("COPY")])[r]
    assert mask_after([DISP, QUOTE, *text_ids("Copy")])[[r, IDX["BBC2"]]].all()
    assert mask_after([DISP, QUOTE, *text_ids("(C")])[rp]
    assert mask_after([DISP, QUOTE, *text_ids("2009 (c")])[rp]
    assert not mask_after([DISP, QUOTE, *text_ids("(CM")])[rp]
    # outside strings nothing of this applies: B*Y, (C) as arithmetic
    assert not mask_after([*text_ids("B")])[y]
    assert not mask_after([DISP, *text_ids("(C")])[rp]
    # a closed string resets the automaton
    assert not mask_after([DISP, QUOTE, *text_ids("COPY"), QUOTE, *text_ids("+R")])[r]
    # level 1 never bans these
    assert not mask_after([DISP, QUOTE, *text_ids("B")], level=1)[y]


def test_only_the_banned_class_goes():
    from model import safety as S
    b = mask_after([DISP, QUOTE, *text_ids("B")])
    base = mask_after([DISP, QUOTE, *text_ids("B")], level=1)
    cls = S.str_classes(VOCAB)
    extra = np.flatnonzero(b & ~base)
    always = set(ids(*S.ALWAYS_HEX))
    assert {int(i) for i in extra} - always == {i for i in range(2, len(VOCAB))
                                                if cls[i] == S.LETTER_CLASS["y"]}


def test_calls_and_settings_are_always_banned_at_level_2():
    from model import safety as S
    for prefix in ([], [DISP, QUOTE], [*text_ids("If A")], [*text_ids("A"), STORE]):
        b2, b1 = mask_after(prefix), mask_after(prefix, level=1)
        assert b2[ids(*S.ALWAYS_HEX)].all(), prefix
        assert not b1[ids(*S.ALWAYS_HEX)].any(), prefix


def test_window_variables_cannot_be_assigned():
    from model import safety as S
    w = S.window_ids(VOCAB)
    xmin, xmax = IDX["630A"], IDX["630B"]
    assert mask_after([*text_ids("5"), STORE])[w].all()                    # 5->Xmin
    assert mask_after([DISP, QUOTE, *text_ids("A"), STORE])[w].all()       # "A"->...
    assert mask_after([PROMPT])[w].all()                                   # Prompt Xmin
    assert mask_after([PROMPT, *text_ids("A,")])[w].all()                  # Prompt A,Xmin
    assert mask_after([INPUT, QUOTE, *text_ids("X"), QUOTE, *text_ids(",")])[w].all()
    assert mask_after([FOR])[w].all()                                      # For(Xmin,...
    assert mask_after(ids("DA"))[w].all() and mask_after(ids("DB"))[w].all()   # IS>( DS<(
    # reads stay allowed
    assert not mask_after([DISP])[w].any()                                 # Disp Xmax
    assert not mask_after([DISP, xmax, IDX["71"]])[w].any()                # Disp Xmax-Xmin
    assert not mask_after([FOR, *text_ids("I,")])[w].any()                 # For(I,Xmin,...
    assert not mask_after([*text_ids("A"), STORE, *text_ids("B"), NL])[w].any()
    assert not mask_after([PROMPT, *text_ids("A"), NL])[w].any()           # next statement
    assert not mask_after([*text_ids("5"), STORE], level=1)[[xmin, xmax]].any()


def test_level_2_keeps_some_id_allowed_and_the_v23_bans():
    """Level 2 bans a superset of level 1 on real programs, and never empties the vocab."""
    if not VAL.exists():
        pytest.skip("data/tok/val.bin not present")
    from model.intmask import IntMask
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)[:40]
    m1, m2 = IntMask(VOCAB, level=1), IntMask(VOCAB, level=2)
    n = 0
    for a, e in zip(starts, starts[1:]):
        m1.reset()
        m2.reset()
        for t in stream[a:min(e, a + 300)]:
            m1.step(int(t))
            m2.step(int(t))
            b1, b2 = m1.banned(), m2.banned()
            assert not (b1 & ~b2).any()
            allowed = ~b2
            allowed[[0, 1, 3]] = False
            assert allowed.any()
            n += 1
    assert n > 3000


def test_unknown_levels_are_refused():
    from model.intmask import IntMask
    for bad in (0, 4):
        with pytest.raises(ValueError):
            IntMask(VOCAB, level=bad)


def test_mask_ids_carry_the_level_2_tables():
    from model import safety as S
    from model.intmask import mask_ids
    m = mask_ids(VOCAB)
    assert m["always"] == ids(*S.ALWAYS_HEX)
    assert m["window"] == S.window_ids(VOCAB)
    assert m["loopvar"] == ids("D3", "DA", "DB") and m["io"] == ids("DC", "DD")
    sid, scls = m["str_ids"], m["str_cls"]
    assert sid == sorted(sid) and len(sid) == len(scls) and S.C_W not in scls
    cls = S.str_classes(VOCAB)
    assert {i: c for i, c in zip(sid, scls)} == {i: int(c) for i, c in enumerate(cls)
                                                 if c != S.C_W}


# ---- review fixes: concatenated strings, table invariants ---------------------------------------

PLUS = IDX["70"]


def test_credit_lines_split_by_concatenation_are_caught():
    """"B"+"Y...": the automaton carries its state across a closing quote, + and a reopening
    quote, so the second string continues the first."""
    y, r = IDX["59"], IDX["52"]
    assert mask_after([DISP, QUOTE, *text_ids("B"), QUOTE, PLUS, QUOTE])[y]
    assert mask_after([DISP, QUOTE, *text_ids("MADE b"), QUOTE, PLUS, QUOTE])[IDX["BBC9"]]
    assert mask_after([DISP, QUOTE, *text_ids("COP"), QUOTE, PLUS, QUOTE, *text_ids("Y")])[r]
    # anything else between the strings breaks the chain
    assert not mask_after([DISP, QUOTE, *text_ids("B"), QUOTE, NL, QUOTE])[y]
    assert not mask_after([DISP, QUOTE, *text_ids("B"), QUOTE, *text_ids(","), QUOTE])[y]
    assert not mask_after([DISP, QUOTE, *text_ids("B"), QUOTE, PLUS, *text_ids("A+"), QUOTE])[y]
    assert not mask_after([DISP, QUOTE, *text_ids("B"), STORE, PLUS, QUOTE])[y]


def test_the_automaton_never_bans_class_w():
    """The C core walks only the listed (non-W) ids when a string state bans something."""
    from model import safety as S
    assert all(not (bits >> S.C_W) & 1 for bits in S.STR_BAN)


def test_letter_plus_subscript_tokens_complete_by():
    y1, Y1 = IDX["621E"], IDX["5E10"]
    assert mask_after([DISP, QUOTE, *text_ids("B")])[[y1, Y1]].all()


# ---- level 3: the destructive and safety bans alone (the calculator's Masks row off) -------------

def test_level_3_is_the_safety_rules_without_the_v23_grammar_masks():
    """Level 3 (clm.h CLM_MASKS_SAFETY) bans what level 2 adds to level 1 and v2.3's
    data-destroying tokens (DelVar ... ClrList), but none of v2.3's grammar rules (End, Else,
    undefined reads, lists): the Masks row turns those off, never a destructive or safety ban."""
    from model import safety as S
    from model.intmask import SAFETY_HEX
    assert S.SAFETY_ONLY == 3
    for prefix in ([], [DISP, QUOTE], [*text_ids("If A")], [*text_ids("A"), STORE]):
        b3 = mask_after(prefix, level=3)
        assert b3[ids(*S.ALWAYS_HEX)].all(), prefix
        assert b3[ids(*SAFETY_HEX)].all(), prefix              # DelVar ... ClrList: always on
    w = S.window_ids(VOCAB)
    assert mask_after([*text_ids("5"), STORE], level=3)[w].all()
    assert mask_after([PROMPT, *text_ids("A,")], level=3)[w].all()
    assert not mask_after([DISP], level=3)[w].any()
    end = IDX["D4"]
    assert mask_after([], level=2)[end] and not mask_after([], level=3)[end]   # End, unmatched
    by = [DISP, QUOTE, *text_ids("MADE B")]
    assert mask_after(by, level=3)[IDX["59"]]                            # the Y of BY


def test_level_3_equals_level_2_minus_v23_on_real_programs():
    """At every position of val programs: level 3's bans are exactly the bans level 2 adds,
    i.e. level 2 = level 1 | level 3 (apart from the nothing-left fallback, never hit here),
    and level 3 includes v2.3's data-destroying bans."""
    if not VAL.exists():
        pytest.skip("data/tok/val.bin not present")
    from model.intmask import IntMask
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)[:30]
    from model.intmask import SAFETY_HEX
    ms = {lv: IntMask(VOCAB, level=lv) for lv in (1, 2, 3)}
    destroy = ids(*SAFETY_HEX)
    n = 0
    for a, e in zip(starts, starts[1:]):
        for m in ms.values():
            m.reset()
        for t in stream[a:min(e, a + 300)]:
            for m in ms.values():
                m.step(int(t))
            b1, b2, b3 = (ms[lv].banned() for lv in (1, 2, 3))
            assert np.array_equal(b2, b1 | b3)
            assert b3[destroy].all()
            n += 1
    assert n > 2000
