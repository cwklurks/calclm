"""Unit tests per TI-BASIC construct, run against both implementations (spec and C).

Features at position i are the state AFTER token i. Semantics follow the TI-OS behaviour
measured in tracker/probe_os.py (see README "Evidence from the emulator").
"""
from __future__ import annotations

from tracker import feats as F
from tracker.tests.conftest import EOS, INDEX, ids_of, raw, run

G = F.GROUP_INDEX


def cat(feat, group: str) -> int:
    return feat[0][G[group]]


def bag_bits(feat) -> set[int]:
    return {b for b in range(64) if (feat[1] >> b) & 1}


def hint(feat) -> set[str]:
    """Hinted next name characters, plus '$' for the END bit."""
    out = {F.NAME_CHARS[b] for b in range(F.HINT_BITS) if (feat[1] >> b) & 1}
    return out | ({"$"} if (feat[1] >> F.HINT_END) & 1 else set())


def last(impl, src: str, extra: list[int] | None = None):
    ids = ids_of(src) + (extra or [])
    return run(impl, ids)[-1]


def summary(impl, src: str, eos: bool = True) -> dict:
    run(impl, ids_of(src) + ([EOS] if eos else []))
    return impl.summary()


# ---- statement position, strings, parens --------------------------------------------------

def test_bos_state(impl):
    f = run(impl, [1])[0]
    assert f[0] == (F.CTX_START, 0, 0, F.LIT_NONE, F.MENU_NONE, F.BLK_NONE, 0, F.IF_NONE, 0)
    assert f[1] == 0


def test_statement_start_after_newline_and_colon(impl):
    fs = run(impl, ids_of("1→A:Disp A\nA"))
    ctx = [cat(f, "ctx") for f in fs]
    # <bos> 1 → A : Disp A \n A
    assert ctx == [F.CTX_START, F.CTX_EXPR, F.CTX_STORE, F.CTX_EXPR, F.CTX_START, F.CTX_EXPR,
                   F.CTX_EXPR, F.CTX_START, F.CTX_EXPR]


def test_string_contents_are_literal(impl):
    # ':' 'Lbl' 'Goto' '(' inside a string change nothing; the closing quote ends it
    ids = ids_of('Disp "') + raw("3E", "D6", "41", "D7", "42", "10") + raw("2A")
    fs = run(impl, ids)
    assert [cat(f, "ctx") for f in fs[2:-1]] == [F.CTX_STRING] * 7
    assert cat(fs[-1], "ctx") == F.CTX_EXPR
    assert all(cat(f, "paren") == 0 for f in fs)
    assert all(cat(f, "unres") == 0 for f in fs)


def test_optional_closing_quote_newline_ends_string(impl):
    fs = run(impl, ids_of('Disp "HI\nGoto A\n'))
    nl = fs[5]  # <bos> Disp " H I \n
    assert cat(nl, "ctx") == F.CTX_START
    assert cat(fs[-1], "unres") == 1      # the Goto after the string still counts


def test_store_closes_string_and_parens(impl):
    fs = run(impl, ids_of('"ABC→Str1'))
    assert cat(fs[-2], "ctx") == F.CTX_STORE
    fs = run(impl, ids_of("2(3(4→A"))
    assert cat(fs[-3], "paren") == 2
    assert cat(fs[-2], "paren") == 0 and cat(fs[-2], "ctx") == F.CTX_STORE


def test_optional_closing_parens_reset_at_line_end(impl):
    fs = run(impl, ids_of("Disp (1+(2+(3+(4\nDisp 1"))
    assert max(cat(f, "paren") for f in fs) == 3          # bucket 3+
    assert cat(fs[ids_of("Disp (1+(2+(3+(4\n").__len__() - 1], "paren") == 0
    assert cat(fs[-1], "paren") == 0


def test_function_tokens_open_parens(impl):
    fs = run(impl, ids_of("Output(1,sub(Str1,1,1)"))
    assert cat(fs[2], "paren") == 1 and cat(fs[5], "paren") == 2
    assert cat(fs[-1], "paren") == 1


def test_list_and_matrix_literals(impl):
    fs = run(impl, ids_of("{1,2}→L₁\n[[1,2][3,4]]→[A]"))
    assert cat(fs[1], "lit") == F.LIT_BRACE
    assert cat(fs[5], "lit") == F.LIT_NONE
    i = len(ids_of("{1,2}→L₁\n"))
    assert cat(fs[i], "lit") == F.LIT_BRACKET


# ---- blocks ---------------------------------------------------------------------------------

def test_if_then_end(impl):
    fs = run(impl, ids_of("If A\nThen\nDisp 1\nEnd\n"))
    then = fs[4]  # <bos> If A \n Then
    assert cat(then, "blk_top") == F.BLK_IF and cat(then, "blk_depth") == 1
    assert cat(fs[-2], "blk_depth") == 0 and cat(fs[-2], "blk_top") == F.BLK_NONE


def test_if_cond_then_same_line(impl):
    fs = run(impl, ids_of("If A:Then:Disp 1:End"))
    assert cat(fs[4], "blk_top") == F.BLK_IF
    assert cat(fs[-1], "blk_depth") == 0


def test_else(impl):
    fs = run(impl, ids_of("If A\nThen\n1\nElse\n2\nEnd"))
    assert cat(fs[8], "blk_top") == F.BLK_ELSE and cat(fs[8], "blk_depth") == 1
    assert cat(fs[-1], "blk_depth") == 0
    assert summary(impl, "If A\nThen\n1\nElse\n2\nEnd")["n_bad"] == 0


def test_else_without_if_then_is_bad(impl):
    s = summary(impl, "Else\nDisp 1")
    assert s["n_bad"] == 1 and s["first_bad_pos"] == 1
    s = summary(impl, "For(I,1,2\nElse\nEnd")
    assert s["n_bad"] == 1 and s["depth"] == 0


def test_nested_loops(impl):
    src = "While 1\nFor(I,1,5\nRepeat K\nEnd\nEnd\nEnd"
    fs = run(impl, ids_of(src))
    tops = [(cat(f, "blk_top"), cat(f, "blk_depth")) for f in fs]
    assert (F.BLK_WHILE, 1) in tops and (F.BLK_FOR, 2) in tops and (F.BLK_REPEAT, 3) in tops
    assert tops[-1] == (F.BLK_NONE, 0)
    # after the first End the innermost open block is For( again
    first_end = ids_of(src).index(INDEX["D4"])
    assert tops[first_end] == (F.BLK_FOR, 2)


def test_for_counts_as_paren(impl):
    fs = run(impl, ids_of("For(I,1,5"))
    assert cat(fs[1], "paren") == 1 and cat(fs[1], "blk_top") == F.BLK_FOR


def test_depth_bucket_and_deep_stack(impl):
    src = "\n".join(["While 1"] * 40)
    fs = run(impl, ids_of(src))
    assert cat(fs[-1], "blk_depth") == 4 and cat(fs[-1], "blk_top") == F.BLK_DEEP
    src2 = src + "\n" + "\n".join(["End"] * 8)
    fs = run(impl, ids_of(src2))
    assert cat(fs[-1], "blk_top") == F.BLK_WHILE          # depth 32: types known again
    assert summary(impl, src2)["depth"] == 32


def test_extra_end_counted(impl):
    s = summary(impl, "Disp 1\nEnd\nDisp 2")
    assert s["n_extra_end"] == 1 and s["depth"] == 0


def test_unclosed_blocks_in_summary(impl):
    s = summary(impl, "If A\nThen\nWhile 1\nFor(I,1,2")
    assert s["depth"] == 3


def test_openers_only_at_statement_start(impl):
    s = summary(impl, 'Disp "While\nDisp 1')
    assert s["depth"] == 0
    fs = run(impl, ids_of("If 1:While 0"))
    assert cat(fs[-1], "blk_top") == F.BLK_WHILE


def test_stray_then_is_bad_but_opens_block(impl):
    s = summary(impl, "Then\nDisp 1\nEnd")
    assert s["n_bad"] == 1 and s["depth"] == 0 and s["n_extra_end"] == 0
    s = summary(impl, "If 1\nDisp 1\nThen\nEnd")     # THENLATE probe: SYNTAX on the OS
    assert s["n_bad"] == 1


# ---- single-line If ----------------------------------------------------------------------

def test_single_line_if_states(impl):
    fs = run(impl, ids_of("If A\nDisp 1\nDisp 2"))
    st = [cat(f, "if_state") for f in fs]
    # <bos> If A \n Disp 1 \n Disp 2
    assert st == [F.IF_NONE, F.IF_COND, F.IF_COND, F.IF_AFTER, F.IF_BODY, F.IF_BODY, F.IF_NONE,
                  F.IF_NONE, F.IF_NONE]


def test_if_then_clears_pending(impl):
    fs = run(impl, ids_of("If A:Then"))
    assert cat(fs[-1], "if_state") == F.IF_NONE and cat(fs[-1], "blk_top") == F.BLK_IF


def test_empty_statement_is_the_if_body(impl):
    # IFEMPTYCO / IFNLTHEN probes: an empty statement is the skipped one, so Then is stray
    fs = run(impl, ids_of("If A::Then"))
    assert cat(fs[4], "if_state") == F.IF_NONE
    assert summary(impl, "If A:\nThen\nEnd")["n_bad"] == 1


def test_nested_single_line_ifs(impl):
    fs = run(impl, ids_of("If A:If B:Disp 1"))
    st = [cat(f, "if_state") for f in fs]
    assert st[4] == F.IF_COND and st[6] == F.IF_AFTER and st[7] == F.IF_BODY


def test_is_ds_skip_like_if(impl):
    fs = run(impl, ids_of("IS>(A,5\nDisp 1"))
    assert cat(fs[1], "if_state") == F.IF_COND and cat(fs[1], "paren") == 1
    assert cat(fs[5], "if_state") == F.IF_AFTER


def test_single_line_if_body_may_open_block(impl):
    fs = run(impl, ids_of("If A:For(I,1,2"))
    assert cat(fs[-1], "if_state") == F.IF_BODY and cat(fs[-1], "blk_top") == F.BLK_FOR


# ---- labels -------------------------------------------------------------------------------

def test_goto_registers_on_terminator(impl):
    fs = run(impl, ids_of("Goto AB\n"))
    assert [cat(f, "ctx") for f in fs[1:4]] == [F.CTX_REF] * 3
    assert [cat(f, "name_len") for f in fs[1:4]] == [0, 1, 2]
    assert cat(fs[3], "unres") == 0 and cat(fs[4], "unres") == 1


def test_lbl_resolves(impl):
    fs = run(impl, ids_of("Goto AB\nGoto C\nLbl AB\nLbl C\n"))
    assert cat(fs[7], "unres") == 2
    assert cat(fs[11], "unres") == 1 and cat(fs[-1], "unres") == 0


def test_back_jump_never_unresolved(impl):
    fs = run(impl, ids_of("Lbl A\nGoto A\n"))
    assert all(cat(f, "unres") == 0 for f in fs)


def test_unres_bucket(impl):
    fs = run(impl, ids_of("Goto A\nGoto B\nGoto C\nGoto A\n"))
    assert cat(fs[-1], "unres") == 2
    assert summary(impl, "Goto A\nGoto B\nGoto C\nGoto A\n")["n_unres"] == 3


def test_label_hint_for_lbl(impl):
    fs = run(impl, ids_of("Goto Q\nGoto R5\nGoto 7\nLbl "))
    assert cat(fs[-1], "ctx") == F.CTX_LBL
    assert hint(fs[-1]) == {"Q", "R", "7"}
    fs = run(impl, ids_of("Goto Q\nGoto R5\nGoto R\nLbl R"))
    assert hint(fs[-1]) == {"5", "$"}
    fs = run(impl, ids_of("Goto Q\nLbl Q"))
    assert hint(fs[-1]) == {"$"}


def test_label_hint_for_goto_lists_defined(impl):
    fs = run(impl, ids_of("Lbl A1\nLbl B\nGoto Z\nGoto "))
    assert cat(fs[-1], "ctx") == F.CTX_REF and hint(fs[-1]) == {"A", "B"}
    fs = run(impl, ids_of("Lbl A1\nLbl B\nGoto A"))
    assert hint(fs[-1]) == {"1"}


def test_hint_only_in_name_context(impl):
    fs = run(impl, ids_of("Goto Q\nDisp 1"))
    assert all(f[1] & ((1 << 38) - 1) == 0 for f in fs[3:])


def test_two_char_labels_digits_theta(impl):
    s = summary(impl, "Goto 10\nGoto θ\nLbl 10\nLbl θ\n")
    assert s["n_unres"] == 0 and s["n_bad"] == 0


def test_malformed_label_names(impl):
    # LBL3, GOTO3, LBLSPC, GOTOSPC, LBLEMPTY, GOTOEMPTY probes: all ERR:SYNTAX
    for src in ("Lbl ABC\n", "Goto ABC\n", "Lbl \n", "Goto \n"):
        s = summary(impl, src)
        assert s["n_bad"] == 1 and s["n_unres"] == 0, src
    s = summary(impl, "Goto A\nLbl ABC\n")
    assert s["n_unres"] == 1                     # a malformed Lbl defines nothing
    ids = ids_of("Goto A\nLbl A") + raw("29", "3F")  # Lbl A<space>
    run(impl, ids + [EOS])
    assert impl.summary()["n_unres"] == 1 and impl.summary()["n_bad"] == 1


def test_label_at_eos_registers(impl):
    run(impl, ids_of("Lbl A\nGoto B") + [EOS])
    assert impl.summary()["n_unres"] == 1


def test_lbl_goto_mid_statement_ignored(impl):
    # LBLMID / GOTOMID probes: not a label definition, not a reference (and bad)
    s = summary(impl, "Goto A\nDisp 2Lbl A")
    assert s["n_unres"] == 1 and s["n_bad"] == 1
    s = summary(impl, "Disp 1Goto A")
    assert s["n_unres"] == 0 and s["n_bad"] == 1


def test_label_after_colon_and_in_if_body(impl):
    s = summary(impl, "Goto A\nGoto B\n1:Lbl A\nIf 0:Lbl B")
    assert s["n_unres"] == 0


def test_label_in_string_is_not_a_label(impl):
    ids = ids_of("Goto A\n") + raw("2A", "D6", "41", "3F")
    run(impl, ids + [EOS])
    assert impl.summary()["n_unres"] == 1


def test_menu_labels(impl):
    src = 'Menu("T","A",A,"B",BC,"Q",Q'
    fs = run(impl, ids_of(src))
    menu = [cat(f, "menu") for f in fs]
    ids = ids_of(src)
    first_label = ids.index(INDEX["41"], 8)  # A after the 3rd comma
    assert F.MENU_TITLE in menu and F.MENU_TEXT in menu
    assert menu[first_label] == F.MENU_LABEL and cat(fs[first_label], "ctx") == F.CTX_REF
    run(impl, ids + raw("11") + [EOS])
    s = impl.summary()
    assert s["n_unres"] == 3 and s["n_bad"] == 0


def test_menu_label_resolution_and_terminators(impl):
    s = summary(impl, 'Menu("T","A",A,"B",B)\nLbl A\nLbl B')
    assert s["n_unres"] == 0
    s = summary(impl, 'Menu("T","A",A\nLbl A')          # no closing paren
    assert s["n_unres"] == 0 and s["n_bad"] == 0
    s = summary(impl, 'Menu("T","A",A )')                # MENUSPC probe: SYNTAX
    assert s["n_bad"] == 1 and s["n_unres"] == 0


def test_menu_ends_with_its_paren(impl):
    fs = run(impl, ids_of('Menu("T","A",A)+1'))
    assert cat(fs[-1], "menu") == F.MENU_NONE


# ---- variables ----------------------------------------------------------------------------

def str_bit(k):
    return F.BAG_STR + k


def mat_bit(k):
    return F.BAG_MAT + k


def test_string_assignment_bag(impl):
    f = last(impl, '"HI"→Str1\n')
    assert str_bit(0) in bag_bits(f)
    f = last(impl, '"HI"→Str0\n')
    assert str_bit(9) in bag_bits(f)


def test_string_read_before_assign(impl):
    s = summary(impl, "Disp Str1")
    assert s["n_undef"] == 1 and s["first_undef_pos"] == 2
    s = summary(impl, '"A"→Str1\nDisp Str1')
    assert s["n_undef"] == 0


def test_input_prompt_define(impl):
    assert summary(impl, 'Input "NAME",Str1\nDisp Str1')["n_undef"] == 0
    assert summary(impl, "Input Str1\nDisp Str1")["n_undef"] == 0
    assert summary(impl, "Prompt Str2\nDisp Str2")["n_undef"] == 0
    assert summary(impl, "Input Str1,A")["n_undef"] == 1     # Str1 is the prompt here
    assert summary(impl, "Prompt ʟAB\nDisp ʟAB")["n_undef"] == 0


def test_delvar_undefines_and_chains(impl):
    s = summary(impl, '"A"→Str1\nDelVar Str1\nDisp Str1')
    assert s["n_undef"] == 1
    f = run(impl, ids_of('"A"→Str1\nDelVar Str1'))[-1]
    assert str_bit(0) not in bag_bits(f)
    fs = run(impl, ids_of("DelVar ADisp A"))
    assert cat(fs[1], "ctx") == F.CTX_DELVAR
    assert cat(fs[2], "ctx") == F.CTX_START           # DVCHAIN probe: chained statement
    fs = run(impl, ids_of("If A:DelVar BWhile 1"))
    assert cat(fs[-1], "blk_top") == F.BLK_WHILE and cat(fs[-1], "if_state") == F.IF_NONE


def test_delvar_list_chains(impl):
    # DVLIST, DVLIST2, DVLIST3 probes (raw tokens: tivars reads "ʟABDisp" as one list name)
    ids = ids_of("{1}→ʟAB\n") + raw("BB54", "EB", "41", "42", "DE", "35")
    fs = run(impl, ids + [EOS])
    assert impl.summary()["n_bad"] == 0
    assert cat(fs[-4], "ctx") == F.CTX_LIST and cat(fs[-3], "ctx") == F.CTX_EXPR  # Disp
    ids = ids_of("{1}→ʟA\n{2}→ʟB\n") + raw("BB54", "EB", "41", "BB54", "EB", "42", "3F", "DE",
                                             "EB", "42")
    run(impl, ids + [EOS])
    assert impl.summary()["n_undef"] == 1 and impl.summary()["n_bad"] == 0
    fs = run(impl, ids_of("DelVar ʟA") + raw("D1", "31"))            # DelVar ʟAWhile 1
    assert cat(fs[-1], "blk_top") == F.BLK_WHILE
    s = summary(impl, "{1}→ʟAB\nDelVar ʟAB:Disp ʟAB")
    assert s["n_undef"] == 1 and s["n_bad"] == 0


def test_custom_lists(impl):
    assert summary(impl, "Disp ʟAB")["n_undef"] == 1
    assert summary(impl, "{1,2}→ʟAB\nDisp ʟAB")["n_undef"] == 0
    assert summary(impl, "5→ʟAB(1)\nDisp ʟAB")["n_undef"] == 0       # LSTELEM
    assert summary(impl, "3→dim(ʟAB)\nDisp ʟAB")["n_undef"] == 0     # LSTDIM
    assert summary(impl, "SetUpEditor ʟAB\nDisp ʟAB")["n_undef"] == 0
    assert summary(impl, "SetUpEditor AB,CD\nDisp ʟCD")["n_undef"] == 0
    assert summary(impl, "ClrList ʟAB")["n_undef"] == 1              # LSTCLR
    assert summary(impl, "Disp dim(ʟAB)")["n_undef"] == 1            # LSTDIMRD
    assert summary(impl, "ʟAB+1→ʟAB\nDisp ʟAB")["n_undef"] == 1      # LSTREADST
    assert summary(impl, "{1}→AB\nDisp ʟAB")["n_undef"] == 0         # LSTNOL
    assert summary(impl, "{1}→A\nDisp ʟA")["n_undef"] == 0           # LSTNOL1
    assert summary(impl, "5→A\nDisp ʟA")["n_undef"] == 1
    assert summary(impl, "[[1,2]]→[A]\nMatr►list([A],ʟAB\nDisp ʟAB")["n_undef"] == 0
    assert summary(impl, "Disp L₁")["n_undef"] == 0


def test_list_name_context_and_hint(impl):
    fs = run(impl, ids_of("{1}→ʟSCORE\n{2}→ʟSAVE\nDisp ʟS"))
    assert cat(fs[-2], "ctx") == F.CTX_LIST and cat(fs[-1], "name_len") == 1
    assert hint(fs[-1]) == {"C", "A"}
    fs = run(impl, ids_of("{1}→ʟAB\nDisp ʟAB"))
    assert hint(fs[-1]) == {"$"}


def test_list_bucket_bits(impl):
    f = last(impl, "{1}→ʟHS\n")
    assert F.BAG_LIST + F.list_bucket("HS") in bag_bits(f)
    f = last(impl, "{1}→ʟHS\nDelVar ʟHS\n")
    assert not any(F.BAG_LIST <= b < 64 for b in bag_bits(f))


def test_list_name_too_long_is_bad(impl):
    assert summary(impl, "{1}→ʟABCDEF")["n_bad"] == 1                # LST6


def test_matrices(impl):
    assert summary(impl, "Disp [A]")["n_undef"] == 1
    assert summary(impl, "[[1]]→[A]\nDisp [A]")["n_undef"] == 0
    assert summary(impl, "{2,2}→dim([A])\nDisp [A]")["n_undef"] == 0
    assert summary(impl, "5→[A](1,1)")["n_undef"] == 1               # MATELEM
    assert summary(impl, "Fill(0,[A])")["n_undef"] == 1              # MATFILL
    assert summary(impl, "{1,2}→L₁\nList►matr(L₁,[A]\nDisp [A]")["n_undef"] == 0
    f = last(impl, "[[1]]→[C]\n")
    assert mat_bit(2) in bag_bits(f)


def test_matrix_store_resolves_on_next_token(impl):
    fs = run(impl, ids_of("[[1]]→[A]\n"))
    assert mat_bit(0) not in bag_bits(fs[-2])          # [A]: could still be an element store
    assert mat_bit(0) in bag_bits(fs[-1])


def test_equ_string_defines(impl):
    assert summary(impl, '"X"→Y₁\nEqu►String(Y₁,Str1\nDisp Str1')["n_undef"] == 0


def test_pics_and_gdbs(impl):
    assert summary(impl, "RecallPic 1")["n_undef"] == 1
    assert summary(impl, "StorePic 1\nRecallPic Pic1")["n_undef"] == 0
    assert summary(impl, "StorePic 0\nRecallPic Pic0")["n_undef"] == 0
    assert summary(impl, "StorePic 1\nDelVar Pic1RecallPic 1")["n_undef"] == 1
    assert summary(impl, "RecallGDB 2")["n_undef"] == 1
    assert summary(impl, "StoreGDB 2\nRecallGDB GDB2")["n_undef"] == 0


def test_prgm_calls_counted(impl):
    fs = run(impl, ids_of("prgmABC\nDisp 1"))
    assert cat(fs[1], "ctx") == F.CTX_PRGM and cat(fs[2], "ctx") == F.CTX_PRGM
    s = summary(impl, "prgmABC\nprgmD")
    assert s["n_prgm"] == 2


def test_eos_finishes_statement(impl):
    fs = run(impl, ids_of("Goto A") + [EOS])
    assert cat(fs[-1], "unres") == 1 and cat(fs[-1], "ctx") == F.CTX_START


def test_pad_is_ignored(impl):
    a = run(impl, ids_of("If A\nThen"))
    b = run(impl, ids_of("If A\n") + [0] + raw("CF"))
    assert a[-1] == b[-1]


def test_unknown_ids_are_ordinary_tokens(impl):
    fs = run(impl, ids_of("Disp ") + [3, 5000])
    assert cat(fs[-1], "ctx") == F.CTX_EXPR


def test_archive_reads_its_operand(impl):
    # UNARCPIC, UNARCL, ARCSTR, ARCMAT probes: Archive/UnArchive of a missing var is UNDEFINED
    assert summary(impl, "UnArchive Pic1")["n_undef"] == 1
    assert summary(impl, "UnArchive ʟAB")["n_undef"] == 1
    assert summary(impl, "Archive Str1")["n_undef"] == 1
    assert summary(impl, "Archive [A]")["n_undef"] == 1
    assert summary(impl, "StorePic 1\nArchive Pic1")["n_undef"] == 0
    assert summary(impl, "{1}→ʟAB\nArchive ʟAB\nUnArchive ʟAB")["n_undef"] == 0


def test_statistics_variables(impl):
    # ST62xx probes: every stat result variable except 𝑛 (6221) is UNDEFINED in a clean state
    run(impl, ids_of("Disp ") + raw("6212") + [EOS])                  # r
    assert impl.summary()["n_undef"] == 1
    run(impl, ids_of("Disp ") + raw("6221") + [EOS])                  # sequence 𝑛
    assert impl.summary()["n_undef"] == 0
    run(impl, ids_of("{1,2}→L₁\n1-Var Stats L₁\nDisp ") + raw("6203") + [EOS])   # x̄
    assert impl.summary()["n_undef"] == 0
    run(impl, ids_of("Disp ") + raw("6203") + ids_of("\nLinReg(ax+b) L₁,L₂", bos=False) + [EOS])
    assert impl.summary()["n_undef"] == 1                             # read before the stats


def test_paren_stat_commands_open_parens(impl):
    fs = run(impl, ids_of("Z-Test(0,1,L₁"))
    assert cat(fs[1], "paren") == 1


def test_program_call_statement_syntax(impl):
    # PRGMSPC, PRGM9, PRGMPLUS, PRGMEMPTY, PRGMLOW probes: ERR:SYNTAX; PRGMOK: fine
    for src in ("prgmABC Disp 1", "prgmABCDEFGHI", "prgmAB+1", "prgm\nDisp 1"):
        assert summary(impl, src)["n_bad"] == 1, src
    run(impl, ids_of("prgmAB") + raw("BBB4", "3F") + [EOS])
    assert impl.summary()["n_bad"] == 1
    s = summary(impl, "prgmAB:Disp 1\nprgmABCDEFGH")
    assert s["n_bad"] == 0 and s["n_prgm"] == 2
    # inside an expression the OS says DATA TYPE or INVALID, not SYNTAX: no structure error
    assert summary(impl, "1+prgmAB")["n_bad"] == 0
    assert summary(impl, "Asm(prgmAB)")["n_bad"] == 0


def test_single_line_if_at_end_of_program_is_bad(impl):
    # IFEOF*, ISEOF probes: ERR:SYNTAX whatever the condition; THENEOF: fine
    for ids in (ids_of("Disp 1\nIf 1"), ids_of("Disp 1\nIf 1") + raw("3F"),
                ids_of("Disp 1\nIf 0") + raw("3E"), ids_of("0→A\nIS>(A,5)")):
        run(impl, ids + [EOS])
        assert impl.summary()["n_bad"] == 1
    for src in ("If 1\nThen", "If 1:Disp 1", "If 1\nDisp 1\n"):
        assert summary(impl, src)["n_bad"] == 0, src
