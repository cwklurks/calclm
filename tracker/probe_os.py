"""Probe TI-OS semantics that the tracker depends on, on the emulated TI-84 Plus CE.

Runs small hand-written programs through the eval harness (eval/run_programs.py, clean
state per program) and prints each outcome next to the tracker's assumption. The results are
recorded in tracker/README.md ("Evidence from the emulator").

    uv run python -m tracker.probe_os            # writes eval/runs/w2-probe/
"""
from __future__ import annotations

import sys
from pathlib import Path

from tivars import TIProgram

from eval import inputs, run_programs
from eval import emu as E

OUT = Path(__file__).resolve().parent.parent / "eval" / "runs" / "w2-probe"

# raw tokens for byte-level probes
LBL, GOTO, QUOTE, NL, SPACE, COLON = b"\xd6", b"\xd7", b"\x2a", b"\x3f", b"\x29", b"\x3e"
DISP = b"\xde"


def src(text: str) -> bytes:
    p = TIProgram(name="X")
    p.load_string(text)
    return p.data


# (name, body, what the tracker assumes)
PROBES: list[tuple[str, bytes, str]] = [
    ("LSTUNDEF", src("Disp ʟAB"), "read of unassigned custom list: ERR:UNDEFINED"),
    ("LSTSTORE", src("{1,2}→ʟAB\nDisp ʟAB"), "→ʟNAME assigns: OK"),
    ("LSTELEM", src("5→ʟAB(1)\nDisp ʟAB"), "element store to a missing list: ?"),
    ("LSTDIM", src("3→dim(ʟAB)\nDisp ʟAB"), "→dim(ʟNAME) creates: OK"),
    ("LSTSETUP", src("SetUpEditor ʟAB\nDisp dim(ʟAB)"), "SetUpEditor creates: ?"),
    ("LSTSETUP2", src("SetUpEditor AB\nDisp dim(ʟAB)"), "SetUpEditor without ʟ: ?"),
    ("LSTCLR", src("ClrList ʟAB\nDisp dim(ʟAB)"), "ClrList creates: ?"),
    ("LSTNOL", src("{1,2}→AB\nDisp ʟAB"), "store a list without ʟ: ?"),
    ("L1EXISTS", src("Disp dim(L₁)"), "L1-L6 exist in the clean state: OK"),
    ("LSTDIMRD", src("Disp dim(ʟAB)"), "dim( of missing list: ?"),
    ("LST6", src("{1}→ʟABCDEF\nDisp ʟABCDE"), "6th name char: ?"),
    ("STRUNDEF", src("Disp Str1"), "unassigned Str: ERR:UNDEFINED"),
    ("STRDEF", src("\"HI\"→Str1\nDisp Str1"), "OK"),
    ("STRINPUT", src("Input Str1\nDisp Str1"), "Input defines Str: OK"),
    ("STRPROMPT", src("Prompt Str1\nDisp Str1"), "Prompt Str: ?"),
    ("MATUNDEF", src("Disp [A]"), "unassigned matrix: ERR:UNDEFINED"),
    ("MATELEM", src("5→[A](1,1)\nDisp [A]"), "element store to missing matrix: ?"),
    ("MATDIM", src("{2,2}→dim([A])\nDisp [A]"), "→dim([A]) creates: OK"),
    ("MATFILL", src("Fill(0,[A])"), "Fill( on missing matrix: ?"),
    ("DVCHAIN", src("1→A\nDelVar ADisp A"), "DelVar chains without ':': OK, shows 0"),
    # raw bytes: tivars would read "ʟABDisp" as a list name plus lowercase letters
    ("DVLIST", src("{1}→ʟAB\n") + b"\xbb\x54\xeb\x41\x42" + DISP + b"5",
     "DelVar ʟNAME chains: OK"),
    ("DVLIST2", src("{1}→ʟA\n{2}→ʟB\n") + b"\xbb\x54\xeb\x41\xbb\x54\xeb\x42" + NL
     + DISP + b"\xeb\x42", "DelVar ʟA chains into DelVar ʟB: ERR:UNDEFINED at Disp ʟB"),
    ("DVLIST3", b"\xbb\x54\xeb\x4e\xe0\x31\x2b\x31\x2b" + QUOTE + b"X" + QUOTE + b"\x11",
     "DelVar ʟN chains into Output(: OK"),
    ("DVMISSING", src("DelVar ʟAB\nDisp 3"), "DelVar of a missing var: OK"),
    ("DVSTR", src("\"A\"→Str1\nDelVar Str1\nDisp Str1"), "DelVar undefines: ERR:UNDEFINED"),
    ("LBL3", src("Goto AB\nLbl ABC\nDisp 1"), "3-char Lbl: ?"),
    ("GOTO3", src("Goto ABC\nDisp 0\nLbl AB\nDisp 1"), "3-char Goto: ?"),
    ("LBLSPC", GOTO + b"A" + NL + DISP + b"0" + NL + LBL + b"A" + SPACE + NL + DISP + b"1",
     "Lbl A<space>: ?"),
    ("GOTOSPC", GOTO + b"A" + SPACE + NL + DISP + b"0" + NL + LBL + b"A" + NL + DISP + b"1",
     "Goto A<space>: ?"),
    ("LBLCOLON", src("Goto A\nDisp 0\n1:Lbl A\nDisp 1"), "Lbl after ':' is found: OK"),
    ("LBLINSTR", GOTO + b"A" + NL + QUOTE + LBL + b"A" + NL + DISP + b"1",
     "Lbl inside a string is not a label: ERR:LABEL"),
    ("LBLINSTR2", GOTO + b"A" + NL + DISP + QUOTE + b"1" + COLON + LBL + b"A" + NL + DISP + b"1",
     "Lbl after ':' inside a string: ERR:LABEL"),
    ("LBLEMPTY", src("Lbl \nDisp 1"), "empty Lbl name: ?"),
    ("LBLDUP", src("Lbl A\nLbl A\nDisp 1"), "duplicate Lbl: OK"),
    ("LBLMID", src("Goto A\nDisp 0\nDisp 2Lbl A\nDisp 1"), "Lbl in mid-statement: ?"),
    ("IFEMPTYNL", src("0→A\nIf 0\n\n1→A\nDisp A"), "empty line after If: shows 0 or 1?"),
    ("IFEMPTYCO", src("0→A\nIf 0::1→A\nDisp A"), "empty statement after If: shows 0 or 1?"),
    ("IFNOEND0", src("If 0\nThen\nDisp 1"), "unclosed false If-Then: ?"),
    ("IFNOEND1", src("If 1\nThen\nDisp 1"), "unclosed true If-Then: ?"),
    ("WHNOEND0", src("While 0\nDisp 1"), "unclosed false While: ?"),
    ("WHNOEND1", src("0→I\nWhile I<2\nI+1→I"), "unclosed true While: ?"),
    ("FORNOEND", src("For(I,1,3)\nDisp I"), "unclosed For: ?"),
    ("RPNOEND", src("Repeat 1\nDisp 4"), "unclosed Repeat: ?"),
    ("EXTRAEND", src("Disp 1\nEnd\nDisp 2"), "End with nothing open: ?"),
    ("ELSENOIF", src("Else\nDisp 1"), "Else with no If-Then: ?"),
    ("THENNOIF", src("Then\nDisp 1"), "Then with no If: ?"),
    ("ELSEFALSE", src("If 0\nThen\nDisp 1\nElse\nDisp 2"), "false If-Then-Else, no End: ?"),
    ("YEMPTY", src("Disp Y₁(2)"), "empty Y1: ?"),
    ("PICUNDEF", src("RecallPic 1"), "missing Pic: ERR:UNDEFINED"),
    ("PICDEF", src("StorePic 1\nRecallPic 1"), "OK"),
    ("GDBUNDEF", src("RecallGDB 1"), "missing GDB: ERR:UNDEFINED"),
    ("MENUHALF", src("Menu(\"T\",\"A\",A,\"B\",B)\nLbl A\nDisp 1"),
     "Menu( with one missing target, option 1 chosen: OK (checked only when chosen)"),
    ("MENUMISS", src("Menu(\"T\",\"A\",A,\"B\",B)\nLbl B\nDisp 1"),
     "Menu( option 1 target missing: ERR:LABEL"),
    ("STRSTORE", src("\"ABC→Str1\nDisp Str1"), "→ closes a string: OK"),
    ("PARENSTORE", src("2(3→A\nDisp A"), "→ closes parens: OK"),
    ("IFTHEN1L", src("If 1:Then:Disp 1:End"), "OK"),
    ("ISGT", src("0→A\nIS>(A,5)\nDisp 1\nDisp 2"), "IS>( skips like If: shows 1 and 2"),
    ("THENLATE", src("If 1\nDisp 1\nThen\nDisp 2\nEnd"), "Then not right after If: ?"),
    ("EQUSTR", src("\"X\"→Y₁\nEqu►String(Y₁,Str1\nDisp Str1"), "Equ►String( defines Str: OK"),
    ("LISTMATR", src("{1,2}→L₁\nList►matr(L₁,[A]\nDisp [A]"), "List►matr( defines matrix: OK"),
    ("MATRLIST", src("[[1,2]]→[A]\nMatr►list([A],ʟAB\nDisp ʟAB"), "Matr►list( defines list: OK"),
    # round 2
    ("LSTNOL1", src("{1,2}→A\nDisp ʟA"), "list stored to a 1-char name without ʟ: ?"),
    ("REALNOL2", src("5→AB\nDisp A"), "real stored to a 2-char name: ?"),
    ("LSTNOL5", src("{3}→ABCDE\nDisp ʟABCDE"), "5-char list name without ʟ: ?"),
    ("DVSTRCH", src("\"A\"→Str1\nDelVar Str1Disp 5"), "DelVar Str1 chains: ?"),
    ("DVMATCH", src("{1,1}→dim([A])\nDelVar [A]Disp 5"), "DelVar [A] chains: ?"),
    ("DVL1CH", src("DelVar L₁Disp 5"), "DelVar L1 chains: ?"),
    ("DVAB", src("{1}→ʟAB\nDelVar AB\nDisp ʟAB"), "DelVar AB (no ʟ) deletes list AB: ?"),
    ("DVABCH", src("1→A\n2→B\nDelVar ABDisp 5"), "DelVar AB then a command: ?"),
    ("DVLSTCO", src("{1}→ʟAB\nDelVar ʟAB:Disp ʟAB"), "DelVar ʟAB undefines: ERR:UNDEFINED"),
    ("SETUP2", src("SetUpEditor AB,CD\nDisp dim(ʟCD)"), "SetUpEditor creates every list: OK"),
    ("PROMPTL", src("Prompt ʟAB\nDisp ʟAB"), "Prompt ʟ: ?"),
    ("INPUTL", src("Input ʟAB\nDisp ʟAB"), "Input ʟ: ?"),
    ("GOTOEMPTY", src("Goto \nDisp 1"), "empty Goto name: ?"),
    ("LBLNUM", src("Goto 10\nDisp 0\nLbl 10\nDisp 1"), "2-digit label: OK"),
    ("LBLTHETA", src("Goto θ\nDisp 0\nLbl θ\nDisp 1"), "θ label: OK"),
    ("GOTOMID", src("Disp 1Goto A\nLbl A"), "Goto in mid-statement: ?"),
    ("IFNLTHEN", src("If 1:\nThen\nDisp 1\nEnd"), "If cond: then newline Then: ?"),
    ("MENUSPC", src("Menu(\"T\",\"A\",A )\nLbl A\nDisp 1"), "Menu( label with a space: ?"),
    ("MENUCLOSE", src("Menu(\"T\",\"A\",A\nLbl A\nDisp 1"), "Menu( without ')': OK"),
    ("LSTREADST", src("ʟAB+1→ʟAB"), "read before store in one statement: ERR:UNDEFINED"),
    ("LSTSORT", src("SortA(ʟAB"), "SortA( on missing list: ?"),
    ("LSTFILL", src("Fill(0,ʟAB"), "Fill( on missing list: ?"),
    ("STRELEM", src("Disp sub(Str1,1,1"), "sub( of missing Str: ERR:UNDEFINED"),
    ("PICSTNUM", src("StorePic 1\nRecallPic 1\nDisp 1"), "StorePic n then RecallPic n: OK"),
    ("PICPIC", src("StorePic Pic1\nRecallPic 1\nDisp 1"), "Pic1 token and number 1 agree: ?"),
    ("PICZERO", src("StorePic 0\nRecallPic Pic0\nDisp 1"), "Pic0 token and number 0: ?"),
    ("DVPIC", src("StorePic 1\nDelVar Pic1RecallPic 1"), "DelVar Pic1 undefines: ?"),
    # round 3: Archive/UnArchive, statistics variables
    ("UNARCPIC", src("UnArchive Pic1"), "UnArchive of a missing Pic: ERR:UNDEFINED"),
    ("UNARCL", src("UnArchive ʟAB"), "UnArchive of a missing list: ?"),
    ("ARCSTR", src("Archive Str1"), "Archive of a missing Str: ?"),
    ("ARCMAT", src("Archive [A]"), "Archive of a missing matrix: ?"),
    ("ARCLDEF", src("{1}→ʟAB\nArchive ʟAB\nUnArchive ʟAB\nDisp 1"), "OK"),
    ("STATAFT", src("{1,2}→L₁\n1-Var Stats L₁\nDisp x̄"), "x̄ after 1-Var Stats: OK"),
    ("STATR", src("{1,2}→L₁\n1-Var Stats L₁\nDisp Sx"), "Sx after 1-Var Stats: ?"),
    # round 4: program calls
    ("PRGMSPC", src("prgmABC Disp 1"), "prgm name then a space: ?"),
    ("PRGM9", src("prgmABCDEFGHI"), "9-character prgm name: ?"),
    ("PRGMPLUS", src("prgmAB+1"), "prgm name then an operator: ?"),
    ("PRGMOK", src("prgmAB:Disp 1"), "missing subprogram: ERR:UNDEFINED"),
    ("PRGMEMPTY", src("prgm\nDisp 1"), "empty prgm name: ?"),
    ("PRGMMID", src("1+prgmAB"), "prgm inside an expression: ?"),
    ("PRGMLOW", b"\x5f\x41\x42\xbb\xbb\x3f\xde\x31", "prgm name with lowercase: ?"),
    # round 5: a single-line If with nothing after it
    ("IFEOF1", src("Disp 1\nIf 1"), "true If at the end of the program: ?"),
    ("IFEOF0", src("Disp 1\nIf 0"), "false If at the end of the program: ?"),
    ("IFEOFNL", src("Disp 1\nIf 1") + NL, "If, newline, end: ?"),
    ("IFEOFNL0", src("Disp 1\nIf 0") + NL, "false If, newline, end: ?"),
    ("IFEOFCO", src("Disp 1\nIf 1") + COLON, "If, colon, end: ?"),
    ("ISEOF", src("0→A\nIS>(A,5)"), "IS>( at the end: ?"),
    ("THENEOF", src("If 1\nThen"), "If-Then at the end: ?"),
]
# every statistics-result variable (62xx) read in a clean state
STAT_HEX = ["6201", "6202", "6203", "6204", "6205", "6206", "6207", "6208", "6209", "620A",
            "620B", "620C", "620D", "620E", "620F", "6210", "6211", "6212", "6213", "6214",
            "6215", "6216", "6217", "6218", "6219", "621A", "621B", "621C", "621D", "621E",
            "621F", "6220", "6221", "6222", "6223", "6224", "6225", "6226", "6227", "6228",
            "6229", "622A", "622B", "622C", "622D", "622E", "622F", "6230", "6231", "6232",
            "6233", "6234", "6235", "6236", "6238", "6239", "623B"]
PROBES += [(f"ST{h}", DISP + bytes.fromhex(h), f"Disp stat var {h} in a clean state: ?")
           for h in STAT_HEX]


def main() -> int:
    if not E.rom_available():
        print("ROM not found", file=sys.stderr)
        return 2
    names = [f"Q{n[:7]}" for n, _, _ in PROBES]
    specs = inputs.from_bodies(list(zip(names, [b for _, b, _ in PROBES])), OUT / "inputs",
                               source="probe")
    res = run_programs.run_specs(specs, OUT, workers=1, shots=0, title="W2 OS probes")
    for (name, _, why), r in zip(PROBES, res):
        home = " | ".join(t.strip() for t in (r.get("home_text") or []) if t.strip())
        print(f"{name:10s} {r['label']:18s} home=[{home[:40]}]  assumption: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
