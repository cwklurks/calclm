"""Hand-made TI-BASIC programs covering every outcome class and input-driving path.

Used by eval/tests/test_driver.py and as a smoke set from the CLI (`--handmade`).
Each case: program name, TI-BASIC source (tivars load_string syntax), expected label, and
optional checks (timeout state, text on the final home screen, graph screen shown).
"""
from __future__ import annotations

from dataclasses import dataclass

from tivars import TIProgram


@dataclass(frozen=True)
class Case:
    name: str
    src: str
    expect: str
    timeout_state: str | None = None
    screen_has: str | None = None
    graph: bool | None = None
    what: str = ""

    def body(self) -> bytes:
        p = TIProgram(name=self.name)
        p.load_string(self.src)
        return p.data


CASES: tuple[Case, ...] = (
    Case("OKDISP", "Disp 42", "OK", screen_has="42", graph=False, what="valid"),
    Case("OKLOOP", "0→S\nFor(I,1,100)\nS+I→S\nEnd\nDisp S", "OK", screen_has="5050"),
    Case("OKOUT", "ClrHome\nOutput(5,5,\"HELLO\")", "OK", screen_has="HELLO"),
    Case("OKSTOP", "Disp 5\nStop\nDisp 6", "OK", screen_has="5"),
    Case("OKWAIT", "Wait 1\nDisp 7", "OK", screen_has="7"),
    Case("ESYNTAX", "Disp 1+", "ERR:SYNTAX", what="syntax error"),
    Case("EDOMAIN", "Disp ln(0)", "ERR:DOMAIN", what="domain error"),
    Case("EDIV", "Disp 1/0", "ERR:DIVIDE BY 0"),
    Case("EUNDEF", "prgmNOPE", "ERR:UNDEFINED", what="missing subprogram"),
    Case("EDIMMIS", "{1,2}+{1,2,3}", "ERR:DIM MISMATCH"),
    Case("EINVDIM", "{1,2}→L₁\nDisp L₁(5)", "ERR:INVALID DIM"),
    Case("ETYPE", "Disp \"A\"+1", "ERR:DATA TYPE"),
    Case("ELABEL", "Goto Z", "ERR:LABEL"),
    Case("EMEMORY", "{99,99}→dim([A])\n{99,99}→dim([B])", "ERR:MEMORY"),
    Case("EASM", "Asm(prgmA)", "ERR:INVALID", what="Asm( is out of scope: an error"),
    Case("TLOOP", "While 1\nEnd", "TIMEOUT", timeout_state="computing", what="infinite loop"),
    Case("INPUT", "Input A\nDisp A*2", "OK", screen_has="2", what="Input"),
    Case("PROMPT", "Prompt A,B\nDisp A+B", "OK", screen_has="2"),
    Case("INSTR", "Input \"NAME?\",Str1\nDisp Str1", "OK", screen_has="1"),
    Case("PAUSE", "Disp 1\nPause \nDisp 2", "OK", screen_has="2", what="Pause"),
    Case("MENU", "Lbl 0\nMenu(\"T\",\"GO\",A,\"QUIT\",Q)\nLbl A\nDisp 1\nGoto 0\nLbl Q\nDisp 9",
         "OK", screen_has="9", what="Menu(: option 1, then the last option"),
    Case("GETKEY", "Repeat K\ngetKey→K\nEnd\nDisp K", "OK", what="getKey loop"),
    Case("GKCLEAR", "Repeat K=45\ngetKey→K\nEnd\nDisp 45", "OK", screen_has="45"),
    Case("GKREP", "Repeat getKey\nEnd\nDisp 8", "OK", screen_has="8", what="getKey as condition"),
    Case("GKANS", "0\nRepeat Ans\ngetKey\nEnd\nDisp 9", "OK", screen_has="9"),
    Case("GKWHILE", "While not(getKey)\nEnd\nDisp 6", "OK", screen_has="6"),
    Case("WINDOW", "Prompt Xmin,Xmax,Ymin\nDisp 5", "OK", screen_has="5",
         what="errNo set by the OS without an ERR: screen (Xmax=Xmin) is not an error"),
    Case("SLOWMENU", "GetCalc(A)\nMenu(\"T\",\"A\",A,\"B\",B)\nLbl A\nDisp 1/0\nLbl B\nDisp 2",
         "ERR:DIVIDE BY 0", what="Menu( slow to take its key: never answer twice"),
    Case("TINPUT", "Lbl A\nInput A\nGoto A", "TIMEOUT", timeout_state="input"),
    Case("TGETKEY", "Repeat 0\ngetKey→K\nEnd", "TIMEOUT", timeout_state="getkey"),
    Case("TMENU", "Lbl 0\nMenu(\"T\",\"A\",0,\"B\",0)", "TIMEOUT", timeout_state="menu"),
    Case("GRPAUSE", "ClrDraw\nLine(⁻5,⁻5,5,5)\nPause \nDisp 1", "OK", graph=True),
    Case("GREND", "ClrDraw\nDispGraph", "OK", graph=True),
    Case("GRINPUT", "Input \nDisp 1", "OK", graph=True, what="Input with no variable"),
)


def case(name: str) -> Case:
    return next(c for c in CASES if c.name == name)
