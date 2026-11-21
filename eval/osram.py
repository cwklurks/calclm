"""TI-84 Plus CE OS 5.8 RAM facts the evaluator reads (equates from CEmu's
tests/ti84pce.inc, behaviour checked in CEmu on the author's OS 5.8.0.0022 ROM).

IY-relative system flags live at FLAGS (0xD00080). While a TI-BASIC program runs the OS keeps
bit 1 of (flags+8) set: at the home screen it is clear, during Input, Prompt, Pause, Menu(,
getKey loops, computation and the ERR: menu it is set.
"""
from __future__ import annotations

from dataclasses import dataclass

FLAGS = 0xD00080
FLAGS_LEN = 0x40
PLOTFLAGS, PLOT_DISP = 0x02, 0x04          # plotDisp: the graph screen is displayed
NEWDISPF, PROG_EXECUTING = 0x08, 0x02      # a BASIC program is executing
GRAPHFLAGS2, KEY_READY = 0x1F, 0x20        # injected OS keycode not yet taken (CEmu sendKey)
CX_CUR_APP = 0xD007E0                      # context: 0x40 home, 0x50 programmed Input, 0x52 error
MENU_CURRENT = 0xD00824                    # 0x14 Menu( from a program, 0x3F the ERR: menu
ERR_NO = 0xD008DF
BEG_PC, CUR_PC, END_PC = 0xD02317, 0xD0231A, 0xD0231D
EDIT_TOP, EDIT_CURSOR = 0xD02437, 0xD0243A    # program editor buffer (after ERR: 2:Goto)
ERR_GOTO = 0x80                                # errNo bit 7: the ERR: menu offers Goto

CX_HOME, CX_PRGM_EDIT, CX_PRGM_INPUT, CX_ERROR = 0x40, 0x46, 0x50, 0x52
MENU_PROGRAM = 0x14

# OS error numbers (errNo & 0x7F) -> name as the ERROR: screen shows it (short forms where the
# OS 5.8 screen spells them out: DIM MISMATCH, INVALID DIM, NONREAL ANS, SINGULAR MAT).
ERRORS = {
    1: "OVERFLOW", 2: "DIVIDE BY 0", 3: "SINGULAR MAT", 4: "DOMAIN", 5: "INCREMENT",
    6: "BREAK", 7: "NONREAL ANS", 8: "SYNTAX", 9: "DATA TYPE", 10: "ARGUMENT",
    11: "DIM MISMATCH", 12: "INVALID DIM", 13: "UNDEFINED", 14: "MEMORY", 15: "INVALID",
    16: "ILLEGAL NEST", 17: "BOUND", 18: "WINDOW RANGE", 19: "ZOOM", 20: "LABEL", 21: "STAT",
    22: "SOLVER", 23: "SINGULARITY", 24: "NO SIGN CHANGE", 25: "ITERATIONS", 26: "BAD GUESS",
    27: "STAT PLOT", 28: "TOL NOT MET", 29: "RESERVED", 30: "MODE", 31: "LINK",
    32: "LINK MEMORY", 33: "LINK TRANSMISSION", 34: "LINK DUPLICATE", 35: "LINK MEMORY FULL",
    36: "UNKNOWN", 37: "SCALE", 38: "ID NOT FOUND", 39: "NO MODE", 40: "VALIDATION",
    41: "LENGTH", 42: "APPLICATION", 43: "APP ERR1", 44: "APP ERR2", 45: "EXPIRED",
    46: "BAD ADDRESS", 47: "ARCHIVED", 48: "VERSION", 49: "ARCHIVE FULL", 50: "VARIABLE",
    51: "DUPLICATE",
}


def error_name(errno: int) -> str | None:
    code = errno & 0x7F
    if not code:
        return None
    return ERRORS.get(code, f"ERROR {code}")


@dataclass(frozen=True)
class OsState:
    prog: bool        # a BASIC program is executing (or stopped at its ERR: menu)
    plot: bool        # graph screen displayed
    key_ready: bool   # an injected keycode is still waiting for the OS
    cx: int
    menu: int
    errno: int

    @property
    def error(self) -> str | None:
        """Last error number's name. The OS can leave errNo set after an error it handled
        itself, so only `error_shown` means the program stopped with an error."""
        return error_name(self.errno)

    @property
    def error_shown(self) -> bool:
        """Stopped with an error: the ERR: screen, or the program editor that its 2:Goto
        opens (only reachable through an error)."""
        return bool(self.error) and self.cx in (CX_ERROR, CX_PRGM_EDIT)

    def wait_kind(self) -> str:
        """What a stopped (CPU idle) program is waiting for."""
        if self.error_shown:
            return "error"
        if not self.prog:
            return "home"
        if self.cx == CX_PRGM_INPUT:
            return "input"          # Input / Prompt
        if self.menu == MENU_PROGRAM:
            return "menu"           # Menu(
        if self.plot:
            return "graph_wait"     # Pause or Input with no variable on the graph screen
        return "pause"              # Pause (or any other key wait)


def decode(flags: bytes, cx: int, menu: int, errno: int) -> OsState:
    return OsState(prog=bool(flags[NEWDISPF] & PROG_EXECUTING),
                   plot=bool(flags[PLOTFLAGS] & PLOT_DISP),
                   key_ready=bool(flags[GRAPHFLAGS2] & KEY_READY),
                   cx=cx, menu=menu, errno=errno)


def read_state(emu) -> OsState:
    return decode(emu.read(FLAGS, FLAGS_LEN), emu.read(CX_CUR_APP, 1)[0],
                  emu.read(MENU_CURRENT, 1)[0], emu.read(ERR_NO, 1)[0])


def read_pcs(emu) -> tuple[int, int, int]:
    """(begPC, curPC, endPC) of the program the parser is executing."""
    return emu.u24(BEG_PC), emu.u24(CUR_PC), emu.u24(END_PC)
