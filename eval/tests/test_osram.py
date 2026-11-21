"""Decoding OS RAM bytes (flags, context, menu, error number) into a program state."""
from eval import osram as R


def flags(**bits):
    b = bytearray(0x40)
    if bits.get("prog"):
        b[R.NEWDISPF] |= R.PROG_EXECUTING
    if bits.get("plot"):
        b[R.PLOTFLAGS] |= R.PLOT_DISP
    if bits.get("key_ready"):
        b[R.GRAPHFLAGS2] |= R.KEY_READY
    return bytes(b)


def test_error_names_mask_goto_bit():
    assert R.error_name(0) is None
    assert R.error_name(2) == "DIVIDE BY 0"
    assert R.error_name(0x80 | 8) == "SYNTAX"
    assert R.error_name(11) == "DIM MISMATCH"
    assert R.error_name(12) == "INVALID DIM"
    assert R.error_name(13) == "UNDEFINED"
    assert R.error_name(14) == "MEMORY"
    assert R.error_name(47) == "ARCHIVED"
    assert R.error_name(120) == "ERROR 120"


def test_decode_home():
    s = R.decode(flags(), cx=0x40, menu=0, errno=0)
    assert not s.prog and not s.plot and s.error is None and s.wait_kind() == "home"


def test_decode_waits():
    assert R.decode(flags(prog=True), cx=0x50, menu=0, errno=0).wait_kind() == "input"
    assert R.decode(flags(prog=True), cx=0x40, menu=0x14, errno=0).wait_kind() == "menu"
    graph = R.decode(flags(prog=True, plot=True), cx=0x40, menu=0, errno=0)
    assert graph.wait_kind() == "graph_wait"
    assert R.decode(flags(prog=True), cx=0x40, menu=0, errno=0).wait_kind() == "pause"


def test_decode_error_wins():
    s = R.decode(flags(prog=True), cx=0x52, menu=0x3F, errno=0x84)
    assert s.error == "DOMAIN" and s.wait_kind() == "error"


def test_key_ready():
    assert R.decode(flags(key_ready=True), cx=0x40, menu=0, errno=0).key_ready


def test_errno_without_error_screen_is_not_an_error_wait():
    # the OS can leave errNo set after an error it handled itself (e.g. Xmax=Xmin at Prompt)
    s = R.decode(flags(prog=True), cx=0x50, menu=0, errno=18)
    assert s.wait_kind() == "input" and not s.error_shown
    assert R.decode(flags(prog=True), cx=0x52, menu=0x3F, errno=18).error_shown
