"""CalcLM v2.5's start screen, src/menu.c, linked on the host with the core and stand-ins for
the screen calls (as test_calclm_os.py links src/model.c): keys move, jump and choose; topics
are hidden when the masks' vocab, the measured model, LD2 or the full context is missing; a
topic runs at its fixed settings with the mixed seed; another try advances the seed by one;
the menu rows, check lines and reasons fit the 40-column screen."""
from __future__ import annotations

import ctypes
import subprocess

import pytest

from ce.tools.clm import CALCLM, CORE_DIR, HOST_SRC, TRACKER_DIR
import json

from model import topics as TP

(K_NONE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ENTER, K_CLEAR, K_DEL, K_MODE, K_2ND,
 K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9) = range(20)
MENU_NONE, MENU_PICK, MENU_SETTINGS, MENU_QUIT = range(4)
TOPICS_OK, NO_MASKS, OTHER_MODEL, NO_LD2, SHORT_CTX, OTHER_FIT = range(6)
RANDOM = 0xFF
FIX = TP.load_fixture()
MENU = TP.menu_entries(FIX)
TOPICS = MENU[:-1]
SHA6 = bytes.fromhex(FIX["export_sha256"])[:6]
FIT = TP.FIT                                                  # the measured LD2 fit
E_FIT = (ctypes.c_uint32 * 16)(*json.loads(FIT.read_text())["E"])     # the measured LD2 fit
E_OTHER = (ctypes.c_uint32 * 16)(*[e + (i == 5) for i, e in enumerate(E_FIT)])


class Run(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_uint32), ("temperature_q8", ctypes.c_uint16),
                ("cap", ctypes.c_uint16), ("top_k", ctypes.c_uint8), ("greedy", ctypes.c_uint8),
                ("masks", ctypes.c_uint8), ("ld2", ctypes.c_uint8),
                ("prompt", ctypes.POINTER(ctypes.c_uint16)), ("n_prompt", ctypes.c_uint16)]


class Menu(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint8), ("n", ctypes.c_uint8)]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("menu")
    (tmp / "stubs.c").write_text('''
#include <stdint.h>
#include <string.h>
char screen[16][41]; int lines;
void ui_text(uint8_t c, uint8_t y, const char *s) { (void)y; if (lines < 16) {
    memset(screen[lines], 0, 41); strncpy(screen[lines] + 0, s, 40 - c); lines++; } }
void ui_line_clear(uint8_t y) { (void)y; }
''')
    so = tmp / "menu.so"
    subprocess.run(["gcc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", f"-I{CALCLM / 'src'}",
                    f"-I{CORE_DIR}", f"-I{TRACKER_DIR}", *map(str, HOST_SRC),
                    str(CALCLM / "src" / "menu.c"), str(tmp / "stubs.c"), "-o", str(so)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    lib.menu_key.restype = ctypes.c_uint8
    lib.menu_topic.restype = ctypes.c_uint8
    lib.menu_topics_reason.restype = ctypes.c_uint8
    lib.menu_topics_reason.argtypes = [ctypes.c_uint8, ctypes.c_char_p,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
    lib.menu_reason_text.restype = ctypes.c_char_p
    lib.menu_label.restype = lib.menu_check.restype = ctypes.c_char_p
    lib.menu_line.restype = ctypes.c_char_p
    lib.menu_retry_seed.restype = ctypes.c_uint32
    lib.menu_retry_seed.argtypes = [ctypes.c_uint32]
    lib.menu_run.argtypes = [ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint8,
                             ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8,
                             ctypes.c_uint8, ctypes.POINTER(Run)]
    return lib


def menu(lib, topics_ok=True) -> Menu:
    m = Menu()
    lib.menu_init(ctypes.byref(m), topics_ok)
    return m


def keys(lib, m, *ks) -> list[int]:
    return [lib.menu_key(ctypes.byref(m), k) for k in ks]


# ---- entries and keys --------------------------------------------------------------------------

def test_entries_are_the_measured_menu_then_random(lib):
    m = menu(lib)
    assert m.n == len(MENU) and m.sel == 0
    assert [lib.menu_label(i).decode() for i in range(len(TOPICS))] == [e["label"] for e in TOPICS]
    assert lib.menu_label(RANDOM).decode() == "Random program"
    assert menu(lib, topics_ok=False).n == 1


def test_up_down_move_within_the_list(lib):
    m = menu(lib)
    assert keys(lib, m, K_UP) == [MENU_NONE] and m.sel == 0
    keys(lib, m, *[K_DOWN] * 30)
    assert m.sel == m.n - 1 and lib.menu_topic(ctypes.byref(m)) == RANDOM
    keys(lib, m, K_UP)
    assert m.sel == m.n - 2 and lib.menu_topic(ctypes.byref(m)) == len(TOPICS) - 1


def test_digits_jump_and_enter_picks(lib):
    m = menu(lib)
    keys(lib, m, K_3)
    assert m.sel == 2 and lib.menu_topic(ctypes.byref(m)) == 2
    keys(lib, m, K_0)
    assert lib.menu_topic(ctypes.byref(m)) == RANDOM
    keys(lib, m, K_1)
    assert m.sel == 0
    assert keys(lib, m, K_ENTER, K_MODE, K_CLEAR, K_2ND, K_LEFT) == [
        MENU_PICK, MENU_SETTINGS, MENU_QUIT, MENU_NONE, MENU_NONE]
    assert m.sel == 0


def test_nine_digits_reach_nine_topics_and_a_tenth_needs_the_arrows(lib):
    """`9` jumps to the ninth entry; with ten topics (v2.5, v2.6-rc2) no digit reaches the tenth,
    as the hardware-check sheet and INSTALL say (review of v2.6-rc2, finding 9)."""
    m = menu(lib)
    keys(lib, m, K_9)
    assert m.sel == 8 and lib.menu_topic(ctypes.byref(m)) == 8
    reached = set()
    for k in (K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_0):
        keys(lib, m, k)
        reached.add(m.sel)
    assert (len(TOPICS) - 1 in reached) == (len(TOPICS) <= 9)


def _doc_menu(path, sheet: bool) -> list[tuple[str, str, str, str]]:
    """(digit, label, rate, time) rows of a menu table: the sheet's `| key | entry | right | time |`
    or INSTALL's `| key entry | right | time |`."""
    import re
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if sheet and len(cells) == 4 and re.fullmatch(r"\d?", cells[0]) and (
                cells[2].endswith("in 10") or cells[2] == "rarely"):
            rows.append(tuple(cells))
        elif not sheet and len(cells) == 3 and (cells[1].endswith("in 10") or cells[1] == "rarely"):
            m = re.fullmatch(r"(?:(\d) )?(.+)", cells[0])
            rows.append((m.group(1) or "", m.group(2), cells[1], cells[2]))
    return rows


def test_the_v26_sheet_and_install_show_the_measured_menu():
    """The hardware-check sheet's and INSTALL-v2.6.md's menu tables are the measured menu: digit,
    label, rate and time per row, in order (review of v2.6-rc2)."""
    want = [("0" if e["key"] == "RANDOM" else str(i + 1) if i < 9 else "", e["label"], e["rate"],
             e["time"]) for i, e in enumerate(MENU)]
    sheet = CALCLM.parents[1] / "hw" / "checks" / "2026-11-12-v2.6-rc2" / "CHECK.md"
    install = CALCLM / "INSTALL-v2.6.md"
    if not (sheet.exists() and install.exists()):
        pytest.skip("the hardware-check sheet or INSTALL-v2.6.md is absent (public export)")
    assert _doc_menu(sheet, True) == want
    assert _doc_menu(install, False) == want


def test_without_topics_only_random_is_offered(lib):
    m = menu(lib, topics_ok=False)
    keys(lib, m, K_1, K_5, K_DOWN, K_UP, K_9)
    assert m.sel == 0 and lib.menu_topic(ctypes.byref(m)) == RANDOM
    assert keys(lib, m, K_ENTER) == [MENU_PICK]


@pytest.mark.parametrize("masks_ok,sha,ld2,ctx,want", [
    (1, SHA6, E_FIT, 256, TOPICS_OK),
    (0, SHA6, E_FIT, 256, NO_MASKS),
    (1, b"\0" * 6, E_FIT, 256, OTHER_MODEL),
    (1, SHA6, None, 256, NO_LD2),
    (1, SHA6, E_OTHER, 256, OTHER_FIT),
    (1, SHA6, E_FIT, 248, SHORT_CTX),
    (0, b"\0" * 6, None, 32, NO_MASKS)])
def test_topics_need_masks_the_measured_model_and_fit_and_the_full_context(lib, masks_ok, sha,
                                                                          ld2, ctx, want):
    """ld2: the accepted CLMLD2's offsets E (None: no LD2); a refitted CLMLD2 for the same
    model is not the measured fit (review M1)."""
    import hashlib
    assert hashlib.sha256(FIT.read_bytes()).hexdigest() == FIX["offsets_sha256"]
    assert lib.menu_topics_reason(masks_ok, sha, ld2, ctx) == want
    text = lib.menu_reason_text(want).decode()
    assert (text == "") == (want == TOPICS_OK) and len(text) <= 39


def test_the_kv4_build_offers_no_topics(tmp_path):
    """The E20 4-bit K/V build writes other programs than the measured ones (review M1)."""
    (tmp_path / "stubs.c").write_text('''
#include <stdint.h>
void ui_text(uint8_t c, uint8_t y, const char *s) { (void)c; (void)y; (void)s; }
void ui_line_clear(uint8_t y) { (void)y; }
uint32_t clm_mix_seed(uint32_t s) { return s; }
''')
    so = tmp_path / "kv4menu.so"
    subprocess.run(["gcc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                    "-DCLM_KV4", f"-I{CALCLM / 'src'}", f"-I{CORE_DIR}", f"-I{TRACKER_DIR}",
                    str(CALCLM / "src" / "menu.c"), str(tmp_path / "stubs.c"), "-o", str(so)],
                   check=True, capture_output=True)
    kv4 = ctypes.CDLL(str(so))
    kv4.menu_topics_reason.restype = ctypes.c_uint8
    kv4.menu_topics_reason.argtypes = [ctypes.c_uint8, ctypes.c_char_p,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
    assert kv4.menu_topics_reason(1, SHA6, E_FIT, 256) == OTHER_MODEL


# ---- what a run uses ---------------------------------------------------------------------------

def run(lib, topic, seed=20, temp=128, k=5, greedy=0, cap=512, row=1, masks_ok=1, ld2=1) -> Run:
    r = Run()
    lib.menu_run(topic, seed, temp, k, greedy, cap, row, masks_ok, ld2, ctypes.byref(r))
    return r


@pytest.mark.parametrize("i", range(len(TOPICS)))
def test_a_topic_runs_at_its_fixed_settings_with_the_mixed_seed(lib, i):
    for seed in (0, 1, 20, 999999):
        r = run(lib, i, seed=seed, greedy=1, row=0)        # the settings rows do not apply
        assert r.seed == TP.mix_seed(seed)
        assert (r.temperature_q8, r.top_k, r.greedy, r.cap, r.masks, r.ld2) == (
            205, 20, 0, 256, 2, 1)
        assert list(r.prompt[:r.n_prompt]) == TP.prompt(TOPICS[i]["key"])


def test_random_uses_the_settings_rows_and_keeps_the_safety_rules(lib):
    r = run(lib, RANDOM, seed=20, temp=128, k=5, greedy=0, cap=512, row=1)
    assert (r.seed, r.temperature_q8, r.top_k, r.greedy, r.cap, r.masks, r.ld2) == (
        20, 128, 5, 0, 512, 2, 1)
    assert not r.prompt and r.n_prompt == 0
    assert (run(lib, RANDOM, row=0).masks, run(lib, RANDOM, row=0).ld2) == (3, 0)  # safety only
    assert run(lib, RANDOM, masks_ok=0).masks == 0         # no mask tables for this vocab
    assert run(lib, RANDOM, greedy=1).ld2 == 0 and run(lib, RANDOM, ld2=0).ld2 == 0


def test_another_try_advances_the_seed_by_one(lib):
    assert [lib.menu_retry_seed(s) for s in (0, 20, 999999)] == [1, 21, 1000000]


# ---- text --------------------------------------------------------------------------------------

def test_check_lines_are_model_topics_and_fit_the_status_line(lib):
    for i, e in enumerate(TOPICS):
        line = lib.menu_check(i).decode()
        assert line == TP.check_line(e["key"]) and len(line) <= 40
    assert lib.menu_check(RANDOM) == b""


def test_menu_rows_fit_and_show_rate_and_time(lib):
    m = menu(lib)
    buf = ctypes.create_string_buffer(64)
    for i, e in enumerate(MENU):
        row = lib.menu_line(buf, ctypes.byref(m), i).decode()
        assert len(row) <= 40
        assert e["label"] in row and row.rstrip().endswith(e["time"])
        assert e["rate"] in row
        assert row[0] == (">" if i == 0 else " ")
        assert row[1] == ("0" if e["key"] == "RANDOM" else str(i + 1) if i < 9 else " ")


def test_the_app_checks_the_reset_and_the_feed():
    """src/main.c refuses to run when clm_reset or clm_feed is refused (review L1)."""
    import re
    src = (CALCLM / "src" / "main.c").read_text()
    assert re.search(r"if \(clm_reset\(&M,[^;]*\) != CLM_OK\)", src)
    assert "clm_feed(&M, r.prompt[i]) != CLM_OK" in src
