"""v2.4 safety masks (mask level 2, docs/agents/phase1-safety-masks.md) in the C core against
model.intmask.IntMask(level=2) and model.linker.generate(mask_level=2), on the W build with
rc2's model (E10's L4-s23) and its LD2 fit, through the shared host/eZ80 driver: host (gcc +
UBSan), CEmu with the routines in RAM, CEmu with CLMCODE attached.

Bit-exact means identical ban bitmaps on every position of the programs fed, and identical
tokens for the same seed and settings. Level 1 (v2.3) stays covered by the existing modules,
including rc2's hardware-check sheet."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from ce.tools.clm import CALCLM, W_RC2_EXPORT, EmuCore, Export, HostCore, emu_available

ROOT = Path(__file__).parents[2]
FIT = Path(__file__).parent / "fixtures" / "ld2_offsets_L4-s23.json"
VAL = ROOT / "data" / "tok" / "val.bin"
BOS, EOS, OK, E_ARG = 1, 2, 0, -11

pytestmark = pytest.mark.skipif(not (W_RC2_EXPORT / "model.bin").exists(),
                                reason=f"{W_RC2_EXPORT} not present")
_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module")
def ex() -> Export:
    return Export.load(W_RC2_EXPORT)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex, horizons=True)
    return EmuCore(ex, flash=request.param == "emuflash", horizons=True)


def is_emu(core) -> bool:
    return isinstance(core, EmuCore)


def E() -> list[int]:
    return json.loads(FIT.read_text())["E"]


def vocab(ex) -> list[str]:
    return ex.meta["vocab"]


def hexes(ex, *hs) -> list[int]:
    idx = {h: i for i, h in enumerate(vocab(ex))}
    return [idx[h] for h in hs]


def text(ex, s: str) -> list[int]:
    """Uppercase, digits, space and simple punctuation as one-byte tokens; a-z as BB tokens."""
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            k = ord(ch) - ord("a")
            out.append("BB%02X" % (0xB0 + k + (k >= 11)))
        else:
            out.append({" ": "29", "(": "10", ")": "11", ",": "2B", ".": "3A", "-": "71",
                        '"': "2A", ">": "04", "\n": "3F"}.get(ch, "%02X" % ord(ch)))
    return hexes(ex, *out)


# crafted programs covering every level 2 rule; '>' is the store arrow, '"' a quote
def crafted(ex) -> list[list[int]]:
    t = lambda s: text(ex, s)                                     # noqa: E731
    h = lambda *x: hexes(ex, *x)                                  # noqa: E731
    disp, inp, prompt, for_, prgm = h("DE", "DC", "DD", "D3", "5F")
    xmin, xmax = h("630A", "630B")
    return [
        [BOS, disp, *t('"BY JOE, MADE by X, COPYRIGHT (C) 2009 Copyr (cm)"\n')],
        [BOS, disp, *t('"ABBEY STANDBY (c"\n'), *t("B+Y>A\n"), *t("(C)\n")],
        [BOS, *t("5>"), xmin, *t("\n"), disp, xmax, *t("-"), xmin, *t("\n")],
        [BOS, prompt, *t("A,"), xmin, *t("\n"), inp, *t('"X",'), xmax, *t("\n")],
        [BOS, for_, xmin, *t(",1,2\n"), for_, *t("I,"), xmin, *t(",9\n"), *h("DA"), xmin],
        [BOS, prgm, *t("ABC\n"), *t('"COPY"'), *h("04"), xmin, *t("\n"), *h("73"), *t("2\n")],
        [BOS, disp, *t('"by'), *t("\n"), disp, *t('"V1.0 B'), *t("Y\n")],
    ]


def ours(ex, level=2):
    from model.intmask import IntMask
    m = IntMask(vocab(ex), level=level)
    m.reset()
    m.step(BOS)
    return m


def decisions_match(core, ex, programs, length=10_000, level=2) -> int:
    """Ban bitmaps of the core (the level set by a reset) and IntMask(level) before every
    token of each program; returns the number of positions compared."""
    assert core.load()[0] == OK
    assert core.reset(1, 205, 20, cap=8, masks=level) == OK
    checked = 0
    for prog in programs:
        m = ours(ex, level)
        core.track([], reset=True)
        for tok in prog[1:length]:
            assert np.array_equal(core.mask(), m.banned()), (checked, prog[:8])
            core.track([int(tok)])
            m.step(int(tok))
            checked += 1
    return checked


# ---- settings ------------------------------------------------------------------------------------

def test_reset_takes_levels_0_to_3_only(core, ex):
    assert core.load()[0] == OK
    for level in (0, 1, 2, 3):
        assert core.reset(1, 205, 20, cap=8, masks=level) == OK
    assert core.reset(1, 205, 20, cap=8, masks=4) == E_ARG


def test_level_1_masks_are_v23s(core, ex):
    """A reset at level 1 leaves the ban set v2.3's (the level 2 rules stay off)."""
    assert core.load()[0] == OK
    assert core.reset(1, 205, 20, cap=8, masks=1) == OK
    for prog in crafted(ex)[:3]:
        m = ours(ex, level=1)
        core.track([], reset=True)
        for tok in prog[1:]:
            assert np.array_equal(core.mask(), m.banned())
            core.track([int(tok)])
            m.step(int(tok))


# ---- ban bitmaps --------------------------------------------------------------------------------

def test_level_2_decisions_on_crafted_programs(core, ex):
    progs = crafted(ex)
    n = decisions_match(core, ex, progs)
    assert n == sum(len(p) - 1 for p in progs)
    # each rule decides somewhere in them
    m, seen = ours(ex), {"str": 0, "window": 0}
    from model import safety as S
    for prog in progs:
        m.reset()
        m.step(BOS)
        for tok in prog[1:]:
            b2 = m.banned()
            seen["str"] += bool(m.ctx == 2 and S.STR_BAN[m.str_state])
            seen["window"] += bool(b2[hexes(ex, "630A")[0]])
            m.step(int(tok))
    assert seen["str"] >= 8 and seen["window"] >= 5


def _val_programs_with_strings(ex, n):
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)
    ends = np.append(starts[1:], len(stream))
    quote, store = hexes(ex, "2A", "04")
    out = []
    for a, b in zip(starts, ends):
        p = stream[a:b].tolist()
        if p.count(quote) >= 6 and store in p:
            out.append(p)
        if len(out) == n:
            break
    return out


def test_level_2_decisions_on_val_programs(core, ex):
    if not VAL.exists():
        pytest.skip("data/tok/val.bin not present")
    if is_emu(core):
        assert decisions_match(core, ex, _val_programs_with_strings(ex, 2), 90) >= 150
    else:
        assert decisions_match(core, ex, _val_programs_with_strings(ex, 30), 500) > 5000


# ---- generation ---------------------------------------------------------------------------------

def _linker(ex, seed, cap, level=2):
    from model.linker import ARMS, generate
    return generate(ex.model_hz, ARMS["LD2"], seed, E=E(), cap=cap, mask_level=level)["ids"][1:]


@pytest.mark.parametrize("seed", [3, 4, 7])
def test_level_2_generation_matches_linker(core, ex, seed):
    """LD2 + horizons at level 2; these seeds' programs differ from level 1's."""
    cap = 48 if is_emu(core) else 1024
    want = _linker(ex, seed, cap)
    assert want != _linker(ex, seed, cap, level=1)
    assert core.load()[0] == OK
    assert core.load_E(E()) == OK
    assert core.set_ld2(True) == OK
    assert core.reset(seed, 205, 20, cap=cap, masks=2) == OK
    assert core.generate(1024) == want
    assert core.step() == -1


@pytest.mark.slow
def test_level_2_whole_program_in_cemu(core, ex):
    """One whole level 2 program through the OS program's load policy (seed 4: 840 tokens at
    level 2, 59 at level 1)."""
    if not is_emu(core) or core.flash is False:
        pytest.skip("CEmu with CLMCODE only (the host covers whole programs above)")
    from ce.calclm.tools.make_ld2 import payload
    want = _linker(ex, 4, 1024)
    assert core.load()[0] == OK
    assert core.program_decoder(payload(ex.path, FIT)) == OK
    assert core.reset(4, 205, 20, cap=1024, masks=2) == OK
    assert core.generate(1100) == want


def test_the_os_program_uses_level_2():
    """src/main.c starts every generation with the masks src/menu.c chooses: level 2 with the
    Masks row on, level 3 (destructive and safety bans) with it off (test_calclm_menu.py runs
    menu_run on the host)."""
    src = (CALCLM / "src" / "main.c").read_text()
    args = re.search(r"clm_reset\(&M,([^)]*)\)", src).group(1)
    assert args.strip().endswith("r.masks")
    menu = (CALCLM / "src" / "menu.c").read_text()
    assert "masks_row ? CLM_MASKS_V24 : CLM_MASKS_SAFETY" in menu


# ---- the v2.4 hardware-check reference -----------------------------------------------------------

def test_hw_check_reference_at_level_2(ex):
    """ce.tools.hw_check writes the level 2 program with --mask-level 2 and rc2's sheet (level 1)
    by default."""
    from ce.tools.hw_check import reference, sheet_line
    from ce.tests.test_calclm_w_ld2 import check_sheet, check_sheets
    v24 = reference(ex.path, FIT, 20, mask_level=2)
    assert v24["mask_level"] == 2 and v24["ids"][1:] == _linker(ex, 20, 1024)
    assert "masks=2" in sheet_line(v24)
    v23 = reference(ex.path, FIT, 20)
    assert "mask_level" not in v23 and v23["ids"] != v24["ids"]   # level 1: as rc2 wrote it
    assert "masks=" not in sheet_line(v23)
    assert v23["ids_sha256"] == check_sheet(check_sheets(ex.path)[0])["ids_sha256"]


def test_level_2_decisions_on_concatenations_and_lookalikes(core, ex):
    t = lambda s: text(ex, s)                                     # noqa: E731
    h = lambda *x: hexes(ex, *x)                                  # noqa: E731
    disp, plus = h("DE", "70")
    progs = [[BOS, disp, *t('"B"'), plus, *t('"Y"\n'), disp, *t('"COP"'), plus, *t('"YR"\n')],
             [BOS, disp, *t('"'), *h("6217"), *h("6223"), *t(' '), *h("6218"), *t('"\n')],
             [BOS, disp, *t('"(C"'), plus, *t('"X"\n'), *t('"B"'), plus, *t("A+"), *t('"Y"')]]
    assert decisions_match(core, ex, progs) == sum(len(p) - 1 for p in progs)


def test_level_2_mask_cost_stays_small(core, ex):
    """CEmu cycles of MASK at level 2 stay within 40 k of level 1's (mean, crafted programs)."""
    if not is_emu(core):
        pytest.skip("cycle counts are CEmu's")
    progs = crafted(ex)
    cyc = {}
    for level in (1, 2):
        assert core.load()[0] == OK
        assert core.reset(1, 205, 20, cap=8, masks=level) == OK
        got = []
        for prog in progs:
            core.track([], reset=True)
            for tok in prog[1:]:
                core.mask()
                got.append(core.last.cycles)
                core.track([int(tok)])
        cyc[level] = sum(got) / len(got)
    assert cyc[2] - cyc[1] < 40_000, cyc


# ---- level 3: destructive and safety bans alone (the Masks row off, phase 3 decision 1) ------------

def test_level_3_keeps_v23s_destructive_bans(core, ex):
    """The Masks row off never lifts a destructive ban: DelVar, ClrAllLists, Archive, UnArchive
    and ClrList stay banned at level 3 at every position of the crafted programs, while v2.3's
    grammar rule for an unmatched End does not apply."""
    destroy = hexes(ex, "BB54", "BB52", "BB68", "BB69", "FA")
    end = hexes(ex, "D4")[0]
    assert core.load()[0] == OK
    assert core.reset(1, 205, 20, cap=8, masks=3) == OK
    core.track([], reset=True)
    assert not core.mask()[end]
    for prog in crafted(ex):
        core.track([], reset=True)
        for tok in prog[1:]:
            assert core.mask()[destroy].all()
            core.track([int(tok)])


def test_level_3_decisions_on_crafted_programs(core, ex):
    progs = crafted(ex)
    assert decisions_match(core, ex, progs, level=3) == sum(len(p) - 1 for p in progs)


def test_level_3_decisions_on_val_programs(core, ex):
    if not VAL.exists():
        pytest.skip("data/tok/val.bin not present")
    if is_emu(core):
        assert decisions_match(core, ex, _val_programs_with_strings(ex, 2), 90, level=3) >= 150
    else:
        assert decisions_match(core, ex, _val_programs_with_strings(ex, 20), 400,
                               level=3) > 3000


@pytest.mark.parametrize("seed", [3, 4])
def test_level_3_generation_matches_intref(core, ex, seed):
    """Masks row off: plain sampling (no LD2) with only the safety rules, as model.intref
    generates with IntMask(level=3)."""
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    cap = 40 if is_emu(core) else 400
    want = generate(ex.model_hz, [], cap, SamplerConfig(205, 20, False, seed), cap=cap,
                    mask=IntMask(vocab(ex), level=3))[1:]
    assert core.load()[0] == OK
    assert core.reset(seed, 205, 20, cap=cap, masks=3) == OK
    assert core.generate(1024) == want
