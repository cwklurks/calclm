"""LD2 + horizons on the wide model shape W (the v2.3 port) against model.linker, through the
shared host/eZ80 driver, on each W export (W_EXPORTS) with its own LD2 fit
(fixtures/ld2_offsets_<export>.json, or CALCLM_W_OFFSETS for W_EXPORT: fitted as E2b fitted
v2's, `model.linker fit --data data/tok --n-programs 2000 --horizons --hazard-full`).

Also the OS program's load path (src/model.c with fileioc stand-ins, test_calclm_os.py's
fixture) compiled for W: 16 model AppVars, the V cache and the LD2 workspace in CLMTMP; and
each export's hardware checks (the hw/checks/*/CHECK.md that names it: rc1 for W-s11, rc2 and
v2.4/v2.5-rc1 for L4-s23, v2.6-rc2 for E24's L4c-s11) as the core's own output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from ce.tests import test_calclm_os as osmod
from ce.tests.test_calclm_os import os_model  # noqa: F401  (compiled for W below)
from ce.tools.clm import SHAPES, W_EXPORT, W_EXPORTS, EmuCore, Export, HostCore, emu_available
from model.linker import ARMS, generate

ROOT = Path(__file__).parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
SHEET = re.compile(r"<!-- hardware-check (.*?) -->")


def w_offsets(export: Path) -> Path:
    """The export's LD2 fit: fixtures/ld2_offsets_<export dir>.json (CALCLM_W_OFFSETS can name
    another for W_EXPORT)."""
    if export == W_EXPORT and "CALCLM_W_OFFSETS" in os.environ:
        return Path(os.environ["CALCLM_W_OFFSETS"])
    return FIXTURES / f"ld2_offsets_{export.name}.json"


def _fit_matches(export: Path) -> bool:
    if not (export / "model.json").exists() or not w_offsets(export).exists():
        return False
    sha = json.loads((export / "model.json").read_text())["sha256"]
    return json.loads(w_offsets(export).read_text()).get("export_sha256") == sha


def sheet_lines(path: Path) -> list[dict]:
    """A sheet's machine-readable lines: seed, tokens and lines as ints, masks (the mask
    level; 1 when the line names none) and, for a v2.5 topic, topic and seed_mix."""
    out = []
    for m in SHEET.finditer(path.read_text()):
        line = dict(kv.split("=", 1) for kv in m.group(1).split())
        out.append(line | {k: int(line[k]) for k in ("seed", "tokens", "lines")}
                   | {"masks": int(line.get("masks", 1))})
    return out


def check_sheets(export: Path, mask_level: int = 1) -> list[Path]:
    """The hardware-check sheets (hw/checks/*/CHECK.md) with a machine-readable line for this
    export (its export_sha256 prefix) at this mask level. One sheet per (export, mask level):
    v2.3's checks are level 1, the combined v2.4/v2.5 check and v2.6's level 2."""
    sha = json.loads((export / "model.json").read_text())["sha256"][:16]
    return [p for p in sorted((ROOT / "hw" / "checks").glob("*/CHECK.md"))
            if any(x["export_sha256"] == sha and x["masks"] == mask_level
                   for x in sheet_lines(p))]


# Hardware-check sheets per W export (by export sha256 prefix) at mask levels 1, 2 and 3: W-s11
# (rc1) and L4-s23 (rc2) shipped at level 1, L4-s23 again in the v2.4/v2.5 sheet at level 2,
# E24's L4c-s11 (v2.6-rc2) at level 2 only.
SHEETS_PER_LEVEL = {"a189dd1ca90063c6": [1, 0, 0], "fb4bc774f7e13b28": [1, 1, 0],
                    "e68eca508872b7a5": [0, 1, 0]}

W_FIT_PARAMS = [pytest.param(p, id=p.name, marks=pytest.mark.skipif(
    not _fit_matches(p), reason=f"no LD2 fit for {p} ({w_offsets(p).name})")) for p in W_EXPORTS]
BACKENDS = ["host", pytest.param("emu", marks=pytest.mark.skipif(
    not emu_available(), reason="needs the CE ROM and ~/CEdev"))]


@pytest.mark.parametrize("export", [pytest.param(p, id=p.name, marks=pytest.mark.skipif(
    not (p / "model.bin").exists(), reason=f"{p} not present")) for p in W_EXPORTS])
def test_each_w_export_has_its_fit_and_one_check_sheet(export):
    """A W model ships with its own LD2 fit (bound to its model.bin) and a hardware-check
    sheet, at most one per mask level (v2.3's models at level 1, v2.6's L4c-s11 at level 2
    only: the calculator has run level 2 with the Masks row on since v2.4); the tests below
    then run on it."""
    assert _fit_matches(export), w_offsets(export)
    fit = json.loads(w_offsets(export).read_text())
    assert fit["n_programs"] == 2000 and fit["hazard_full"] and fit["horizons"]
    assert fit["programs_sha256"] == json.loads(
        (FIXTURES / "ld2_offsets_W-s11.json").read_text())["programs_sha256"]   # E2b's programs
    sha = json.loads((export / "model.json").read_text())["sha256"][:16]
    per_level = [len(check_sheets(export, level)) for level in (1, 2, 3)]
    assert per_level == SHEETS_PER_LEVEL[sha], (export.name, per_level)
    sha = json.loads((export / "model.json").read_text())["sha256"][:16]
    for sheet in sorted((ROOT / "hw" / "checks").glob("*/CHECK.md")):
        mine = [(x["masks"], x.get("topic")) for x in sheet_lines(sheet)
                if x["export_sha256"] == sha]
        assert len(mine) == len(set(mine)), sheet         # one line per level and topic


def test_the_v24_v25_sheet_is_level_2_for_rc2():
    """The combined v2.4/v2.5 hardware check (rc2's model at mask level 2): Random at
    phase 1's seed 48 (264 tokens) and the Cylinder volume topic."""
    from ce.tools.clm import W_RC2_EXPORT
    if not (W_RC2_EXPORT / "model.json").exists():
        pytest.skip(f"{W_RC2_EXPORT} not present")
    sheets = check_sheets(W_RC2_EXPORT, 2)
    assert len(sheets) == 1
    lines = sheet_lines(sheets[0])
    random = [x for x in lines if "topic" not in x]
    assert [(x["seed"], x["tokens"]) for x in random] == [(48, 264)]
    cyl = [x for x in lines if x.get("topic") == "CYL"]
    assert len(cyl) == 1 and cyl[0]["seed_mix"] == "fmix32" and cyl[0]["masks"] == 2


def test_the_v26_sheet_is_level_2_for_l4c_s11():
    """v2.6-rc2's hardware check (E24's L4c-s11 at mask level 2, as v2.5-rc1's and rc1's): one
    Random line and the Cylinder volume topic, no level 1 or 3 sheet."""
    from ce.tools.clm import W_V26_EXPORT
    if not (W_V26_EXPORT / "model.json").exists():
        pytest.skip(f"{W_V26_EXPORT} not present")
    assert [len(check_sheets(W_V26_EXPORT, level)) for level in (1, 2, 3)] == [0, 1, 0]
    sheet = check_sheets(W_V26_EXPORT, 2)[0]
    assert sheet.parent.name == "2026-11-12-v2.6-rc2"
    lines = sheet_lines(sheet)
    mine = [x for x in lines if x["export_sha256"] == "e68eca508872b7a5"]
    assert len(mine) == len(lines) == 2
    assert len([x for x in mine if "topic" not in x]) == 1
    cyl = [x for x in mine if x.get("topic") == "CYL"]
    assert len(cyl) == 1 and cyl[0]["seed_mix"] == "fmix32" and cyl[0]["masks"] == 2


@pytest.fixture(scope="module", params=W_FIT_PARAMS)
def ex(request):
    return Export.load(request.param)


@pytest.fixture(scope="module")
def offsets(ex) -> Path:
    return w_offsets(ex.path)


@pytest.fixture(scope="module")
def E(offsets):
    return json.loads(offsets.read_text())["E"]


@pytest.fixture(scope="module")
def data(ex, offsets):
    from ce.calclm.tools.make_ld2 import payload
    return payload(ex.path, offsets)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    return (HostCore if request.param == "host" else EmuCore)(ex, horizons=True)


def _linker(ex, seed, E, cap):
    return generate(ex.model_hz, ARMS["LD2"], seed, E=E, cap=cap)["ids"][1:]


@pytest.mark.parametrize("cap", [2, 8, 32, 64])
def test_short_caps_match_linker(core, ex, E, cap):
    assert core.load()[0] == 0
    assert core.load_E(E) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=cap, masks=True) == 0
    assert core.generate(1024) == _linker(ex, 1, E, cap)
    assert core.step() == -1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_generation_matches_linker(core, ex, E, seed):
    if isinstance(core, EmuCore) and seed == 3:
        pytest.skip("host covers seed 3; CEmu covers seeds 1 and 2")
    assert core.load()[0] == 0
    assert core.load_E(E) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == 0
    assert core.generate(1024) == _linker(ex, seed, E, 1024)
    assert core.step() == -1


def test_offset_appvar_binds_to_the_w_export(core, ex, E, data):
    """W's CLMLD2 loads on W's core; a damaged byte anywhere, or v2's offsets, are refused."""
    assert len(data) == 83
    assert core.load()[0] == 0
    assert core.load_ld2(data) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=8, masks=True) == 0
    assert core.generate(8) == _linker(ex, 1, E, 8)
    for i in range(0, len(data), 1 if not isinstance(core, EmuCore) else 7):
        damaged = data[:i] + bytes([data[i] ^ 1]) + data[i + 1:]
        assert core.load_ld2(damaged) == -11, i
    from ce.calclm.tools.make_ld2 import payload
    from ce.tools.clm import V2_EXPORT
    if (V2_EXPORT / "model.bin").exists():
        v2 = payload(V2_EXPORT, ROOT / "ce/tests/fixtures/ld2_offsets_full.json")
        assert core.load_ld2(v2) == -11


@pytest.mark.parametrize("seed", [1, 2])
def test_program_load_enables_ld2_after_every_reload(core, ex, E, data, seed):
    for cap in (8, 1024):
        assert core.load()[0] == 0
        assert core.program_decoder(data) == 0
        if isinstance(core, EmuCore):
            assert core.last.cycles < 4_800_000  # binding + defaults under 0.1 s at 48 MHz
        assert core.reset(seed, 205, 20, cap=cap, masks=True) == 0
        assert core.generate(1024) == _linker(ex, seed, E, cap)


def test_program_load_falls_back_to_v21(core, ex, data):
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate as baseline
    want = baseline(ex.model_hz, [], 31, SamplerConfig(205, 20, False, 1), cap=32,
                    mask=IntMask(ex.meta["vocab"]))[1:]
    for optional in (None, b"?" + data[1:]):
        assert core.load()[0] == 0
        assert core.program_decoder(data) == 0
        assert core.program_decoder(optional) == 0
        assert core.reset(1, 205, 20, cap=32, masks=True) == 0
        assert core.generate(1024) == want


# ---- the v2.3 hardware checks -----------------------------------------------------------------

def check_sheet(path: Path, mask_level: int = 1, topic: str | None = None) -> dict:
    """CHECK.md's machine-readable line at this mask level (a Random line, or the topic's):
    seed, tokens, lines, ids_sha256, text_sha256, body_sha256."""
    got = [x for x in sheet_lines(path) if x["masks"] == mask_level and x.get("topic") == topic]
    assert len(got) == 1, (path, mask_level, topic)
    return got[0]


@pytest.mark.slow
def test_the_hardware_check_is_what_the_core_writes(core, ex, E, data):
    """The export's sheet's seed at INSTALL check 3's settings (T 0.8, top-k 20, sampled, masks,
    1024) through the program's load policy: the core's tokens are the linker's, and their
    count, line count and hashes are the sheet's (the program text itself is not committed)."""
    from model.intref import render
    sheets = check_sheets(ex.path)
    if not sheets:
        pytest.skip(f"no hardware-check sheet for {ex.path.name}")
    CHECK = sheets[0]
    sheet = check_sheet(CHECK)
    assert sheet["export_sha256"] == ex.meta["sha256"][:16]
    assert core.load()[0] == 0
    assert core.program_decoder(data) == 0
    assert core.reset(sheet["seed"], 205, 20, cap=1024, masks=True) == 0
    got = core.generate(1100)
    assert got == _linker(ex, sheet["seed"], E, 1024)
    assert len(got) == sheet["tokens"]
    ids = b"".join(int(t).to_bytes(2, "little") for t in [1] + got)
    assert hashlib.sha256(ids).hexdigest() == sheet["ids_sha256"]
    text = render(ex.model_hz, got)
    assert len(text.split("\n")) == sheet["lines"]
    assert hashlib.sha256(text.encode()).hexdigest() == sheet["text_sha256"]
    body = b"".join(bytes.fromhex(ex.meta["vocab"][t]) for t in got if t > 3)
    assert hashlib.sha256(body).hexdigest() == sheet["body_sha256"]
    assert f"done: {sheet['tokens']} tok" in CHECK.read_text()


@pytest.mark.slow
def test_the_level_2_hardware_check_is_what_the_core_writes(core, ex, E, data):
    """The v2.4 part of the combined sheet: Random at the Masks row on (level 2) through the
    program's load policy, against the linker, the sheet's counts and hashes."""
    from model.intref import render
    sheets = check_sheets(ex.path, 2)
    if not sheets:
        pytest.skip(f"no level 2 hardware-check sheet for {ex.path.name}")
    sheet = check_sheet(sheets[0], 2)
    assert core.load()[0] == 0
    assert core.program_decoder(data) == 0
    assert core.reset(sheet["seed"], 205, 20, cap=1024, masks=2) == 0
    got = core.generate(1100)
    want = generate(ex.model_hz, ARMS["LD2"], sheet["seed"], E=E, cap=1024, mask_level=2)
    assert got == want["ids"][1:] and len(got) == sheet["tokens"]
    assert got != _linker(ex, sheet["seed"], E, 1024)       # level 2 changes this program
    ids = b"".join(int(t).to_bytes(2, "little") for t in [1] + got)
    assert hashlib.sha256(ids).hexdigest() == sheet["ids_sha256"]
    text = render(ex.model_hz, got)
    assert len(text.split("\n")) == sheet["lines"]
    assert hashlib.sha256(text.encode()).hexdigest() == sheet["text_sha256"]
    body = b"".join(bytes.fromhex(ex.meta["vocab"][t]) for t in got if t > 3)
    assert hashlib.sha256(body).hexdigest() == sheet["body_sha256"]
    assert f"done: {sheet['tokens']} tok" in sheets[0].read_text()


# ---- the OS program's load path, compiled for W ----------------------------------------------

@pytest.fixture(scope="module")
def os_model_w(request):
    """test_calclm_os.py's fileioc/RAM stand-ins and src/model.c, with W's shape forced in."""
    mp = pytest.MonkeyPatch()
    mp.setattr(osmod, "HOST_SRC", ["-include", str(SHAPES["w"]), *osmod.HOST_SRC])
    try:
        return request.getfixturevalue("os_model")
    finally:
        mp.undo()


@pytest.mark.parametrize("free_ram", [60000, 25000])
def test_os_load_finds_16_appvars_and_enables_ld2(os_model_w, ex, E, data, free_ram):
    import ctypes
    lib = os_model_w
    keep = osmod.register(lib, data, free_ram, export=ex)
    ctx = lib.model_ctx()
    assert len(ex.appvars) == 16
    assert ctx == (256 if free_ram == 60000 else (free_ram - 4096 - 64 - 2652) // 192 & ~7)
    assert lib.test_enabled() == lib.test_horizons() == lib.test_workspace_fits() == 1
    assert lib.test_tmp_size() == 6 * ctx * 32 + 2652
    fast = ctypes.c_uint8()
    assert lib.model_prepare(ctypes.byref(fast)) == 0
    out = (ctypes.c_uint16 * 1024)()
    n = lib.test_generate(2, 1024 if ctx == 256 else ctx, 1, 0, out)
    want = _linker(ex, 2, E, 1024 if ctx == 256 else ctx)
    assert list(out[:n]) == want
    assert keep
    lib.model_close()
