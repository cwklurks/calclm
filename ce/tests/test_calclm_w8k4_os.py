"""E20 step P on the host: the OS program's load path (src/model.c with test_calclm_os.py's
fileioc/RAM stand-ins) compiled for W8 with 4-bit K/V (core/clm_shape_w8k4.h): 20 model AppVars
(CLM_MAX_APPVARS 24), CLMKV4 found, checked and loaded after every load, CLMTMP sized for the
packed V cache, the scratch and the RAM tables (placement B), and generations equal to Kv4Model.
No LD2 here: CLMLD2 binds to its export's sha and W8 has no fit (the emulator tests run LD2 with
borrowed offsets through the driver).
"""
from __future__ import annotations

import ctypes

import pytest

from ce.calclm.tools.make_vocab import build
from ce.tests import test_calclm_os as osmod
from ce.tests.test_calclm_os import os_model  # noqa: F401  (compiled for w8k4 below)
from ce.tools.clm import ROOT
from ce.tools.clm_kv4 import KV4_SHAPES, Kv4Export, vc_bytes

W8 = ROOT / "runs" / "e16" / "export" / "W8-s10"
W8_S11 = ROOT / "runs" / "e16" / "export" / "W8-s11"
MARGIN = 4096 + 64


@pytest.fixture(scope="module")
def ex() -> Kv4Export:
    if not (W8 / "model.json").exists():
        pytest.skip(f"needs the export {W8.relative_to(ROOT)} (model files, not in this checkout)")
    return Kv4Export.load(W8)


@pytest.fixture(scope="module")
def os_w8k4(request):
    mp = pytest.MonkeyPatch()
    mp.setattr(osmod, "HOST_SRC", ["-include", str(KV4_SHAPES["w8k4"]), *osmod.HOST_SRC])
    try:
        return request.getfixturevalue("os_model")
    finally:
        mp.undo()


def _register(lib, ex, kv4: bytes | None, free_ram: int, archived: bool = True):
    lib.test_clear(free_ram)
    keep = []
    items = [(f"CLM{i:02d}", d, True) for i, d in enumerate(ex.appvars)]
    items.append(("CLMVOC", build(ex.path)[0], True))
    if kv4 is not None:
        items.append(("CLMKV4", kv4, archived))
    for name, raw, arc in items:
        buf = ctypes.create_string_buffer(raw)
        keep.append(buf)
        lib.test_register(name.encode(), buf, len(raw), arc)
    msg = ctypes.create_string_buffer(200)
    return lib.model_open(msg), msg.value.decode(), keep


def _want(ex, n_cap: int) -> list[int]:
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    return generate(ex.model_hz, [], n_cap - 1, SamplerConfig(205, 20, False, 2), cap=n_cap,
                    mask=IntMask(ex.meta["vocab"]))[1:]


@pytest.mark.parametrize("free_ram", [70000, 40000])
def test_os_load_finds_20_appvars_and_clmkv4(os_w8k4, ex, free_ram):
    lib = os_w8k4
    rc, msg, keep = _register(lib, ex, ex.payload, free_ram)
    assert rc == 0, msg
    ctx = lib.model_ctx()
    assert len(ex.appvars) == 20
    per = 8 * 16 + 67
    want_ctx = 256 if free_ram == 70000 else (free_ram - MARGIN - 8192 - 512) // per & ~7
    assert ctx == want_ctx
    assert lib.test_tmp_size() == vc_bytes(8, ctx, True)          # no LD2 workspace
    assert lib.test_horizons() == 1 and lib.test_enabled() == 0
    fast = ctypes.c_uint8()
    assert lib.model_prepare(ctypes.byref(fast)) == 0            # reload: CLMKV4 again
    cap = 64 if ctx == 256 else ctx
    out = (ctypes.c_uint16 * 1024)()
    n = lib.test_generate(2, cap, 1, 0, out)
    assert list(out[:n]) == _want(ex, cap)
    assert keep
    lib.model_close()


def test_os_load_refuses_a_missing_unarchived_or_foreign_clmkv4(os_w8k4, ex):
    lib = os_w8k4
    other = Kv4Export.load(W8_S11).payload
    for kv4, archived in [(None, True), (ex.payload, False), (other, True),
                          (ex.payload[:-1], True)]:
        rc, msg, keep = _register(lib, ex, kv4, 70000, archived)
        assert rc != 0 and "CLMKV4" in msg, (kv4 is None, archived)
        assert keep


def _borrowed_ld2(ex) -> tuple[bytes, list[int]]:
    """A CLMLD2 payload bound to W8-s10 (its sha and blob size) carrying rc2's fitted offsets,
    the borrowed LD2 of PLAN-E20 Decision 6 (make_ld2.payload refuses an unfitted export)."""
    import json

    from ce.calclm.tools.make_ld2 import checksum
    E = json.loads((ROOT / "ce/tests/fixtures/ld2_offsets_L4-s23.json").read_text())["E"]
    data = (b"LD2\x02" + len(ex.meta["vocab"]).to_bytes(2, "little")
            + ex.meta["blob_bytes"].to_bytes(3, "little") + bytes.fromhex(ex.meta["sha256"])[:6]
            + b"".join(x.to_bytes(4, "little") for x in E))
    return data + checksum(data).to_bytes(4, "little"), E


@pytest.mark.parametrize("free_ram", [70000, 40000])
def test_os_load_enables_borrowed_ld2_after_the_kv4_region(os_w8k4, ex, free_ram):
    """The LD2 workspace follows the kv4 region in CLMTMP (vc + CLM_KV4_VC_BYTES), aligned for
    its uint32 offsets (UBSan checks the stores), and the program's generation is the linker's
    on Kv4Model with the borrowed offsets."""
    from model.linker import ARMS, generate
    lib = os_w8k4
    data, E = _borrowed_ld2(ex)
    lib.test_clear(free_ram)
    keep = []
    items = [(f"CLM{i:02d}", d) for i, d in enumerate(ex.appvars)]
    items += [("CLMVOC", build(ex.path)[0]), ("CLMKV4", ex.payload), ("CLMLD2", data)]
    for name, raw in items:
        buf = ctypes.create_string_buffer(raw)
        keep.append(buf)
        lib.test_register(name.encode(), buf, len(raw), 1)
    msg = ctypes.create_string_buffer(200)
    assert lib.model_open(msg) == 0, msg.value
    ctx = lib.model_ctx()
    assert lib.test_enabled() == 1 and lib.test_horizons() == 1
    assert lib.test_tmp_size() == vc_bytes(8, ctx, True) + 2652
    fast = ctypes.c_uint8()
    assert lib.model_prepare(ctypes.byref(fast)) == 0 and lib.test_enabled() == 1
    cap = 1024 if ctx == 256 else ctx
    out = (ctypes.c_uint16 * 1024)()
    n = lib.test_generate(2, cap, 1, 0, out)
    assert list(out[:n]) == generate(ex.model_hz, ARMS["LD2"], 2, E=E, cap=cap)["ids"][1:]
    assert keep
    lib.model_close()
