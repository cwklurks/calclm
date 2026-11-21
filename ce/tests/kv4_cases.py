"""E20 spike (PLAN-E20.md Tests): the kv4 test cases T2-T7, shared by the w8k4 and wk4 test
modules (test_calclm_w8k4*.py, test_calclm_wk4*.py), with the fixtures they share: the test
patches of model.intref.IntModel.load and test_calclm_horizons._intref_lut_max (Design 7), the
reference caches kept apart per export, and the placement-B scope in CEmu.

Backends (fixture param of `core`): host and host-ram (gcc + UBSan, tables in flash = in place,
and copied to RAM: placement A and B), emu (CEmu, routines in RAM), emuflash (CLMCODE attached),
emuflash-ram (placement B, W8 only). Placement B runs everything on the host; in CEmu only T4,
T5's deployment case and T6 (revision 2.1 B), the rest is skipped by design there.

Every reference is Kv4Model with E19's stored tables; the int8 references of the non-vacuity
asserts are built with IntModel(load_export(path), horizons=...) (Kv4Export.int8_model), which the
patch of IntModel.load never reaches.
"""
from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from ce.tests import test_calclm_horizons as hz
from ce.tests import test_calclm_v2 as v2
from ce.tools.clm import ROOT, emu_available
from ce.tools.clm_kv4 import (
    BLK_ADDR, KC_ADDR, KROW, E_ARG, EmuKv4Core, HostKv4Core, Kv4Export, kc_bytes, tbl_bytes,
    vc_bytes,
)

OK = 0
BOS, EOS, DONE = 1, 2, -1
VAL = ROOT / "data" / "tok" / "val.bin"
LD2_BORROWED = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"   # rc2's fit
EMU_POSITIONS = hz.EMU_POSITIONS
B_EXTRA = [767, 768, 769, 1023]         # sequence B's third wrap and its last position
T6_EMU_POSITIONS = [0, 17, 255, 256, 257, 300]

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", "host-ram", pytest.param("emu", marks=_EMU),
            pytest.param("emuflash", marks=_EMU)]
BACKENDS_B = BACKENDS + [pytest.param("emuflash-ram", marks=_EMU)]
# the tests placement B runs in CEmu (PLAN-E20 Tests, revision 2.1 B)
B_EMU_TESTS = {"test_t4_logits_through_the_ring", "test_t5_deployment_with_ld2",
               "test_t6_unpack_matches_kv4model"}


def make_core(param: str, ex: Kv4Export, horizons: bool):
    ram = param.endswith("-ram")
    if param.startswith("host"):
        return HostKv4Core(ex, horizons=horizons, ram_tables=ram)
    return EmuKv4Core(ex, flash=param.startswith("emuflash"), horizons=horizons, ram_tables=ram)


def is_emu(core) -> bool:
    return isinstance(core, EmuKv4Core)


@functools.lru_cache(maxsize=None)
def sequence_a() -> list[int]:
    """rc2's hardware-check ids (seed 20, LD2 with its own fit), <bos> first."""
    from ce.tools.hw_check import reference
    return reference(ROOT / "runs" / "e10" / "export" / "L4-s23",
                     ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json", 20)["ids"]


@functools.lru_cache(maxsize=None)
def sequence_b() -> list[int]:
    """The first program of model/kv4.py's NLL set (default_rng(1902)), 1,024 tokens."""
    from model.kv4 import nll_programs
    return [int(t) for t in nll_programs(VAL)[0]]


@functools.lru_cache(maxsize=None)
def val_601() -> list[int]:
    return hz.val_program(700)[:601]


SEQUENCES = {"B": sequence_b, "val": val_601}


def borrowed_E() -> list[int]:
    return json.loads(LD2_BORROWED.read_text())["E"]


# ---- fixtures shared by the modules ------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _kv4_patches(ex):
    """Design 7: IntModel.load returns a Kv4Model with the export's stored tables (only for the
    export under test); test_calclm_horizons._intref_lut_max builds a Kv4Model with the replaced
    lut_max and horizons; both modules' reference caches (_REF) are saved, cleared and restored
    per export."""
    from model import intref
    from model.export import load_export
    from model.intref import alibi_horizons
    from model.kv4 import Kv4Model

    mine = ex.path.resolve()
    orig = intref.IntModel.__dict__["load"].__func__

    def load(cls, out_dir, horizons=False):
        if cls is intref.IntModel and Path(out_dir).resolve() == mine:
            return Kv4Model(load_export(out_dir), ex.tables, "kv4", horizons=horizons)
        return orig(cls, out_dir, horizons)

    def lut_max(ex_, lut_max, horizons=True):
        im = Kv4Model(load_export(ex_.path), ex.tables, "kv4", horizons=horizons)
        im.cfg = dataclasses.replace(im.cfg, lut_max=lut_max)
        if horizons:
            im.horizons = alibi_horizons(im.alibi_exponents, lut_max)
        return im

    mp = pytest.MonkeyPatch()
    mp.setattr(intref.IntModel, "load", classmethod(load))
    mp.setattr(hz, "_intref_lut_max", lut_max)
    saved = (dict(v2._REF), dict(hz._REF))
    v2._REF.clear()
    hz._REF.clear()
    yield
    mp.undo()
    v2._REF.clear()
    hz._REF.clear()
    v2._REF.update(saved[0])
    hz._REF.update(saved[1])


@pytest.fixture(autouse=True)
def _placement_b_scope(request):
    spec = getattr(request.node, "callspec", None)
    if spec is not None and spec.params.get("core") == "emuflash-ram" and \
            request.node.originalname not in B_EMU_TESTS:
        pytest.skip("placement B in CEmu runs T4, T5's deployment case and T6 only "
                    "(PLAN-E20 Tests, revision 2.1 B)")


# ---- references (cached per export) ------------------------------------------------------------

_LOGITS: dict = {}


def ref_logits(ex: Kv4Export, seq: str, horizons: bool, int8: bool = False) -> np.ndarray:
    from model.intref import sequence_logits
    key = (str(ex.path), seq, horizons, int8)
    if key not in _LOGITS:
        m = ex.int8_model(horizons) if int8 else (ex.model_hz if horizons else ex.model)
        _LOGITS[key] = sequence_logits(m, SEQUENCES[seq]())
    return _LOGITS[key]


_GEN: dict = {}


def ref_ld2(ex: Kv4Export, seed: int, int8: bool = False) -> list[int]:
    from model.linker import ARMS, generate
    key = (str(ex.path), seed, int8)
    if key not in _GEN:
        m = ex.int8_model(True) if int8 else ex.model_hz
        _GEN[key] = generate(m, ARMS["LD2"], seed, E=borrowed_E(), cap=1024)["ids"][1:]
    return _GEN[key]


def _with_horizons(core, on: bool):
    class _Ctx:
        def __enter__(self):
            self.saved = core.horizons
            core.horizons = on

        def __exit__(self, *exc):
            core.horizons = self.saved
    return _Ctx()


# ---- the patches themselves --------------------------------------------------------------------

def test_patches_build_kv4_references(ex):
    """IntModel.load and _intref_lut_max give Kv4Model for this export (with its tables); the
    non-vacuity references are the plain int8 IntModel."""
    from model.intref import IntModel
    from model.kv4 import Kv4Model
    a = IntModel.load(ex.path, horizons=True)
    assert isinstance(a, Kv4Model) and a.tables is ex.tables and a.horizons is not None
    b = hz._intref_lut_max(ex, 129, horizons=True)
    assert isinstance(b, Kv4Model) and b.cfg.lut_max == 129 and list(b.horizons)[-1] == 129
    assert isinstance(ex.model, Kv4Model) and ex.model.horizons is None
    assert isinstance(ex.model_hz, Kv4Model) and ex.model_hz.horizons is not None
    for h in (False, True):
        r = ex.int8_model(h)
        assert type(r) is IntModel and not isinstance(r, Kv4Model)


# ---- T2: CLMKV4 in the core --------------------------------------------------------------------

def _rechecked(data: bytes) -> bytes:
    from ce.calclm.tools.make_ld2 import checksum
    return data[:14] + checksum(data[18:]).to_bytes(4, "little") + data[18:]


def test_t2_core_refuses_a_bad_clmkv4(core, ex):
    """Wrong sha, blob size, layer count, head_dim, size, format or checksum: CLM_E_ARG (each
    damage with a valid checksum, so the field check is the one that refuses)."""
    good = ex.payload
    assert core.load(kv4=False)[0] == OK
    sha = bytes.fromhex(ex.meta["sha256"])[:6]
    blob = ex.meta["blob_bytes"]
    cases = {
        "sha arg": (good, bytes([sha[0] ^ 1]) + sha[1:]),
        "sha field": (_rechecked(good[:5] + bytes([good[5] ^ 1]) + good[6:]), sha),
        "blob size": (_rechecked(good[:11] + (blob + 1).to_bytes(3, "little") + good[14:]), sha),
        "layers": (_rechecked(good[:3] + bytes([good[3] + 1]) + good[4:]), sha),
        "head_dim": (_rechecked(good[:4] + bytes([16]) + good[5:]), sha),
        "format": (_rechecked(good[:2] + bytes([2]) + good[3:]), sha),
        "magic": (_rechecked(b"KW" + good[2:]), sha),
        "short": (good[:-1], sha),
        "long": (good + b"\x00", sha),
        "checksum": (good[:-1] + bytes([good[-1] ^ 1]), sha),
    }
    for name, (data, s) in cases.items():
        assert core.kv4_load(data, sha=s) == E_ARG, name
    assert core.kv4_load() == OK


def _sentinel_forward(core, n_tap: int) -> bool:
    """FORWARD at position 0 with the taps and the K cache pre-filled: True if the core left both
    untouched (it ran nothing)."""
    taps = bytes([0x5A]) * n_tap
    kc = bytes([0x3C]) * kc_bytes(core.ex.n_layers, core.ctx_cap)
    core._write(core._tap, taps)
    core._write(core._kc, kc)
    core.forward(BOS, 0, copy=False)
    return core._read(core._tap, n_tap) == taps and core._read(core._kc, len(kc)) == kc


def test_t2_no_tables_no_logits_and_a_reload_needs_kv4_load(core, ex):
    """A kv4 build without accepted tables runs no forward pass; clm_load clears acceptance, so
    a reload without a new KV4_LOAD runs none either (R2); with KV4_LOAD it runs (logits =
    Kv4Model's)."""
    n_tap = 4 * 2 * ex.n_layers * ex.dim
    assert core.load(tap=True, kv4=False)[0] == OK
    assert _sentinel_forward(core, n_tap)
    assert core.kv4_load() == OK
    assert not _sentinel_forward(core, n_tap)
    assert core.load(tap=True)[0] == OK                  # with KV4_LOAD
    want = ref_logits(ex, "B", False)[0]
    np.testing.assert_array_equal(core.forward(sequence_b()[0], 0), want)
    assert core.load(tap=True, kv4=False)[0] == OK       # reload, no KV4_LOAD
    assert _sentinel_forward(core, n_tap)


def test_t2_tables_are_refused_for_another_model(core, ex, other_ex):
    """The export's tables after another export's AppVars (same shape) are loaded: refused,
    whether the caller passes the other export's sha (from its CLMVOC) or this one's."""
    assert core.load(other_ex.appvars, kv4=False)[0] == OK
    other_sha = bytes.fromhex(other_ex.meta["sha256"])[:6]
    assert core.kv4_load(ex.payload, sha=other_sha) == E_ARG
    assert core.kv4_load(ex.payload) == E_ARG                    # sha matches, blob size not
    assert core.kv4_load(other_ex.payload, sha=other_sha) == OK  # its own tables load


# ---- T3: exhaustive encode ---------------------------------------------------------------------

def test_t3_exhaustive_encode(core, ex):
    """Every (layer, kind, channel) of the stored tables, all 256 int8 inputs through the core's
    encode (KV4_ENCODE) and table: kv4.encode's code and kv4.decode's value (+ 128)."""
    from model.kv4 import LEVELS, decode, encode
    if getattr(core, "flash", False):
        pytest.skip("the encode is C (no CLMCODE routine): covered on emu")
    assert core.load()[0] == OK
    x = np.arange(-128, 128, dtype=np.int64)
    m, t = ex.tables.m, ex.tables.t
    n = 0
    for layer in range(ex.n_layers):
        for kind in range(2):
            for c in range(32):
                code, val = core.kv4_encode(layer, kind, c)
                j = encode(x, m[layer, kind, c])
                np.testing.assert_array_equal(code, j + LEVELS, err_msg=f"{layer} {kind} {c}")
                want = decode(j, np.broadcast_to(t[layer, kind, c], (256, 15))) + 128
                np.testing.assert_array_equal(val, want, err_msg=f"{layer} {kind} {c}")
                n += 256
    assert n == ex.n_layers * 2 * 32 * 256


# ---- T4: logits through the ring ---------------------------------------------------------------

def _check_positions(core, seq: str, n: int) -> list[int]:
    if not is_emu(core):
        return list(range(n))
    return [p for p in EMU_POSITIONS + (B_EXTRA if seq == "B" else []) if p < n]


@pytest.mark.parametrize("seq", ["B", "val"])
@pytest.mark.parametrize("horizons", [True, False], ids=["hz", "nohz"])
def test_t4_logits_through_the_ring(core, ex, seq, horizons):
    """Teacher-forced FORWARD over sequence B (1,024 tokens, three wraps) and a val program past
    600: int64 logits equal Kv4Model's at every position (host) or at EMU_POSITIONS (CEmu),
    horizons on and off. Non-vacuity: the int8 IntModel's logits differ at a checked position."""
    toks = SEQUENCES[seq]()
    want = ref_logits(ex, seq, horizons)
    check = _check_positions(core, seq, len(toks))
    int8 = ref_logits(ex, seq, horizons, int8=True)
    assert not np.array_equal(want[check], int8[check]), "vacuous: 4-bit K/V changes nothing"
    with _with_horizons(core, horizons):
        assert core.load()[0] == OK
        for pos, tok in enumerate(toks):
            got = core.forward(tok, pos, copy=pos in check)
            if pos in check:
                np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")
    assert core.canaries_intact()


def test_t4_a_reduced_cache_does_not_slide(core, ex):
    """ctx_cap 40 (no ring), horizons on: the first 40 positions of sequence B as with the full
    window; position 40 is refused (FORWARD does nothing past the cache)."""
    toks = sequence_b()[:40]
    want = ref_logits(ex, "B", True)
    check = list(range(40)) if not is_emu(core) else [0, 4, 8, 17, 33, 39]
    assert not np.array_equal(want[check], ref_logits(ex, "B", True, int8=True)[check])
    with _with_horizons(core, True):
        assert core.load(ctx_cap=40)[0] == OK
        for pos, tok in enumerate(toks):
            got = core.forward(tok, pos, copy=pos in check)
            if pos in check:
                np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")
    assert core.canaries_intact()


# ---- T5: generations ---------------------------------------------------------------------------

def test_t5_deployment_with_ld2(core, ex):
    """Seed 20 at CHECK.md's settings (T 0.8, top-k 20, sampled, masks, cap 1024, horizons, LD2
    with rc2's offsets, borrowed through the driver's E command): token for token equal to
    model.linker.generate on Kv4Model. Non-vacuity: the int8 IntModel generates another program."""
    want = ref_ld2(ex, 20)
    assert want != ref_ld2(ex, 20, int8=True), "vacuous: 4-bit K/V changes nothing"
    with _with_horizons(core, True):
        assert core.load()[0] == OK
        assert core.load_E(borrowed_E()) == OK
        assert core.set_ld2(True) == OK
        assert core.reset(20, 205, 20, cap=1024, masks=True) == OK
        assert core.generate(1100) == want
        assert core.step() == DONE
    assert core.canaries_intact()


@pytest.mark.parametrize("horizons", [True, False], ids=["hz", "nohz"])
def test_t5_some_imported_generation_differs_from_int8(ex, horizons):
    """Non-vacuity of the imported generation tests: on their first four cases (cap 300;
    test_calclm_horizons' with horizons, test_calclm_v2's without) at least one Kv4Model
    generation differs from the int8 IntModel's."""
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    cases = hz.CASES[:4] if horizons else v2.CASES[:4]
    kv4, int8 = (ex.model_hz, ex.int8_model(True)) if horizons else (ex.model, ex.int8_model(False))
    differs = 0
    for seed, tq, k, greedy, masks in cases:
        cfg = SamplerConfig(tq, k, greedy, seed)
        mk = (lambda: IntMask(ex.meta["vocab"])) if masks else (lambda: None)
        a = generate(kv4, [], 299, cfg, cap=300, mask=mk())
        b = generate(int8, [], 299, cfg, cap=300, mask=mk())
        differs += a != b
    assert differs >= 1


# ---- T6: the unpack ----------------------------------------------------------------------------

def _kv4_cache_rows(ex, horizons: bool, positions: list[int]) -> dict:
    """Kv4Model's decoded cache rows after feeding sequence B through each position: {pos:
    (K (L, n, 35) biased with the sum term, V (L, n, 32) biased)}."""
    m = ex.model_hz if horizons else ex.model
    cache = m.new_cache()
    toks, out = sequence_b(), {}
    for p in range(max(positions) + 1):
        m.step(int(toks[p]), p, cache)
        if p in positions:
            n = min(p + 1, 256)
            k = (cache.k.a[:, :n] + 128).astype(np.int64)
            s = (-128 * k.sum(axis=2)) & 0xFFFFFF
            krow = np.concatenate([k, (s[..., None] >> np.array([0, 8, 16])) & 0xFF], axis=2)
            out[p] = (krow.astype(np.uint8), (cache.v.a[:, :n] + 128).astype(np.uint8))
    return out


_ROWS: dict = {}


def test_t6_unpack_matches_kv4model(core, ex):
    """KV4_UNPACK(layer) decodes the slots held into the scratch window: equal to Kv4Model's cache
    rows (QuantRing values + 128, with the K sum) for every layer, with the build's unpack and
    with the C one (the assembly unpack, where the eZ80 build has one, gives the same bytes); the
    RAM tables (placement B) are the payload's tables."""
    positions = EMU_POSITIONS[:-1] if not is_emu(core) else T6_EMU_POSITIONS
    key = (str(ex.path), core.horizons, tuple(positions))
    if key not in _ROWS:
        _ROWS[key] = _kv4_cache_rows(ex, core.horizons, positions)
    rows = _ROWS[key]
    assert core.load()[0] == OK
    if core.ram_tables:
        n = ex.n_layers * 64
        assert core.read_ram_tables() == ex.payload[18 + 2 * n:18 + 18 * n]
    toks = sequence_b()
    for pos in range(max(positions) + 1):
        core.forward(toks[pos], pos, copy=False)
        if pos not in positions:
            continue
        n = min(pos + 1, 256)
        wk, wv = rows[pos]
        for layer in range(ex.n_layers):
            cycles = {}
            for c_only in (False, True):
                assert core.kv4_unpack(layer, n, c_only) == OK
                cycles[c_only] = core.last.cycles if is_emu(core) else 0
                k, v = core.read_scratch(n)
                np.testing.assert_array_equal(k, wk[layer], err_msg=f"K pos {pos} l {layer}")
                np.testing.assert_array_equal(v, wv[layer], err_msg=f"V pos {pos} l {layer}")
            if is_emu(core) and n >= 8:
                assert cycles[False] * 3 < cycles[True], "the eZ80 build ran its C unpack"
    assert core.kv4_unpack(ex.n_layers, 1) == E_ARG
    assert core.kv4_unpack(0, core.ctx_cap + 1) == E_ARG
    assert core.canaries_intact()


def test_t6_tables_across_a_64k_boundary_fall_back_to_c(core, ex):
    """CEmu: CLMKV4 placed so that layer 0's K tables cross a 64 KB boundary in flash: the
    assembly unpack refuses that block and the C unpack decodes it (same bytes, C's time)."""
    if not is_emu(core) or core.ram_tables:
        pytest.skip("CEmu, tables in place only: the host has no assembly unpack, RAM tables "
                    "sit inside one bank")
    n_m = ex.n_layers * 2 * 32 * 2
    core.kv4_flash = 0x320000 - 256 - (18 + n_m)       # layer 0's K block: 256 B on each side
    try:
        rows = _kv4_cache_rows(ex, core.horizons, [16])
        assert core.load()[0] == OK
        for pos, tok in enumerate(sequence_b()[:17]):
            core.forward(tok, pos, copy=False)
        cycles = {}
        for c_only in (False, True):
            assert core.kv4_unpack(0, 17, c_only) == OK
            cycles[c_only] = core.last.cycles
            k, v = core.read_scratch(17)
            np.testing.assert_array_equal(k, rows[16][0][0])
            np.testing.assert_array_equal(v, rows[16][1][0])
        assert cycles[True] > cycles[False] > cycles[True] // 2     # K in C, V in assembly
    finally:
        del core.kv4_flash


# ---- T7: the memory contract -------------------------------------------------------------------

def test_t7_memory_contract(core, ex):
    """The packed K cache ends below the driver block; the test binary ends below the kv4 V
    region, which ends below the taps; canaries around kc, vc, the scratch and the RAM tables
    are unchanged after a fed sequence and a generation (T4 and T5 assert them too)."""
    L = ex.n_layers
    if is_emu(core):
        assert KC_ADDR + kc_bytes(L, 256) <= BLK_ADDR
        assert core.org + len(core.bin) + 0x1000 <= core.vc_addr - 16
        assert core.vc_addr + vc_bytes(L, 256, True) + 16 <= core._tap
    assert tbl_bytes(L) == L * 1024
    assert core.load()[0] == OK
    assert core.canaries_intact()
    for pos, tok in enumerate(sequence_b()[:12]):
        core.forward(tok, pos, copy=False)
    assert core.reset(1, 205, 20, cap=40, masks=True) == OK
    assert len(core.generate(40)) >= 1
    assert core.canaries_intact()
    if core.ram_tables:
        assert core.ram_tables_addr % 512 == 0      # every (layer, kind) block in one 64 KB bank
        assert core.ram_tables_addr >= core.scratch + core.ctx_cap * (KROW + 32)


# ---- replacement for an imported test whose premise fails under Kv4Model ----------------------

def test_tracker_state_after_a_masked_generation_to_eos(core, ex):
    """test_calclm_v2's tracker check (seed 42, T 1.0, top-k 40, masks; 200 tokens on the host,
    12 in CEmu) without its premise that no <eos> comes first: under Kv4Model, W8-s10 ends that
    program with <eos> as its 153rd token, which clm_step does not feed to the tracker, so the
    comparison stops before it. The tokens are compared too."""
    n = 12 if is_emu(core) else 200
    assert core.load()[0] == OK
    assert core.reset(42, 256, 40, cap=300, masks=True) == OK
    got = core.generate(n)
    assert got == v2.ref_tokens(ex, 42, 256, 40, cap=300)[:n]
    fed = got[:-1] if got and got[-1] == EOS else got
    v2._assert_tracker_state(core, ex, [BOS] + fed)


N_POS = "pos_scheme"     # placeholder: the format 2 pos_scheme field, after the layer fields


@pytest.mark.parametrize("fields,code", [
    ({1: 3}, -5), ({N_POS: 2}, -5), ({N_POS: 0}, -5), ({16 + 2: 14}, -9), ({13: 4097}, -9),
    ({13: 0}, -9), ({13: 4096, 16 + 2: 10}, OK), ({13: 4096, 16 + 2: 11}, -9),
])
def test_load_checks_format_2_headers_at_this_depth(core, ex, fields, code):
    """test_calclm_v2's header checks with the pos_scheme field placed for this export's depth
    (16 globals + 6 fields per layer: field 64 for W8; v2's test hard-codes 6 layers, field 52,
    which on W8 is a layer field, so that premise fails there)."""
    n_hdr = 16 + 6 * ex.n_layers
    patched = {n_hdr if k == N_POS else k: v for k, v in fields.items()}
    assert core.load(v2._header_patched(ex, patched))[0] == code
