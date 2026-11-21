"""CalcLM inference core (ce/calclm/core) vs the integer reference model/intref.py.

Every test runs on two backends: `host` (the core compiled with gcc + UBSan, fast) and `emu`
(the eZ80 build, i.e. the exact core.o the CE program links, run in CEmu with the AppVars in
emulated flash and buffers at the program's memory map). Bit-exact means: identical logits
and residuals, and identical tokens for the same seed and sampler settings.
"""
from __future__ import annotations

import numpy as np
import pytest

from ce.tools.clm import EXPORTS, EmuCore, Export, HostCore, emu_available

# v1 is the shipping model; v0 (same shape, different vocab) is kept to show the program is
# not tied to one export
AVAILABLE = [k for k, p in EXPORTS.items() if (p / "model.bin").exists()]
pytestmark = pytest.mark.skipif(not AVAILABLE, reason="no trained export under runs/")

# error codes (clm.h)
OK, E_COUNT, E_APPVAR, E_GAP, E_MAGIC, E_VERSION, E_SHAPE, E_SIZE, E_SPLIT, E_RANGE, E_MEM, \
    E_ARG = 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11
EOS, DONE = 2, -1

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
# emu: the core's routines run from RAM (as linked); emuflash: the flash-resident copy
# (CLMCODE) is attached, as on a calculator with CLMCODE.8xv installed
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module", params=AVAILABLE)
def ex(request) -> Export:
    return Export.load(EXPORTS[request.param])


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex)
    return EmuCore(ex, flash=request.param == "emuflash")


def is_emu(core) -> bool:
    return isinstance(core, EmuCore)


_REF: dict = {}


def ref_tokens(ex: Export, seed=1, tq=256, k=40, greedy=False, ctx=256) -> list[int]:
    """intref.generate from an empty prompt: the generated ids (after <bos>), cached."""
    from model.intref import SamplerConfig, generate
    key = (str(ex.path), seed, tq, k, greedy)
    if key not in _REF:
        _REF[key] = generate(ex.model, [], 255, SamplerConfig(tq, k, greedy, seed))[1:]
    return _REF[key][:ctx - 1]


def find_seed(ex: Export, tq: int, k: int, eos: bool) -> int:
    """A seed whose intref run ends with <eos> (eos=True) or fills the 256-token window."""
    for seed in range(1, 200):
        toks = ref_tokens(ex, seed, tq, k)
        if (toks[-1] == EOS and len(toks) < 200) if eos else len(toks) == 255:
            return seed
    raise AssertionError("no suitable seed")


def ref_taps(ex: Export, tokens: list[int]):
    """Residual after each sublayer and logits for the last of `tokens` (fed from position 0)."""
    from model.intops import NULL_COUNTER as NC
    m = ex.model
    cache = m.new_cache()
    for pos, tok in enumerate(tokens):
        r = m.embed(tok, pos, NC)
        taps = []
        for i, lp in enumerate(m.layers):
            r = m.attention(i, lp, r, pos, cache, NC)
            taps.append(r)
            r = m.ffn(lp, r, NC)
            taps.append(r)
    from model.intops import norm_quant, scale_out
    from model.pack import lut_matvec
    x8, sx = norm_quant(r, m.norm_g, m.norm_eg, NC, m.sqrtd)
    o = m.output
    logits = scale_out(lut_matvec(o.w, x8), o.m, o.e, sx, 8, -(1 << 23), (1 << 23) - 1)
    return np.stack(taps), logits


# ---- loading and validation ------------------------------------------------------------------

def test_load_valid_export(core):
    assert core.load() == (OK, -1)


def _patched(ex, i, off, data):
    avs = [bytearray(a) for a in ex.appvars]
    avs[i][off:off + len(data)] = data
    return [bytes(a) for a in avs]


def test_load_rejects_wrong_appvar_count(core, ex):
    assert core.load(ex.appvars[:-1])[0] == E_COUNT      # CLM00 says 12
    assert core.load([])[0] == E_COUNT
    assert core.load(_patched(ex, 0, 3, bytes([13]))) == (E_COUNT, 0)


@pytest.mark.parametrize("i,off,data,code", [
    (3, 0, b"CX", E_APPVAR),           # AppVar magic
    (5, 3, bytes([13]), E_APPVAR),     # AppVar count byte
    (5, 2, bytes([4]), E_APPVAR),      # AppVar index byte
    (6, 4, bytes([0xFF]), E_GAP),      # blob_start does not continue the previous slice
])
def test_load_rejects_bad_appvar_headers(core, ex, i, off, data, code):
    assert core.load(_patched(ex, i, off, data)) == (code, i)


def test_load_rejects_swapped_and_short_appvars(core, ex):
    avs = list(ex.appvars)
    avs[4], avs[5] = avs[5], avs[4]
    assert core.load(avs) == (E_APPVAR, 4)
    avs = list(ex.appvars)
    avs[2] = avs[2][:6]
    assert core.load(avs) == (E_APPVAR, 2)


@pytest.mark.parametrize("field,value,code", [
    (0, 0x304D4C44, E_MAGIC),          # "DLM0"
    (1, 2, E_VERSION),
    (2, 128, E_SHAPE),                 # dim
    (7, 1025, E_SHAPE),                # vocab above CLM_VOCAB_MAX
    (7, 67, E_SHAPE),                  # vocab too small for top-k 64
    (7, 719, E_SIZE),                  # a valid vocab size, but not the one these AppVars hold
    (8, 128, E_SHAPE),                 # ctx
    (10, 1024, E_RANGE),               # silu_len must be 1025
])
def test_load_rejects_bad_blob_header(core, ex, field, value, code):
    data = (value & 0xFFFFFFFF).to_bytes(4, "little")
    assert core.load(_patched(ex, 0, 7 + 4 * field, data))[0] == code


def test_load_rejects_size_mismatch(core, ex):
    avs = list(ex.appvars)
    assert core.load(avs[:-1] + [avs[-1][:-1]])[0] == E_SIZE
    assert core.load(avs[:-1] + [avs[-1] + b"\0"])[0] == E_SIZE


def _recut(ex, k: int, delta: int) -> list[bytes]:
    """Move the boundary between AppVar k and k+1 by `delta` bytes (headers rewritten)."""
    meta = ex.meta["appvars"]
    starts = [a["blob_start"] for a in meta] + [meta[-1]["blob_end"]]
    starts[k + 1] += delta
    n = len(meta)
    return [b"CL" + bytes([i, n]) + starts[i].to_bytes(3, "little") + ex.blob[starts[i]:starts[i + 1]]
            for i in range(n)]


def test_load_rejects_tensor_split_across_appvars(core, ex):
    # CLM00 ends right after layers.0.w3; cutting 100 bytes earlier splits w3.e across AppVars
    assert core.load(_recut(ex, 0, -100)) == (E_SPLIT, 0)
    # an embedding row cut in the middle cannot be resolved either
    assert core.load(_recut(ex, 8, 96))[0] == E_SPLIT


def test_other_row_aligned_split_gives_same_tokens(core, ex):
    # tok_emb rows may be split anywhere between rows: move a boundary by 3 rows
    assert core.load(_recut(ex, 8, 3 * 192)) == (OK, -1)
    assert core.reset(seed=5, temperature_q8=256, top_k=40) == OK
    n = 6 if is_emu(core) else 40
    assert core.generate(n) == ref_tokens(ex, 5, 256, 40)[:n]


@pytest.mark.parametrize("k", [0, 1])
def test_appvar_boundary_between_row_scales_and_exponents(core, ex, k):
    """The exporter may cut an AppVar between a matrix's .m and .e (review H1): CLM00 ends
    after layers.0.w3.e and CLM01 after layers.1.w1.e, so moving the cut back 512 bytes leaves
    .m and .e in different AppVars. The result must not change."""
    avs = _recut(ex, k, -512)
    assert core.load(avs) == (OK, -1)
    from model.intref import sequence_logits
    seq = [1] + ref_tokens(ex, 3, 205, 20)[:2]
    want = sequence_logits(ex.model, seq)
    for pos, tok in enumerate(seq):
        np.testing.assert_array_equal(core.forward(tok, pos), want[pos], err_msg=f"pos {pos}")


def test_load_rejects_bad_memory(core):
    assert core.load(tbl_misalign=0x100)[0] == E_MEM
    assert core.load(ctx_cap=0)[0] == E_MEM
    assert core.load(ctx_cap=257)[0] == E_MEM
    assert core.load(ctx_cap=25)[0] == E_MEM     # must be a multiple of 8 (P.V reads in 8s)


def _tensor_patch(ex, name, index, data: bytes):
    """AppVar payloads with bytes of a tensor changed."""
    off = ex.meta["tensors"][name]["offset"] + index
    for i, a in enumerate(ex.meta["appvars"]):
        if a["blob_start"] <= off < a["blob_end"]:
            return _patched(ex, i, 7 + off - a["blob_start"], data)
    raise AssertionError(name)


@pytest.mark.parametrize("name,index,data", [
    ("layers.2.attn.lut", 0, b"\xfe"),   # the max key must weigh 255 (else Z can be 0)
    ("expw", 0, b"\0\0"),               # EXPW[0] must be nonzero (else the total can be 0)
    ("layers.3.wq.m", 5, b"\x80"),       # PF7 row mantissas are 64..127
    ("output.m", 0, b"\x3f"),
])
def test_load_rejects_broken_tables(core, ex, name, index, data):
    assert core.load(_tensor_patch(ex, name, index, data))[0] == E_RANGE


def test_reset_rejects_bad_sampler_settings(core):
    assert core.load()[0] == OK
    assert core.reset(temperature_q8=32) == E_ARG
    assert core.reset(top_k=0) == E_ARG
    assert core.reset(top_k=65) == E_ARG
    assert core.reset(temperature_q8=32, top_k=0, greedy=True) == OK
    assert core.reset(temperature_q8=33, top_k=64) == OK


# ---- bit-exact forward pass --------------------------------------------------------------

def test_step0_residuals_and_logits(core, ex):
    """Localize mismatches: residual after every sublayer, then logits, for <bos> at 0."""
    assert core.load(tap=True)[0] == OK
    want_taps, want_logits = ref_taps(ex, [1])
    logits = core.forward(1, 0)
    taps = core.taps()
    for i in range(len(want_taps)):
        np.testing.assert_array_equal(taps[i], want_taps[i], err_msg=f"sublayer {i}")
    np.testing.assert_array_equal(logits, want_logits)


def test_logits_over_a_sequence(core, ex):
    seq = [1] + ref_tokens(ex, 3, 205, 20)[:7]
    if not is_emu(core):
        seq = [1] + ref_tokens(ex, 3, 205, 20)[:40]
    assert core.load(tap=True)[0] == OK
    from model.intref import sequence_logits
    want = sequence_logits(ex.model, seq)
    for pos, tok in enumerate(seq):
        np.testing.assert_array_equal(core.forward(tok, pos), want[pos], err_msg=f"pos {pos}")
    want_taps, _ = ref_taps(ex, seq)
    np.testing.assert_array_equal(core.taps(), want_taps)


# ---- token-for-token generation ----------------------------------------------------------

CASES = [  # seed, temperature_q8, top_k, greedy
    (1, 256, 40, True),
    (1, 205, 20, False),
    (42, 205, 20, False),
    (7, 512, 64, False),
    (0, 128, 1, False),        # seed 0 -> 2463534242; top_k 1 equals greedy
    (2463534242, 33, 5, False),
    (0xFFFFFFFF, 65535, 64, False),
]


@pytest.mark.parametrize("seed,tq,k,greedy", CASES)
def test_generation_matches_intref(core, ex, seed, tq, k, greedy):
    n = 12 if is_emu(core) else 255
    assert core.load()[0] == OK
    assert core.reset(seed, tq, k, greedy) == OK
    want = ref_tokens(ex, seed, tq, k, greedy)
    assert core.generate(n) == want[:n]


def test_generation_stops_at_eos(core, ex):
    seed = find_seed(ex, 256, 40, eos=True)
    want = ref_tokens(ex, seed, 256, 40)      # ends with <eos> before the window is full
    assert want[-1] == EOS and len(want) < 255
    assert core.load()[0] == OK
    assert core.reset(seed, 256, 40) == OK
    if is_emu(core):  # run the first part in one call, the rest token by token
        got = core.generate(len(want) - 3)
        got += [core.step() for _ in range(3)]
    else:
        got = core.generate(255)
    assert got == want
    assert core.step() == DONE


def test_reduced_context_is_a_prefix(core, ex):
    cap = 24
    assert core.load(ctx_cap=cap)[0] == OK
    assert core.reset(3, 256, 40) == OK
    got = core.generate(255)
    assert got == ref_tokens(ex, 3, 256, 40)[:cap - 1]
    assert core.step() == DONE


def test_reset_restarts_identically(core, ex):
    assert core.load()[0] == OK
    n = 5 if is_emu(core) else 30
    core.reset(42, 205, 20)
    a = core.generate(n)
    core.reset(42, 205, 20)
    assert core.generate(n) == a == ref_tokens(ex, 42, 205, 20)[:n]


@pytest.mark.slow
def test_full_context_run(core, ex):
    """One whole 256-token window (255 generated tokens) token for token."""
    seed = find_seed(ex, 205, 20, eos=False)
    want = ref_tokens(ex, seed, 205, 20)
    assert len(want) == 255
    assert core.load()[0] == OK
    assert core.reset(seed, 205, 20) == OK
    assert core.generate(255) == want
    assert core.step() == DONE


# ---- the flash-resident code loader ----------------------------------------------------

@pytest.fixture(scope="module")
def emuf():
    if not emu_available():
        pytest.skip("needs the CE ROM and ~/CEdev")
    return EmuCore(Export.load(), flash=False)


@pytest.mark.parametrize("patch,code", [
    ((0, b"CX"), -1),                       # magic
    ((2, b"\x02"), -1),                     # format
    ((9, b"\x00\x00\x00\x00"), -2),          # build id
    ((13, b"\xff\xff\x7f"), -3),            # entry offset beyond the code
])
def test_code_attach_rejects_bad_payloads(emuf, patch, code):
    assert emuf.load()[0] == OK
    bad = bytearray(emuf.clmcode)
    bad[patch[0]:patch[0] + len(patch[1])] = patch[1]
    assert emuf.attach(bytes(bad)) == code
    assert emuf.attach(emuf.clmcode[:20]) == -1     # truncated
    # a refused payload changes nothing: the RAM routines still give the reference tokens
    assert emuf.reset(42, 205, 20) == OK
    assert emuf.generate(3) == ref_tokens(Export.load(), 42, 205, 20)[:3]


def test_code_trampoline_outside_window_is_refused(emuf):
    assert emuf.load()[0] == OK
    p = bytearray(emuf.clmcode)
    nt = p[4] | p[5] << 8
    assert nt > 0
    off = 13 + 3 * 17                                   # first trampoline record
    p[off:off + 3] = (0xD40000).to_bytes(3, "little")   # the kernel table bank
    assert emuf.attach(bytes(p)) == -3


def test_code_attach_then_detach(emuf):
    ex = Export.load()
    assert emuf.load()[0] == OK
    assert emuf.attach() == OK
    assert emuf.reset(7, 256, 40) == OK
    fast = emuf.generate(4)
    c_fast = emuf.last.cycles
    assert emuf.detach() == OK
    assert emuf.reset(7, 256, 40) == OK
    slow = emuf.generate(4)
    assert fast == slow == ref_tokens(ex, 7, 256, 40)[:4]
    assert c_fast < 0.8 * emuf.last.cycles             # the flash copy is the faster one
