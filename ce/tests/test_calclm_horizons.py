"""CalcLM inference core with ALiBi horizons (PLAN-v2.1.md E1) vs model/intref.py with
`IntModel(ex, horizons=True)`, on the format 2 export runs/v2/export (S0-s10).

Same design as test_calclm_v2.py (v2's C1): every test runs on `host` (gcc + UBSan), `emu` (the
eZ80 core.o in CEmu, routines in RAM) and `emuflash` (CLMCODE attached). Bit-exact means
identical logits and identical tokens for the same seed, sampler settings, cap and masks.

With horizons, head h attends only to the keys at distance d < ceil(lut_max / 2^(5-h)): 16, 32,
64, 128, 256, 512 for v2's lut_max 512, so heads 0-3 drop keys. Past position 255 the ring
wraps inside a head's horizon while slot + 1 < horizon (slots 0..14 for head 0, 0..126 for
head 3); there the core splits the head's keys into two runs. The positions below cover that
case and the one-run case for every head, at the first and the second wrap.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ce.tools.clm import EXPORTS, V2_EXPORT, EmuCore, Export, HostCore, emu_available

pytestmark = pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(),
                                reason="runs/v2/export not present")

OK = 0
BOS, EOS, DONE = 1, 2, -1
VAL = V2_EXPORT.parents[2] / "data" / "tok" / "val.bin"

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]

# Positions compared in the emulator (all positions are compared on the host). Slot = pos mod
# 256. Head 0 (horizon 16) has two runs at slots 0..14, head 1 at 0..30, head 2 at 0..62,
# head 3 at 0..126; one run at slot >= horizon - 1 (and before the first wrap).
EMU_POSITIONS = [0, 15, 16, 17, 40, 255, 256, 257, 263, 270, 271, 272, 286, 287, 288, 300, 318,
                 319, 320, 350, 381, 382, 383, 384, 400, 509, 510, 511, 512, 513, 526, 527, 528,
                 600]


@pytest.fixture(scope="module")
def ex() -> Export:
    return Export.load(V2_EXPORT)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex, horizons=True)
    return EmuCore(ex, flash=request.param == "emuflash", horizons=True)


def is_emu(core) -> bool:
    return isinstance(core, EmuCore)


def _mask(ex):
    from model.intmask import IntMask
    return IntMask(ex.meta["vocab"])


_REF: dict = {}


def ref_tokens(ex: Export, seed=1, tq=205, k=20, greedy=False, cap=300, masks=True,
               horizons=True) -> list[int]:
    """intref.generate from <bos> (sliding window, cap tokens in all): the generated ids."""
    from model.intref import SamplerConfig, generate
    key = (seed, tq, k, greedy, cap, masks, horizons)
    if key not in _REF:
        mask = _mask(ex) if masks else None
        im = ex.model_hz if horizons else ex.model
        _REF[key] = generate(im, [], cap - 1, SamplerConfig(tq, k, greedy, seed), cap=cap,
                             mask=mask)[1:]
    return _REF[key]


def val_program(min_len: int) -> list[int]:
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)
    ends = np.append(starts[1:], len(stream))
    return next(stream[a:b].tolist() for a, b in zip(starts, ends) if b - a >= min_len)


def _header_patched(ex, fields: dict):
    """AppVars with int32 header fields of CLM00 replaced (field k at byte 7 + 4k)."""
    avs = [bytearray(a) for a in ex.appvars]
    for k, value in fields.items():
        avs[0][7 + 4 * k:7 + 4 * k + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")
    return [bytes(a) for a in avs]


def _intref_lut_max(ex, lut_max, horizons=True):
    """intref with the header's lut_max replaced (as in the patched AppVars)."""
    from model.intref import IntModel, alibi_horizons
    im = IntModel.load(ex.path, horizons=horizons)
    im.cfg = dataclasses.replace(im.cfg, lut_max=lut_max)
    if horizons:
        im.horizons = alibi_horizons(im.alibi_exponents, lut_max)
    return im


# ---- the flag ---------------------------------------------------------------------------------

def test_load_turns_horizons_off(core, ex):
    """With the flag set the core is intref with horizons; a load right after, without setting
    it, runs the shipped v2 contract (intref without horizons): clm_load clears the flag. The
    two differ from position 16 (head 0's horizon)."""
    from model.intref import sequence_logits
    prog = val_program(40)[:20]
    on, off = sequence_logits(ex.model_hz, prog), sequence_logits(ex.model, prog)
    assert not np.array_equal(on[16:], off[16:])
    try:
        for want in (on, off):
            core.horizons = want is on              # the harness sets the flag after a load
            assert core.load()[0] == OK
            for pos, tok in enumerate(prog):
                got = core.forward(tok, pos, copy=pos in (15, 16, 19))
                if pos in (15, 16, 19):
                    np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")
    finally:
        core.horizons = True


def test_horizons_are_ignored_for_format_1(core):
    """Learned positions have no ALiBi penalty to cap: the flag changes nothing."""
    from model.intref import sequence_logits
    if not (EXPORTS["v1"] / "model.bin").exists():
        pytest.skip("runs/v1/export not present")
    v1 = Export.load(EXPORTS["v1"])
    other = type(core)(v1, horizons=True, **({"flash": core.flash} if is_emu(core) else {}))
    prog = val_program(40)[:24]
    want = sequence_logits(v1.model, prog)
    assert other.load()[0] == OK
    for pos, tok in enumerate(prog):
        got = other.forward(tok, pos, copy=pos in (0, 17, 23) or not is_emu(core))
        if got is not None:
            np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")


# ---- bit-exact forward pass through the ring -------------------------------------------------

@pytest.mark.parametrize("lut_max", [None, 500, 129, pytest.param(510, marks=pytest.mark.slow)])
def test_logits_through_the_ring(core, ex, lut_max):
    """Every position of a val program past the second wrap (host), or EMU_POSITIONS (emu: the
    other positions are fed, their logits not compared). Patched lut_max values give horizons
    that are not multiples of 8 and cut more heads: 500 (16, 32, 63, 125, 250, 500: head 4),
    129 (5, 9, 17, 33, 65, 129: heads 4 and 5) and 510 (16, 32, 64, 128, 255, 510: head 4 with
    255 keys, whose two runs at slot 253 fill all CTX + 8 weight bytes). 500 runs on the host
    only."""
    from model.intref import sequence_logits
    if is_emu(core) and lut_max == 500:
        pytest.skip("500 is covered on the host; 129 covers the emulator")
    prog = val_program(700)
    n = 601
    if lut_max is None:
        im, im_off, avs = ex.model_hz, ex.model, ex.appvars
    else:
        im, im_off = _intref_lut_max(ex, lut_max), _intref_lut_max(ex, lut_max, False)
        avs = _header_patched(ex, {13: lut_max})
    want = sequence_logits(im, prog[:n])
    check = EMU_POSITIONS if is_emu(core) else list(range(n))
    off = sequence_logits(im_off, prog[:n])
    assert not np.array_equal(want[EMU_POSITIONS[4:]], off[EMU_POSITIONS[4:]])   # not vacuous
    assert core.load(avs)[0] == OK
    for pos, tok in enumerate(prog[:n]):
        got = core.forward(tok, pos, copy=pos in check)
        if pos in check:
            np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")


# ---- token-for-token generation with horizons ---------------------------------------------------

CASES = [  # seed, temperature_q8, top_k, greedy, masks
    (1, 205, 20, False, True),          # the deployment setting
    (1, 205, 20, False, False),
    (42, 256, 40, False, True),
    (7, 512, 64, False, True),
    (0, 128, 1, False, True),           # seed 0 -> 2463534242; top_k 1 equals greedy
    (0xFFFFFFFF, 65535, 64, False, True),
    (1, 256, 40, True, True),
]


@pytest.mark.parametrize("seed,tq,k,greedy,masks", CASES)
def test_generation_matches_intref(core, ex, seed, tq, k, greedy, masks):
    """cap 300 (past the first wrap) on the host; in the emulator the deployment setting only,
    40 tokens (heads 0 and 1 cut). The first four cases give other tokens than intref without
    horizons; top-k 1, greedy and the T 256 case happen to give the same ones."""
    if is_emu(core) and (seed, tq, k, greedy, masks) != CASES[0]:
        pytest.skip("covered on the host; the deployment setting covers the emulator")
    cap = 300
    n = 40 if is_emu(core) else cap - 1
    assert core.load()[0] == OK
    assert core.reset(seed, tq, k, greedy, cap=cap, masks=masks) == OK
    assert core.generate(n) == ref_tokens(ex, seed, tq, k, greedy, cap, masks)[:n]


def test_a_reduced_cache_does_not_slide(core, ex):
    """A 40-entry cache (no ring): one run per head, starting past slot 0 once the horizon is
    shorter than the keys held; generation stops when the cache is full, as a prefix of the
    full-window output."""
    assert core.load(ctx_cap=40)[0] == OK
    assert core.reset(1, 205, 20, cap=300, masks=True) == OK
    assert core.generate(1000) == ref_tokens(ex, 1, 205, 20, cap=300)[:39]
    assert core.step() == DONE


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_1024_token_generations_with_masks(core, ex, seed):
    """cap 1024 with masks (the calculator's start setting), T 0.8 / top-k 20: token for token,
    then DONE (or <eos>); the tracker state matches. Seed 1 is the C2 benchmark program."""
    if is_emu(core) and seed == 3:
        pytest.skip("covered on the host; seeds 1 and 2 cover the emulator")
    want = ref_tokens(ex, seed, 205, 20, cap=1024, masks=True)
    assert want != ref_tokens(ex, seed, 205, 20, cap=1024, masks=True, horizons=False)
    assert core.load()[0] == OK
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == OK
    got = core.generate(1100)
    assert got == want
    assert core.step() == DONE
    if want[-1] != EOS:
        assert len(want) == 1023
    # <eos> ends generation without being fed to the tracker (clm_step): compare up to it
    _assert_tracker_state(core, [BOS] + (want[:-1] if want[-1] == EOS else want))


def _assert_tracker_state(core, fed):
    """The core's tracker state after reading `fed` equals the host tracker's (tracker.bind)."""
    import ctypes

    from tracker import bind
    lib = bind.lib()
    st = ctypes.create_string_buffer(bind.state_size())
    feat = ctypes.create_string_buffer(bind.feat_size())
    lib.tracker_reset(st)
    for t in fed:
        lib.tracker_step(st, int(t), feat)
    state, got_feat, prev = core.tracker()
    assert state == st.raw and got_feat == feat.raw and prev == fed[-1]
