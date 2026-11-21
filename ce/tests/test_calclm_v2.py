"""CalcLM inference core, export format 2 (ALiBi, 256-token sliding window, ring K/V cache,
tracker-driven logit masks) vs the integer reference model/intref.py + model/intmask.py.

Same design as test_calclm.py (format 1): every test runs on `host` (gcc + UBSan), `emu` (the
eZ80 core.o in CEmu, routines in RAM) and `emuflash` (CLMCODE attached). Bit-exact means
identical logits and residuals, identical tokens for the same seed, sampler settings, cap and
masks, identical mask decisions, and an identical tracker state.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ce.tools.clm import V2_EXPORT, EmuCore, Export, HostCore, emu_available

pytestmark = pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(),
                                reason="runs/v2/export not present")

OK, E_VERSION, E_RANGE, E_ARG = 0, -5, -9, -11
BOS, EOS, DONE = 1, 2, -1
VAL = V2_EXPORT.parents[2] / "data" / "tok" / "val.bin"

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module")
def ex() -> Export:
    return Export.load(V2_EXPORT)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex)
    return EmuCore(ex, flash=request.param == "emuflash")


def is_emu(core) -> bool:
    return isinstance(core, EmuCore)


def _mask(ex):
    from model.intmask import IntMask
    return IntMask(ex.meta["vocab"])


_REF: dict = {}


def ref_tokens(ex: Export, seed=1, tq=205, k=20, greedy=False, cap=300, masks=True) -> list[int]:
    """intref.generate from <bos> (sliding window, cap tokens in all): the generated ids."""
    from model.intref import SamplerConfig, generate
    key = (seed, tq, k, greedy, cap, masks)
    if key not in _REF:
        mask = _mask(ex) if masks else None
        _REF[key] = generate(ex.model, [], cap - 1, SamplerConfig(tq, k, greedy, seed), cap=cap,
                             mask=mask)[1:]
    return _REF[key]


def find_seed(ex, eos_before: int, masks=True) -> int:
    for seed in range(1, 400):
        toks = ref_tokens(ex, seed, 256, 40, cap=eos_before, masks=masks)
        if toks[-1] == EOS:
            return seed
    raise AssertionError("no seed ends early")


def val_programs(n=None, need=()):
    stream = np.fromfile(VAL, dtype="<u2")
    starts = np.flatnonzero(stream == BOS)
    ends = np.append(starts[1:], len(stream))
    out = [stream[a:b].tolist() for a, b in zip(starts, ends)]
    return [p for p in out if all(t in p for t in need)][:n]


# ---- loading ----------------------------------------------------------------------------

def _patched(ex, i, off, data):
    avs = [bytearray(a) for a in ex.appvars]
    avs[i][off:off + len(data)] = data
    return [bytes(a) for a in avs]


def _header_field(ex, k):
    """Byte offset of int32 header field k inside CLM00 (after the 7-byte AppVar header)."""
    return 7 + 4 * k


N_HDR = 16 + 6 * 6          # globals + per-layer fields; format 2's pos_scheme follows


def test_load_valid_v2_export(core, ex):
    assert ex.meta["version"] == 2 and ex.meta["positions"]["scheme"] == "alibi"
    assert core.load() == (OK, -1)


def _header_patched(ex, fields: dict):
    avs = ex.appvars
    for field, value in fields.items():
        data = (value & 0xFFFFFFFF).to_bytes(4, "little")
        avs = _patched(type(ex)(ex.meta, ex.blob, avs, ex.path), 0, _header_field(ex, field), data)
    return avs


@pytest.mark.parametrize("fields,code", [
    ({1: 3}, E_VERSION),                    # unknown format
    ({N_HDR: 2}, E_VERSION),                # format 2 with NoPE: not supported by this core
    ({N_HDR: 0}, E_VERSION),
    ({16 + 2: 14}, E_RANGE),                # layer 0 attn_sh 14: ALiBi scores would leave int24
    ({13: 4097}, E_RANGE),                  # lut_max outside 1..4096
    ({13: 0}, E_RANGE),
    ({13: 4096, 16 + 2: 10}, OK),           # largest ALiBi penalty 4096 << 10 = 2^22: allowed
    ({13: 4096, 16 + 2: 11}, E_RANGE),      # 2^23: refused
])
def test_load_checks_format_2_headers(core, ex, fields, code):
    assert core.load(_header_patched(ex, fields))[0] == code


def test_alibi_penalty_bound_does_not_apply_to_format_1(core):
    from ce.tools.clm import EXPORTS
    if not (EXPORTS["v1"] / "model.bin").exists():
        pytest.skip("runs/v1/export not present")
    v1 = Export.load(EXPORTS["v1"])
    assert core.load(_header_patched(v1, {13: 4096, 16 + 2: 11}))[0] == OK


# ---- bit-exact forward pass past the window -------------------------------------------------

def test_step0_residuals_and_logits(core, ex):
    """Format 2 embeds the token row only; residual after every sublayer, then logits."""
    from model.intops import NULL_COUNTER as NC
    m = ex.model
    assert core.load(tap=True)[0] == OK
    cache = m.new_cache()
    r = m.embed(BOS, 0, NC)
    want = []
    for i, lp in enumerate(m.layers):
        r = m.attention(i, lp, r, 0, cache, NC)
        want.append(r)
        r = m.ffn(lp, r, NC)
        want.append(r)
    logits = core.forward(BOS, 0)
    np.testing.assert_array_equal(core.taps(), np.stack(want))
    from model.intref import sequence_logits
    np.testing.assert_array_equal(logits, sequence_logits(m, [BOS])[0])


@pytest.mark.parametrize("lut_max", [None, 500, 129])
def test_logits_through_the_ring(core, ex, lut_max):
    """Every position of a whole val program longer than the window (host), or the positions
    around the first wrap (emu: the earlier ones are fed, their logits not compared). lut_max
    500 and 129 (header patched in both the AppVars and intref) are not multiples of 2^e, so
    the capped runs start where d << e first reaches lut_max (a rounded-up distance)."""
    import dataclasses

    from model.intref import IntModel, sequence_logits
    if is_emu(core) and lut_max == 500:
        pytest.skip("500 is covered on the host; 129 covers the emulator")
    prog = next(p for p in val_programs() if 270 <= len(p) <= 330)
    n = len(prog) if not is_emu(core) else 262
    im, avs = ex.model, ex.appvars
    if lut_max is not None:
        im = IntModel.load(ex.path)
        im.cfg = dataclasses.replace(im.cfg, lut_max=lut_max)
        avs = _header_patched(ex, {13: lut_max})
    want = sequence_logits(im, prog[:n])
    assert core.load(avs)[0] == OK
    check = range(n) if not is_emu(core) else [0, 1, 254, 255, 256, 257, 261]
    for pos, tok in enumerate(prog[:n]):
        got = core.forward(tok, pos, copy=pos in check)
        if pos in check:
            np.testing.assert_array_equal(got, want[pos], err_msg=f"pos {pos}")


# ---- token-for-token generation with the sliding window and masks -----------------------------

CASES = [  # seed, temperature_q8, top_k, greedy, masks
    (1, 205, 20, False, True),          # the deployment setting
    (1, 205, 20, False, False),
    (42, 256, 40, False, True),
    (7, 512, 64, False, True),
    (0, 128, 1, False, True),           # seed 0 -> 2463534242; top_k 1 equals greedy
    (2463534242, 33, 5, False, False),
    (0xFFFFFFFF, 65535, 64, False, True),
    (1, 256, 40, True, True),
]


@pytest.mark.parametrize("seed,tq,k,greedy,masks", CASES)
def test_generation_matches_intref(core, ex, seed, tq, k, greedy, masks):
    cap = 300
    n = 10 if is_emu(core) else cap - 1
    assert core.load()[0] == OK
    assert core.reset(seed, tq, k, greedy, cap=cap, masks=masks) == OK
    assert core.generate(n) == ref_tokens(ex, seed, tq, k, greedy, cap, masks)[:n]


def test_generation_stops_at_eos(core, ex):
    seed = find_seed(ex, 120)
    want = ref_tokens(ex, seed, 256, 40, cap=120)
    assert want[-1] == EOS and len(want) < 119
    assert core.load()[0] == OK
    assert core.reset(seed, 256, 40, cap=300, masks=True) == OK
    got = core.generate(len(want) - 2)
    got += [core.step() for _ in range(2)]
    assert got == want
    assert core.step() == DONE


@pytest.mark.slow
@pytest.mark.parametrize("cap,masks", [(300, True), (300, False), (600, True), (600, False)])
def test_runs_well_past_the_window(core, ex, cap, masks):
    """cap - 1 generated tokens, token for token, then DONE; the tracker state matches."""
    if is_emu(core) and cap == 600 and not masks:
        pytest.skip("covered on the host; 600 tokens with masks covers the emulator")
    want = ref_tokens(ex, 3, 205, 20, cap=cap, masks=masks)
    assert len(want) == cap - 1 and EOS not in want
    assert core.load()[0] == OK
    assert core.reset(3, 205, 20, cap=cap, masks=masks) == OK
    assert core.generate(cap - 1) == want
    assert core.step() == DONE
    if masks:
        _assert_tracker_state(core, ex, [BOS] + want)


def _assert_tracker_state(core, ex, fed):
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
    assert len(state) == len(st.raw) and state == st.raw
    assert got_feat == feat.raw and prev == fed[-1]


def test_tracker_state_after_a_masked_generation(core, ex):
    n = 12 if is_emu(core) else 200
    assert core.load()[0] == OK
    assert core.reset(42, 256, 40, cap=300, masks=True) == OK
    got = core.generate(n)
    # <eos> ends generation without being fed to the tracker (clm_step): compare up to it, as
    # test_calclm_horizons does. E24's L4c-s11 ends this program with <eos> at 34 tokens (rc1's
    # E22 L4c-s17 at 71).
    _assert_tracker_state(core, ex, [BOS] + (got[:-1] if got and got[-1] == EOS else got))


def test_a_smaller_cap_is_a_prefix(core, ex):
    assert core.load()[0] == OK
    assert core.reset(1, 205, 20, cap=24, masks=True) == OK
    assert core.generate(1000) == ref_tokens(ex, 1, 205, 20, cap=300)[:23]
    assert core.step() == DONE


def test_a_reduced_cache_does_not_slide(core, ex):
    """Less RAM for the K/V caches (a smaller window than the model's) stops generation when
    the cache is full, as for format 1: a prefix of the full-window output."""
    assert core.load(ctx_cap=24)[0] == OK
    assert core.reset(1, 205, 20, cap=300, masks=True) == OK
    assert core.generate(1000) == ref_tokens(ex, 1, 205, 20, cap=300)[:23]


def test_default_cap_is_1024_within_the_token_buffer(core, ex):
    assert core.load(tok_cap=40)[0] == OK
    assert core.reset(1, 205, 20, masks=True) == OK            # cap 0: the default
    assert len(core.generate(1000)) == 39                      # the buffer holds 40 tokens
    if not is_emu(core):
        assert core.load()[0] == OK
        assert core.reset(1, 256, 40, masks=False) == OK
        got = core.generate(1100)
        assert len(got) == 1023 or got[-1] == EOS


def test_reset_rejects_bad_settings(core, ex):
    assert core.load()[0] == OK
    assert core.reset(cap=1) == E_ARG                           # no room after <bos>
    assert core.reset(temperature_q8=32) == E_ARG
    assert core.reset(top_k=65, masks=True) == E_ARG
    assert core.reset(cap=2, masks=True) == OK


def test_masks_need_the_tracker_vocab(core):
    """The tracker's token classes and the mask ids are compiled for the 781-token vocab."""
    from ce.tools.clm import EXPORTS
    if not (EXPORTS["v0"] / "model.bin").exists():
        pytest.skip("runs/v0/export not present")
    other = type(core)(Export.load(EXPORTS["v0"]), **({"flash": False} if is_emu(core) else {}))
    assert other.load()[0] == OK
    assert other.reset(1, 205, 20, masks=True) == E_ARG
    assert other.reset(1, 205, 20, masks=False) == OK


# ---- mask decisions and the tracker on given prefixes ------------------------------------------

def _decisions_match(core, ex, programs, length):
    assert core.load()[0] == OK
    checked = 0
    for prog in programs:
        ours = _mask(ex)
        ours.reset()
        ours.step(BOS)
        core.track([], reset=True)
        for tok in prog[1:length]:
            assert np.array_equal(core.mask(), ours.banned()), (checked, prog[:10])
            core.track([tok])
            ours.step(tok)
            checked += 1
    return checked


def test_pending_read_masks(core, ex):
    """Rule 4 on the core: `,` after Input Str1 / Str0 and `(` after ->[A] / ->[J] are banned,
    and nothing once the variable is assigned or the statement is over (model/tests/
    test_intref_v2.py PENDING_CASES); the whole ban set equals intmask's."""
    from model.tests.test_intref_v2 import PENDING_CASES, word_ids
    vocab = ex.meta["vocab"]
    assert core.load()[0] == OK
    for words, tok, want in PENDING_CASES:
        ours = _mask(ex)
        ours.reset()
        prefix = word_ids(vocab, *words)
        for t in [BOS, *prefix]:
            ours.step(t)
        core.track(prefix, reset=True)
        got = core.mask()
        assert bool(got[word_ids(vocab, tok)[0]]) == want, (words, tok)
        assert np.array_equal(got, ours.banned()), words


def test_mask_decisions_match_intmask_on_val_prefixes(core, ex):
    """Programs with lists, strings and matrices, so every rule decides something."""
    need = [ex.meta["vocab"].index(h) for h in ("EB", "AA00")]
    progs = val_programs(None, need)
    if is_emu(core):
        assert _decisions_match(core, ex, progs[:2], 90) >= 150
    else:
        assert _decisions_match(core, ex, progs[:12] + val_programs(6), 400) > 3000


def _rare_positions(ex, prog):
    """Positions where the rarer rules decide: inside a list name, a pending read after a
    Str/matrix token, or every id but <eos> banned (the forced-<eos> rule). Returns
    {position: banned} from intmask (position p = the mask before token p)."""
    from model.intmask import CTX_LIST
    ours = _mask(ex)
    ours.reset()
    ours.step(BOS)
    out = {}
    for p, tok in enumerate(prog[1:], 1):
        b = ours.banned()
        allowed = ~b
        allowed[[0, 1, 3]] = False
        if ours.ctx == CTX_LIST or ours.prev in ours.vars or (allowed.sum() == 1 and allowed[EOS]):
            out[p] = b
        ours.step(tok)
    return out


def test_mask_decisions_in_list_names_and_rare_rules(core, ex):
    """The eZ80 build on the positions where list names, pending reads and the forced <eos>
    are decided (the tracker reads the whole prefix in one go between them)."""
    need = [ex.meta["vocab"].index("EB")]
    progs = val_programs(None, need)[: 3 if is_emu(core) else 40]
    assert core.load()[0] == OK
    kinds = {"list": 0, "forced": 0}
    for prog in progs:
        want = _rare_positions(ex, prog)
        core.track([], reset=True)
        fed = 1
        for p, b in want.items():
            while fed < p:
                chunk = prog[fed:min(p, fed + 256)]
                core.track(chunk)
                fed += len(chunk)
            assert np.array_equal(core.mask(), b), (p, prog[:p][-8:])
            allowed = ~b
            allowed[[0, 1, 3]] = False
            kinds["forced"] += int(allowed.sum() == 1 and allowed[EOS])
            kinds["list"] += 1
    assert kinds["list"] >= (20 if is_emu(core) else 500)
    if not is_emu(core):
        assert kinds["forced"] > 0


def test_mask_decisions_match_on_generated_programs(core, ex):
    path = V2_EXPORT.parents[2] / "runs" / "lab" / "S0-s1" / "samples.jsonl"
    if not path.exists():
        pytest.skip("S0-s1 samples not present")
    rows = [json.loads(line)["ids"] for line in path.read_text().splitlines()[:8]]
    n = 1 if is_emu(core) else 8
    assert _decisions_match(core, ex, rows[:n], 100 if is_emu(core) else 1024) >= 99


@pytest.mark.slow
def test_the_install_md_hardware_check_is_what_the_core_writes(core, ex):
    """INSTALL.md's token-for-token check (seed, token count, line count, first and last
    lines) is the core's own output for the start settings: T 0.8, top-k 20, masks, 1024, with
    ALiBi horizons and LD2 on through v2.2's program load policy (src/model.c)."""
    import re

    from ce.calclm.tools.make_ld2 import payload
    from model.intref import render
    from model.linker import ARMS, generate

    offsets = V2_EXPORT.parents[2] / "ce/tests/fixtures/ld2_offsets_full.json"
    fit = json.loads(offsets.read_text())
    install = V2_EXPORT.parents[2] / "ce" / "calclm" / "INSTALL.md"
    if not install.exists():
        pytest.skip("public export: INSTALL.md, the guide to an unpublished bundle, is not published")
    text = install.read_text()
    m = re.search(r"<!-- hardware-check seed=(\d+) tokens=(\d+) lines=(\d+) -->\n```\n(.*?)```",
                  text, re.S)
    assert m is not None
    seed, n_tok, n_lines = (int(x) for x in m.groups()[:3])
    head, tail = (part.strip("\n").split("\n") for part in m.group(4).split("...\n"))
    assert len(head) == 10 and len(tail) == 6
    assert all(line == line.rstrip(" ") for line in head + tail)
    assert core.load()[0] == OK
    assert core.program_decoder(payload(ex.path, offsets)) == OK
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == OK
    got = core.generate(1100)
    want = generate(ex.model_hz, ARMS["LD2"], seed, E=fit["E"], cap=1024)["ids"][1:]
    assert got == want
    # The same program policy without offsets must retain v2.1's decoder.
    assert core.load()[0] == OK
    assert core.program_decoder(None) == OK
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == OK
    assert core.generate(1100) != got
    assert len(got) == n_tok and got[-1] == EOS
    lines = render(ex.model_hz, [BOS] + got).split("\n")
    assert len(lines) == n_lines and lines[:len(head)] == head and lines[-len(tail):] == tail
    assert f"done: {n_tok} tok" in text and f"Seed  {seed}`" in text
    assert f"It has {n_lines} lines" in text
