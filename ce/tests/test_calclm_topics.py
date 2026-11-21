"""CalcLM v2.5 topics (PHASE3-topic-menu.md 4b, 5) in the C core: clm_feed (a topic's opening
fed after <bos> with E14's rules) and clm_mix_seed (fmix32) against model.topics, on the W build
with each model the menu has shipped with (rc2's E10 L4-s23 for v2.5, E24's L4c-s11 for v2.6-rc2)
and its own LD2 fit, through the shared host/eZ80 driver: host (gcc + UBSan), CEmu with the
routines in RAM, CEmu with CLMCODE attached through the OS program's load policy.

Bit-exact: every candidate topic's program equals model.topics.generate_topic token for token,
for seeds 1, 2, 20, 600001 and 999999 on the host and 1 and 20 in each CEmu mode; a topic
program that reaches the 256-token cap stops there."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from ce.tools.clm import W_RC2_EXPORT, W_V26_EXPORT, EmuCore, Export, HostCore, emu_available
from model import topics as TP

FIXTURES = Path(__file__).parent / "fixtures"
BOS, EOS, OK, E_ARG, DONE = 1, 2, 0, -11, -1
KEYS = [t.key for t in TP.CANDIDATES]
HOST_SEEDS, EMU_SEEDS = (1, 2, 20, 600001, 999999), (1, 20)

MODELS = [pytest.param(p, id=p.name, marks=pytest.mark.skipif(
    not (p / "model.bin").exists(), reason=f"{p} not present"))
    for p in (W_RC2_EXPORT, W_V26_EXPORT)]
_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module", params=MODELS)
def ex(request) -> Export:
    return Export.load(request.param)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex, horizons=True)
    return EmuCore(ex, flash=request.param == "emuflash", horizons=True)


def fit(export: Path) -> Path:
    """The export's own LD2 fit (fixtures/ld2_offsets_<export>.json, as test_calclm_w_ld2)."""
    return FIXTURES / f"ld2_offsets_{export.name}.json"


def E(ex) -> list[int]:
    return json.loads(fit(ex.path).read_text())["E"]


@lru_cache(maxsize=None)
def reference(export: Path, key: str, seed: int, cap: int = TP.CAP) -> tuple[int, ...]:
    m = Export.load(export)
    return tuple(TP.generate_topic(m.model_hz, E(m), key, seed, cap=cap)["ids"])


def start(core, ex, seed: int, cap: int = TP.CAP, masks: int = 2, ld2: bool = True,
          mix: bool = False) -> None:
    """Load, the decoder (the OS policy with CLMCODE attached, else the driver's E), reset;
    mix: the reset gets the core's clm_mix_seed(seed), asked after the load (before the first
    load an emulated core has no driver to ask)."""
    assert core.load()[0] == OK
    if ld2 and isinstance(core, EmuCore) and core.flash:
        from ce.calclm.tools.make_ld2 import payload
        assert core.program_decoder(payload(ex.path, fit(ex.path))) == OK
    elif ld2:
        assert core.load_E(E(ex)) == OK
        assert core.set_ld2(True) == OK
    if mix:
        seed = core.mix_seed(seed)
    assert core.reset(seed, TP.TEMPERATURE_Q8, TP.TOP_K, cap=cap, masks=masks) == OK


def run_topic(core, ex, key: str, seed: int, cap: int = TP.CAP) -> list[int]:
    """What src/main.c does for a topic: the mixed seed, the opening fed, then generation."""
    start(core, ex, seed, cap, mix=True)
    assert core.feed(TP.prompt(key)) == OK
    return [BOS, *TP.prompt(key), *core.generate(1100)]


# ---- the seed mix ------------------------------------------------------------------------------

def test_mix_seed_is_model_topics_mix_seed(core, ex):
    assert core.load()[0] == OK
    seeds = [0, 1, 2, 20, 48, 600001, 999999, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    assert [core.mix_seed(s) for s in seeds] == [TP.mix_seed(s) for s in seeds]


# ---- clm_feed's contract -----------------------------------------------------------------------

def test_feed_refuses_specials_and_ids_outside_the_vocab(core, ex):
    start(core, ex, 5, cap=64)
    for bad in (0, 1, 2, 3, ex.vocab, ex.vocab + 1, 0xFFFF):
        assert core.feed([bad]) == E_ARG, bad
    assert core.feed([TP.prompt("CYL")[0]]) == OK            # nothing above was taken


def test_feed_refuses_a_prompt_that_fills_the_cap(core, ex):
    """<bos> + prompt must leave room for one decided token (lab.e14.check_prompt)."""
    p = TP.prompt("CYL")
    start(core, ex, 5, cap=8)
    assert core.feed(p[:6]) == OK                            # 7 tokens of 8
    assert core.feed(p[6:7]) == E_ARG
    assert len(core.generate(10)) == 1                       # the 8th: decided, then the cap
    assert core.step() == DONE


def test_feed_refuses_after_the_first_step(core, ex):
    p = TP.prompt("CIRC")
    start(core, ex, 5, cap=64)
    assert core.feed(p[:3]) == OK
    assert core.step() >= 0
    assert core.feed(p[3:4]) == E_ARG
    start(core, ex, 5, cap=2)                                # a step that only says DONE
    assert core.step() >= 0 and core.step() == DONE
    assert core.feed(p[:1]) == E_ARG


def test_an_empty_prompt_with_the_raw_seed_is_todays_generation(core, ex):
    from lab.e14 import generate_prompted
    from model.linker import ARMS, generate
    cap = 48 if isinstance(core, EmuCore) else 256
    want = generate(ex.model_hz, ARMS["LD2"], 20, E=E(ex), cap=cap, mask_level=2)["ids"]
    assert generate_prompted(ex.model_hz, ARMS["LD2"], 20, E(ex), [], cap=cap,
                             mask_level=2)["ids"] == want
    start(core, ex, 20, cap=cap)
    assert core.feed([]) == OK
    assert [BOS, *core.generate(1100)] == want


@pytest.mark.parametrize("key", ["CYL", "HEAT", "QUAD"])
def test_ban_bitmap_after_the_opening_is_intmask_level_2(core, ex, key):
    from model.intmask import IntMask
    m = IntMask(ex.meta["vocab"], level=2)
    m.reset()
    for t in [BOS, *TP.prompt(key)]:
        m.step(t)
    start(core, ex, 5, cap=TP.CAP)
    assert core.feed(TP.prompt(key)) == OK
    assert np.array_equal(core.mask(), m.banned())


def test_tracker_state_after_the_opening_is_the_tracks(core, ex):
    """clm_feed runs the tracker exactly as TRACK does for the same tokens."""
    p = TP.prompt("HYP")
    start(core, ex, 5, cap=TP.CAP)
    assert core.feed(p) == OK
    fed = core.tracker()
    start(core, ex, 5, cap=TP.CAP)
    core.track([], reset=True)
    core.track(p)
    assert core.tracker() == fed


# ---- whole topic programs ----------------------------------------------------------------------

@pytest.mark.parametrize("key", KEYS)
def test_topic_programs_match_the_reference(core, ex, key):
    seeds = EMU_SEEDS if isinstance(core, EmuCore) else HOST_SEEDS
    for seed in seeds:
        got = run_topic(core, ex, key, seed)
        assert got == list(reference(ex.path, key, seed)), (key, seed)
        assert len(got) <= TP.CAP
        assert core.step() == DONE


def test_hw_check_writes_the_topic_reference(ex):
    """ce.tools.hw_check --topic: the program the calculator writes for a topic and a seed,
    the opening included in prgmGEN's body; the sheet line names topic, seed mix and level."""
    from ce.tools.hw_check import reference as hw_reference
    from ce.tools.hw_check import sheet_line
    ref = hw_reference(ex.path, fit(ex.path), 1, topic="CYL")
    assert ref["ids"] == list(reference(ex.path, "CYL", 1))
    assert (ref["topic"], ref["seed_mix"], ref["mask_level"], ref["cap"]) == ("CYL", "fmix32", 2,
                                                                               256)
    assert ref["tokens"] == len(ref["ids"]) - 1
    want = b"".join(bytes.fromhex(ex.meta["vocab"][t]) for t in ref["ids"] if t > 3)
    assert bytes.fromhex(ref["body_hex"]) == want
    line = sheet_line(ref)
    assert " topic=CYL seed_mix=fmix32 masks=2 " in line and line.startswith(
        "<!-- hardware-check seed=1 ")
    with pytest.raises(ValueError):                  # a topic is level 2 only (review L2)
        hw_reference(ex.path, fit(ex.path), 1, mask_level=1, topic="CYL")
    assert hw_reference(ex.path, fit(ex.path), 1, mask_level=2, topic="CYL")["ids"] == ref["ids"]


@pytest.mark.slow
def test_the_topic_hardware_check_is_what_the_core_writes(core, ex):
    """The topic lines of the model's level 2 sheet (v2.5's combined sheet for L4-s23, v2.6's
    for E24's L4c-s11): the programs, counts and hashes."""
    import hashlib

    from ce.tests.test_calclm_w_ld2 import check_sheets, sheet_lines
    from model.intref import render
    sheets = check_sheets(ex.path, 2)
    lines = [x for s in sheets for x in sheet_lines(s) if "topic" in x]
    if not lines:
        pytest.skip("no topic line on a hardware-check sheet")
    for sheet in lines:
        got = run_topic(core, ex, sheet["topic"], sheet["seed"])
        assert got == list(reference(ex.path, sheet["topic"], sheet["seed"]))
        assert len(got) - 1 == sheet["tokens"]
        ids = b"".join(int(t).to_bytes(2, "little") for t in got)
        assert hashlib.sha256(ids).hexdigest() == sheet["ids_sha256"]
        text = render(ex.model_hz, got[1:])
        assert len(text.split("\n")) == sheet["lines"]
        assert hashlib.sha256(text.encode()).hexdigest() == sheet["text_sha256"]
        body = b"".join(bytes.fromhex(ex.meta["vocab"][t]) for t in got if t > 3)
        assert hashlib.sha256(body).hexdigest() == sheet["body_sha256"]
        assert f"done: {sheet['tokens']} tok" in sheets[0].read_text()


def test_topic_cap_1024_matches_the_reference_on_the_host(core, ex):
    if isinstance(core, EmuCore):
        pytest.skip("the host covers cap 1024")
    for key in ("QUAD", "HYP"):
        assert run_topic(core, ex, key, 600001, cap=1024) == list(
            reference(ex.path, key, 600001, 1024))


def _capped_seed(export: Path, key: str = "QUAD") -> int:
    """The lowest seed whose topic program is cut by the 256-token cap (QUAD reaches it most)."""
    for seed in range(1, 41):
        ids = reference(export, key, seed)
        if len(ids) == TP.CAP and ids[-1] != EOS:
            return seed
    raise AssertionError(f"no {key} program at the cap in seeds 1-40 on {export.name}")


def test_a_topic_program_stops_at_the_256_token_cap(core, ex):
    """The cap counts <bos> and the opening: a program that reaches it is exactly 256 tokens,
    the reference's, and the next step says DONE (PHASE3-topic-menu.md 2c)."""
    seed = _capped_seed(ex.path)
    got = run_topic(core, ex, "QUAD", seed)
    assert len(got) == TP.CAP and got[-1] != EOS
    assert got == list(reference(ex.path, "QUAD", seed))
    assert core.step() == DONE
