"""CalcLM v2.6-rc2 (docs/agents/release-v2.6-rc2.md; rc1's tests, docs/agents/release-v2.6-rc1.md):
the masks' generation on the new model, E24's L4c-s11 with its own LD2 fit, through the shared host/eZ80 driver (host, CEmu RAM, CEmu
with CLMCODE). The ban logic itself is the model's vocab and the core's code, both v2.5-rc1's
(test_calclm_safety.py runs it on rc2's model); here the model-dependent part: programs at
level 2 (Masks on, LD2) and level 3 (Masks off) equal the Python references token for token
(1,024 and 400 tokens on the host, 48 and 40 in CEmu; the sheet's whole level 2 program runs in
CEmu in test_calclm_w_ld2.py).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from ce.tools.clm import W_V26_EXPORT, EmuCore, Export, HostCore, emu_available

FIT = Path(__file__).parent / "fixtures" / f"ld2_offsets_{W_V26_EXPORT.name}.json"
OK, DONE = 0, -1

pytestmark = pytest.mark.skipif(not (W_V26_EXPORT / "model.bin").exists() or not FIT.exists(),
                                reason=f"{W_V26_EXPORT} or its fit not present")
_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module")
def ex() -> Export:
    return Export.load(W_V26_EXPORT)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex, horizons=True)
    return EmuCore(ex, flash=request.param == "emuflash", horizons=True)


def E() -> list[int]:
    return json.loads(FIT.read_text())["E"]


def _linker(ex, seed: int, cap: int, level: int) -> list[int]:
    from model.linker import ARMS, generate
    return generate(ex.model_hz, ARMS["LD2"], seed, E=E(), cap=cap, mask_level=level)["ids"][1:]


@lru_cache(maxsize=None)
def _changed_seeds(n: int = 3) -> tuple[int, ...]:
    """The lowest seeds whose first 48 tokens at level 2 differ from level 1's, so the level 2
    rules decide part of each program the core writes below."""
    ex = Export.load(W_V26_EXPORT)
    out = [s for s in range(1, 60) if _linker(ex, s, 48, 2) != _linker(ex, s, 48, 1)][:n]
    assert len(out) == n
    return tuple(out)


@pytest.mark.parametrize("i", range(3))
def test_level_2_generation_matches_linker(core, ex, i):
    seed = _changed_seeds()[i]
    cap = 48 if isinstance(core, EmuCore) else 1024
    want = _linker(ex, seed, cap, 2)
    assert core.load()[0] == OK
    if isinstance(core, EmuCore) and core.flash:
        from ce.calclm.tools.make_ld2 import payload
        assert core.program_decoder(payload(ex.path, FIT)) == OK    # the OS program's policy
    else:
        assert core.load_E(E()) == OK
        assert core.set_ld2(True) == OK
    assert core.reset(seed, 205, 20, cap=cap, masks=2) == OK
    assert core.generate(1100) == want
    assert core.step() == DONE


@pytest.mark.parametrize("seed", [3, 4])
def test_level_3_generation_matches_intref(core, ex, seed):
    """Masks row off: plain sampling (no LD2) with the safety and destructive bans only."""
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate
    cap = 40 if isinstance(core, EmuCore) else 400
    want = generate(ex.model_hz, [], cap, SamplerConfig(205, 20, False, seed), cap=cap,
                    mask=IntMask(ex.meta["vocab"], level=3))[1:]
    assert core.load()[0] == OK
    assert core.reset(seed, 205, 20, cap=cap, masks=3) == OK
    assert core.generate(1024) == want
