"""CalcLM core with ALiBi horizons on the wide model shape W (the v2.3 port), vs model/intref.py
with `IntModel(ex, horizons=True)`: the tests of test_calclm_horizons.py, run unchanged on each
W export (W_EXPORTS: W-s11, L4-s23 and E24's L4c-s11) with the core built for W, on host / emu /
emuflash.

With 8 heads the ALiBi exponents are 7..0, so for lut_max 512 the horizons are 4, 8, 16, 32,
64, 128, 256 and 512: heads 0-5 drop keys (v2's six heads: 0-3). Past position 255 the ring
wraps inside head h's horizon at slots below horizon - 1 (0..2 for head 0, 0..126 for head 5);
test_calclm_horizons.EMU_POSITIONS (256, 257, 263, 270, 286-288, 318-320, 381-384, ...) cover
the two-run and one-run cases of every cutting head. The v1 check is left out (another shape).
"""
from __future__ import annotations

import pytest

from ce.tests import test_calclm_horizons as hz
from ce.tests.test_calclm_horizons import (  # noqa: F401  (collected here, on the W export)
    test_1024_token_generations_with_masks,
    test_a_reduced_cache_does_not_slide,
    test_generation_matches_intref,
    test_load_turns_horizons_off,
    test_logits_through_the_ring,
)
from ce.tests.test_calclm_w import W_PARAMS
from ce.tools.clm import EmuCore, Export, HostCore, emu_available

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module", params=W_PARAMS)
def ex(request) -> Export:
    return Export.load(request.param)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex, horizons=True)
    return EmuCore(ex, flash=request.param == "emuflash", horizons=True)


@pytest.fixture(scope="module", autouse=True)
def _own_reference_cache(ex):
    """hz.ref_tokens caches intref runs by sampler settings only: keep each W export's runs
    apart (this fixture is set up again for every export)."""
    saved = dict(hz._REF)
    hz._REF.clear()
    yield
    hz._REF.clear()
    hz._REF.update(saved)


def test_w_horizons_cut_six_heads(ex):
    """The premise above: W's horizons for its lut_max."""
    from model.intref import alibi_horizons
    m = ex.model_hz
    assert m.alibi_exponents == [7, 6, 5, 4, 3, 2, 1, 0]
    assert list(alibi_horizons(m.alibi_exponents, 512)) == [4, 8, 16, 32, 64, 128, 256, 512]
    assert m.cfg.lut_max == 512
