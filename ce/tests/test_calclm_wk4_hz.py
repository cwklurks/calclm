"""E20 spike (PLAN-E20.md): the horizon tests of test_calclm_horizons.py on the wk4 core (L4-s10
with 4-bit K/V, 6 layers) with ALiBi horizons on, against Kv4Model(horizons=True); the same
imports, patches and premises as test_calclm_w8k4_hz.py (see its table). Backends host,
host-ram, emu, emuflash.
"""
from __future__ import annotations

import pytest

from ce.tests.kv4_cases import BACKENDS, _kv4_patches, make_core  # noqa: F401
from ce.tests.test_calclm_horizons import (  # noqa: F401  (collected here, on the kv4 build)
    test_1024_token_generations_with_masks,
    test_a_reduced_cache_does_not_slide,
    test_generation_matches_intref,
    test_load_turns_horizons_off,
    test_logits_through_the_ring,
)
from ce.tools.clm import ROOT
from ce.tools.clm_kv4 import Kv4Export

L4 = ROOT / "runs" / "e15" / "export" / "L4-s10"


@pytest.fixture(scope="module")
def ex() -> Kv4Export:
    if not (L4 / "model.json").exists():
        pytest.skip(f"needs the export {L4.relative_to(ROOT)} (model files, not in this checkout)")
    return Kv4Export.load(L4)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    return make_core(request.param, ex, horizons=True)
