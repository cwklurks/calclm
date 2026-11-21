"""E20 spike (PLAN-E20.md): the horizon tests of test_calclm_horizons.py on the w8k4 core (W8-s10
with 4-bit K/V) with ALiBi horizons on, against Kv4Model(horizons=True); IntModel.load and
_intref_lut_max patched to Kv4Model, the reference cache kept apart (kv4_cases._kv4_patches).
Backends host, host-ram, emu, emuflash (placement B in CEmu runs only T4-T6, in
test_calclm_w8k4.py).

| test (test_calclm_horizons) | premise under Kv4Model on W8-s10, asserted by the test |
|---|---|
| test_load_turns_horizons_off | logits with and without horizons differ from position 16 |
| test_logits_through_the_ring (lut_max None, 500, 129, 510) | horizons change the logits at EMU_POSITIONS[4:] |
| test_generation_matches_intref | the reference is generate on Kv4Model(horizons=True) |
| test_a_reduced_cache_does_not_slide | a prefix of the full-window output |
| test_1024_token_generations_with_masks (seeds 1-3) | horizons change the 1,024-token program |

Left out: test_horizons_are_ignored_for_format_1 (v1's shape).
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

W8 = ROOT / "runs" / "e16" / "export" / "W8-s10"


@pytest.fixture(scope="module")
def ex() -> Kv4Export:
    if not (W8 / "model.json").exists():
        pytest.skip(f"needs the export {W8.relative_to(ROOT)} (model files, not in this checkout)")
    return Kv4Export.load(W8)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    return make_core(request.param, ex, horizons=True)
