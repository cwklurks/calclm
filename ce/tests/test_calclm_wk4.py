"""E20 spike (PLAN-E20.md): the core built for the W shape with 4-bit K/V (6 layers,
core/clm_shape_wk4.h, wk4) against model/kv4.py's Kv4Model with E19's stored tables, on L4-s10
(runs/e15/export/L4-s10, tables runs/e19/kv4/L4-s10.json): the same kv4 tests and imported
test_calclm_v2 tests as test_calclm_w8k4.py (see its table; premises asserted by the tests), but
no placement B in CEmu (W8 only) and no mask-only tests (run once, on w8k4). Horizon tests:
test_calclm_wk4_hz.py. T2's other model: E15's L4-s11 (same shape, its own E19 tables).
"""
from __future__ import annotations

import pytest

from ce.tests.kv4_cases import (  # noqa: F401  (fixtures and the kv4 tests, collected here)
    BACKENDS,
    _kv4_patches,
    _placement_b_scope,
    make_core,
    test_patches_build_kv4_references,
    test_t2_core_refuses_a_bad_clmkv4,
    test_t2_no_tables_no_logits_and_a_reload_needs_kv4_load,
    test_t2_tables_are_refused_for_another_model,
    test_t3_exhaustive_encode,
    test_t4_a_reduced_cache_does_not_slide,
    test_t4_logits_through_the_ring,
    test_t5_deployment_with_ld2,
    test_t5_some_imported_generation_differs_from_int8,
    test_t6_tables_across_a_64k_boundary_fall_back_to_c,
    test_t6_unpack_matches_kv4model,
    test_t7_memory_contract,
)
from ce.tests.test_calclm_v2 import (  # noqa: F401  (collected here, on the kv4 build)
    test_a_reduced_cache_does_not_slide,
    test_a_smaller_cap_is_a_prefix,
    test_default_cap_is_1024_within_the_token_buffer,
    test_generation_matches_intref,
    test_generation_stops_at_eos,
    test_load_checks_format_2_headers,
    test_load_valid_v2_export,
    test_logits_through_the_ring,
    test_reset_rejects_bad_settings,
    test_step0_residuals_and_logits,
    test_tracker_state_after_a_masked_generation,
)
from ce.tools.clm import ROOT
from ce.tools.clm_kv4 import Kv4Export

L4 = ROOT / "runs" / "e15" / "export" / "L4-s10"
L4_S11 = ROOT / "runs" / "e15" / "export" / "L4-s11"


@pytest.fixture(scope="module")
def ex() -> Kv4Export:
    if not (L4 / "model.json").exists():
        pytest.skip(f"needs the export {L4.relative_to(ROOT)} (model files, not in this checkout)")
    e = Kv4Export.load(L4)
    c = e.meta["config"]
    assert (c["dim"], c["n_heads"], c["hidden_dim"], c["n_layers"]) == (256, 8, 576, 6)
    return e


@pytest.fixture(scope="module")
def other_ex() -> Kv4Export:
    return Kv4Export.load(L4_S11)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    return make_core(request.param, ex, horizons=False)
