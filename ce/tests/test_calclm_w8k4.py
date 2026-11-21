"""E20 spike (PLAN-E20.md): the core built for W8 with 4-bit K/V (core/clm_shape_w8k4.h, w8k4)
against model/kv4.py's Kv4Model with E19's stored tables, on W8-s10 (runs/e16/export/W8-s10,
tables runs/e19/kv4/W8-s10.json).

Backends: host, host-ram (gcc + UBSan; tables in place and in RAM), emu, emuflash (CEmu; CLMCODE
attached) and emuflash-ram (placement B in CEmu: T4, T5's deployment case and T6 only). Bit-exact
means identical int64 logits and identical tokens for the same seed and settings.

The kv4 tests T2-T7 are in ce/tests/kv4_cases.py; the horizon tests run in
test_calclm_w8k4_hz.py (a core with horizons on). Imported from test_calclm_v2 (the tests
test_calclm_w.py imports), with IntModel.load patched to Kv4Model and the reference cache kept
apart (kv4_cases._kv4_patches); each premise is asserted by the test itself or checked here:

| test (test_calclm_v2) | premise under Kv4Model on W8-s10 | runs on |
|---|---|---|
| test_load_valid_v2_export | format 2, ALiBi | all but emuflash-ram |
| test_load_checks_format_2_headers | FAILS: hard-codes 6 layers (pos_scheme at field 52; W8's is 64); left out, replaced by kv4_cases.test_load_checks_format_2_headers_at_this_depth | - |
| test_step0_residuals_and_logits | Kv4Model's cache quantizes at the write | idem |
| test_logits_through_the_ring | IntModel.load patched (lut_max 500 / 129) | idem |
| test_generation_matches_intref | ref_tokens on ex.model (Kv4Model) | idem |
| test_generation_stops_at_eos | find_seed finds an <eos> before 120 under Kv4Model | idem |
| test_a_smaller_cap_is_a_prefix, test_a_reduced_cache_does_not_slide | prefixes | idem |
| test_default_cap_is_1024_within_the_token_buffer, test_reset_rejects_bad_settings | none | idem |
| test_tracker_state_after_a_masked_generation | FAILS: seed 42 ends with <eos> at token 153 under Kv4Model (int8 W8-s10: none in 299), and the test feeds that <eos> to the host tracker; left out, replaced by kv4_cases.test_tracker_state_after_a_masked_generation_to_eos | - |
| test_mask_decisions_* , test_pending_read_masks | mask-only (no K/V): once on w8k4, placement A | host, emu, emuflash |

Left out: test_runs_well_past_the_window (a v2-model premise: seed 3 runs to the cap; also left
out for W), the v1/v0 checks (another shape) and v2's INSTALL check (v2's own model).
"""
from __future__ import annotations

import pytest

from ce.tests.kv4_cases import (  # noqa: F401  (fixtures and the kv4 tests, collected here)
    BACKENDS_B,
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
    test_load_checks_format_2_headers_at_this_depth,
    test_t7_memory_contract,
    test_tracker_state_after_a_masked_generation_to_eos,
)
from ce.tests.test_calclm_v2 import (  # noqa: F401  (collected here, on the kv4 build)
    test_a_reduced_cache_does_not_slide,
    test_a_smaller_cap_is_a_prefix,
    test_default_cap_is_1024_within_the_token_buffer,
    test_generation_matches_intref,
    test_generation_stops_at_eos,
    test_load_valid_v2_export,
    test_logits_through_the_ring,
    test_mask_decisions_in_list_names_and_rare_rules,
    test_mask_decisions_match_intmask_on_val_prefixes,
    test_mask_decisions_match_on_generated_programs,
    test_pending_read_masks,
    test_reset_rejects_bad_settings,
    test_step0_residuals_and_logits,
)
from ce.tools.clm import ROOT
from ce.tools.clm_kv4 import Kv4Export

W8 = ROOT / "runs" / "e16" / "export" / "W8-s10"
W8_S11 = ROOT / "runs" / "e16" / "export" / "W8-s11"
MASK_ONLY = {"test_mask_decisions_in_list_names_and_rare_rules",
             "test_mask_decisions_match_intmask_on_val_prefixes",
             "test_mask_decisions_match_on_generated_programs", "test_pending_read_masks"}


@pytest.fixture(scope="module")
def ex() -> Kv4Export:
    if not (W8 / "model.json").exists():
        pytest.skip(f"needs the export {W8.relative_to(ROOT)} (model files, not in this checkout)")
    e = Kv4Export.load(W8)
    c = e.meta["config"]
    assert (c["dim"], c["n_heads"], c["hidden_dim"], c["n_layers"]) == (256, 8, 576, 8)
    return e


@pytest.fixture(scope="module")
def other_ex() -> Kv4Export:
    return Kv4Export.load(W8_S11)


@pytest.fixture(scope="module", params=BACKENDS_B)
def core(request, ex):
    return make_core(request.param, ex, horizons=False)


@pytest.fixture(autouse=True)
def _mask_tests_once(request):
    spec = getattr(request.node, "callspec", None)
    if request.node.originalname in MASK_ONLY and spec is not None and \
            spec.params.get("core", "").endswith("-ram"):
        pytest.skip("mask-only (no K/V): run once on w8k4, placement A")


def test_w8_has_20_appvars_and_the_spike_cap(ex):
    """D5: 20 model AppVars, inside the spike header's CLM_MAX_APPVARS 24 (16 elsewhere)."""
    assert len(ex.appvars) == 20 and len(ex.payload) == 18 + 1024 + 8192
