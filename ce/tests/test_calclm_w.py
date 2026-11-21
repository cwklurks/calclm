"""CalcLM core for the wide model shape W (dim 256, 8 heads of 32, hidden 576; lab/E7.md), the
v2.3 port, vs model/intref.py.

The format 2 tests of test_calclm_v2.py run here unchanged on each W export in W_EXPORTS (E7's
W-s11, or any export of that shape named by CALCLM_W_EXPORT; E10's L4-s23, the v2.3-rc2 model;
E24's L4c-s11, the v2.6-rc2 model),
with the core built for W (core/clm_shape_w.h; ce/tools/clm.py picks the build from the
export's dims). Backends as there:
`host` (gcc + UBSan), `emu` (the W core.o in CEmu, routines in RAM), `emuflash` (W's CLMCODE).
Left out: the v1/v0 checks (another shape) and v2's INSTALL check (v2's own model). The long
runs are v2's test on another seed (v2's seed 3 ends early on W-s11).

W-only: the program build for W, and each build refusing the other shape's AppVars.

Contract note: INTREF.md's matvec bound |acc| <= 127 * n_in < 2^16 holds for any weights only
for n_in <= 516; W's w2 has 576 inputs. The core's scale_out assumes the bound (intref raises
OverflowError past it). It holds by construction when no row has more than 516 nonzero
weights: W-s11's w2 rows have at most 403 (|acc| <= 51,181), L4-s23's 394 (|acc| <= 50,038),
E24's L4c-s11's 404 (|acc| <= 51,308);
ce/tools/stage_export.py refuses an export where it fails (test_bundle_w.py).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ce.tests import test_calclm_v2 as v2
from ce.tests.test_calclm_v2 import (  # noqa: F401  (collected here, on the W export)
    test_a_reduced_cache_does_not_slide,
    test_a_smaller_cap_is_a_prefix,
    test_default_cap_is_1024_within_the_token_buffer,
    test_generation_matches_intref,
    test_generation_stops_at_eos,
    test_load_checks_format_2_headers,
    test_load_valid_v2_export,
    test_logits_through_the_ring,
    test_mask_decisions_in_list_names_and_rare_rules,
    test_mask_decisions_match_intmask_on_val_prefixes,
    test_mask_decisions_match_on_generated_programs,
    test_pending_read_masks,
    test_reset_rejects_bad_settings,
    test_step0_residuals_and_logits,
    test_tracker_state_after_a_masked_generation,
)
from ce.tools.clm import CALCLM, W_EXPORTS, EmuCore, Export, HostCore, emu_available, out_dir

# One parameter per W export; a missing export is skipped, not failed.
W_PARAMS = [pytest.param(p, id=p.name, marks=pytest.mark.skipif(
    not (p / "model.bin").exists(), reason=f"{p} not present")) for p in W_EXPORTS]

_EMU = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")
BACKENDS = ["host", pytest.param("emu", marks=_EMU), pytest.param("emuflash", marks=_EMU)]


@pytest.fixture(scope="module", params=W_PARAMS)
def ex(request) -> Export:
    e = Export.load(request.param)
    c = e.meta["config"]
    assert (c["dim"], c["n_heads"], c["hidden_dim"], c["n_layers"]) == (256, 8, 576, 6)
    return e


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    if request.param == "host":
        return HostCore(ex)
    return EmuCore(ex, flash=request.param == "emuflash")


@pytest.fixture(scope="module", autouse=True)
def _own_reference_cache(ex):
    """v2.ref_tokens caches intref runs by sampler settings only: keep each W export's runs
    apart (this fixture is set up again for every export)."""
    saved = dict(v2._REF)
    v2._REF.clear()
    yield
    v2._REF.clear()
    v2._REF.update(saved)


# ---- long runs (test_calclm_v2's, on a W seed that reaches the cap) --------------------------

_LONG_SEED: dict = {}
LONG_SEED = {"W-s11": 4, "L4-s23": 4, "L4c-s11": 7}     # pinned per export (long_seed.log)


def _long_seed(ex) -> int:
    """The lowest seed from 4 whose programs run to 600 tokens without <eos> with masks on and
    off: 4 on W-s11 and L4-s23; on E24's L4c-s11 seed 4 ends at 561 with masks off (v2.6-rc2)."""
    if ex.path not in _LONG_SEED:
        _LONG_SEED[ex.path] = next(
            s for s in range(4, 40)
            if all(len(w) == 599 and v2.EOS not in w
                   for w in (v2.ref_tokens(ex, s, 205, 20, cap=600, masks=m) for m in (True, False))))
    return _LONG_SEED[ex.path]


@pytest.mark.slow
@pytest.mark.parametrize("cap,masks", [(300, True), (300, False), (600, True), (600, False)])
def test_runs_well_past_the_window(core, ex, cap, masks):
    """cap - 1 generated tokens, token for token, then DONE; the tracker state matches. v2's
    test uses seed 3, which ends with <eos> at 378 tokens on W-s11; here each W export's long
    seed (_long_seed: 4 on W-s11 and L4-s23) runs to 600 with masks on and off."""
    seed = _long_seed(ex)
    assert seed == LONG_SEED[ex.path.name]
    if v2.is_emu(core) and cap == 600 and not masks:
        pytest.skip("covered on the host; 600 tokens with masks covers the emulator")
    want = v2.ref_tokens(ex, seed, 205, 20, cap=cap, masks=masks)
    assert len(want) == cap - 1 and v2.EOS not in want
    assert core.load()[0] == 0
    assert core.reset(seed, 205, 20, cap=cap, masks=masks) == 0
    assert core.generate(cap - 1) == want
    assert core.step() == v2.DONE
    if masks:
        v2._assert_tracker_state(core, ex, [v2.BOS] + want)


# ---- the calculator program for W --------------------------------------------------------

@pytest.mark.skipif(not (Path.home() / "CEdev" / "bin" / "cedev-config").exists(),
                    reason="needs ~/CEdev")
def test_w_program_builds_and_fits():
    """`make SHAPE=w`: bin/w/CALCLM.8xp links W's core.o, its BSS (the K cache, the same size
    as v2.2's: MQA keeps one 32-wide K/V head) fits the toolchain's region, and bin/w's
    CLMCODE.8xv is the payload the W emulator tests attach."""
    env = dict(os.environ, PATH=f"{Path.home() / 'CEdev' / 'bin'}:{os.environ['PATH']}")
    subprocess.run(["make", "SHAPE=w"], cwd=CALCLM, env=env, check=True, capture_output=True)
    prog = CALCLM / "bin" / "w" / "CALCLM.8xp"
    assert prog.exists() and prog.stat().st_size <= 60000
    mp = (CALCLM / "bin" / "w" / "CALCLM.map").read_text()
    bss = int(mp.split("___bss_len = (___bss_high - ___bss_low)")[0].split()[-1], 16)
    assert bss <= 0xD13FD8 - 0xD052C6
    av = (CALCLM / "bin" / "w" / "CLMCODE.8xv").read_bytes()
    payload = (out_dir("w") / "clmcode.bin").read_bytes()
    assert av[74:74 + int.from_bytes(av[72:74], "little")] == payload
    code_id = int((out_dir("w") / "obj" / "code_id").read_text().strip(), 16)
    assert int.from_bytes(payload[9:13], "little") == code_id
    v22 = out_dir() / "obj" / "code_id"
    if v22.exists():
        assert v22.read_text() != (out_dir("w") / "obj" / "code_id").read_text()


def test_v22_core_refuses_a_w_export_and_back(ex):
    """Each build checks the dims it was compiled for: a W export on v2.2's core (and v2's
    export on W's) is CLM_E_SHAPE, which CALCLM reports as 'Model shape differs'."""
    from ce.tools.clm import V2_EXPORT
    if not (V2_EXPORT / "model.bin").exists():
        pytest.skip("runs/v2/export not present")
    w, v = ex, Export.load(V2_EXPORT)
    assert HostCore(v).load(w.appvars)[0] == -6
    assert HostCore(w).load(v.appvars)[0] == -6
