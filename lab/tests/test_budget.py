"""lab.budget: calculator cost of a ModelConfig (cycles, flash, RAM) against v1's measurements."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lab import budget as B
from lab.tests.conftest import ROOT, TINY, V1_EXPORT, have_v1, pattern_batch
from model.export import export_model, write_appvars
from model.model import ModelConfig, Transformer

V1 = ModelConfig(vocab_size=781, dim=192, n_layers=6, n_heads=6, n_kv_heads=1, hidden_dim=512,
                 max_seq_len=256)
# ce/build/calclm/clm_bench_flash.json (flash-resident code, LCD DMA on), cycles per token
MEASURED = {0: 44_177_872, 64: 47_439_772, 128: 50_731_779, 255: 57_231_289}
V1_LUT_LENS = [370, 467, 501, 442, 509, 359]
BENCH_JSON = ROOT / "ce" / "build" / "calclm" / "clm_bench_flash.json"


def measured_average(n_tokens=255):
    """Measured cycles/token over positions 0..n-1, linear between the benchmarked positions."""
    pos = sorted(MEASURED)
    return float(np.mean(np.interp(np.arange(n_tokens), pos, [MEASURED[p] for p in pos])))


# ---- cycles ----------------------------------------------------------------------------------

@pytest.mark.parametrize("pos", [0, 64, 128, 255])
def test_v1_cycles_within_3_percent_of_measured(pos):
    got = B.cycles_per_token(V1, pos)
    assert abs(got / MEASURED[pos] - 1) < 0.03, (pos, got, MEASURED[pos])


def test_v1_average_and_tokens_per_second():
    avg = B.average_cycles(V1)
    assert abs(avg / measured_average() - 1) < 0.03
    rep = B.budget(V1)
    assert rep["cycles"]["avg"] == pytest.approx(avg)
    assert rep["tokens_per_s"]["avg"] == pytest.approx(B.CPU_HZ / avg)
    assert set(rep["cycles"]) >= {0, 128, 255, "avg"}
    # a full 255-token program takes about 4.5 minutes on the calculator
    assert 240 < rep["seconds_full_program"] < 300


def test_measured_constants_match_bench_json_when_present():
    if not BENCH_JSON.exists():
        pytest.skip("ce/build/calclm/clm_bench_flash.json not built")
    rows = json.loads(BENCH_JSON.read_text())["rows"]
    assert {r["pos"]: r["cycles"] for r in rows} == MEASURED


def test_kernel_terms_are_predicted_not_fitted():
    """build + rows come from the M2 kernel benchmark (other shapes), not from v1's phases;
    they must still land within 1.5% of v1's measured matvec phases."""
    rows = B.cycles_breakdown(V1, 0)
    measured_matvec = 6_472_444 + 17_939_929  # build + rows at position 0, hooked run
    assert abs((rows["build"] + rows["rows"]) / measured_matvec - 1) < 0.015


def test_phase_breakdown_sums_to_total_and_uses_bench_phase_names():
    br = B.cycles_breakdown(V1, 128)
    assert set(br) == {"embed", "norm", "build", "rows", "scale_out", "scores", "softmax", "pv",
                       "quant", "ffn_act", "sample"}
    assert sum(br.values()) == pytest.approx(B.cycles_per_token(V1, 128))


def test_attention_grows_with_position_other_phases_do_not():
    a, b = B.cycles_breakdown(V1, 0), B.cycles_breakdown(V1, 255)
    for k in ("scores", "softmax", "pv"):
        assert b[k] > a[k]
    for k in ("build", "rows", "scale_out", "norm", "quant", "ffn_act", "embed", "sample"):
        assert b[k] == a[k]


def test_structure_scales_with_config():
    deeper = ModelConfig(**{**V1.to_dict(), "n_layers": 12})
    assert B.cycles_per_token(deeper, 0) > 1.8 * B.cycles_per_token(V1, 0) - B.cycles_breakdown(V1, 0)["sample"]
    smaller_vocab = ModelConfig(**{**V1.to_dict(), "vocab_size": 400})
    d = {k: B.cycles_breakdown(V1, 0)[k] - B.cycles_breakdown(smaller_vocab, 0)[k]
         for k in B.cycles_breakdown(V1, 0)}
    assert d["rows"] > 0 and d["scale_out"] > 0 and d["sample"] > 0
    assert d["norm"] == d["quant"] == d["ffn_act"] == d["scores"] == 0


def test_extra_cost_terms_can_be_added():
    def lookup_table(cfg, pos):
        return {"my_feature": 1000.0 * cfg.n_layers}

    base = B.cycles_per_token(V1, 0)
    assert B.cycles_per_token(V1, 0, extra_terms=[lookup_table]) == pytest.approx(base + 6000)
    assert B.budget(V1, extra_terms=[lookup_table])["phases"][0]["my_feature"] == 6000


# ---- flash ------------------------------------------------------------------------------------

def test_v1_flash_matches_export_exactly():
    fl = B.flash_bytes(V1, lut_lens=V1_LUT_LENS)
    assert fl["exact"]
    assert fl["blob_bytes"] == 726_927
    assert fl["n_appvars"] == 12
    # what INSTALL.md lists for CLM00..CLM11: the .8xv files (payload + 7-byte CL header + 76)
    assert fl["appvar_file_bytes"] == 727_923
    parts = fl["parts"]
    assert parts["ternary_packed"] == 6 * (7488 + 1248 + 1248 + 7488 + 19968 + 19968 + 19776) + 30459
    assert parts["embeddings"] == 781 * 192 + 256 * 192
    assert sum(parts.values()) == fl["blob_bytes"]


@have_v1
def test_flash_reads_lut_lengths_from_an_export():
    fl = B.flash_bytes_from_export(V1_EXPORT)
    meta = json.loads((V1_EXPORT / "model.json").read_text())
    assert fl["blob_bytes"] == meta["blob_bytes"]
    assert fl["n_appvars"] == len(meta["appvars"])
    assert fl["appvar_file_bytes"] == sum(p.stat().st_size for p in V1_EXPORT.glob("CLM*.8xv"))


def test_flash_without_lut_lengths_is_an_upper_bound():
    fl = B.flash_bytes(V1)
    assert not fl["exact"]
    assert fl["blob_bytes"] == 726_927 + sum(512 - n for n in V1_LUT_LENS)


def test_flash_layout_matches_a_fresh_export_of_another_shape(tmp_path):
    cfg = ModelConfig(vocab_size=50, dim=45, n_layers=3, n_heads=3, n_kv_heads=1,
                      hidden_dim=70, max_seq_len=40)
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    calib = pattern_batch(np.random.default_rng(0), 4, 40, 50)[:, :-1]
    model.calibrate(torch.from_numpy(calib))
    meta = export_model(model, calib, tmp_path, appvar_capacity=4000)
    paths = write_appvars(tmp_path)
    luts = [meta["tensors"][f"layers.{i}.attn.lut"]["shape"][0] for i in range(3)]
    fl = B.flash_bytes(cfg, lut_lens=luts, appvar_capacity=4000)
    assert fl["blob_bytes"] == meta["blob_bytes"]
    assert fl["n_appvars"] == len(meta["appvars"])
    assert fl["appvar_file_bytes"] == sum(Path(p).stat().st_size for p in paths)


# ---- RAM --------------------------------------------------------------------------------------

def test_v1_ram_matches_the_memory_map():
    ram = B.ram_bytes(V1)
    assert ram["k_cache"] == 53_760          # 6 x 256 x 35 (BSS)
    assert ram["v_cache"] == 49_152          # 6 x 256 x 32 (CLMTMP AppVar)
    assert ram["activations"] == 12_689      # CLM_ACT_BYTES (VRAM)
    assert ram["table_bank"] == 62_160       # pages 0..242 x columns 0..206 (VRAM)
    assert ram["total"] == 53_760 + 49_152 + 12_689 + 62_160
    assert ram["fits"], ram["problems"]


@pytest.mark.parametrize("change, problem", [
    ({"max_seq_len": 512}, "k_cache"),       # K cache no longer fits the toolchain BSS
    ({"hidden_dim": 768}, "table_bank"),     # 154 groups: table columns overflow a page
    ({"vocab_size": 1100}, "vocab"),         # CLM_VOCAB_MAX 1024
    ({"max_seq_len": 260}, "ctx"),           # the core wants a multiple of 8
    ({"n_heads": 3}, "head_dim"),            # head_dim 64: P.V compaction buffer is 96 x 32
    ({"dim": 320, "n_heads": 10, "hidden_dim": 256}, "hidden"),   # v32 holds qd values in hidden slots
])
def test_ram_reports_what_does_not_fit(change, problem):
    ram = B.ram_bytes(ModelConfig(**{**V1.to_dict(), **change}))
    assert not ram["fits"]
    assert any(problem in p for p in ram["problems"]), ram["problems"]


def test_budget_report_is_json_serializable_and_complete():
    rep = B.budget(TINY)
    json.dumps(rep)
    for key in ("config", "cycles", "tokens_per_s", "phases", "flash", "ram", "params",
                "seconds_full_program", "fits"):
        assert key in rep
    assert rep["params"]["ternary"] == Transformer(TINY).ternary_param_count()
    assert rep["params"]["embedding"] == Transformer(TINY).embedding_param_count()


# ---- v2 extras: feature embeddings, tracker, logit masks ----------------------------------------

FEATS = ModelConfig(**{**V1.to_dict(), "feat_cats": (9, 3, 4, 3, 4, 7, 5, 4, 3), "feat_bag": 64})


def test_feature_rows_cost_one_embedding_row_add_each():
    rep = B.budget(FEATS, bag_bits=4.0)
    row = B.COEFFS.embed_elem * FEATS.dim
    assert rep["phases"][0]["feat_embed"] == pytest.approx(row * (9 + 4.0))
    assert rep["phases"][0]["tracker"] == B.TRACKER_CYCLES_MEASURED == 6200
    assert rep["extras"]["tracker_cycles"]["source"].startswith("measured")
    worst = B.budget(FEATS)                                   # no measurement: every bag bit set
    assert worst["phases"][0]["feat_embed"] == pytest.approx(row * (9 + 64))
    assert worst["extras"]["bag_bits"] == {"per_token": 64, "source": "bound"}
    assert rep["params"]["embedding"] == (781 + 256 + 42 + 64) * 192


def test_v1_budget_has_no_extra_phases():
    rep = B.budget(V1)
    assert "feat_embed" not in rep["phases"][0] and "tracker" not in rep["phases"][0]
    assert rep["ram"] == B.ram_bytes(V1)


def test_masks_cost_the_tracker_and_a_check_per_maskable_id():
    rep = B.budget(V1, mask_ids=30, tracker_cycles=12_000)
    assert rep["phases"][0]["mask"] == pytest.approx(30 * B.MASK_ID_CYCLES)
    assert rep["phases"][0]["tracker"] == 12_000
    assert rep["extras"]["tracker_cycles"] == {"per_token": 12_000, "source": "given"}
    assert rep["ram"]["bss"] == B.ram_bytes(V1)["bss"] + B.TRACKER_STATE_BYTES


def test_flash_layout_matches_a_fresh_export_with_features(tmp_path):
    cfg = ModelConfig(vocab_size=50, dim=45, n_layers=2, n_heads=3, n_kv_heads=1, hidden_dim=70,
                      max_seq_len=40, feat_cats=(3, 5), feat_bag=6)
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    g = np.random.default_rng(0)
    calib = pattern_batch(g, 4, 40, 50)[:, :-1]
    feats = (np.stack([g.integers(0, 3, (4, 40)), g.integers(0, 5, (4, 40))], -1),
             g.integers(0, 64, (4, 40)))
    model.calibrate(torch.from_numpy(calib), *map(torch.as_tensor, feats))
    meta = export_model(model, calib, tmp_path, appvar_capacity=4000, calib_feats=feats)
    paths = write_appvars(tmp_path)
    luts = [meta["tensors"][f"layers.{i}.attn.lut"]["shape"][0] for i in range(2)]
    fl = B.flash_bytes(cfg, lut_lens=luts, appvar_capacity=4000)
    assert fl["blob_bytes"] == meta["blob_bytes"]
    assert fl["parts"]["embeddings"] == (50 + 40 + 14) * 45
    assert fl["appvar_file_bytes"] == sum(Path(p).stat().st_size for p in paths)
    assert B.flash_bytes_from_export(tmp_path)["blob_bytes"] == meta["blob_bytes"]


# ---- position schemes and the sliding window (G2a) ---------------------------------------------

ALIBI = ModelConfig(**{**V1.to_dict(), "pos": "alibi"})
NOPE = ModelConfig(**{**V1.to_dict(), "pos": "nope"})


def test_relative_schemes_add_one_embedding_row_and_alibi_pays_per_key():
    v1, al, no = (B.cycles_breakdown(c, 100) for c in (V1, ALIBI, NOPE))
    assert al["embed"] == no["embed"] == pytest.approx(v1["embed"] / 2)
    assert "alibi" not in v1 and "alibi" not in no
    assert al["alibi"] == pytest.approx(B.ALIBI_KEY_CYCLES * 6 * 6 * 101)


def test_attention_stops_growing_at_the_window():
    a, b = B.cycles_breakdown(ALIBI, 255), B.cycles_breakdown(ALIBI, 900)
    assert a == b
    with pytest.raises(ValueError):
        B.cycles_breakdown(V1, 256)          # learned positions stop at the table


def test_program_time_sums_every_position_of_a_long_program():
    sec = B.program_seconds(ALIBI, 1024)
    want = sum(B.cycles_per_token(ALIBI, p) for p in range(1023)) / B.CPU_HZ
    assert sec == pytest.approx(want)
    assert B.budget(ALIBI, cap=1024)["seconds_cap_program"] == pytest.approx(sec)
    assert B.budget(V1)["seconds_full_program"] == pytest.approx(B.program_seconds(V1, 256))


def test_relative_schemes_drop_the_position_table_from_flash(tmp_path):
    cfg = ModelConfig(vocab_size=50, dim=45, n_layers=2, n_heads=3, n_kv_heads=1, hidden_dim=70,
                      max_seq_len=40, pos="alibi")
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    calib = pattern_batch(np.random.default_rng(0), 4, 40, 50)[:, :-1]
    model.calibrate(torch.from_numpy(calib))
    meta = export_model(model, calib, tmp_path, appvar_capacity=4000)
    luts = [meta["tensors"][f"layers.{i}.attn.lut"]["shape"][0] for i in range(2)]
    fl = B.flash_bytes(cfg, lut_lens=luts, appvar_capacity=4000)
    assert fl["blob_bytes"] == meta["blob_bytes"] and fl["parts"]["embeddings"] == 50 * 45
    assert B.ram_bytes(ALIBI) == B.ram_bytes(V1)                       # 256-entry K/V caches
