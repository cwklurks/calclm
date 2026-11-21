import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model.export import APPVAR_CAPACITY, APPVAR_HEADER, export_model, load_export, write_appvars
from model.intops import sqrt_pf15
from model.model import ModelConfig, Transformer
from model.pack import unpack_ternary
from model.quant import weight_quant

TINY = ModelConfig(vocab_size=40, dim=24, n_layers=2, n_heads=3, n_kv_heads=1,
                   hidden_dim=40, max_seq_len=16)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = Transformer(TINY)
    m.calibrate(torch.randint(4, TINY.vocab_size, (2, 16)))
    return m.eval()


@pytest.fixture(scope="module")
def calib():
    return np.random.default_rng(0).integers(4, TINY.vocab_size, size=(3, 16))


def test_export_is_deterministic(tiny_model, calib, tmp_path):
    a = export_model(tiny_model, calib, tmp_path / "a")
    b = export_model(tiny_model, calib, tmp_path / "b")
    assert (tmp_path / "a" / "model.bin").read_bytes() == (tmp_path / "b" / "model.bin").read_bytes()
    assert a["sha256"] == b["sha256"]
    assert json.loads((tmp_path / "a" / "model.json").read_text()) == a


def test_packed_weights_match_model_ternary(tiny_model, calib, tmp_path):
    export_model(tiny_model, calib, tmp_path)
    ex = load_export(tmp_path)
    w = tiny_model.layers[1].ffn.w2.weight
    wt, bm, be = weight_quant(w)
    assert np.array_equal(unpack_ternary(ex.t("layers.1.w2.w"), w.shape[1]), wt.numpy())
    assert ex.t("layers.1.w2.m").tolist() == bm.int().tolist()
    assert ex.t("layers.1.w2.e").tolist() == be.int().tolist()
    s = tiny_model.layers[0].attn.scales()
    _, bq, eq = weight_quant(tiny_model.layers[0].attn.wq.weight, 1.0 / s[0])
    assert ex.t("layers.0.wq.m").tolist() == bq.int().tolist()
    assert ex.t("layers.0.wq.e").tolist() == eq.int().tolist()


def test_header_tensor_is_self_describing(tiny_model, calib, tmp_path):
    meta = export_model(tiny_model, calib, tmp_path)
    ex = load_export(tmp_path)
    hdr = ex.header()
    assert hdr["dim"] == 24 and hdr["n_layers"] == 2 and hdr["vocab"] == 40
    assert hdr["F"] == meta["scalars"]["F"] and 0 <= hdr["F"] <= 22
    assert ex.blob[:4] == b"CLM0"
    assert (hdr["sqrtd_m"], hdr["sqrtd_e"]) == sqrt_pf15(24)  # host-computed sqrt(dim) as PF15


def test_tables_and_scales_have_documented_ranges(tiny_model, calib, tmp_path):
    export_model(tiny_model, calib, tmp_path)
    ex = load_export(tmp_path)
    assert ex.t("silu").shape == (1025,) and ex.t("silu")[512] == 0 and ex.t("silu")[1024] == 2047
    expw = ex.t("expw")
    assert expw[0] == 65535 and expw[-1] == 0 and (np.diff(expw.astype(int)) <= 0).all()
    em = ex.t("tok_emb.m")
    assert ((em >= 128) & (em <= 255)).all()
    assert (np.abs(ex.t("tok_emb").astype(int)) <= 127).all()
    lut = ex.t("layers.0.attn.lut")
    assert lut[0] == 255 and lut[-1] == 0 and len(lut) <= TINY.lut_max
    for name in ("layers.0.wq.m", "layers.1.w1.m", "output.m"):
        m = ex.t(name)
        assert ((m >= 64) & (m <= 127)).all()


def test_appvar_plan_covers_blob_without_splitting_rows(tiny_model, calib, tmp_path):
    meta = export_model(tiny_model, calib, tmp_path, appvar_capacity=600)
    covered = np.zeros(meta["blob_bytes"], dtype=int)
    for av in meta["appvars"]:
        assert av["payload_bytes"] <= 600
        for part in av["parts"]:
            info = meta["tensors"][part["tensor"]]
            row = info["row_bytes"]
            lo = info["offset"] + part["row_start"] * row
            hi = info["offset"] + part["row_end"] * row
            covered[lo:hi] += 1
    assert (covered == 1).all()
    assert APPVAR_CAPACITY <= 65505


def test_appvar_files_carry_their_blob_offset(tiny_model, calib, tmp_path):
    from tivars.types import TIAppVar

    meta = export_model(tiny_model, calib, tmp_path, appvar_capacity=900)
    blob = (tmp_path / "model.bin").read_bytes()
    paths = write_appvars(tmp_path)
    assert len(paths) == len(meta["appvars"]) > 1 and APPVAR_HEADER == 7
    for i, (path, av) in enumerate(zip(paths, meta["appvars"])):
        data = bytes(TIAppVar.open(str(path)).data)
        assert data[:4] == bytes([ord("C"), ord("L"), i, len(paths)])
        assert int.from_bytes(data[4:7], "little") == av["blob_start"]
        assert data[7:] == blob[av["blob_start"] : av["blob_end"]]


# ---- extra input features (v2) ----------------------------------------------------------------

FEAT = ModelConfig(**{**TINY.to_dict(), "feat_cats": (3, 4), "feat_bag": 5})


def _feat_model_and_calib():
    torch.manual_seed(0)
    m = Transformer(FEAT)
    g = np.random.default_rng(0)
    idx = g.integers(4, FEAT.vocab_size, size=(3, 16))
    cats = np.stack([g.integers(0, 3, (3, 16)), g.integers(0, 4, (3, 16))], -1)
    bag = g.integers(0, 32, (3, 16))
    m.calibrate(torch.as_tensor(idx), torch.as_tensor(cats), torch.as_tensor(bag))
    return m.eval(), idx, (cats, bag)


def test_feature_table_is_exported_after_the_v1_tensors(tmp_path):
    m, idx, feats = _feat_model_and_calib()
    meta = export_model(m, idx, tmp_path, calib_feats=feats)
    names = list(meta["tensors"])
    assert names[-3:] == ["feat_emb", "feat_emb.m", "feat_emb.e"]
    assert names[:-3][-2:] == ["silu", "expw"]                    # v1 layout unchanged before it
    ex = load_export(tmp_path)
    w8, s = m.feat_emb.quantized()
    assert ex.t("feat_emb").shape == (12, 24)
    assert np.array_equal(ex.t("feat_emb"), w8.numpy().astype(np.int8))
    assert meta["features"] == {"cats": [3, 4], "bag": 5, "row_offsets": [0, 3], "bag_row0": 7,
                                "rows": 12}
    assert meta["counts"]["embedding_params"] == (40 + 16 + 12) * 24


def test_the_residual_peak_includes_the_feature_rows(tmp_path):
    m, idx, feats = _feat_model_and_calib()
    from model.export import residual_peak
    with torch.no_grad():
        m.feat_emb.weight.mul_(50.0)          # features dominate the embedding sum
    assert residual_peak(m, idx, feats) > 2 * residual_peak(m, idx, (feats[0] * 0, feats[1] * 0))


def test_a_feature_model_needs_calibration_features(tmp_path):
    m, idx, _ = _feat_model_and_calib()
    with pytest.raises(ValueError):
        export_model(m, idx, tmp_path)


def test_intref_refuses_a_feature_export_until_it_implements_features(tmp_path):
    from model.intref import IntModel
    m, idx, feats = _feat_model_and_calib()
    export_model(m, idx, tmp_path, calib_feats=feats)
    with pytest.raises(NotImplementedError):
        IntModel.load(tmp_path)


# ---- relative position schemes (G2a) -----------------------------------------------------------

@pytest.mark.parametrize("pos", ["alibi", "nope"])
def test_relative_position_exports_have_no_position_table(tmp_path, pos):
    from model.intref import IntModel
    cfg = ModelConfig(**{**TINY.to_dict(), "pos": pos})
    torch.manual_seed(0)
    m = Transformer(cfg)
    calib = np.random.default_rng(0).integers(4, 40, size=(3, 16))
    m.calibrate(torch.as_tensor(calib))
    meta = export_model(m.eval(), calib, tmp_path)
    assert not [n for n in meta["tensors"] if n.startswith("pos_emb")]
    want = {"scheme": pos, "window": 16}
    if pos == "alibi":
        want["alibi_exponents"] = [2, 1, 0]
    assert meta["positions"] == want
    assert meta["counts"]["embedding_params"] == 40 * 24
    assert meta["version"] == 2 and IntModel.load(tmp_path).pos_scheme == pos   # INTREF.md 12


def test_worker_modules_do_not_import_torch():
    """The sampling and NLL pools spawn workers that import these modules, and torch costs each
    worker ~290 MB that none of them use. Checked in a fresh interpreter: this test process
    already has torch."""
    code = "import sys, model.intref, model.linker; print('torch' in sys.modules)"
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(root)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
