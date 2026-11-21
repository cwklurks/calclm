"""The v2.3 release-candidate inputs: ce/tools/bundle.py's folder name and shape check (on the
stand-in files of test_bundle.py) and ce/tools/stage_export.py (the AppVars of an export)."""
from __future__ import annotations

import json
import zipfile

import pytest

from ce.tests.test_bundle import inputs  # noqa: F401  (the stand-in fixture)
from ce.tools import bundle
from ce.tools.clm import SHAPES, V2_EXPORT, W_EXPORTS

CFG = {"dim": 192, "n_layers": 6, "n_heads": 6, "hidden_dim": 512, "max_seq_len": 256}


def _with_config(inputs, cfg):  # noqa: F811
    prog, ex, install, extras, build, offsets = inputs
    meta = json.loads((ex / "model.json").read_text())
    (ex / "model.json").write_text(json.dumps(meta | {"config": cfg}))
    return inputs


def test_bundle_folder_names_the_release(tmp_path, inputs):  # noqa: F811
    prog, ex, install, extras, build, offsets = inputs
    out = tmp_path / "dist" / "rc.zip"
    bundle.build_bundle(out, prog, ex, install, extras, build, offsets, folder="CalcLM-v2.3-rc1")
    with zipfile.ZipFile(out) as z:
        assert {n.split("/")[0] for n in z.namelist()} == {"CalcLM-v2.3-rc1"}


@pytest.mark.parametrize("tag,ok", [("", True), ("w", False)])
def test_bundle_refuses_a_program_built_for_another_shape(tmp_path, inputs, tag, ok):  # noqa: F811
    prog, ex, install, extras, build, offsets = _with_config(inputs, CFG)
    (build / "clm_shape.h").write_text(SHAPES[tag].read_text())
    out = tmp_path / "dist" / "a.zip"
    if ok:
        bundle.build_bundle(out, prog, ex, install, extras, build, offsets)
    else:
        with pytest.raises(bundle.BundleError, match="model shape"):
            bundle.build_bundle(out, prog, ex, install, extras, build, offsets)
        assert not out.exists()


@pytest.mark.skipif(not (V2_EXPORT / "CLM00.8xv").exists(), reason="runs/v2/export not present")
def test_staging_reproduces_v2s_shipped_appvars(tmp_path):
    from ce.tools.stage_export import stage
    meta = json.loads((V2_EXPORT / "model.json").read_text())
    stage(V2_EXPORT, tmp_path / "v2")
    for name in [f"{a['name']}.8xv" for a in meta["appvars"]] + ["CLMVOC.8xv"]:
        assert (tmp_path / "v2" / name).read_bytes() == (V2_EXPORT / name).read_bytes(), name


def test_staging_refuses_to_write_into_the_export(tmp_path):
    from ce.tools.stage_export import stage
    blob = b"weights"
    (tmp_path / "model.bin").write_bytes(blob)
    (tmp_path / "model.json").write_text(json.dumps({"sha256": "00"}))
    with pytest.raises(ValueError, match="differ"):
        stage(tmp_path, tmp_path / ".")
    assert (tmp_path / "model.bin").read_bytes() == blob and not (tmp_path / "model.bin").is_symlink()


def _present(path):
    return pytest.mark.skipif(not (path / "model.bin").exists(), reason=f"{path} not present")


# The most nonzero weights in one row of any matrix (always a w2 row, 576 inputs), by the
# export's sha256 prefix: E7's W-s11, E10's L4-s23 and E24's L4c-s11.
W_NONZERO = {"a189dd1ca90063c6": 403, "fb4bc774f7e13b28": 394, "e68eca508872b7a5": 404}
W_PRESENT = [pytest.param(p, id=p.name, marks=_present(p)) for p in W_EXPORTS]


@pytest.mark.parametrize("export", W_PRESENT)
def test_accumulator_bound_by_construction(tmp_path, export):
    """INTREF's |acc| < 2^16 holds by construction when 127 times the most nonzero weights in
    any row stays below 2^16: W-s11's w2 rows (576 inputs) have at most 403, L4-s23's 394,
    E24's L4c-s11's 404. A row of 576 nonzero weights is refused by stage_export."""
    import hashlib

    from ce.tools.stage_export import accumulator_bound, stage
    meta = json.loads((export / "model.json").read_text())
    blob = (export / "model.bin").read_bytes()
    if meta["sha256"][:16] not in W_NONZERO:
        pytest.skip(f"no recorded row bound for {export}")
    assert accumulator_bound(meta, blob) == 127 * W_NONZERO[meta["sha256"][:16]] < 1 << 16
    w2 = meta["tensors"]["layers.0.w2.w"]
    bad = bytearray(blob)
    bad[w2["offset"]:w2["offset"] + w2["row_bytes"]] = bytes([242]) * w2["row_bytes"]  # +1 x 5
    assert accumulator_bound(meta, bytes(bad)) == 127 * 576
    ex = tmp_path / "ex"
    ex.mkdir()
    (ex / "model.bin").write_bytes(bytes(bad))
    (ex / "model.json").write_text(json.dumps(meta | {"sha256": hashlib.sha256(bad).hexdigest()}))
    with pytest.raises(ValueError, match="2\\^16"):
        stage(ex, tmp_path / "out")


@pytest.mark.parametrize("export", W_PRESENT)
def test_w_appvars_fit_the_calculator(tmp_path, export):
    """16 model AppVars (CLM_MAX_APPVARS), each within an AppVar's 65,505 data bytes."""
    from ce.tools.stage_export import APPVAR_MAX, stage
    sizes = stage(export, tmp_path / "w")
    model = {n: s for n, s in sizes.items() if n != "CLMVOC.8xv"}
    assert len(model) == 16 and max(sizes.values()) <= APPVAR_MAX
    meta = json.loads((export / "model.json").read_text())
    assert sum(model.values()) == meta["blob_bytes"] + 7 * 16      # 7-byte CL headers
