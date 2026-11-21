"""ce/tools/bundle.py: the CalcLM v2.2 download (dist/CalcLM-v2.2.zip), built here from stand-in
files so the test needs neither the trained export nor the third-party extras."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from ce.calclm.tools.make_ld2 import payload
from ce.tools import bundle
from ce.tools.clm import CALCLM

sys.path.insert(0, str(CALCLM / "tools"))
from build_flash import appvar_8xv  # noqa: E402

EXTRAS = {"clibs.8xg": b"clibs stand-in" * 50, "arTIfiCE_v2.1.8xp": b"extra stand-in" * 10}


BLOB = b"model weight" * 25
SHA = hashlib.sha256(BLOB).hexdigest()                     # the stand-in export's sha256


def _clmvoc(blob_bytes: int, sha: str = SHA) -> bytes:
    data = (b"CV\x01\x00" + (5).to_bytes(2, "little") + blob_bytes.to_bytes(3, "little")
            + bytes.fromhex(sha[:12]) + bytes(40))
    return appvar_8xv("CLMVOC", data)


def _extras_zip(path, extras=EXTRAS, sums=None, extra_members=()):
    sums = {n: hashlib.sha256(d).hexdigest() for n, d in EXTRAS.items()} if sums is None else sums
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("CalcLM/", "")
        for name, data in extras.items():
            z.writestr(f"CalcLM/{name}", data)
        z.writestr("CalcLM/CLM00.8xv", b"v1 weights, not copied")
        z.writestr("CalcLM/SHA256SUMS.txt", "".join(f"{h}  {n}\n" for n, h in sums.items()))
        for name, data in extra_members:
            z.writestr(name, data)


@pytest.fixture
def inputs(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "core.o").write_bytes(b"core")
    (build / "clmcode.bin").write_bytes(b"code" * 100)
    prog = tmp_path / "bin"
    prog.mkdir()
    (prog / "CLMCODE.8xv").write_bytes(appvar_8xv("CLMCODE", b"code" * 100))
    (prog / "CALCLM.8xp").write_bytes(b"program" * 100)
    t = os.stat(build / "core.o").st_mtime
    os.utime(prog / "CALCLM.8xp", (t + 10, t + 10))          # linked after the core
    ex = tmp_path / "export"
    ex.mkdir()
    meta = {"blob_bytes": len(BLOB), "vocab": list(range(5)), "sha256": SHA,
            "appvars": [{"name": "CLM00"}, {"name": "CLM01"}]}
    (ex / "model.json").write_text(json.dumps(meta))
    (ex / "model.bin").write_bytes(BLOB)
    for a in meta["appvars"]:
        (ex / f"{a['name']}.8xv").write_bytes(appvar_8xv(a["name"], a["name"].encode() * 20))
    (ex / "CLMVOC.8xv").write_bytes(_clmvoc(300))
    install = tmp_path / "INSTALL.md"
    install.write_text("# Installing CalcLM v2.2\n")
    extras = tmp_path / "downloads" / "CalcLM.zip"
    extras.parent.mkdir()
    _extras_zip(extras)
    offsets = tmp_path / "offsets.json"
    fit = json.loads(bundle.ROOT.joinpath("ce/tests/fixtures/ld2_offsets_full.json").read_text())
    offsets.write_text(json.dumps(fit | {"export_sha256": SHA}))
    return prog, ex, install, extras, build, offsets


def _build(tmp_path, inputs, name="a.zip"):
    prog, ex, install, extras, build, offsets = inputs
    out = tmp_path / "dist" / name
    return out, bundle.build_bundle(out, prog, ex, install, extras, build, offsets)


def test_bundle_holds_the_program_model_extras_install_and_sums(tmp_path, inputs):
    out, sums = _build(tmp_path, inputs)
    with zipfile.ZipFile(out) as z:
        names = sorted(z.namelist())
        files = {n.split("/", 1)[1]: z.read(n) for n in names if not n.endswith("/")}
    assert all(n.startswith("CalcLM-v2.2/") for n in names)
    assert sorted(files) == sorted(["CALCLM.8xp", "CLMCODE.8xv", "CLM00.8xv", "CLM01.8xv",
                                   "CLMVOC.8xv", "CLMLD2.8xv", "INSTALL.md",
                                   "SHA256SUMS.txt", *EXTRAS])
    assert files["CLMLD2.8xv"] == appvar_8xv("CLMLD2", payload(inputs[1], inputs[5]))
    for name, data in EXTRAS.items():
        assert files[name] == data                      # copied as they are
    lines = files["SHA256SUMS.txt"].decode().splitlines()
    assert len(lines) == len(files) - 1
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256(files[name]).hexdigest() == digest == sums[name]


def test_bundle_is_deterministic_and_leaves_the_extras_zip_alone(tmp_path, inputs):
    before = inputs[3].read_bytes()
    a, _ = _build(tmp_path, inputs, "a.zip")
    b, _ = _build(tmp_path, inputs, "b.zip")
    assert a.read_bytes() == b.read_bytes()
    assert inputs[3].read_bytes() == before
    with zipfile.ZipFile(io.BytesIO(a.read_bytes())) as z:
        assert z.testzip() is None


def test_bundle_refuses_a_missing_appvar(tmp_path, inputs):
    (inputs[1] / "CLM01.8xv").unlink()
    with pytest.raises(bundle.BundleError, match="CLM01"):
        _build(tmp_path, inputs)


@pytest.mark.parametrize("voc", [_clmvoc(301), _clmvoc(300, "cd" * 32)])
def test_bundle_refuses_a_clmvoc_from_another_export(tmp_path, inputs, voc):
    (inputs[1] / "CLMVOC.8xv").write_bytes(voc)              # other blob size, other sha256
    with pytest.raises(bundle.BundleError, match="CLMVOC"):
        _build(tmp_path, inputs)


def test_bundle_refuses_extras_that_are_missing(tmp_path, inputs):
    _extras_zip(inputs[3], {"clibs.8xg": EXTRAS["clibs.8xg"]})
    with pytest.raises(bundle.BundleError, match="arTIfiCE"):
        _build(tmp_path, inputs)


def test_bundle_refuses_extras_that_differ_from_the_v1_checksums(tmp_path, inputs):
    _extras_zip(inputs[3], sums={"clibs.8xg": "0" * 64, "arTIfiCE_v2.1.8xp": "0" * 64})
    with pytest.raises(bundle.BundleError, match="SHA256SUMS"):
        _build(tmp_path, inputs)


def test_bundle_refuses_two_extras_with_one_name(tmp_path, inputs):
    _extras_zip(inputs[3], extra_members=[("Other/clibs.8xg", b"another copy")])
    with pytest.raises(bundle.BundleError, match="clibs"):
        _build(tmp_path, inputs)


def test_bundle_refuses_a_program_older_than_the_core(tmp_path, inputs):
    prog, build = inputs[0], inputs[4]
    t = os.stat(prog / "CALCLM.8xp").st_mtime
    os.utime(build / "core.o", (t + 10, t + 10))               # core rebuilt, program not relinked
    with pytest.raises(bundle.BundleError, match="make"):
        _build(tmp_path, inputs)


def test_bundle_refuses_a_clmcode_that_is_not_the_tested_payload(tmp_path, inputs):
    (inputs[0] / "CLMCODE.8xv").write_bytes(appvar_8xv("CLMCODE", b"old code" * 50))
    with pytest.raises(bundle.BundleError, match="CLMCODE"):
        _build(tmp_path, inputs)


@pytest.mark.skipif(not shutil.which("unzip"), reason="no unzip")
def test_the_zip_unpacks_into_a_readable_folder(tmp_path, inputs):
    out, _ = _build(tmp_path, inputs)
    dest = tmp_path / "unpacked"
    subprocess.run(["unzip", "-q", str(out), "-d", str(dest)], check=True)
    folder = dest / "CalcLM-v2.2"
    assert os.stat(folder).st_mode & 0o755 == 0o755
    assert (folder / "INSTALL.md").read_text().startswith("# Installing")
    with zipfile.ZipFile(out) as z:                            # as the v1 download's entries
        attrs = {i.filename.endswith("/"): i.external_attr for i in z.infolist()}
    assert attrs == {True: (0o40755 << 16) | 0x10, False: 0o100644 << 16}


def test_dist_is_ignored_by_git():
    root = CALCLM.parents[1]
    res = subprocess.run(["git", "check-ignore", "-q", "dist/CalcLM-v2.2.zip"], cwd=root)
    assert res.returncode == 0


@pytest.mark.parametrize("changed", ["model.bin", "model.json", "offsets.json"])
def test_bundle_refuses_ld2_that_does_not_match_the_export(tmp_path, inputs, changed):
    if changed == "model.bin":
        (inputs[1] / changed).write_bytes(BLOB + b"changed")
    else:
        path = inputs[5] if changed == "offsets.json" else inputs[1] / changed
        meta = json.loads(path.read_text())
        meta["export_sha256" if changed == "offsets.json" else "sha256"] = "cd" * 32
        path.write_text(json.dumps(meta))
        if changed == "model.json":
            (inputs[1] / "CLMVOC.8xv").write_bytes(_clmvoc(300, "cd" * 32))
    with pytest.raises(bundle.BundleError, match="LD2.*SHA-256 mismatch"):
        _build(tmp_path, inputs)
    assert not (tmp_path / "dist/a.zip").exists()


def test_cli_defaults_to_the_v22_zip(tmp_path, inputs, monkeypatch):
    prog, ex, install, extras, build, offsets = inputs
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    bundle.main(["--program", str(prog), "--export", str(ex), "--install", str(install),
                 "--extras", str(extras), "--build", str(build), "--offsets", str(offsets)])
    assert (tmp_path / "dist/CalcLM-v2.2.zip").is_file()
