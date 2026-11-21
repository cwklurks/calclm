"""Phase 6, the local browser demo (docs/agents/phase6-browser-demo.md): the WebAssembly build of
the device core, the model package, the page's offline guarantees, and the Node checks that run
the module against the Python reference (A1-A7 of the plan).

The model export is the deployed one, v2.6-rc2's E24 L4c-s11 (runs/e24/export/L4c-s11 of this
checkout or of the main checkout of a worktree; CALCLM_DEMO_EXPORT overrides), with its LD2 fit
model.topics.FIT, the fit topics.h is bound to. Tests that need it, clang's wasm32 target
or node skip when they are missing. No GPU, emulator or network is used.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from demo.tools import build_model

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
BUILD = DEMO / "build"
PAGE_FILES = ["index.html", "app.js", "calclm.js", "style.css"]
NODE = shutil.which("node")
EXPORT = build_model.default_export()
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

needs_export = pytest.mark.skipif(EXPORT is None, reason="E24 L4c-s11 export not found")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _wasm_toolchain() -> bool:
    if not shutil.which("clang") or not shutil.which("wasm-ld"):
        return False
    out = subprocess.run(["clang", "-print-targets"], capture_output=True, text=True).stdout
    return "wasm32" in out


needs_wasm = pytest.mark.skipif(not _wasm_toolchain(), reason="clang wasm32 / wasm-ld missing")


def _run(cmd, **kw):
    res = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, **kw)
    assert res.returncode == 0, f"{cmd} failed:\n{res.stdout}\n{res.stderr}"
    return res.stdout


@pytest.fixture(scope="module")
def wasm() -> Path:
    if not _wasm_toolchain():
        pytest.skip("clang wasm32 / wasm-ld missing")
    _run(["bash", str(DEMO / "tools" / "build_wasm.sh")])
    return BUILD / "calclm.wasm"


@pytest.fixture(scope="module")
def model_js() -> Path:
    if EXPORT is None:
        pytest.skip("E24 L4c-s11 export not found")
    _run([sys.executable, "-m", "demo.tools.build_model", "--export", str(EXPORT)])
    return BUILD / "calclm-model.js"


def test_the_demo_defaults_to_the_deployed_model():
    """The package's default export and fit are the ones topics.h is bound to (v2.6-rc2: E24's
    L4c-s11)."""
    from ce.tools.clm import W_V26_EXPORT
    from model import topics as TP
    assert build_model.FIT == TP.FIT
    assert build_model.EXPORT_REL == W_V26_EXPORT.relative_to(ROOT)
    assert build_model.EXPORT_REL == Path("runs") / "e24" / "export" / "L4c-s11"


# ---- A6: offline and private ------------------------------------------------------------------

NETWORK = re.compile(r"https?:|\bfetch\s*\(|XMLHttpRequest|WebSocket|\bimport\s*\(|sendBeacon|"
                     r"EventSource|<link\b[^>]*\bhref\s*=\s*['\"]?//|src\s*=\s*['\"]?//")


@pytest.mark.parametrize("name", PAGE_FILES)
def test_the_page_makes_no_network_calls(name):
    text = (DEMO / name).read_text()
    assert not NETWORK.findall(text), f"{name}: {NETWORK.findall(text)}"


def test_the_page_loads_only_local_scripts():
    html = (DEMO / "index.html").read_text()
    srcs = re.findall(r"<script[^>]*\bsrc\s*=\s*\"([^\"]+)\"", html)
    assert srcs, "index.html loads its scripts with <script src>"
    for s in srcs:
        assert not re.match(r"[a-z]+:|/", s), s       # relative paths only
    assert "build/calclm-core.js" in srcs and "build/calclm-model.js" in srcs


def test_the_build_folder_is_ignored():
    for name in ("calclm-model.js", "calclm.wasm", "reference.json"):
        res = subprocess.run(["git", "check-ignore", "-q", f"demo/build/{name}"], cwd=ROOT)
        assert res.returncode == 0, f"demo/build/{name} is not git-ignored"


def _wasm_imports(data: bytes) -> int:
    """Number of entries in a WebAssembly module's import section (0 if it has none)."""
    def leb(i):
        v = s = 0
        while True:
            b = data[i]
            i += 1
            v |= (b & 0x7F) << s
            s += 7
            if b < 0x80:
                return v, i
    assert data[:4] == b"\0asm"
    i = 8
    while i < len(data):
        sid = data[i]
        size, j = leb(i + 1)
        if sid == 2:
            return leb(j)[0]
        i = j + size
    return 0


@needs_wasm
def test_the_module_imports_nothing(wasm):
    """No host calls at all: the core cannot reach the network, the DOM or the clock."""
    assert _wasm_imports(wasm.read_bytes()) == 0


@needs_wasm
def test_the_core_script_holds_the_module(wasm):
    js = (BUILD / "calclm-core.js").read_text()
    import base64
    b64 = re.search(r'"([A-Za-z0-9+/=]+)"', js).group(1)
    assert base64.b64decode(b64) == wasm.read_bytes()


# ---- the model package ------------------------------------------------------------------------

@needs_export
def test_the_model_package_is_the_calculators_files(model_js):
    from ce.calclm.tools.make_ld2 import payload
    from ce.tools.clm import Export
    pkg = build_model.read_package(model_js)
    ex = Export.load(EXPORT)
    import base64
    assert [base64.b64decode(a) for a in pkg["appvars"]] == ex.appvars
    assert base64.b64decode(pkg["ld2"]) == payload(EXPORT, build_model.FIT)
    assert pkg["export_sha256"] == ex.meta["sha256"]
    assert pkg["vocab_hex"][4:] == ex.meta["vocab"][4:] and pkg["vocab_hex"][:4] == [""] * 4


@needs_export
def test_the_detokenizer_table_is_render(model_js):
    """A7: tivars decodes token by token, so the per-id text joined (ids >= 4, trailing
    newlines stripped) is model.intref.render, here on 200 random sequences."""
    from model.intref import IntModel, render
    pkg = build_model.read_package(model_js)
    m = IntModel.load(EXPORT)
    table = pkg["vocab_text"]
    rng = random.Random(6)
    for _ in range(200):
        ids = [rng.randrange(0, len(table)) for _ in range(rng.randrange(1, 80))]
        if rng.random() < 0.3:
            ids += [4, 4]                                   # trailing newlines
        assert build_model.join_text(table, ids) == render(m, ids)


@needs_export
def test_build_model_refuses_a_fit_for_another_export(tmp_path):
    fit = json.loads(build_model.FIT.read_text())
    fit["export_sha256"] = "0" * 64
    other = tmp_path / "fit.json"
    other.write_text(json.dumps(fit))
    with pytest.raises(ValueError):
        build_model.package(EXPORT, other)


# ---- the Node checks (A1-A5) ------------------------------------------------------------------

@needs_node
@needs_export
def test_the_engine_unit_tests_pass(wasm, model_js):
    _run([NODE, "--test", str(DEMO / "tests" / "engine.test.js")], timeout=900)


@needs_node
@needs_export
@pytest.mark.slow
def test_the_grid_is_byte_identical_to_the_reference(wasm, model_js):
    """A1: every topic x seeds {1, 2, 3, 48} x levels {1, 2, 3}, and Random at the same seeds
    and levels: 12 cases per menu topic (v2.6-rc2: ten) plus 12, ids and text equal to the
    Python reference."""
    from model import topics as TP
    n_topics = len(TP.menu_entries(TP.load_fixture())) - 1
    _run([sys.executable, "-m", "demo.tools.reference", "--export", str(EXPORT)], timeout=1800)
    out = _run([NODE, str(DEMO / "tests" / "check_grid.js")], timeout=1800)
    grid = json.loads((BUILD / "grid.json").read_text())
    assert grid["cases"] == 12 * n_topics + 12 and grid["match"] == grid["cases"], out
    assert grid["topic_cases"] == 12 * n_topics and grid["random_cases"] == 12


@needs_node
@needs_export
def test_the_page_writes_the_sheets_topic_from_file_url(wasm, model_js):
    """A8: index.html from file:// in a local headless Chromium shows "done: 103 tok", the check
    line and the ids sha256 of the v2.6-rc2 sheet's CYL seed 1 (skips without a Chromium)."""
    res = subprocess.run([NODE, str(DEMO / "tests" / "page_check.js")], cwd=ROOT, env=ENV,
                         capture_output=True, text=True, timeout=300)
    if res.returncode == 2:
        pytest.skip(res.stdout.strip())
    assert res.returncode == 0, res.stdout + res.stderr
