"""T3: the tracker builds warning-free with the CE toolchain, is bit-exact on the emulated
eZ80, and every step costs < 0.1% of CALCLM v1's per-token cycles (44.2 M at position 0)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tracker import ce_bench

ROOT = Path(__file__).resolve().parents[2]
need_ce = pytest.mark.skipif(not ce_bench.ce_available(), reason="CE toolchain not installed")


def _emu_ready() -> bool:
    from ce.tools.bench import DEFAULT_ROM
    return ce_bench.ce_available() and DEFAULT_ROM is not None and DEFAULT_ROM.exists() \
        and (ROOT / "data" / "tok" / "val.bin").exists()


@need_ce
@pytest.mark.parametrize("opt", ["-Oz", "-O2"])
def test_ce_build_is_clean(opt):
    obj = ce_bench.compile_obj(ce_bench.HERE / "tracker.c", opt)   # raises on any warning
    sizes = ce_bench.object_size(obj)
    assert sizes["data"] == 0 and sizes["bss"] == 0                 # no writable globals
    out = subprocess.run([str(ce_bench.CEDEV / "binutils" / "bin" / "z80-none-elf-nm"), "-u",
                          str(obj)], capture_output=True, text=True, check=True).stdout
    undefined = [ln.split()[-1] for ln in out.splitlines() if ln.strip()]
    assert all(u.startswith("__") for u in undefined), undefined  # only libcrt helpers


@pytest.mark.skipif(not _emu_ready(), reason="CE toolchain, ROM or corpus missing")
def test_cemu_bit_exact_and_cycles(tmp_path):
    out = tmp_path / "bench.json"
    # own process: the CEmu core is a per-process global
    res = subprocess.run([sys.executable, "-m", "tracker.ce_bench", "--exact", "30",
                          "--steps", "2", "--json", str(out)], cwd=ROOT, capture_output=True,
                         text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    r = json.loads(out.read_text())
    assert r["exact_error"] is None
    assert r["state_bytes"] <= 512 and r["feat_bytes"] == 18
    budget = 0.001 * r["v1_cycles_per_token"]
    assert r["stream_mean"] < budget and r["step_max"] < budget
