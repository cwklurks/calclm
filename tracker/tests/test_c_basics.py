"""T1 and host-build checks for tracker/tracker.c: warning-free C99, fixed-size state
(<= 512 B), no allocation, no globals, no library calls."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tracker import bind, feats as F

HERE = Path(__file__).resolve().parents[1]


def _gcc(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["gcc", *args], capture_output=True, text=True, cwd=cwd)


def test_compiles_warning_free_c99(tmp_path):
    for opt in ("-O0", "-O2"):
        res = _gcc("-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror", "-Wconversion",
                   "-Wshadow", opt, "-c", str(HERE / "tracker.c"), "-o",
                   str(tmp_path / "t.o"), cwd=tmp_path)
        assert res.returncode == 0, res.stderr
        assert res.stderr == ""


def test_no_allocation_no_globals_no_calls(tmp_path):
    obj = tmp_path / "t.o"
    res = _gcc("-std=c99", "-O2", "-fno-builtin", "-c", str(HERE / "tracker.c"), "-o", str(obj),
               cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    nm = subprocess.run(["nm", str(obj)], capture_output=True, text=True, check=True).stdout
    undefined = [ln.split()[-1] for ln in nm.splitlines() if ln.split()[0] == "U"]
    assert undefined == [], f"tracker.c calls external code: {undefined}"
    kinds = {ln.split()[-2] for ln in nm.splitlines() if len(ln.split()) == 3}
    assert not kinds & set("DdBbCcGgSs"), f"writable globals in tracker.c: {nm}"


def test_state_size_limit():
    assert bind.state_size() <= 512
    assert bind.feat_size() == F.NCAT + 5 + 4


def test_header_constants_match_python():
    h = (HERE / "tracker.h").read_text()
    assert f"#define TRK_NCAT {F.NCAT}" in h
    assert f"#define TRK_MAX_BLOCKS {F.MAX_BLOCKS}" in h
    assert f"#define TRK_MAX_LISTS {F.MAX_LISTS}" in h


def test_out_of_vocab_ids_are_safe():
    t = bind.Tracker()
    t.reset()
    for tok in (0xFFFF, 781, 1023, 65000):
        cats, _ = t.step(tok)
        assert cats[F.GROUP_INDEX["ctx"]] == F.CTX_EXPR


@pytest.mark.parametrize("n", [0, 1, 7])
def test_batch_handles_short_streams(n):
    import numpy as np
    ids = np.array([1, 11, 12, 13, 2, 1, 11, 12][:n], dtype=np.uint16)
    cats, bags = bind.batch(ids)
    assert cats.shape == (n, F.NCAT) and bags.shape == (n,)
