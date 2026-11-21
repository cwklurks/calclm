"""model/tests/linker_ref.py imports the original scratch modules (experiments/v3-angles odec.py
and bind.py). Those put the main checkout first on sys.path when they are imported, and a
spawned worker process inherits sys.path: a later test's workers then imported the main
checkout's model.linker instead of this tree's (3 linker CLI tests failed only in the full
suite). scratch() must leave sys.path as it found it, apart from the scratch directories it
appends."""
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from model.tests import linker_ref

ROOT = Path(__file__).resolve().parents[2]


def _linker_file() -> str:
    import model.linker
    return model.linker.__file__


def test_scratch_import_leaves_sys_path_and_workers_alone(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("odec", "bind"):                 # import them afresh (restored afterwards)
        monkeypatch.delitem(sys.modules, name, raising=False)
    before = list(sys.path)
    odec, bind = linker_ref.scratch()
    assert odec.__name__ == "odec" and bind.__name__ == "bind"
    assert sys.path[:len(before)] == before
    assert all(p.startswith(str(linker_ref.SCRATCH)) for p in sys.path[len(before):])
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        where = pool.submit(_linker_file).result()
    assert Path(where).resolve() == ROOT / "model" / "linker.py"
