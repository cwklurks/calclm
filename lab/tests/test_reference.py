"""lab.reference: human val programs judged like samples (G2a reference row)."""
import json

import numpy as np
import pytest

from lab import reference as Ref
from lab.tests.conftest import DATA, VOCAB, have_data


def test_programs_are_split_at_bos_and_kept_whole():
    stream = np.array([1, 5, 6, 2, 1, 7, 2, 1, 8, 8, 8, 2], dtype=np.uint16)
    assert Ref.programs(stream) == [[1, 5, 6, 2], [1, 7, 2], [1, 8, 8, 8, 2]]
    assert Ref.programs(stream, limit=2) == [[1, 5, 6, 2], [1, 7, 2]]


@have_data
def test_reference_row_runs_val_programs_in_the_emulator(tmp_path):
    from eval import emu as E
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    res = Ref.main(["--name", "ref", "--out-root", str(tmp_path), "--limit", "3", "--workers", "2",
                    "--cap", "1024", "--data", str(DATA), "--vocab", str(VOCAB)])
    saved = json.loads((tmp_path / "ref" / "result.json").read_text())
    assert saved == res and res["cap"] == 1024 and res["score"]["n_samples"] == 3
    assert res["score"]["n_run"] + res["score"]["n_empty"] == 3
