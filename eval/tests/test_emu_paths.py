"""Where the CEmu harness and the ROM are found: CALCLM_CEMU_ROOT and CALCLM_ROM, else the
setup's default (none in the public tree), else nowhere, and the emulator is unavailable."""
from pathlib import Path

import pytest

from eval import emu as E


def test_the_harness_root_and_rom_come_from_the_environment(tmp_path):
    root, rom = E.locate({"CALCLM_CEMU_ROOT": str(tmp_path)}, None)
    assert (root, rom) == (tmp_path, tmp_path / "rom" / "ti84pce.rom")
    root, rom = E.locate({"CALCLM_CEMU_ROOT": str(tmp_path), "CALCLM_ROM": "/x/ce.rom"},
                         Path("/setup"))
    assert (root, rom) == (tmp_path, Path("/x/ce.rom"))


def test_unset_variables_fall_back_to_the_setup_default_or_to_nothing():
    assert E.locate({}, Path("/setup")) == (Path("/setup"), Path("/setup/rom/ti84pce.rom"))
    assert E.locate({"CALCLM_CEMU_ROOT": ""}, None) == (None, None)
    assert E.locate({"CALCLM_ROM": "/x/ce.rom"}, None) == (None, Path("/x/ce.rom"))


def test_without_a_harness_the_emulator_is_unavailable(monkeypatch):
    monkeypatch.setattr(E, "CEMU_ROOT", None)
    monkeypatch.setattr(E, "ROM", None)
    assert E.rom_available() is False
    with pytest.raises(RuntimeError, match="CALCLM_CEMU_ROOT"):
        E.cemu()
