"""LD2 + horizons against model.linker, through the shared host/eZ80 driver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ce.tools.clm import V2_EXPORT, EmuCore, Export, HostCore, emu_available
from model.linker import ARMS, generate

pytestmark = pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(),
                                reason="runs/v2/export not present")
E = json.loads((Path(__file__).parent / "fixtures/ld2_offsets_full.json").read_text())["E"]
BACKENDS = ["host", pytest.param("emu", marks=pytest.mark.skipif(
    not emu_available(), reason="needs the CE ROM and ~/CEdev"))]


@pytest.fixture(scope="module")
def ex():
    return Export.load(V2_EXPORT)


@pytest.fixture(scope="module", params=BACKENDS)
def core(request, ex):
    return (HostCore if request.param == "host" else EmuCore)(ex, horizons=True)


@pytest.mark.parametrize("cap", [2, 8, 32, 64])
def test_short_caps_match_linker(core, ex, cap):
    assert core.load()[0] == 0
    assert core.load_E(E) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=cap, masks=True) == 0
    want = generate(ex.model_hz, ARMS["LD2"], 1, E=E, cap=cap)["ids"][1:]
    assert core.generate(1024) == want
    assert core.step() == -1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_generation_matches_linker(core, ex, seed):
    if isinstance(core, EmuCore) and seed == 3:
        pytest.skip("host covers seed 3; CEmu covers seeds 1 and 2")
    assert core.load()[0] == 0
    assert core.load_E(E) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(seed, 205, 20, cap=1024, masks=True) == 0
    want = generate(ex.model_hz, ARMS["LD2"], seed, E=E, cap=1024)["ids"][1:]
    assert core.generate(1024) == want
    assert core.step() == -1


def test_load_clears_ld2(core, ex):
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate as baseline

    assert core.load()[0] == 0
    assert core.load_E(E) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=16, masks=True) == 0
    first = core.generate(16)
    assert core.reset(1, 205, 20, cap=16, masks=True) == 0
    assert core.generate(16) == first
    assert core.load()[0] == 0
    assert core.reset(1, 205, 20, cap=16, masks=True) == 0
    want = baseline(ex.model_hz, [], 15, SamplerConfig(205, 20, False, 1), cap=16,
                    mask=IntMask(ex.meta["vocab"]))[1:]
    assert core.generate(16) == want
    assert first != want


def test_ld2_requires_workspace_and_masks(core):
    assert core.load()[0] == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=16, masks=True) == -11
    assert core.load_E(E) == 0
    assert core.reset(1, 205, 20, cap=16, masks=False) == -11
    assert core.reset(1, 205, 20, cap=16, masks=True) == 0


def test_offset_appvar_checks_the_loaded_export(core, ex):
    from ce.calclm.tools.make_ld2 import payload

    data = payload(ex.path, Path(__file__).parent / "fixtures/ld2_offsets_full.json")
    assert len(data) == 83
    assert core.load()[0] == 0
    assert core.load_ld2(data) == 0
    assert core.set_ld2(True) == 0
    assert core.reset(1, 205, 20, cap=8, masks=True) == 0
    assert core.generate(8) == generate(ex.model_hz, ARMS["LD2"], 1, E=E, cap=8)["ids"][1:]
    # Every byte, including E and the checksum, must be protected.
    for i in range(len(data)):
        damaged = data[:i] + bytes([data[i] ^ 1]) + data[i + 1:]
        assert core.load_ld2(damaged) == -11
        assert core.set_ld2(True) == 0
        assert core.reset(1, 205, 20, cap=8, masks=True) == -11

    # A valid checksum cannot make offsets from a different export acceptable.
    import struct
    from ce.calclm.tools.make_ld2 import checksum
    for field in (4, 6, 9):  # vocabulary, blob size, export SHA prefix
        wrong = bytearray(data)
        wrong[field] ^= 1
        wrong[-4:] = struct.pack("<I", checksum(wrong[:-4]))
        assert core.load_ld2(bytes(wrong)) == -11
    assert core.load_ld2(data[:-1]) == -11
    assert core.load_ld2(data + b"\0") == -11


@pytest.mark.parametrize("seed", [1, 2])
def test_program_load_enables_ld2_after_every_reload(core, ex, seed):
    """Use the program's shared post-load policy, not a test-only LD2 flag toggle."""
    from ce.calclm.tools.make_ld2 import payload

    data = payload(ex.path, Path(__file__).parent / "fixtures/ld2_offsets_full.json")
    for cap in (8, 1024):
        assert core.load()[0] == 0
        assert core.program_decoder(data) == 0
        if isinstance(core, EmuCore):
            assert core.last.cycles < 4_800_000  # binding + defaults under 0.1 s at 48 MHz
        assert core.reset(seed, 205, 20, cap=cap, masks=True) == 0
        assert core.generate(1024) == generate(ex.model_hz, ARMS["LD2"], seed,
                                               E=E, cap=cap)["ids"][1:]


def test_program_load_falls_back_to_v21(core, ex):
    from ce.calclm.tools.make_ld2 import payload
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate as baseline

    data = payload(ex.path, Path(__file__).parent / "fixtures/ld2_offsets_full.json")
    want = baseline(ex.model_hz, [], 31, SamplerConfig(205, 20, False, 1), cap=32,
                    mask=IntMask(ex.meta["vocab"]))[1:]
    for optional in (None, b"?" + data[1:]):
        assert core.load()[0] == 0
        assert core.program_decoder(data) == 0
        assert core.program_decoder(optional) == 0
        assert core.reset(1, 205, 20, cap=32, masks=True) == 0
        assert core.generate(1024) == want


def test_offset_packager_rejects_wrong_fit(ex, tmp_path):
    from ce.calclm.tools.make_ld2 import payload

    fit = json.loads((Path(__file__).parent / "fixtures/ld2_offsets_full.json").read_text())
    for key, value in [("hazard_full", False), ("horizons", False),
                       ("export_sha256", "00" * 32), ("E", [1])]:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(fit | {key: value}))
        with pytest.raises(ValueError):
            payload(ex.path, bad)

    # Full model hashing still happens on the host, even though the device uses CLMVOC.
    changed = bytearray(ex.blob)
    changed[-1] ^= 1
    (tmp_path / "model.bin").write_bytes(changed)
    (tmp_path / "model.json").write_text(json.dumps(ex.meta))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        payload(tmp_path, Path(__file__).parent / "fixtures/ld2_offsets_full.json")


def test_benchmark_ld2_and_json(ex, monkeypatch, capsys, tmp_path):
    """The benchmark configures LD2 after load, before reset, and saves the measured JSON."""
    from ce.tools import clm_bench

    calls = []

    class Core:
        last = type("Result", (), {"cycles": 96000000})()

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            calls.append("load")
            return 0, -1

        def program_decoder(self, data):
            assert data[:4] == b"LD2\x02"
            calls.append("program defaults")
            return 0

        def reset(self, *args, **kwargs):
            calls.append("reset")
            return 0

        def generate(self, cap):
            return [4, 2]

    monkeypatch.setattr(clm_bench, "EmuCore", Core)
    out = tmp_path / "cycles.json"
    clm_bench.main(["--v2", "--flash", "--program", "1", "--cap", "1024", "--masks",
                    "--horizons", "--ld2", "--json", str(out)])
    assert calls == ["load", "program defaults", "reset"]
    measured = json.loads(capsys.readouterr().out)
    assert measured == json.loads(out.read_text())
    assert measured["ld2"] is True
    assert measured["cycles_per_token"] == 48000000
