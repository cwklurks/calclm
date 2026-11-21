"""Use the CEmu harness (class Emu) without writing anything into that repo.

The harness checkout (tools/cemu/, and the ROM at rom/ti84pce.rom) is CALCLM_CEMU_ROOT and the
ROM file CALCLM_ROM; see locate(). The checkout is imported read-only through sys.path
(bytecode writing is turned off so no __pycache__ lands there) and its state directory is
redirected to eval/runs/_cache/, which git ignores: the booted home state and the OCR glyph
templates contain ROM-derived data and never leave this box.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The harness checkout used when CALCLM_CEMU_ROOT is unset: none.
SETUP_CEMU_ROOT: Path | None = None


def locate(env: Mapping[str, str], setup: Path | None) -> tuple[Path | None, Path | None]:
    """(harness root, ROM): CALCLM_CEMU_ROOT, else `setup`; CALCLM_ROM, else rom/ti84pce.rom
    under the root. None where neither is set (the emulator is then unavailable)."""
    root = Path(env["CALCLM_CEMU_ROOT"]) if env.get("CALCLM_CEMU_ROOT") else setup
    if env.get("CALCLM_ROM"):
        return root, Path(env["CALCLM_ROM"])
    return root, (root / "rom" / "ti84pce.rom" if root else None)


CEMU_ROOT, ROM = locate(os.environ, SETUP_CEMU_ROOT)
CACHE = Path(__file__).resolve().parent / "runs" / "_cache"


def rom_available() -> bool:
    return ROM is not None and ROM.exists()


def import_wrapper(root: Path):
    """root/tools/cemu as the module `tools.cemu`, read-only: bytecode writing stays off for
    the rest of the process, because the wrapper imports its sibling modules (tools.build,
    ...) lazily. This repository has its own top-level `tools` package, which a sys.path entry
    cannot outrank (a regular package beats namespace portions anywhere on the path), so the
    wrapper's tools directory becomes a second portion of the `tools` package's path: this
    repository's modules resolve first, the wrapper's after them."""
    import importlib  # noqa: PLC0415
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"   # spawned children too
    if str(root) not in sys.path:
        sys.path.append(str(root))             # the wrapper's own top-level imports
    import tools as pkg  # noqa: PLC0415
    portion = str(Path(root) / "tools")
    if portion not in list(pkg.__path__):
        pkg.__path__.append(portion)
    return importlib.import_module("tools.cemu")


def cemu():
    """The tools.cemu module, state directory redirected to CACHE."""
    if CEMU_ROOT is None:
        raise RuntimeError("no CEmu harness: set CALCLM_CEMU_ROOT to its checkout")
    sys.dont_write_bytecode = True
    mod = import_wrapper(CEMU_ROOT)
    CACHE.mkdir(parents=True, exist_ok=True)
    mod.STATE_DIR = CACHE
    return mod


@dataclass
class Env:
    emu: object          # tools.cemu.Emu
    clean_state: Path    # loaded before every program
    preloaded: frozenset  # program names already on the calculator in clean_state


def _atomic_save(emu, path: Path) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    emu.save_state(tmp)
    tmp.replace(path)


def prepare(library: Path | None = None) -> Env:
    """One Emu for this process. Without `library` the clean state is the freshly booted
    calculator (RAM cleared, OS defaults). With `library`, every .8xp in it is sent to
    Archive first (cached state keyed by the files' bytes): used for program suites whose
    programs call each other."""
    mod = cemu()
    emu = mod.Emu(rom=ROM)
    emu.glyphs  # noqa: B018 - calibrate the screen OCR templates once (cached)
    if library is None:
        return Env(emu, emu.booted_state, frozenset())
    files = sorted(Path(library).glob("*.8xp"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    state = CACHE / f"lib-{emu.digest}-{h.hexdigest()[:16]}.img"
    if not state.exists():
        emu.load_state(emu.booted_state)
        emu.send(*files, archive=True)
        _atomic_save(emu, state)
    emu.load_state(state)
    names = {v["name"].rstrip(b"\0").decode("latin-1").replace("[", "θ")
             for v in emu.vat() if v["type"] in (5, 6)}
    return Env(emu, state, frozenset(names))
