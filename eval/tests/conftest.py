import pytest

from eval import emu as E


@pytest.fixture(scope="session")
def emu_env():
    """Emulator plus the clean saved state; skips when the ROM is not on this box."""
    if not E.rom_available():
        pytest.skip("TI-84 Plus CE ROM not available")
    return E.prepare()
