"""Cycle-exact benchmark harness for eZ80 kernels in headless CEmu (CalcLM M2).

This is emulator instrumentation only, like a debugger: the emulated CPU is pointed at our
code. Flow per benchmark:

    b = Bench()                      # boots the ROM to the home screen once (cached state)
    b.reset()                        # reload that state (RAM, flash, CPU, timers)
    b.write_ram(addr, code_or_data)
    b.write_flash(0x300000, packed)  # emulator's private flash copy; reset() restores it
    r = b.call(fn_addr, args=[...])  # stub: di; ld sp,..; push args; call fn; halt
    r.cycles                         # exact CPU cycles spent in fn (stub overhead removed)

Booting reuses the existing `Emu` logic from the CEmu harness's tools/cemu (CALCLM_CEMU_ROOT,
see eval/emu.py; imported read-only, with this harness's own library and state directory), so
the OS has configured the CPU (48 MHz) and the flash controller exactly as on the real unit
before we take over. The ROM (CALCLM_ROM) is read in place; states live in ce/build/cemu-state
(git-ignored).
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.emu import CEMU_ROOT, ROM as DEFAULT_ROM   # CALCLM_CEMU_ROOT, CALCLM_ROM

HERE = Path(__file__).resolve().parent
CE = HERE.parent
BUILD = CE / "build"
LIB_PATH = BUILD / "libcebench.so"
STATE_DIR = BUILD / "cemu-state"

# RAM map used by the benchmarks (all inside user RAM; the OS is suspended with DI).
STUB_ADDR = 0xD1A900      # entry stub written per call
RET_ADDR = 0xD1A9F0       # the stub stores HL (C return value) here
NULL_FN = 0xD1A9F8        # a lone RET, used to calibrate the stub overhead
STACK_TOP = 0xD1A87E      # TI-OS stack top (grows down)
FLASH_FREE = 0x300000     # unused archive flash in the booted state (checked on write)

CPU_HZ = 48_000_000


class BenchError(Exception):
    pass


def _u8p(n: int):
    return (ctypes.c_uint8 * n)()


def load_lib() -> ctypes.CDLL:
    if not LIB_PATH.exists() or (HERE / "shim.c").stat().st_mtime > LIB_PATH.stat().st_mtime:
        subprocess.run([str(HERE / "build.sh")], check=True)
    lib = ctypes.CDLL(str(LIB_PATH))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    sigs = {
        # same API as the CEmu harness's shim (so its Emu class can drive this library)
        "shim_load_rom": ([ctypes.c_char_p], ctypes.c_int),
        "shim_load_state": ([ctypes.c_char_p], ctypes.c_int),
        "shim_save_state": ([ctypes.c_char_p], ctypes.c_int),
        "shim_run": ([ctypes.c_uint32], None),
        "shim_key": ([ctypes.c_uint, ctypes.c_uint, ctypes.c_int], None),
        "shim_os_key": ([ctypes.c_uint16], ctypes.c_int),
        "shim_send": ([ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "shim_transfer_done": ([], ctypes.c_int),
        "shim_receive": ([ctypes.c_char_p, ctypes.c_int, u8p, ctypes.c_int], ctypes.c_int),
        "shim_read": ([ctypes.c_uint32, u8p, ctypes.c_uint32], None),
        "shim_write": ([ctypes.c_uint32, u8p, ctypes.c_uint32], None),
        "shim_read_flash": ([ctypes.c_uint32, u8p, ctypes.c_uint32], None),
        "shim_lcd_rgb": ([u8p], None),
        "shim_vat": ([u8p, ctypes.c_int], ctypes.c_int),
        "shim_var_data": ([ctypes.c_int, u8p, ctypes.c_int, u8p, ctypes.c_int], ctypes.c_int),
        "shim_cpu_stats": ([u64p, u64p, ctypes.POINTER(ctypes.c_int)], None),
        "shim_set_verbose": ([ctypes.c_int], None),
        # M2 instrumentation
        "shim_write_flash": ([ctypes.c_uint32, u8p, ctypes.c_uint32], ctypes.c_int),
        "shim_write_ram": ([ctypes.c_uint32, u8p, ctypes.c_uint32], ctypes.c_int),
        "shim_read_ram": ([ctypes.c_uint32, u8p, ctypes.c_uint32], ctypes.c_int),
        "shim_hw_info": ([u32p], None),
        "shim_set_lcd_dma": ([ctypes.c_int], None),
        "shim_get_regs": ([u32p], None),
        "shim_call": ([ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint64, u64p],
                      ctypes.c_int),
        # M4 instrumentation
        "shim_resume": ([ctypes.c_uint64, u64p], ctypes.c_int),
    }
    for name, (args, res) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


def _import_cemu():
    """Import the CEmu harness's tools.cemu read-only (no bytecode written there)."""
    if CEMU_ROOT is None or not (CEMU_ROOT / "tools" / "cemu" / "__init__.py").exists():
        raise BenchError(f"CEmu harness not found under {CEMU_ROOT} (set CALCLM_CEMU_ROOT)")
    from eval.emu import import_wrapper   # past this repository's own `tools` package
    return import_wrapper(CEMU_ROOT)


@dataclass(frozen=True)
class CallResult:
    cycles: int          # CPU cycles inside the called function (stub overhead subtracted)
    raw_cycles: int      # active cycles of the whole stub run
    dma_cycles: int      # cycles the CPU stalled for LCD/USB DMA (included in cycles)
    flash_accesses: int  # serial-flash cache touches (reads + fetches from flash)
    flash_misses: int    # serial-flash cache line misses
    ret: int             # HL on return (24-bit)

    @property
    def ms(self) -> float:
        return self.cycles / CPU_HZ * 1e3


def encode_stub(fn: int, args: list[int], ret_addr: int = RET_ADDR) -> bytes:
    """di; push args right-to-left (CE C ABI, 3 bytes each); call fn; ld (ret_addr),hl; halt."""
    out = bytearray([0xF3])                          # di
    for a in reversed(args):
        out += bytes([0x21]) + (a & 0xFFFFFF).to_bytes(3, "little")   # ld hl,a
        out += bytes([0xE5])                         # push hl
    out += bytes([0xCD]) + fn.to_bytes(3, "little")  # call fn
    out += bytes([0x22]) + ret_addr.to_bytes(3, "little")  # ld (ret_addr),hl
    out += bytes([0x76])                             # halt
    return bytes(out)


class Bench:
    """One emulated TI-84 Plus CE for kernel benchmarks (the CEmu core is per process)."""

    def __init__(self, rom: str | Path | None = DEFAULT_ROM, verbose: bool = False,
                 lcd_dma: bool = False, stub_addr: int = STUB_ADDR):
        """lcd_dma=True turns on CEmu's accurate LCD DMA emulation (the LCD's VRAM reads stall
        CPU RAM accesses, as on hardware with the screen on); off by default in the core.
        stub_addr: 256 bytes of RAM for the call stub, its return slot (+0xF0) and the bare RET
        used for calibration (+0xF8); move it when the code under test occupies 0xD1A900."""
        self.lcd_dma = lcd_dma
        self.stub_addr, self.ret_addr, self.null_fn = stub_addr, stub_addr + 0xF0, stub_addr + 0xF8
        if rom is None or not Path(rom).exists():
            raise BenchError(f"ROM not found: {rom} (set CALCLM_CEMU_ROOT or CALCLM_ROM)")
        self.rom = Path(rom)
        self.lib = load_lib()
        self.lib.shim_set_verbose(1 if verbose else 0)
        digest = hashlib.sha256(self.rom.read_bytes()).hexdigest()[:16]
        self.home_state = STATE_DIR / f"home-{digest}.img"
        if not self.home_state.exists():
            self._boot()
        self._overhead: dict[int, int] = {}
        self.reset()

    def _boot(self) -> None:
        """Boot to the home screen with the CEmu harness's Emu logic, save the state."""
        cemu = _import_cemu()
        lib = self.lib
        orig_load, orig_dir = cemu._lib.load, cemu.STATE_DIR
        cemu._lib.load = lambda: lib     # drive *this* library (a superset of its shim API)
        cemu.STATE_DIR = STATE_DIR       # and keep its cached state under ce/build
        try:
            emu = cemu.Emu(rom=self.rom)  # first boot: ROM, RAM Cleared splash, Clear, idle
            if emu.booted_state != self.home_state:
                raise BenchError(f"unexpected state path {emu.booted_state}")
        finally:
            cemu._lib.load, cemu.STATE_DIR = orig_load, orig_dir

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Reload the booted home-screen state (restores RAM, flash, CPU, peripherals)."""
        if self.lib.shim_load_state(str(self.home_state).encode()) != 0:
            raise BenchError(f"cannot load state {self.home_state}")
        self.lib.shim_set_lcd_dma(1 if self.lcd_dma else 0)
        if self.lcd_dma:
            # CEmu arms LCD DMA only at the next frame start: let the idle OS run past one
            # frame (60 Hz) so the measured call sees steady-state DMA from its first cycle
            self.lib.shim_run(20)

    def hw_info(self) -> dict:
        buf = (ctypes.c_uint32 * 16)()
        self.lib.shim_hw_info(buf)
        v = list(buf)
        return {"asic_rev": "?AIM"[v[0]] if v[0] < 4 else v[0], "serial_flash": bool(v[1]),
                "python": bool(v[2]), "cpu_hz": v[3], "flash_wait_states": v[4],
                "flash_mask": v[5], "flash_mapped": v[6], "privileged": v[7],
                "protected": (v[8], v[9]), "stack_limit": v[10], "cpu_speed": v[11],
                "flash_port05": v[12], "mbase": v[13]}

    def regs(self) -> dict:
        buf = (ctypes.c_uint32 * 12)()
        self.lib.shim_get_regs(buf)
        names = ["AF", "BC", "DE", "HL", "IX", "IY", "SPL", "PC", "MBASE", "ADL", "IEF1", "halted"]
        return dict(zip(names, buf))

    # ------------------------------------------------------------------ memory
    def write_ram(self, addr: int, data: bytes) -> None:
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        if self.lib.shim_write_ram(addr, buf, len(data)) != 0:
            raise BenchError(f"RAM write out of range: {addr:#x}+{len(data)}")

    def read_ram(self, addr: int, n: int) -> bytes:
        buf = _u8p(n)
        if self.lib.shim_read_ram(addr, buf, n) != 0:
            raise BenchError(f"RAM read out of range: {addr:#x}+{n}")
        return bytes(buf)

    def read_flash(self, addr: int, n: int) -> bytes:
        buf = _u8p(n)
        self.lib.shim_read_flash(addr, buf, n)
        return bytes(buf)

    def write_flash(self, addr: int, data: bytes, require_erased: bool = True) -> None:
        """Write into the emulator's private flash array. By default the target range must be
        erased (all 0xFF) in the current state, i.e. unused by the OS and the archive."""
        if require_erased and self.read_flash(addr, len(data)) != b"\xff" * len(data):
            raise BenchError(f"flash {addr:#x}+{len(data)} is in use (not erased)")
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        if self.lib.shim_write_flash(addr, buf, len(data)) != 0:
            raise BenchError(f"flash write out of range: {addr:#x}+{len(data)}")

    # ------------------------------------------------------------------ execution
    def _run_stub(self, fn: int, args: list[int], flush_cache: bool,
                  max_cycles: int) -> tuple[list[int], int]:
        self.write_ram(self.stub_addr, encode_stub(fn, args, self.ret_addr))
        stats = (ctypes.c_uint64 * 8)()
        rc = self.lib.shim_call(self.stub_addr, STACK_TOP, 1 if flush_cache else 0, max_cycles,
                                stats)
        if rc != 0:
            r = self.regs()
            raise BenchError(f"no HALT within {max_cycles} cycles; PC={r['PC']:#x} regs={r}")
        ret = int.from_bytes(self.read_ram(self.ret_addr, 3), "little")
        return list(stats), ret

    def run_code(self, code: bytes, flush_cache: bool = True,
                 max_cycles: int = 2_000_000_000) -> list[int]:
        """Run raw eZ80 code placed at the stub address (ADL, interrupts initially disabled)
        until it executes HALT with interrupts disabled. Returns shim_call's stats list."""
        self.write_ram(self.stub_addr, code)
        stats = (ctypes.c_uint64 * 8)()
        rc = self.lib.shim_call(self.stub_addr, STACK_TOP, 1 if flush_cache else 0, max_cycles,
                                stats)
        if rc != 0:
            r = self.regs()
            raise BenchError(f"no HALT within {max_cycles} cycles; PC={r['PC']:#x} regs={r}")
        return list(stats)

    def start(self, fn: int, args: list[int] | tuple[int, ...] = (), flush_cache: bool = True,
              max_cycles: int = 2_000_000_000) -> tuple[list[int], bool]:
        """Begin fn(args...) from the stub like call(), but return at the first HALT (a
        breakpoint in the code under test, or the stub's own final HALT). Returns (shim stats,
        finished), finished meaning the stub's final HALT was reached; continue with resume()."""
        stub = encode_stub(fn, list(args), self.ret_addr)
        self.write_ram(self.stub_addr, stub)
        self._stub_end = self.stub_addr + len(stub)
        stats = (ctypes.c_uint64 * 8)()
        if self.lib.shim_call(self.stub_addr, STACK_TOP, 1 if flush_cache else 0, max_cycles,
                              stats) != 0:
            raise BenchError(f"no HALT within {max_cycles} cycles; PC={self.regs()['PC']:#x}")
        return list(stats), self.regs()["PC"] == self._stub_end

    def resume(self, max_cycles: int = 2_000_000_000) -> tuple[list[int], bool]:
        """Continue after a breakpoint HALT; same return as start()."""
        stats = (ctypes.c_uint64 * 8)()
        if self.lib.shim_resume(max_cycles, stats) != 0:
            raise BenchError(f"no HALT within {max_cycles} cycles; PC={self.regs()['PC']:#x}")
        return list(stats), self.regs()["PC"] == self._stub_end

    def stub_overhead(self, nargs: int) -> int:
        """Active cycles of the stub with a function that only returns (measured once)."""
        if nargs not in self._overhead:
            saved = self.read_ram(self.null_fn, 1)
            self.write_ram(self.null_fn, b"\xc9")
            stats, _ = self._run_stub(self.null_fn, [0] * nargs, True, 10_000)
            self.write_ram(self.null_fn, saved)
            self._overhead[nargs] = stats[0]
        return self._overhead[nargs]

    def call(self, fn: int, args: list[int] | tuple[int, ...] = (), flush_cache: bool = True,
             max_cycles: int = 2_000_000_000) -> CallResult:
        """Run fn(args...) from the stub (interrupts disabled) until HALT. `cycles` is
        (stub run) - (same stub calling a bare RET): the function's own cycles excluding the
        CALL into it and one RET (a constant ~18 cycles). Flushes the flash cache first (cold
        cache) unless told otherwise."""
        args = list(args)
        overhead = self.stub_overhead(len(args))
        stats, ret = self._run_stub(fn, args, flush_cache, max_cycles)
        return CallResult(cycles=stats[0] - overhead, raw_cycles=stats[0], dma_cycles=stats[3],
                          flash_accesses=stats[4], flash_misses=stats[5], ret=ret)
