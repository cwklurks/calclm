"""Run the CalcLM inference core (ce/calclm/core) on the host or inside CEmu, through the
test driver ce/calclm/test/drv.c, and compare it with model/intref.py.

    from ce.tools.clm import Export, HostCore, EmuCore
    ex = Export.load()                      # runs/v0/export: blob, AppVar payloads, intref model
    core = EmuCore(ex)                      # or HostCore(ex); horizons=True: ALiBi horizons
    core.load()                             # AppVars written into emulated flash; buffers in RAM
    core.reset(seed=1, temperature_q8=205, top_k=20)
    tokens = core.generate(40)

EmuCore is emulator instrumentation only (ce/tools/bench.py): the CPU is pointed at the
driver like a debugger; TI-OS never launches anything. The same core.o that the CE program
links is linked here into a flat binary at the program's own load address.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bench import BUILD, CE, Bench, BenchError

ROOT = CE.parent
EXPORTS = {"v1": ROOT / "runs" / "v1" / "export", "v0": ROOT / "runs" / "v0" / "export"}
EXPORT_DIR = EXPORTS["v1"]        # the shipping model
V2_EXPORT = ROOT / "runs" / "v2" / "export"   # format 2 (ALiBi, sliding window): the v2 port
# The wide model W (dim 256, 8 heads, hidden 576; lab/E7.md): the v2.3 port. Any export of
# that shape can stand in (CALCLM_W_EXPORT); E7's W-s11 is the port vehicle.
W_EXPORT = Path(os.environ.get("CALCLM_W_EXPORT",
                               ROOT / "runs" / "e7" / "export" / "W-s11")).absolute()
# E10's adopted candidate (runs/e10/decision.json), the v2.3-rc2 model. The W tests run on
# every export in W_EXPORTS.
W_RC2_EXPORT = ROOT / "runs" / "e10" / "export" / "L4-s23"
# E24's L4c-s11 (the L4 recipe trained on the cleaned data SEEDc in E24; the topic-menu sweep's
# pick), the v2.6-rc2 model (docs/agents/release-v2.6-rc2.md). rc1's E22 L4c-s17 was not shipped.
W_V26_EXPORT = ROOT / "runs" / "e24" / "export" / "L4c-s11"
W_EXPORTS = tuple(dict.fromkeys([W_EXPORT, W_RC2_EXPORT, W_V26_EXPORT]))
CALCLM = CE / "calclm"
CORE_DIR = CALCLM / "core"
# Core builds by model shape: tag -> shape header (tools/build_core.sh CLM_SHAPE). The default
# tag "" is v2.2's shape; each tag builds into its own directory (out_dir).
SHAPES = {"": CORE_DIR / "clm_shape.h", "w": CORE_DIR / "clm_shape_w.h"}


def out_dir(tag: str = "") -> Path:
    return BUILD / ("calclm" + (f"-{tag}" if tag else ""))


OUT = out_dir()

# commands of drv_cmd
(LOAD, RESET, STEP, GENERATE, FORWARD, SEEK, PROF, ATTACH, DETACH, TRACK, MASK, TRACKER,
 HORIZONS, LD2, LOAD_E, LD2_LOAD, PROGRAM_DECODER) = range(1, 18)
FEED, MIX = 30, 31         # v2.5 topics (fixed numbers, after the kv4 build's 18-20)
APPVAR_HEADER = 7
TOK_CAP = 1024             # token buffer (clm_mem.tok): the largest generation cap

KROW = 35                  # CLM_KROW: 32 biased keys + int24 -128 * sum (clm.h)
FEAT_BYTES = 18            # sizeof(trk_feat) (tracker/tracker.h)

# ---- emulator memory map (mirrors the CE program; see ce/calclm/README.md) -----------------
CODE_ORG = 0xD1A87F        # userMem: where TI-OS places a running program
KC_ADDR = 0xD052C6         # K cache: start of the toolchain's BSS region (OS scratch RAM)
BLK_ADDR = 0xD13000        # driver command block (BSS region, after the K cache)
OUT_TOK = 0xD13400         # GENERATE output tokens (up to 1024)
TRACK_IN = 0xD13C00        # TRACK input tokens (up to 256 per call)
VC_ADDR = 0xD30000         # V cache: user RAM (a temporary AppVar on the calculator)
TAP_ADDR = 0xD3C000        # debug taps (residual after each sublayer)
LOGITS_OUT = 0xD3E400      # FORWARD output; MASK and TRACKER output too
TOK_ADDR = 0xD3F400        # the core's token buffer (TOK_CAP u16)
TBL_ADDR = 0xD40000        # kernel table bank: first 64 KB of VRAM
ACT_ADDR = 0xD4F300        # activations: VRAM between the tables and the flash routines' RAM
SCREEN_ADDR = 0xD52C00     # visible 8bpp framebuffer on the calculator (not touched here)
FLASH_BASE = 0x0C0000      # erased archive flash in the booted state
CODE_FLASH = 0x300123      # CLMCODE payload in emulated flash (an odd address on purpose)
FRAM = 0xD52A00            # RAM window of the flash-resident routines (core/clm_code.h)
STUB = 0xD13E00            # harness call stub (Bench's default 0xD1A900 is inside the binary)
FORWARD_MAX = 3_000_000_000  # safety bound for one forward pass (catches runaway code)


@dataclass
class Export:
    """A trained export: blob, AppVar payloads (exactly as in CLMnn.8xv) and intref."""
    meta: dict
    blob: bytes
    appvars: list[bytes]
    path: Path = EXPORT_DIR
    _model: object = field(default=None, repr=False)
    _model_hz: object = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path = EXPORT_DIR) -> "Export":
        path = Path(path)
        meta = json.loads((path / "model.json").read_text())
        blob = (path / "model.bin").read_bytes()
        n = len(meta["appvars"])
        avs = [b"CL" + bytes([i, n]) + a["blob_start"].to_bytes(3, "little")
               + blob[a["blob_start"]:a["blob_end"]] for i, a in enumerate(meta["appvars"])]
        return cls(meta, blob, avs, path)

    @property
    def model(self):
        if self._model is None:
            from model.intref import IntModel
            self._model = IntModel.load(self.path)
        return self._model

    @property
    def model_hz(self):
        """intref with ALiBi horizons (PLAN-v2.1.md E1), the contract of a core with horizons."""
        if self._model_hz is None:
            from model.intref import IntModel
            self._model_hz = IntModel.load(self.path, horizons=True)
        return self._model_hz

    @property
    def vocab(self) -> int:
        return self.meta["config"]["vocab_size"]

    @property
    def dim(self) -> int:
        return self.meta["config"]["dim"]

    @property
    def n_layers(self) -> int:
        return self.meta["config"]["n_layers"]


SHAPE_KEYS = {"CLM_DIM": "dim", "CLM_LAYERS": "n_layers", "CLM_HEADS": "n_heads",
              "CLM_HIDDEN": "hidden_dim", "CLM_CTX": "max_seq_len"}


def shape_tag(ex: Export) -> str:
    """The core build (SHAPES tag) compiled for this export's dims."""
    import re
    cfg = ex.meta["config"]
    for tag, header in SHAPES.items():
        dims = {k: int(v) for k, v in re.findall(r"#define (CLM_\w+) (\d+)", header.read_text())}
        if all(dims[k] == cfg[c] for k, c in SHAPE_KEYS.items()):
            return tag
    raise BenchError(f"no core shape in SHAPES for {ex.path}: dim {cfg['dim']}, heads "
                     f"{cfg['n_heads']}, hidden {cfg['hidden_dim']} (generate a header with "
                     f"ce/calclm/tools/gen_shape.py)")


VOCAB_MAX = 1024           # CLM_VOCAB_MAX (clm_shape.h)


def act_bytes(ex: Export, i24: int) -> int:
    """CLM_ACT_BYTES of clm.h for an int24 width of i24 bytes."""
    c = ex.meta["config"]
    d, hid, v, ctx = c["dim"], c["hidden_dim"], VOCAB_MAX, c["max_seq_len"]
    gd, gh = -(-d // 5), -(-hid // 5)
    return (4 * d + 5 * gd + 5 * gh + i24 * (max(v, hid) + ctx + v) + d + 4 * hid + 2 * hid
            + i24 + ctx + 8 + 4 * 64)


def _run(cmd: list[str], **kw) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode:
        raise BenchError(f"{' '.join(cmd)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout


class Block:
    """drv_cmd's command block for a target with `psize`-byte pointers."""

    def __init__(self, psize: int):
        self.p = psize
        self.buf = bytearray(512)

    def clear(self, cmd: int) -> None:
        self.buf[:] = bytes(len(self.buf))
        self.buf[0] = cmd

    def put(self, off: int, value: int, n: int) -> None:
        self.buf[off:off + n] = (value & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

    def ptr(self, off: int, value: int) -> None:
        self.put(off, value, self.p)

    def load(self, count: int, ctx_cap: int, tok_cap: int, flags: int, ptrs: list[int],
             avs: list[tuple[int, int]]) -> None:
        """ptrs: kc, vc, tbl, act, tap, tok."""
        self.clear(LOAD)
        self.buf[1] = count
        self.put(2, ctx_cap, 2)
        self.buf[4] = flags
        self.put(6, tok_cap, 2)
        for i, v in enumerate(ptrs):
            self.ptr(8 + i * self.p, v)
        for i, (addr, size) in enumerate(avs):
            off = 8 + 6 * self.p + i * (self.p + 4)
            self.ptr(off, addr)
            self.put(off + self.p, size, 4)


class CoreBase:
    """Common driver protocol; subclasses provide memory and the call mechanism.
    horizons=True turns the core's ALiBi horizons on after every load (clm_load turns them
    off)."""
    psize = 8

    def __init__(self, ex: Export, horizons: bool = False):
        self.ex = ex
        self.horizons = horizons
        self.blk = Block(self.psize)

    # subclass API
    def _call(self) -> int: ...
    def _read(self, addr: int, n: int) -> bytes: ...

    def load(self, appvars: list[bytes] | None = None, ctx_cap: int = 256, tap: bool = False,
             prof: bool = False, tbl_misalign: int = 0, tok_cap: int = TOK_CAP) -> tuple[int, int]:
        """Returns (error code, bad AppVar index)."""
        raise NotImplementedError

    def reset(self, seed: int = 1, temperature_q8: int = 256, top_k: int = 40,
              greedy: bool = False, cap: int = 0, masks: bool | int = False) -> int:
        """cap: tokens in all, <bos> included (0: the core's default, CLM_DEFAULT_CAP).
        masks: False/0 off, True/1 the v2.3 masks, 2 the v2.4 safety masks (clm.h)."""
        self.blk.clear(RESET)
        self.blk.put(4, seed, 4)
        self.blk.put(8, temperature_q8, 2)
        self.blk.buf[10] = top_k & 0xFF
        self.blk.buf[11] = 1 if greedy else 0
        self.blk.put(12, cap, 2)
        self.blk.buf[14] = int(masks)
        return self._call()

    def step(self) -> int:
        self.blk.clear(STEP)
        return self._call()

    def feed(self, tokens: list[int]) -> int:
        """clm_feed each prompt token (at most 256 per call); CLM_OK or the first error."""
        assert len(tokens) <= 256
        self._write(self._track_in, np.asarray(tokens, dtype="<u2").tobytes())
        self.blk.clear(FEED)
        self.blk.put(4, len(tokens), 2)
        self.blk.ptr(8, self._track_in)
        return self._call()

    def mix_seed(self, seed: int) -> int:
        """clm_mix_seed (fmix32), computed by the core."""
        self.blk.clear(MIX)
        self.blk.put(4, seed, 4)
        self._call()
        return int.from_bytes(self.blk.buf[8:12], "little")

    def generate(self, n: int) -> list[int]:
        self.blk.clear(GENERATE)
        self.blk.put(4, n, 2)
        self.blk.ptr(8, self._tok_out)
        k = self._call()
        raw = self._read(self._tok_out, 2 * k)
        return list(np.frombuffer(raw, dtype="<u2").astype(int))

    def forward(self, tok: int, pos: int, copy: bool = True) -> np.ndarray | None:
        self.blk.clear(FORWARD)
        self.blk.put(4, tok, 2)
        self.blk.put(6, pos, 2)
        self.blk.ptr(8, self._logits_out if copy else 0)
        self._call()
        if not copy:
            return None
        return np.frombuffer(self._read(self._logits_out, 4 * self.ex.vocab), "<i4").astype(np.int64)

    def track(self, tokens: list[int], reset: bool = False) -> int:
        """The tracker reads <bos> (reset) and then `tokens` (at most 256 per call)."""
        assert len(tokens) <= 256
        self._write(self._track_in, np.asarray(tokens, dtype="<u2").tobytes())
        self.blk.clear(TRACK)
        self.blk.put(4, len(tokens), 2)
        self.blk.buf[6] = 1 if reset else 0
        self.blk.ptr(8, self._track_in)
        return self._call()

    def mask(self) -> np.ndarray:
        """The masks for the tracker state: (vocab,) bool, True = banned."""
        self.blk.clear(MASK)
        self.blk.ptr(8, self._logits_out)
        n = self._call()
        bits = np.frombuffer(self._read(self._logits_out, (self.ex.vocab + 7) // 8), np.uint8)
        out = np.unpackbits(bits, bitorder="little")[: self.ex.vocab].astype(bool)
        assert out.sum() == n
        return out

    def tracker(self) -> tuple[bytes, bytes, int]:
        """(trk_state bytes, trk_feat bytes, last token read)."""
        self.blk.clear(TRACKER)
        self.blk.ptr(8, self._logits_out)
        n = self._call()
        raw = self._read(self._logits_out, n + FEAT_BYTES + 2)
        return raw[:n], raw[n:n + FEAT_BYTES], int.from_bytes(raw[n + FEAT_BYTES:], "little")

    def seek(self, pos: int, filler: int = 60) -> int:
        self.blk.clear(SEEK)
        self.blk.put(4, pos, 2)
        self.blk.put(6, filler, 2)
        return self._call()

    def set_prof(self, on: bool) -> int:
        self.blk.clear(PROF)
        self.blk.buf[4] = 1 if on else 0
        return self._call()

    def set_horizons(self, on: bool) -> int:
        self.blk.clear(HORIZONS)
        self.blk.buf[4] = 1 if on else 0
        return self._call()

    def set_ld2(self, on: bool) -> int:
        self.blk.clear(LD2)
        self.blk.buf[4] = int(on)
        return self._call()

    def load_E(self, E: list[int]) -> int:
        if len(E) != 16 or any(not 0 <= x <= 0xFFFFFFFF for x in E):
            raise ValueError("LD2 needs 16 uint32 offsets")
        self.blk.clear(LOAD_E)
        for i, value in enumerate(E):
            self.blk.put(4 + 4 * i, value, 4)
        return self._call()

    def program_decoder(self, payload: bytes | None) -> int:
        """The OS program's post-load policy, including optional-offset fallback."""
        return self.load_ld2(payload or b"", program=True)

    def load_ld2(self, payload: bytes, program: bool = False) -> int:
        if len(payload) > 512:
            raise ValueError("LD2 payload exceeds driver input buffer")
        self._write(self._track_in, payload)
        self.blk.clear(PROGRAM_DECODER if program else LD2_LOAD)
        self.blk.buf[16:22] = bytes.fromhex(self.ex.meta["sha256"])[:6]
        self.blk.put(4, len(payload), 4)
        self.blk.ptr(8, self._track_in)
        return self._call()

    def _after_load(self, rc: int) -> None:
        if rc == 0 and self.horizons:
            assert self.set_horizons(True) == 0

    def taps(self) -> np.ndarray:
        n = 2 * self.ex.n_layers * self.ex.dim
        return np.frombuffer(self._read(self._tap, 4 * n), "<i4").astype(np.int64).reshape(
            2 * self.ex.n_layers, self.ex.dim)


# ---- host backend ----------------------------------------------------------------------------

TRACKER_DIR = ROOT / "tracker"
HOST_SRC = [CORE_DIR / f for f in ("clm_int.c", "clm_load.c", "clm_fwd.c", "clm_gen.c", "clm_ld2.c",
                                     "clm_ld2_load.c",
                                     "clm_mask.c", "clm_code.c")] + [
    CORE_DIR / "host" / "tmv_host.c", CALCLM / "test" / "drv.c", TRACKER_DIR / "tracker.c"]


def build_host(tag: str = "") -> Path:
    """gcc -O1 with UBSan (signed overflow, shifts, bounds of fixed arrays) -> shared library;
    the tracker (tracker/tracker.c, the masks' engine) is checked the same way. tag: the model
    shape (SHAPES), force-included as build_core.sh does."""
    out = out_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    lib = out / "calclm_host.so"
    deps = HOST_SRC + list(CORE_DIR.glob("*.h")) + list(TRACKER_DIR.glob("*.h")) + [Path(__file__)]
    if not lib.exists() or max(p.stat().st_mtime for p in deps) > lib.stat().st_mtime:
        shape = ["-include", str(SHAPES[tag])] if tag else []
        _run(["gcc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
              "-fsanitize=undefined", "-fno-sanitize-recover=all", *shape, f"-I{CORE_DIR}",
              f"-I{TRACKER_DIR}", *map(str, HOST_SRC), "-o", str(lib)])
    return lib


class HostCore(CoreBase):
    psize = ctypes.sizeof(ctypes.c_void_p)

    def __init__(self, ex: Export, horizons: bool = False):
        super().__init__(ex, horizons)
        self.lib = ctypes.CDLL(str(build_host(shape_tag(ex))))
        self.lib.drv_cmd.argtypes = [ctypes.c_char_p]
        self.lib.drv_cmd.restype = ctypes.c_int
        self.cbuf = ctypes.create_string_buffer(len(self.blk.buf))
        self._keep: list = []

    def _alloc(self, n: int) -> int:
        buf = ctypes.create_string_buffer(n)
        self._keep.append(buf)
        return ctypes.addressof(buf)

    def _call(self) -> int:
        ctypes.memmove(self.cbuf, bytes(self.blk.buf), len(self.blk.buf))
        rc = self.lib.drv_cmd(self.cbuf)
        self.blk.buf[:] = self.cbuf.raw
        return rc

    def _read(self, addr: int, n: int) -> bytes:
        return ctypes.string_at(addr, n)

    def _write(self, addr: int, data: bytes) -> None:
        ctypes.memmove(addr, data, len(data))

    def load(self, appvars=None, ctx_cap=256, tap=False, prof=False, tbl_misalign=0,
             tok_cap=TOK_CAP):
        appvars = self.ex.appvars if appvars is None else appvars
        self._keep = []
        avs = []
        for data in appvars:
            buf = ctypes.create_string_buffer(bytes(data), len(data) + 2)
            self._keep.append(buf)
            avs.append((ctypes.addressof(buf), len(data)))
        kv = self.ex.n_layers * max(ctx_cap, 1) * 32
        kc, vc = self._alloc(kv * KROW // 32), self._alloc(kv)
        bank = self._alloc(0x20000)
        tbl_addr = ((bank + 0xFFFF) & ~0xFFFF) + tbl_misalign
        act = self._alloc(act_bytes(self.ex, 4))
        tok = self._alloc(2 * max(tok_cap, 1))
        self._tap = self._alloc(4 * 2 * self.ex.n_layers * self.ex.dim)
        self._tok_out = self._alloc(2 * TOK_CAP)
        self._logits_out = self._alloc(4 * VOCAB_MAX)
        self._track_in = self._alloc(2 * 256)
        self.blk.load(len(appvars), ctx_cap, tok_cap, (1 if tap else 0) | (2 if prof else 0),
                      [kc, vc, tbl_addr, act, self._tap, tok], avs)
        rc = self._call()
        bad = ctypes.c_int8(self.blk.buf[1]).value
        self._after_load(rc)
        return rc, bad


# ---- eZ80 build and emulator backend ---------------------------------------------------------

def build_core(force: bool = False, tag: str = "") -> dict[str, str]:
    """ce/calclm/tools/build_core.sh: core.o (what the CE program links) + the test binary,
    for the model shape `tag` (SHAPES) into out_dir(tag)."""
    env = dict(os.environ, PYTHON=sys.executable)
    if tag:
        env.update(CLM_SHAPE=str(SHAPES[tag]), CALCLM_OUT=str(out_dir(tag)),
                   CALCLM_BIN=str(CALCLM / "bin" / tag))
    out = _run([str(CALCLM / "tools" / "build_core.sh"), f"{CODE_ORG:#x}"] + (["-f"] if force else []),
               env=env, cwd=str(ROOT))
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def read_syms(path: Path) -> dict[str, int]:
    syms = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            syms[parts[2].lstrip("_")] = int(parts[0], 16)
    return syms


class EmuCore(CoreBase):
    psize = 3

    def __init__(self, ex: Export, bench: Bench | None = None, lcd_dma: bool = False,
                 flash: bool = False, horizons: bool = False):
        """flash=True: after every load, attach the flash-resident copy of the assembly
        routines (CLMCODE, as on the calculator when CLMCODE.8xv is installed)."""
        super().__init__(ex, horizons)
        paths = build_core(tag=shape_tag(ex))
        self.flash = flash
        self.clmcode = Path(paths["clmcode"]).read_bytes()
        self.bin = Path(paths["test_bin"]).read_bytes()
        self.syms = read_syms(Path(paths["test_sym"]))
        self.org = int(paths["test_org"], 16)
        self.bench = bench or Bench(lcd_dma=lcd_dma, stub_addr=STUB)
        # the taps end where LOGITS_OUT starts and the V cache (full context) sits below them:
        # TAP_ADDR and VC_ADDR for v2.2's shape, lower for a wider model
        self._tap = LOGITS_OUT - 4 * 2 * ex.n_layers * ex.dim
        self.vc_addr = min(VC_ADDR, self._tap - ex.n_layers * 256 * 32)
        self._tok_out, self._logits_out = OUT_TOK, LOGITS_OUT
        self._track_in = TRACK_IN
        self.last = None   # CallResult of the last driver call
        if act_bytes(ex, 3) > FRAM - ACT_ADDR:
            raise BenchError("activations run into the flash routines' RAM window")
        if self.org + len(self.bin) + 0x1000 > self.vc_addr:   # 0x1000: the driver's own BSS
            raise BenchError("the test binary runs into the V cache")

    def _max_cycles(self) -> int:
        cmd = self.blk.buf[0]
        if cmd in (GENERATE, FEED):          # a forward pass per token
            return FORWARD_MAX * (int.from_bytes(self.blk.buf[4:6], "little") + 1)
        return FORWARD_MAX if cmd in (STEP, FORWARD) else 50_000_000

    def _call(self, max_cycles: int | None = None) -> int:
        max_cycles = max_cycles or self._max_cycles()
        self.bench.write_ram(BLK_ADDR, bytes(self.blk.buf))
        self.last = self.bench.call(self.syms["drv_cmd"], [BLK_ADDR], flush_cache=False,
                                    max_cycles=max_cycles)
        self.blk.buf[:] = self.bench.read_ram(BLK_ADDR, len(self.blk.buf))
        ret = self.last.ret
        return ret - (1 << 24) if ret & 0x800000 else ret

    def _read(self, addr: int, n: int) -> bytes:
        return self.bench.read_ram(addr, n)

    def _write(self, addr: int, data: bytes) -> None:
        self.bench.write_ram(addr, data)

    def place_appvars(self, appvars: list[bytes]) -> list[tuple[int, int]]:
        """Write AppVar payloads into erased flash like archived variables (odd gaps between
        them, as the archive's variable headers would leave). Returns (address, size)."""
        addr, out = FLASH_BASE + 0x123, []
        for i, data in enumerate(appvars):
            self.bench.write_flash(addr, bytes(data))
            out.append((addr, len(data)))
            addr += len(data) + 17 + 3 * i
        return out

    def load(self, appvars=None, ctx_cap=256, tap=False, prof=False, tbl_misalign=0,
             tok_cap=TOK_CAP):
        appvars = self.ex.appvars if appvars is None else appvars
        self.bench.reset()
        self.bench.write_ram(self.org, self.bin)
        avs = self.place_appvars(appvars)
        self.blk.load(len(appvars), ctx_cap, tok_cap, (1 if tap else 0) | (2 if prof else 0),
                      [KC_ADDR, self.vc_addr, TBL_ADDR + tbl_misalign, ACT_ADDR, self._tap,
                       TOK_ADDR],
                      avs)
        rc = self._call()
        bad = ctypes.c_int8(self.blk.buf[1]).value
        if self.flash:
            assert self.attach() == 0
        self._after_load(rc)
        return rc, bad

    def attach(self, payload: bytes | None = None) -> int:
        """Put a CLMCODE payload into emulated flash and attach it (driver ATTACH)."""
        payload = self.clmcode if payload is None else payload
        self.bench.write_flash(CODE_FLASH, payload, require_erased=False)
        self.blk.clear(ATTACH)
        self.blk.put(4, len(payload), 3)
        self.blk.ptr(8, CODE_FLASH)
        return self._call()

    def detach(self) -> int:
        self.blk.clear(DETACH)
        return self._call()


# The CE C toolchain: CEDEV, else its usual install folder ~/CEdev.
CEDEV = Path(os.environ.get("CEDEV") or Path.home() / "CEdev")


def emu_available() -> bool:
    from .bench import DEFAULT_ROM
    return (DEFAULT_ROM is not None and DEFAULT_ROM.exists()
            and (CEDEV / "bin" / "ez80-clang").exists())
