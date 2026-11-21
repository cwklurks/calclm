"""E20 spike harness (PLAN-E20.md Design 7): the CalcLM core built with 4-bit K/V on the host
and in CEmu, against model/kv4.py's Kv4Model.

    from ce.tools.clm_kv4 import Kv4Export, HostKv4Core, EmuKv4Core
    ex = Kv4Export.load("runs/e16/export/W8-s10")       # tables: runs/e19/kv4/W8-s10.json
    core = EmuKv4Core(ex, flash=True, horizons=True, ram_tables=True)
    core.load()                                          # LOAD, then KV4_LOAD (CLMKV4)

ce/tools/clm.py is not edited: these classes subclass it. The kv4 builds are picked by the
export's depth (w8k4: core/clm_shape_w8k4.h, 8 layers; wk4: core/clm_shape_wk4.h, 6 layers) and
the header is passed explicitly (clm.shape_tag would pick the int8 W build for L4-s10 and no
build for W8). Memory follows core/clm_kv4.h: kc holds the packed K rows (19 B), vc the packed V
rows (16 B), then the scratch (one layer's decoded window, ctx_cap x 67 B), then, with
ram_tables (placement B), the biased tables at a 512-aligned address (512 B of slack).
Canary bytes sit around kc, around vc's whole region and in the alignment slack (T7).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bench import Bench, BenchError
from .clm import (
    ACT_ADDR, BLK_ADDR, CALCLM, CORE_DIR, FRAM, HOST_SRC, KC_ADDR, KROW, LOGITS_OUT, OUT_TOK,
    ROOT, STUB, TBL_ADDR, TOK_ADDR, TOK_CAP, TRACK_IN, TRACKER_DIR, VOCAB_MAX, CoreBase, EmuCore,
    Export, HostCore, _run, act_bytes, out_dir, read_syms,
)

KV4_SHAPES = {"w8k4": CORE_DIR / "clm_shape_w8k4.h", "wk4": CORE_DIR / "clm_shape_wk4.h"}
E19_KV4 = ROOT / "runs" / "e19" / "kv4"

# driver commands appended to drv_cmd's enum in the kv4 build (test/drv.c)
KV4_LOAD, KV4_UNPACK, KV4_ENCODE = 18, 19, 20
HEAD_DIM = 32
KROW4 = HEAD_DIM // 2 + 3          # packed K row: 16 bytes of codes + int24 sum term
VROW4 = HEAD_DIM // 2              # packed V row
SROW = KROW + HEAD_DIM             # scratch: one decoded K row (35 B) and V row (32 B) per slot
ALIGN = 512                        # placement B: the RAM tables' alignment (one 64 KB bank per
SLACK = 512                        # (layer, kind) block); slack keeps vc's end 8-aligned
CANARY = 16                        # canary bytes on each side of a region (T7)
CANARY_BYTE = 0xA5
KV4_FLASH = 0x310123               # CLMKV4 payload in emulated flash (after CLMCODE; odd on purpose)
E_ARG = -11


def tbl_bytes(n_layers: int) -> int:
    return n_layers * 2 * HEAD_DIM * 16


def kc_bytes(n_layers: int, ctx_cap: int) -> int:
    return n_layers * ctx_cap * KROW4


def vc_bytes(n_layers: int, ctx_cap: int, ram: bool) -> int:
    """vc's size contract in the kv4 build: packed V, scratch and (placement B) the tables with
    the worst-case alignment slack."""
    return (n_layers * ctx_cap * VROW4 + ctx_cap * SROW
            + (tbl_bytes(n_layers) + SLACK if ram else 0))


def scratch_addr(vc: int, n_layers: int, ctx_cap: int) -> int:
    return vc + n_layers * ctx_cap * VROW4


def tables_addr(vc: int, n_layers: int, ctx_cap: int) -> int:
    end = scratch_addr(vc, n_layers, ctx_cap) + ctx_cap * SROW
    return (end + ALIGN - 1) & ~(ALIGN - 1)


def kv4_tag(ex: Export) -> str:
    c = ex.meta["config"]
    if (c["dim"], c["n_heads"], c["hidden_dim"], c["max_seq_len"]) != (256, 8, 576, 256):
        raise BenchError(f"no kv4 build for {ex.path}: dim {c['dim']}, heads {c['n_heads']}")
    tag = {8: "w8k4", 6: "wk4"}.get(c["n_layers"])
    if tag is None:
        raise BenchError(f"no kv4 build for {c['n_layers']} layers ({ex.path})")
    return tag


def default_tables(export: Path) -> Path:
    return E19_KV4 / f"{Path(export).name}.json"


@dataclass
class Kv4Export(Export):
    """An export with its stored E19 tables: model / model_hz are Kv4Model (horizons off / on),
    the contract of the kv4 core; payload is its CLMKV4 (ce/calclm/tools/make_kv4.py)."""
    tables: object = field(default=None, repr=False)
    tables_path: Path | None = None
    payload: bytes = field(default=b"", repr=False)

    @classmethod
    def load(cls, path: Path, tables: Path | None = None) -> "Kv4Export":
        from ce.calclm.tools.make_kv4 import payload
        from model.kv4 import read_tables
        base = Export.load(path)
        tpath = Path(tables) if tables is not None else default_tables(path)
        tab = read_tables(tpath, base.path)          # refuses another export's tables
        return cls(base.meta, base.blob, base.appvars, base.path, tables=tab, tables_path=tpath,
                   payload=payload(base.path, tpath))

    def _kv4(self, horizons: bool):
        from model.export import load_export
        from model.kv4 import Kv4Model
        return Kv4Model(load_export(self.path), self.tables, "kv4", horizons=horizons)

    @property
    def model(self):
        if self._model is None:
            self._model = self._kv4(False)
        return self._model

    @property
    def model_hz(self):
        if self._model_hz is None:
            self._model_hz = self._kv4(True)
        return self._model_hz

    def int8_model(self, horizons: bool):
        """The unchanged int8 intref for non-vacuity checks, built with the constructor (never
        through IntModel.load, which the kv4 tests patch)."""
        from model.export import load_export
        from model.intref import IntModel
        return IntModel(load_export(self.path), horizons=horizons)


# ---- builds -----------------------------------------------------------------------------------

def build_host_kv4(tag: str) -> Path:
    """clm.build_host for a kv4 shape: gcc -O1 + UBSan, the kv4 header force-included."""
    out = out_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    lib = out / "calclm_host.so"
    deps = HOST_SRC + list(CORE_DIR.glob("*.h")) + list(TRACKER_DIR.glob("*.h")) + [Path(__file__)]
    if not lib.exists() or max(p.stat().st_mtime for p in deps) > lib.stat().st_mtime:
        _run(["gcc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
              "-fsanitize=undefined", "-fno-sanitize-recover=all", "-include",
              str(KV4_SHAPES[tag]), f"-I{CORE_DIR}", f"-I{TRACKER_DIR}", *map(str, HOST_SRC),
              "-o", str(lib)])
    return lib


def build_core_kv4(tag: str, force: bool = False) -> dict[str, str]:
    """clm.build_core for a kv4 shape: core.o, test.bin and CLMCODE into ce/build/calclm-<tag>."""
    from .clm import CODE_ORG
    env = dict(os.environ, PYTHON=sys.executable, CLM_SHAPE=str(KV4_SHAPES[tag]),
               CALCLM_OUT=str(out_dir(tag)), CALCLM_BIN=str(CALCLM / "bin" / tag))
    out = _run([str(CALCLM / "tools" / "build_core.sh"), f"{CODE_ORG:#x}"]
               + (["-f"] if force else []), env=env, cwd=str(ROOT))
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


PREBUILT_ENV = "E20_PREBUILT"


def core_paths(tag: str, force: bool = False) -> dict[str, str]:
    """The eZ80 build of `tag` (kv4 tags through build_core_kv4, "" and "w" through
    clm.build_core), its paths recorded in out_dir(tag)/paths.json. With E20_PREBUILT set (the
    parallel CEmu processes ce.tools.clm_bench_kv4 starts after building every core once), the
    record is read instead, so no two processes rebuild the same test.bin at once (PLAN-E20
    Constraints 7)."""
    rec = out_dir(tag) / "paths.json"
    if os.environ.get(PREBUILT_ENV):
        return json.loads(rec.read_text())
    from .clm import build_core
    paths = build_core_kv4(tag, force) if tag in KV4_SHAPES else build_core(force, tag)
    rec.write_text(json.dumps(paths))
    return paths


# ---- the kv4 protocol (shared by both backends) ------------------------------------------------

class _Kv4Mixin:
    """KV4_LOAD after every successful load (unless load(kv4=False)), the test-only KV4_UNPACK
    and KV4_ENCODE, the scratch window, and the canaries."""
    ram_tables: bool = False

    def _kv4_init(self, ram_tables: bool) -> None:
        self.tag = kv4_tag(self.ex)
        self.ram_tables = ram_tables
        self.ctx_cap = 256
        self._canaries: list[tuple[int, int]] = []

    def kv4_load(self, payload: bytes | None = None, ram: bool | None = None,
                 sha: bytes | None = None) -> int:
        payload = self.ex.payload if payload is None else payload
        ram = self.ram_tables if ram is None else ram
        addr = self._place_kv4(payload)
        self.blk.clear(KV4_LOAD)
        self.blk.put(4, len(payload), 4)
        self.blk.ptr(8, addr)
        self.blk.buf[16:22] = (bytes.fromhex(self.ex.meta["sha256"])[:6] if sha is None
                               else sha)
        self.blk.buf[22] = 1 if ram else 0
        return self._call()

    def kv4_unpack(self, layer: int, n: int, c_only: bool = False) -> int:
        self.blk.clear(KV4_UNPACK)
        self.blk.buf[4] = layer
        self.blk.buf[5] = 1 if c_only else 0
        self.blk.put(6, n, 2)
        return self._call()

    def kv4_encode(self, layer: int, kind: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
        """For x = -128..127: (the nibble code j + 7, the biased decoded value T[j] + 128)."""
        self.blk.clear(KV4_ENCODE)
        self.blk.buf[4], self.blk.buf[5], self.blk.buf[6] = layer, kind, channel
        self.blk.ptr(8, self._logits_out)
        rc = self._call()
        if rc != 0:
            raise BenchError(f"KV4_ENCODE returned {rc}")
        raw = np.frombuffer(self._read(self._logits_out, 512), np.uint8)
        return raw[:256].astype(np.int64), raw[256:].astype(np.int64)

    @property
    def scratch(self) -> int:
        return scratch_addr(self._vc, self.ex.n_layers, self.ctx_cap)

    @property
    def ram_tables_addr(self) -> int:
        return tables_addr(self._vc, self.ex.n_layers, self.ctx_cap)

    def read_scratch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The first n decoded rows of the scratch window: K (n, 35) and V (n, 32), uint8."""
        cap = self.ctx_cap
        k = np.frombuffer(self._read(self.scratch, n * KROW), np.uint8).reshape(n, KROW)
        v = np.frombuffer(self._read(self.scratch + cap * KROW, n * HEAD_DIM),
                          np.uint8).reshape(n, HEAD_DIM)
        return k, v

    def read_ram_tables(self) -> bytes:
        return self._read(self.ram_tables_addr, tbl_bytes(self.ex.n_layers))

    def _set_canaries(self, kc: int, vc: int, cap: int) -> None:
        """Canaries before and after kc, before vc, after the scratch (the alignment slack up to
        the tables, or 16 bytes with placement A) and after the RAM tables."""
        L = self.ex.n_layers
        regions = [(kc - CANARY, CANARY), (kc + kc_bytes(L, cap), CANARY), (vc - CANARY, CANARY)]
        s_end = scratch_addr(vc, L, cap) + cap * SROW
        if self.ram_tables:
            t = tables_addr(vc, L, cap)
            if t > s_end:
                regions.append((s_end, t - s_end))
            regions.append((t + tbl_bytes(L), CANARY))
        else:
            regions.append((s_end, CANARY))
        self._canaries = regions
        for a, n in regions:
            self._write(a, bytes([CANARY_BYTE]) * n)

    def canaries_intact(self) -> bool:
        return all(self._read(a, n) == bytes([CANARY_BYTE]) * n for a, n in self._canaries)


# ---- host backend ------------------------------------------------------------------------------

class HostKv4Core(_Kv4Mixin, HostCore):
    def __init__(self, ex: Kv4Export, horizons: bool = False, ram_tables: bool = False):
        CoreBase.__init__(self, ex, horizons)
        self._kv4_init(ram_tables)
        self.lib = ctypes.CDLL(str(build_host_kv4(self.tag)))
        self.lib.drv_cmd.argtypes = [ctypes.c_char_p]
        self.lib.drv_cmd.restype = ctypes.c_int
        self.cbuf = ctypes.create_string_buffer(len(self.blk.buf))
        self._keep: list = []

    def _place_kv4(self, payload: bytes) -> int:
        return self._alloc_bytes(payload)

    def _alloc_bytes(self, data: bytes) -> int:
        buf = ctypes.create_string_buffer(bytes(data), len(data) + 2)
        self._keep.append(buf)
        return ctypes.addressof(buf)

    def load(self, appvars=None, ctx_cap=256, tap=False, prof=False, tbl_misalign=0,
             tok_cap=TOK_CAP, kv4=True):
        appvars = self.ex.appvars if appvars is None else appvars
        self._keep = []
        avs = [(self._alloc_bytes(d), len(d)) for d in appvars]
        L, cap = self.ex.n_layers, max(ctx_cap, 1)
        kc = self._alloc(kc_bytes(L, cap) + 2 * CANARY) + CANARY
        vsize = vc_bytes(L, cap, True)                 # room for either placement
        vc = self._alloc(vsize + 2 * CANARY + ALIGN) + CANARY
        bank = self._alloc(0x20000)
        tbl_addr = ((bank + 0xFFFF) & ~0xFFFF) + tbl_misalign
        act = self._alloc(act_bytes(self.ex, 4))
        tok = self._alloc(2 * max(tok_cap, 1))
        self._tap = self._alloc(4 * 2 * L * self.ex.dim)
        self._tok_out = self._alloc(2 * TOK_CAP)
        self._logits_out = self._alloc(4 * VOCAB_MAX)
        self._track_in = self._alloc(2 * 256)
        self._kc, self._vc, self.ctx_cap = kc, vc, cap
        self._set_canaries(kc, vc, cap)
        self.blk.load(len(appvars), ctx_cap, tok_cap, (1 if tap else 0) | (2 if prof else 0),
                      [kc, vc, tbl_addr, act, self._tap, tok], avs)
        rc = self._call()
        bad = ctypes.c_int8(self.blk.buf[1]).value
        if rc == 0 and kv4:
            assert self.kv4_load() == 0
        self._after_load(rc)
        return rc, bad


# ---- emulator backend --------------------------------------------------------------------------

class EmuKv4Core(_Kv4Mixin, EmuCore):
    kv4_flash = KV4_FLASH          # where load() writes CLMKV4 in emulated flash

    def __init__(self, ex: Kv4Export, bench: Bench | None = None, lcd_dma: bool = False,
                 flash: bool = False, horizons: bool = False, ram_tables: bool = False):
        CoreBase.__init__(self, ex, horizons)
        self._kv4_init(ram_tables)
        paths = core_paths(self.tag)
        self.flash = flash
        self.clmcode = Path(paths["clmcode"]).read_bytes()
        self.bin = Path(paths["test_bin"]).read_bytes()
        self.syms = read_syms(Path(paths["test_sym"]))
        self.org = int(paths["test_org"], 16)
        self.bench = bench or Bench(lcd_dma=lcd_dma, stub_addr=STUB)
        L = ex.n_layers
        self._tap = LOGITS_OUT - 4 * 2 * L * ex.dim
        # vc's whole kv4 region (placement B's size at the full window) ends a canary below
        # the taps
        self.vc_addr = self._tap - CANARY - vc_bytes(L, 256, True)
        self._tok_out, self._logits_out = OUT_TOK, LOGITS_OUT
        self._track_in = TRACK_IN
        self.last = None
        if act_bytes(ex, 3) > FRAM - ACT_ADDR:
            raise BenchError("activations run into the flash routines' RAM window")
        if self.org + len(self.bin) + 0x1000 > self.vc_addr - CANARY:
            raise BenchError("the test binary runs into the kv4 V region")
        if KC_ADDR + kc_bytes(L, 256) + CANARY > BLK_ADDR:
            raise BenchError("the packed K cache runs into the driver block")

    def _place_kv4(self, payload: bytes) -> int:
        self.bench.write_flash(self.kv4_flash, payload, require_erased=False)
        return self.kv4_flash

    def load(self, appvars=None, ctx_cap=256, tap=False, prof=False, tbl_misalign=0,
             tok_cap=TOK_CAP, kv4=True):
        appvars = self.ex.appvars if appvars is None else appvars
        self.bench.reset()
        self.bench.write_ram(self.org, self.bin)
        avs = self.place_appvars(appvars)
        self._kc, self._vc, self.ctx_cap = KC_ADDR, self.vc_addr, max(ctx_cap, 1)
        self._set_canaries(KC_ADDR, self.vc_addr, self.ctx_cap)
        self.blk.load(len(appvars), ctx_cap, tok_cap, (1 if tap else 0) | (2 if prof else 0),
                      [KC_ADDR, self.vc_addr, TBL_ADDR + tbl_misalign, ACT_ADDR, self._tap,
                       TOK_ADDR], avs)
        rc = self._call()
        bad = ctypes.c_int8(self.blk.buf[1]).value
        if self.flash:
            assert self.attach() == 0
        if rc == 0 and kv4:
            assert self.kv4_load() == 0
        self._after_load(rc)
        return rc, bad

    def place_appvars(self, appvars):
        out = super().place_appvars(appvars)
        if out and out[-1][0] + out[-1][1] > KV4_FLASH - 0x10000:
            raise BenchError("the model AppVars run into CLMCODE / CLMKV4 in emulated flash")
        return out



class EmuInt8Core(EmuCore):
    """clm.EmuCore (an int8 build picked by clm.shape_tag) whose build comes from core_paths, so
    parallel bench processes read the one prebuilt test binary (M1-M3 baselines)."""

    def __init__(self, ex: Export, bench: Bench | None = None, lcd_dma: bool = False,
                 flash: bool = False, horizons: bool = False):
        from .clm import STUB, VC_ADDR, shape_tag
        CoreBase.__init__(self, ex, horizons)
        paths = core_paths(shape_tag(ex))
        self.flash = flash
        self.clmcode = Path(paths["clmcode"]).read_bytes()
        self.bin = Path(paths["test_bin"]).read_bytes()
        self.syms = read_syms(Path(paths["test_sym"]))
        self.org = int(paths["test_org"], 16)
        self.bench = bench or Bench(lcd_dma=lcd_dma, stub_addr=STUB)
        self._tap = LOGITS_OUT - 4 * 2 * ex.n_layers * ex.dim
        self.vc_addr = min(VC_ADDR, self._tap - ex.n_layers * 256 * 32)
        self._tok_out, self._logits_out = OUT_TOK, LOGITS_OUT
        self._track_in = TRACK_IN
        self.last = None
        if self.org + len(self.bin) + 0x1000 > self.vc_addr:
            raise BenchError("the test binary runs into the V cache")
