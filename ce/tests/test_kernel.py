"""The eZ80 ternary LUT matvec kernels, run in headless CEmu, must be bit-identical to
model.pack.lut_matvec (SPEC.md kernel contract). Skips when the ROM or ~/CEdev is missing."""
from __future__ import annotations

import numpy as np
import pytest

from ce.tools.bench import DEFAULT_ROM, Bench
from ce.tools.tmv import run_asm, run_c
from model.pack import lut_matvec, pack_ternary
from pathlib import Path

pytestmark = pytest.mark.skipif(
    DEFAULT_ROM is None or not DEFAULT_ROM.exists() or not (Path.home() / "CEdev" / "bin" / "ez80-clang").exists(),
    reason="needs the TI-84 Plus CE ROM and the CE toolchain on this box")

SHAPES = [(192, 192), (192, 512), (512, 192)]   # (n_in, n_out)


@pytest.fixture(scope="module")
def bench() -> Bench:
    return Bench()


def random_case(n_in: int, n_out: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.integers(-1, 2, size=(n_out, n_in), dtype=np.int8)
    x = rng.integers(-127, 128, size=n_in, dtype=np.int8)
    return w, x


def check(bench: Bench, w: np.ndarray, x: np.ndarray, runner=run_asm, **kw):
    packed = pack_ternary(w)
    want = lut_matvec(packed, x)
    assert want.tolist() == (w.astype(np.int64) @ x.astype(np.int64)).tolist()  # sanity
    res = runner(bench, packed, x, **kw)
    assert res.y.dtype == np.int32 and res.y.shape == want.shape
    np.testing.assert_array_equal(res.y, want)
    return res


@pytest.mark.parametrize("n_in,n_out", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_flash_weights(bench, n_in, n_out, seed):
    w, x = random_case(n_in, n_out, 1000 * seed + n_in + n_out)
    res = check(bench, w, x, weights="flash")
    assert res.rows.flash_misses > 0          # the weights really streamed from flash


@pytest.mark.parametrize("n_in,n_out", SHAPES)
def test_random_ram_weights(bench, n_in, n_out):
    w, x = random_case(n_in, n_out, 7)
    res = check(bench, w, x, weights="ram")
    assert res.rows.flash_misses == 0


@pytest.mark.parametrize("n_in,n_out", SHAPES)
def test_flash_resident_code(bench, n_in, n_out):
    # position-independent kernel executing in place from flash (an archived AppVar on the calc)
    w, x = random_case(n_in, n_out, 11)
    check(bench, w, x, weights="flash", code="flash")


@pytest.mark.parametrize("load_at", [0x2A0000, 0x2A1234, 0x0C0011])
def test_flash_code_is_relocatable(bench, load_at):
    # linked at FLASH_CODE_ORG, loaded elsewhere; only the RAM trampolines are patched
    w, x = random_case(512, 33, load_at & 0xFF)
    check(bench, w, x, weights="flash", code="flash", load_at=load_at)


@pytest.mark.parametrize("n_in,n_out", [(192, 192), (13, 7), (512, 3)])
def test_simple_de_variant(bench, n_in, n_out):
    w, x = random_case(n_in, n_out, 13)
    check(bench, w, x, variant="de")


@pytest.mark.parametrize("n_in,n_out", [(192, 192), (13, 7), (512, 5)])
def test_c_reference(bench, n_in, n_out):
    w, x = random_case(n_in, n_out, 17)
    check(bench, w, x, runner=run_c)


@pytest.mark.parametrize("n_in", [1, 2, 4, 6, 7, 13, 191, 193, 509, 511, 513, 634])
@pytest.mark.parametrize("n_out", [1, 2, 7])
def test_n_in_not_multiple_of_five(bench, n_in, n_out):
    # odd and even group counts (G = ceil(n_in/5)), odd and even row counts (tail row path)
    w, x = random_case(n_in, n_out, n_in * 31 + n_out)
    check(bench, w, x)


@pytest.mark.parametrize("n_in", [192, 512, 634])
@pytest.mark.parametrize("sx,sw", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_extremes(bench, n_in, sx, sw):
    # |y| = 127 * n_in: 65024 for 512 inputs needs 17 bits, 80518 for 634
    x = np.full(n_in, 127 * sx, np.int8)
    w = np.full((4, n_in), sw, np.int8)
    res = check(bench, w, x)
    assert abs(int(res.y[0])) == 127 * n_in


@pytest.mark.parametrize("n_in", [192, 512])
def test_mixed_extremes(bench, n_in):
    rng = np.random.default_rng(n_in)
    x = rng.choice(np.array([-127, 127], np.int8), size=n_in)
    w = np.stack([np.sign(x).astype(np.int8), -np.sign(x).astype(np.int8),
                  rng.integers(-1, 2, size=n_in, dtype=np.int8), np.zeros(n_in, np.int8)])
    res = check(bench, w, x)
    assert res.y[:2].tolist() == [127 * n_in, -127 * n_in]


@pytest.mark.parametrize("n_in,n_out", SHAPES)
def test_all_zero_weights(bench, n_in, n_out):
    _, x = random_case(n_in, n_out, 3)
    res = check(bench, np.zeros((n_out, n_in), np.int8), x)
    assert not res.y.any()


def test_zero_input(bench):
    w, _ = random_case(192, 192, 5)
    res = check(bench, w, np.zeros(192, np.int8))
    assert not res.y.any()


def test_more_than_256_row_iterations(bench):
    # the row loop counts iterations in A (inner, mod 256) and B' (outer)
    w, x = random_case(15, 1027, 9)      # G = 3 (odd): 513 two-row iterations + tail row
    check(bench, w, x)
    w, x = random_case(20, 600, 10)      # G = 4 (even): 600 one-row iterations
    check(bench, w, x)


def test_tables_are_reusable_across_matrices(bench):
    # q, k, v share one input vector: one build, several row passes
    from ce.tools import tmv
    rng = np.random.default_rng(21)
    x = rng.integers(-127, 128, size=192, dtype=np.int8)
    ws = [rng.integers(-1, 2, size=(n, 192), dtype=np.int8) for n in (192, 32, 32)]
    k = tmv.asm_kernel(39)
    bench.reset()
    bench.write_ram(k.org, k.code)
    bench.write_ram(tmv.X_ADDR, x.tobytes() + b"\0\0\0")
    bench.call(k["tmv_build"], [tmv.X_ADDR, tmv.TBL_ADDR])
    for w in ws:
        p = pack_ternary(w)
        bench.write_ram(tmv.W_RAM, p.tobytes() + b"\0\0")
        bench.call(k["tmv_rows"], [tmv.W_RAM, tmv.Y_ADDR, len(w), tmv.TBL_ADDR])
        np.testing.assert_array_equal(tmv._read_y(bench, len(w)), lut_matvec(p, x))


def test_rejects_out_of_range_input(bench):
    w, _ = random_case(5, 1, 0)
    with pytest.raises(ValueError):
        run_asm(bench, pack_ternary(w), np.array([-128, 0, 0, 0, 0], np.int8))


def test_restores_enabled_interrupts(bench):
    """Called from C with interrupts on, both functions disable them (SP is a data register)
    and turn them back on before returning."""
    from ce.tools import tmv
    from ce.tools.bench import RET_ADDR
    w, x = random_case(192, 8, 4)
    p = pack_ternary(w)
    k = tmv.asm_kernel(39)
    bench.reset()
    bench.write_ram(k.org, k.code)
    bench.write_ram(tmv.X_ADDR, x.tobytes() + b"\0\0\0")
    bench.write_flash(tmv.W_FLASH, p.tobytes() + b"\0\0")

    def call(fn, args):
        code = bytearray([0xFB])                                  # ei
        for a in reversed(args):
            code += bytes([0x21]) + a.to_bytes(3, "little") + bytes([0xE5])  # ld hl,a; push hl
        code += bytes([0xCD]) + fn.to_bytes(3, "little")          # call fn
        code += bytes([0xED, 0x57])                               # ld a,i  (P/V = IEF1)
        code += bytes([0xF3, 0xF5, 0xE1])                         # di; push af; pop hl
        code += bytes([0x22]) + RET_ADDR.to_bytes(3, "little") + bytes([0x76])  # ld (ret),hl; halt
        bench.run_code(bytes(code))
        return bench.read_ram(RET_ADDR, 1)[0] & 0x04              # P/V flag after return

    assert call(k["tmv_build"], [tmv.X_ADDR, tmv.TBL_ADDR])
    assert call(k["tmv_rows"], [tmv.W_FLASH, tmv.Y_ADDR, 8, tmv.TBL_ADDR])
    np.testing.assert_array_equal(tmv._read_y(bench, 8), lut_matvec(p, x))


# ------------------------------------------------------------------ timing and bounds

def test_cycle_calibration_nop_sled(bench):
    """The harness counts CEmu cycles exactly: an opcode fetch costs 4 from RAM; from flash 2
    per byte plus 195 per missed 32-byte cache line (cold cache), as in core/mem.c, flash.c."""
    for n in (0, 1, 100, 1000):
        bench.reset()
        bench.write_ram(0xD1B000, b"\x00" * n + b"\xc9")          # n x nop; ret
        assert bench.call(0xD1B000).cycles == 4 * n
        bench.write_flash(0x380000, b"\x00" * n + b"\xc9")
        r = bench.call(0x380000)
        assert r.flash_misses == (n + 1 + 31) // 32
        assert r.cycles - (2 * n + 195 * r.flash_misses) == -4      # same constant for every n


def test_timing_is_data_independent(bench):
    cycles = set()
    for seed in range(4):
        w, x = random_case(192, 64, 100 + seed)
        if seed == 3:
            w[:] = 0
        res = check(bench, w, x)
        cycles.add((res.build.cycles, res.rows.cycles))
    assert len(cycles) == 1


@pytest.mark.parametrize("code,build,rows", [("ram", 273_905, 379_570), ("flash", 179_149, 270_847)])
def test_cycle_regression_192x192(bench, code, build, rows):
    # BENCH.md numbers (weights in flash, cold cache, LCD DMA off); 1% band
    w, x = random_case(192, 192, 0)
    res = check(bench, w, x, code=code)
    assert abs(res.build.cycles - build) <= build // 100
    assert abs(res.rows.cycles - rows) <= rows // 100


@pytest.mark.parametrize("G", [1, 39])
def test_zero_rows(bench, G):
    w, x = random_case(5 * G, 1, 0)
    res = run_asm(bench, pack_ternary(w)[:0], x)
    assert res.y.shape == (0,)


@pytest.mark.parametrize("n_in,c0", [(192, 40), (512, 40), (7, 200)])
def test_table_column_offset(bench, n_in, c0):
    w, x = random_case(n_in, 9, c0)
    check(bench, w, x, c0=c0)
    check(bench, w, x, c0=c0, code="flash")


def test_table_pointer_must_be_bank_aligned():
    from ce.tools.tmv import table_ptr
    assert table_ptr(0xD30000, 40) == 0xD30028
    with pytest.raises(ValueError):
        table_ptr(0xD31000)


@pytest.mark.parametrize("n_in,c0", [(192, 0), (512, 30), (634, 0)])
def test_writes_stay_in_bounds(bench, n_in, c0):
    """Tables touch only pages 0..242 x columns C0..C0+2G of their bank; y only 3*n_out bytes."""
    from ce.tools import tmv
    n_out = 5
    G = -(-n_in // 5)
    w, x = random_case(n_in, n_out, 77)
    p = pack_ternary(w)
    k = tmv.asm_kernel(G, c0=c0)
    bench.reset()
    bench.write_ram(k.org, k.code)
    bench.write_ram(tmv.X_ADDR, tmv._pad_x(x, G))
    bench.write_flash(tmv.W_FLASH, p.tobytes() + b"\0\0")
    guard = bytes([0x5A]) * 0x10000
    bench.write_ram(tmv.TBL_ADDR, guard)
    bench.write_ram(tmv.Y_ADDR, b"\xa5" * (3 * n_out + 64))
    tbl = tmv.table_ptr(tmv.TBL_ADDR, c0)
    bench.call(k["tmv_build"], [tmv.X_ADDR, tbl])
    bench.call(k["tmv_rows"], [tmv.W_FLASH, tmv.Y_ADDR, n_out, tbl])
    np.testing.assert_array_equal(tmv._read_y(bench, n_out), lut_matvec(p, x))
    assert bench.read_ram(tmv.Y_ADDR + 3 * n_out, 64) == b"\xa5" * 64
    bank = np.frombuffer(bench.read_ram(tmv.TBL_ADDR, 0x10000), np.uint8).reshape(256, 256)
    inside = np.zeros((256, 256), bool)
    inside[:243, c0:c0 + 2 * G + 1] = True
    assert (bank[~inside] == 0x5A).all()


@pytest.mark.parametrize("n_in,n_out", [(1, 1), (7, 3), (13, 7), (191, 5), (634, 4)])
def test_edge_cases_flash_code(bench, n_in, n_out):
    w, x = random_case(n_in, n_out, n_in + 5)
    check(bench, w, x, code="flash")
    xe = np.full(n_in, -127, np.int8)
    check(bench, np.ones((n_out, n_in), np.int8), xe, code="flash")
