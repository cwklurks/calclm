"""The CalcLM eZ80 assembly hot loops (ce/calclm/core/ez80), called directly in CEmu like a
debugger would, against Python models built from model/intops.py primitives. Random inputs
cover every shift amount the routines accept, plus the clamping edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from ce.tools.clm import EmuCore, Export, emu_available
from model.intops import rsr

pytestmark = pytest.mark.skipif(not emu_available(), reason="needs the CE ROM and ~/CEdev")

IN_ADDR = 0xD30000      # inputs (V-cache region; nothing else runs during these calls)
OUT_ADDR = 0xD38000
AUX_ADDR = 0xD3C000


@pytest.fixture(scope="module", params=["ram", "flash"])
def emu(request):
    """The routines as linked in RAM, or (flash) the CLMCODE copy executing from emulated
    flash: the RAM entry points jump there once it is attached."""
    core = EmuCore(Export.load())
    core.bench.reset()
    core.bench.write_ram(core.org, core.bin)
    if request.param == "flash":
        assert core.attach() == 0
    return core


def call(emu, name, args, max_cycles=200_000_000):
    res = emu.bench.call(emu.syms[name], [a & 0xFFFFFF for a in args], flush_cache=False,
                         max_cycles=max_cycles)
    return res


def clamp(a, lo, hi):
    return max(lo, min(hi, a))


# ---- quant ---------------------------------------------------------------------------------

def norm1(v: int, c: int) -> int:
    return clamp(rsr(v, c), -32767, 32767) if c > 0 else v << -c


def ref_quant(v, c, gm, S):
    return [clamp(rsr(norm1(int(x), c) * gm, S), -127, 127) for x in v]


def quant_case(rng, c, S, n):
    """Random vector with |v| < 2^(c+15) whose quantized values stay within +-128."""
    gm = int(rng.integers(1 << 14, 1 << 15))
    vmax_p = min(32767, (128 << S) // gm)            # largest allowed |v'|
    lim = min((1 << (c + 15)) - 1, (vmax_p << c) if c > 0 else vmax_p >> -c)
    v = rng.integers(-lim, lim + 1, size=n, dtype=np.int64)
    v[:4] = [lim, -lim, 0, -min(1, lim)]
    if c > 0:   # values that round to exactly +-2^15 before the int16 clamp
        e = (1 << (c + 15)) - (1 << (c - 1))
        if e <= lim:
            v[4:6] = [e, -e - (1 << (c - 1))]
    assert all(abs(rsr(norm1(int(x), c) * gm, S)) <= 128 for x in v)
    return v, gm


@pytest.mark.parametrize("c", list(range(-15, 17)))
def test_quant_matches_reference(emu, c):
    rng = np.random.default_rng(1000 + c)
    for S in (17, 20, 22, 23, 24):
        n = int(rng.integers(1, 300))
        v, gm = quant_case(rng, c, S, max(n, 8))
        emu.bench.write_ram(IN_ADDR, v.astype("<i4").tobytes())
        emu.bench.write_ram(OUT_ADDR, b"\x55" * (len(v) + 4))
        call(emu, "clm_a_quant", [IN_ADDR, len(v), c, gm, S, OUT_ADDR])
        got = np.frombuffer(emu.bench.read_ram(OUT_ADDR, len(v)), np.int8).tolist()
        assert got == ref_quant(v, c, gm, S), (c, S, gm)
        assert emu.bench.read_ram(OUT_ADDR + len(v), 4) == b"\x55" * 4   # no overrun


def test_quant_restores_interrupts(emu):
    """Called with interrupts on, the routine disables them (shadow registers) and restores."""
    from ce.tools.bench import encode_stub
    v = np.arange(-50, 50, dtype="<i4") * 1000
    emu.bench.write_ram(IN_ADDR, v.tobytes())
    stub = bytearray([0xFB]) + encode_stub(emu.syms["clm_a_quant"],
                                           [IN_ADDR, 100, 2, 20000, 22, OUT_ADDR],
                                           emu.bench.ret_addr)[1:-5]
    # after the call: ld a,i; di; push af; pop hl; ld (ret),hl; halt
    stub += bytes([0xED, 0x57, 0xF3, 0xF5, 0xE1, 0x22]) + emu.bench.ret_addr.to_bytes(3, "little")
    stub += b"\x76"
    emu.bench.run_code(bytes(stub))
    assert emu.bench.read_ram(emu.bench.ret_addr, 1)[0] & 0x04
    got = np.frombuffer(emu.bench.read_ram(OUT_ADDR, 100), np.int8).tolist()
    assert got == ref_quant(v, 2, 20000, 22)


# ---- scale_out -----------------------------------------------------------------------------

I24 = (-(1 << 23), (1 << 23) - 1)
MODES = {0: ((-127, 127), 1), 1: ((-32768, 32767), 2), 2: (I24, 3), 3: (I24, 3)}


def ref_scale(acc, bm, be, sxm, sb, mode, prev=None):
    (lo, hi), _ = MODES[mode]
    out = []
    for i, a in enumerate(acc):
        r = clamp(rsr(int(a) * int(bm[i]) * sxm, sb - int(be[i])), lo, hi)
        if mode == 3:
            r = clamp(int(prev[i]) + r, *I24)
        out.append(r)
    return out


def write_int(emu, addr, vals, size):
    emu.bench.write_ram(addr, b"".join((int(v) & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
                                       for v in vals))


def read_int(emu, addr, n, size):
    raw = emu.bench.read_ram(addr, n * size)
    out = []
    for i in range(n):
        v = int.from_bytes(raw[i * size:(i + 1) * size], "little")
        out.append(v - (1 << (8 * size)) if v >> (8 * size - 1) else v)
    return out


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 7, 256, 257, 600, 1024])
def test_scale_matches_reference(emu, mode, n):
    rng = np.random.default_rng(n * 10 + mode)
    for trial in range(3):
        sb = int(rng.integers(-10, 40))
        smin, smax = [(1, 24), (1, 8), (16, 24)][trial]
        be = sb - rng.integers(smin, smax + 1, size=n)
        if (be < -128).any() or (be > 127).any():
            be = np.clip(be, -128, 127)
            be = np.where((sb - be >= 1) & (sb - be <= 24), be, sb - 12)
        bm = rng.integers(64, 128, size=n)
        sxm = int(rng.integers(128, 256))
        acc = rng.integers(-65024, 65025, size=n)
        acc[: min(n, 6)] = [65024, -65024, 0, -1, 1, -32768][: min(n, 6)]
        prev = rng.integers(*I24, size=n, endpoint=True)
        prev[: min(n, 2)] = [I24[1], I24[0]][: min(n, 2)]
        (_, size) = MODES[mode]
        write_int(emu, IN_ADDR, acc, 3)
        emu.bench.write_ram(AUX_ADDR, bytes(int(x) & 255 for x in bm) +
                            bytes(int(x) & 255 for x in be))
        write_int(emu, OUT_ADDR, prev if mode == 3 else [0x5A5A5A] * n, size)
        emu.bench.write_ram(OUT_ADDR + n * size, b"\xa5" * 4)
        be_at = AUX_ADDR + n if trial == 0 else AUX_ADDR + 1100   # .e need not follow .m
        emu.bench.write_ram(be_at, bytes(int(x) & 255 for x in be))
        call(emu, "clm_a_scale", [IN_ADDR, AUX_ADDR, n, sxm, sb, OUT_ADDR, mode, 0xD3E000, be_at])
        got = read_int(emu, OUT_ADDR, n, size)
        want = ref_scale(acc, bm, be, sxm, sb, mode, prev)
        bad = [i for i in range(n) if got[i] != want[i]]
        assert not bad, (trial, bad[:5], [(int(acc[i]), int(sb - be[i]), got[i], want[i])
                                          for i in bad[:5]])
        assert emu.bench.read_ram(OUT_ADDR + n * size, 4) == b"\xa5" * 4


# ---- RMSNorm / SwiGLU product passes ----------------------------------------------------

def norm1_24(x: int, c: int) -> int:
    return norm1(x, c)


@pytest.mark.parametrize("c", list(range(-15, 9)))
@pytest.mark.parametrize("want_ss", [0, 1])
def test_mulnorm_matches_reference(emu, c, want_ss):
    rng = np.random.default_rng(77 + c * 3 + want_ss)
    n = int(rng.integers(1, 600))
    lim = (1 << (c + 15)) - 1
    x = rng.integers(-lim, lim + 1, size=n)
    x[: min(n, 4)] = [lim, -lim, 0, -min(1, lim)][: min(n, 4)]
    if c > 0 and n > 6:
        e = (1 << (c + 15)) - (1 << (c - 1))
        if e <= lim:
            x[4:6] = [e, -e - (1 << (c - 1))]
    w = rng.integers(-32768, 32768, size=n)
    w[: min(n, 3)] = [-32768, 32767, 0][: min(n, 3)]
    write_int(emu, IN_ADDR, x, 3)
    write_int(emu, AUX_ADDR, w, 2)
    emu.bench.write_ram(OUT_ADDR + 4 * n, b"\xa5" * 4)
    tmp = 0xD3E000 if want_ss else 0      # with a copy buffer, x must come back unchanged
    r = call(emu, "clm_a_mulnorm", [IN_ADDR, n, c, AUX_ADDR, OUT_ADDR, want_ss, tmp])
    got = read_int(emu, OUT_ADDR, n, 4)
    if tmp:
        assert read_int(emu, IN_ADDR, n, 3) == [int(v) for v in x]
    xp = [norm1_24(int(v), c) for v in x]
    assert got == [a * int(b) for a, b in zip(xp, w)]
    ss = sum(rsr(a, 8) ** 2 for a in xp) if want_ss else 0
    assert r.ret == ss
    assert emu.bench.read_ram(OUT_ADDR + 4 * n, 4) == b"\xa5" * 4


@pytest.mark.parametrize("n", [1, 5, 512, 700])
def test_t3_matches_reference(emu, n):
    rng = np.random.default_rng(n)
    for trial in range(4):
        acc = rng.integers(-65024, 65025, size=n)
        acc[: min(n, 4)] = [65024, -65024, 0, -1][: min(n, 4)]
        bm = rng.integers(64, 128, size=n)
        E3 = int(rng.integers(-12, 0))
        be = E3 - rng.integers(0, 9 if trial else 4, size=n)
        write_int(emu, IN_ADDR, acc, 3)
        emu.bench.write_ram(AUX_ADDR, bytes(int(v) & 255 for v in bm) + bytes(int(v) & 255 for v in be))
        out = IN_ADDR if trial % 2 else OUT_ADDR   # the core runs it in place
        be_at = AUX_ADDR + n if trial < 2 else AUX_ADDR + 900
        emu.bench.write_ram(be_at, bytes(int(v) & 255 for v in be))
        r = call(emu, "clm_a_t3", [IN_ADDR, AUX_ADDR, n, E3, out, be_at])
        want = [rsr(int(a) * int(b), E3 - int(e)) for a, b, e in zip(acc, bm, be)]
        assert read_int(emu, out, n, 3) == want
        assert r.ret == max(abs(v) for v in want)


@pytest.mark.parametrize("n", [1, 2, 193, 512])
def test_maxabs32(emu, n):
    rng = np.random.default_rng(n + 5)
    for scale in (1 << 30, 1 << 20, 300, 1):
        v = rng.integers(-scale, scale, size=n)
        if n > 3:
            v[rng.integers(0, n)] = -scale + 1
        write_int(emu, IN_ADDR, v, 4)
        call(emu, "clm_a_maxabs32", [IN_ADDR, n, OUT_ADDR])
        assert read_int(emu, OUT_ADDR, 1, 4)[0] == max(abs(int(x)) for x in v)


# ---- attention ------------------------------------------------------------------------------

def k_rows(rng, n):
    """Biased K rows (32 x k+128, int24 -128 * sum) and the signed keys."""
    k = rng.integers(-127, 128, size=(n, 32))
    k[0] = 127
    if n > 1:
        k[1] = -127
    rows = bytearray()
    for r in k:
        ku = (r + 128).astype(int)
        nk = (-128 * int(ku.sum())) & 0xFFFFFF
        rows += bytes(ku.tolist()) + nk.to_bytes(3, "little")
    return k, bytes(rows)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 255, 256])
@pytest.mark.parametrize("bias,step", [(0, 0), (-(512 << 7), 0), (-(255 << 12), 1 << 12),
                                       (-(15 << 9), 1 << 9), (-(1 << 22), 0)])
def test_scores(emu, n, bias, step):
    """s_j = q . k_j + bias + j * step: plain scores (0, 0), a capped ALiBi run (constant
    penalty), and linear runs whose penalty falls by `step` per key (INTREF.md 12)."""
    rng = np.random.default_rng(n + 300)
    k, rows = k_rows(rng, n)
    q = rng.integers(-127, 128, size=32)
    q[:2] = [127, -127]
    qu = (q + 128).astype(int)
    ch = 32 * 128 * 128 - 128 * int(qu.sum())
    emu.bench.write_ram(AUX_ADDR, bytes(qu.tolist()))
    emu.bench.write_ram(IN_ADDR, rows)
    r = call(emu, "clm_a_scores", [AUX_ADDR, ch + bias, IN_ADDR, n, OUT_ADDR, step])
    want = (k @ q + bias + step * np.arange(n)).tolist()
    got = read_int(emu, OUT_ADDR, n, 3)[::-1]            # stored last key first
    assert got == want
    mx = r.ret - (1 << 24) if r.ret & 0x800000 else r.ret
    assert mx == max(want)


@pytest.mark.parametrize("sh", [1, 4, 5, 6, 7])
@pytest.mark.parametrize("n", [1, 3, 200])
def test_softmax(emu, sh, n):
    rng = np.random.default_rng(sh * 1000 + n)
    L = int(rng.integers(200, 513))
    lut = np.clip((255 * np.exp(-np.arange(L) / (L / 6))).round(), 0, 255).astype(int)
    lut[-1] = 0
    s = rng.integers(-(1 << 19), 1 << 19, size=n)
    s[0] = s.max() if n > 2 else s[0]
    mx = int(s.max())
    write_int(emu, IN_ADDR, s[::-1], 3)                  # the scores layout: last key first
    emu.bench.write_ram(AUX_ADDR, bytes(lut.tolist()))
    r = call(emu, "clm_a_softmax", [IN_ADDR, n, mx, sh, AUX_ADDR, L, OUT_ADDR])
    idx = [min(rsr(mx - int(v), sh), L - 1) for v in s]
    want = [int(lut[i]) for i in idx]
    assert list(emu.bench.read_ram(OUT_ADDR, n)) == want
    assert r.ret == sum(want)


@pytest.mark.parametrize("n", [1, 8, 9, 100, 256])
def test_pv(emu, n):
    rng = np.random.default_rng(n + 9)
    V = rng.integers(0, 256, size=(n + 8, 32))
    e = np.zeros(n + 8, dtype=int)
    e[:n] = rng.integers(0, 256, size=n)
    e[: min(n, 2)] = 255
    emu.bench.write_ram(IN_ADDR, bytes(V.flatten().tolist()))
    emu.bench.write_ram(AUX_ADDR, bytes(e.tolist()))
    call(emu, "clm_a_pv", [IN_ADDR, AUX_ADDR, n, OUT_ADDR])
    assert [v & 0xFFFFFF for v in read_int(emu, OUT_ADDR, 32, 3)] == (e[:n] @ V[:n]).tolist()


@pytest.mark.parametrize("z", [255, 256, 511, 1000, 4097, 65280])
def test_havg(emu, z):
    rng = np.random.default_rng(z)
    v = rng.integers(-127, 128, size=(min(z // 255 + 1, 256), 32))
    a_signed = rng.integers(-127 * z, 127 * z + 1, size=32)
    a_signed[:2] = [127 * z, -127 * z]
    au = a_signed + 128 * z
    zb = z.bit_length()
    rz = ((1 << (zb + 14)) + (z >> 1)) // z
    write_int(emu, IN_ADDR, au, 3)
    call(emu, "clm_a_havg", [IN_ADDR, z, zb - 8, rz, OUT_ADDR])
    want = [rsr(rsr(int(a), zb - 8) * rz, 14) for a in a_signed]
    assert read_int(emu, OUT_ADDR, 32, 4) == want
    del v


# ---- small loops ------------------------------------------------------------------------------

def test_silu(emu):
    ex = Export.load()
    table = np.frombuffer(ex.blob, "<i2", count=1025, offset=ex.meta["tensors"]["silu"]["offset"])
    y = np.concatenate([np.arange(-2600, 2600, 7), [-32768, 32767, -2049, -2048, 2047, 2048, 0, -1]])
    emu.bench.write_ram(AUX_ADDR, table.astype("<i2").tobytes())
    write_int(emu, IN_ADDR, y, 2)
    call(emu, "clm_a_silu", [IN_ADDR, len(y), AUX_ADDR])
    want = [0 if v < -2048 else v if v >= 2048 else int(table[rsr(int(v), 2) + 512]) for v in y]
    assert read_int(emu, IN_ADDR, len(y), 2) == want


@pytest.mark.parametrize("n", [1, 2, 192, 600])
def test_maxabs24(emu, n):
    rng = np.random.default_rng(n)
    v = rng.integers(-(1 << 23), 1 << 23, size=n)
    for case in (v, np.abs(v) % 300, -np.abs(v)):
        write_int(emu, IN_ADDR, case, 3)
        r = call(emu, "clm_a_maxabs24", [IN_ADDR, n])
        assert r.ret == max(abs(int(x)) for x in case)


@pytest.mark.parametrize("n,density,cap", [(1, 1.0, 96), (50, 0.3, 96), (256, 0.25, 96),
                                           (256, 0.5, 96), (8, 0.0, 96), (100, 1.0, 100)])
def test_compact(emu, n, density, cap):
    rng = np.random.default_rng(n + int(density * 100))
    V = rng.integers(0, 256, size=(n, 32))
    e = np.where(rng.random(n) < density, rng.integers(1, 256, size=n), 0)
    emu.bench.write_ram(IN_ADDR, bytes(V.flatten().tolist()))
    emu.bench.write_ram(AUX_ADDR, bytes(e.tolist()))
    r = call(emu, "clm_a_compact", [IN_ADDR, AUX_ADDR, n, OUT_ADDR, 0xD3E000, cap])
    nz = [j for j in range(n) if e[j]]
    if len(nz) > cap:
        assert r.ret == cap + 1
        nz = nz[:cap]
    else:
        assert r.ret == len(nz)
    assert emu.bench.read_ram(OUT_ADDR, 32 * len(nz)) == bytes(V[nz].flatten().tolist())
    assert list(emu.bench.read_ram(0xD3E000, len(nz))) == [int(e[j]) for j in nz]


@pytest.mark.parametrize("sh", [-8, -1, 0, 1, 3, 9, 15, 16, 23])
def test_emb(emu, sh):
    rng = np.random.default_rng(sh + 50)
    n = 192
    x = rng.integers(-127, 128, size=n)
    x[:4] = [127, -127, 0, -1]
    mm = int(rng.integers(128, 256))
    prev = rng.integers(-(1 << 23), 1 << 23, size=n)
    prev[:2] = [(1 << 23) - 1, -(1 << 23)]
    emu.bench.write_ram(AUX_ADDR, bytes(int(v) & 255 for v in x))
    t = [rsr(int(v) * mm, sh) for v in x]
    call(emu, "clm_a_emb", [AUX_ADDR, n, mm, sh, OUT_ADDR, 0])
    assert read_int(emu, OUT_ADDR, n, 3) == t
    write_int(emu, OUT_ADDR, prev, 3)
    call(emu, "clm_a_emb", [AUX_ADDR, n, mm, sh, OUT_ADDR, 1])
    assert read_int(emu, OUT_ADDR, n, 3) == [clamp(int(a) + b, *I24) for a, b in zip(prev, t)]
