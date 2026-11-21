"""T3: build the tracker with the CE toolchain (~/CEdev) and run it on the emulated eZ80.

    uv run python -m tracker.ce_bench                 # bit-exactness + cycles report
    uv run python -m tracker.ce_bench --exact 666 --steps 20

tracker.c is compiled with the CE program's own flags (as ce/calclm/tools/build_core.sh does,
-Oz by default) and linked with tracker/ce_drv.c into a flat binary at ORG in RAM, where TI-OS
runs a program. The CPU is pointed at the driver through the existing debugger-style harness
(ce/tools/bench.py, used unmodified); cycles are CEmu's exact counts at 48 MHz.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import bind, difftest, feats as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "build" / "ce"
CEDEV = Path(os.environ.get("CEDEV", Path.home() / "CEdev"))
LINK_LD = ROOT / "ce" / "kernel" / "link.ld"

ORG = 0xD1B000          # code (the OS stack grows down from 0xD1A87E; Bench's stub is 0xD1A900)
STATE = 0xD1F000        # trk_state
FEAT1 = 0xD1F400        # one trk_feat
TOKS = 0xD30000         # token ids (uint16), up to 32K
FEATS = 0xD40000        # trk_feat per token (VRAM), up to 8,000
CHUNK = 8000
FEAT_BYTES = F.NCAT + 9
V1_CYCLES_PER_TOKEN = 44_200_000   # RESULTS.md, CALCLM v1 at position 0 (lowest)

CFLAGS = ["-nostdinc", "-isystem", str(CEDEV / "include"), "-fno-autolink", "-fno-addrsig",
          "-fno-threadsafe-statics", "-mllvm", "-profile-guided-section-prefix=false",
          "-mllvm", "-z80-gas-style", "-ffunction-sections", "-fdata-sections",
          "-fno-math-errno", "-D__TICE__=1", "-DNDEBUG=1", "-fno-builtin", "-Wall", "-Wextra",
          "-Werror"]


def ce_available() -> bool:
    return (CEDEV / "bin" / "ez80-clang").exists()


def _run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stdout}{res.stderr}")
    return res.stdout + res.stderr


def compile_obj(src: Path, opt: str = "-Oz") -> Path:
    """ez80-clang -S + z80-none-elf-as, like build_core.sh. Returns the object path."""
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{src.stem}{opt}"
    asm, obj = OUT / f"{stem}.s", OUT / f"{stem}.o"
    warn = _run([str(CEDEV / "bin" / "ez80-clang"), "-S", *CFLAGS, opt, f"-I{HERE}", str(src),
                 "-o", str(asm)])
    if warn.strip():
        raise RuntimeError(f"ez80-clang output for {src.name}:\n{warn}")
    _run([str(CEDEV / "binutils" / "bin" / "z80-none-elf-as"), "-march=ez80+full", str(asm),
          "-o", str(obj)])
    return obj


def object_size(obj: Path) -> dict[str, int]:
    """Bytes per section kind (code, rodata, data, bss) of a relocatable object."""
    out = _run([str(CEDEV / "binutils" / "bin" / "z80-none-elf-size"), "-A", str(obj)])
    sizes = {"text": 0, "rodata": 0, "data": 0, "bss": 0}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            for kind in sizes:
                if parts[0].startswith("." + kind):
                    sizes[kind] += int(parts[1])
    return sizes


def build(opt: str = "-Oz") -> dict:
    bu = CEDEV / "binutils" / "bin"
    trk = compile_obj(HERE / "tracker.c", opt)
    drv = compile_obj(HERE / "ce_drv.c", opt)
    elf, binf, sym = (OUT / f"bench{opt}.{e}" for e in ("elf", "bin", "sym"))
    _run([str(bu / "z80-none-elf-ld"), "-static", "--no-warn-rwx-segments", "--gc-sections",
          "-e", "_trk_drv_run", "-u", "_trk_drv_step", "-u", "_trk_drv_reset",
          "-u", "_trk_drv_run_last", "-u", "_trk_drv_sizes", "-T", str(LINK_LD),
          "--defsym", f"ORG={ORG}", str(drv), str(trk), "--start-group",
          str(CEDEV / "lib" / "crt" / "libcrt.a"), "--end-group", "-o", str(elf)])
    _run([str(bu / "z80-none-elf-objcopy"), "-O", "binary", str(elf), str(binf)])
    sym.write_text(_run([str(bu / "z80-none-elf-nm"), str(elf)]))
    undef = _run([str(bu / "z80-none-elf-nm"), "-u", str(trk)]).split()
    syms = {}
    for line in sym.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            syms[parts[2].lstrip("_")] = int(parts[0], 16)
    return {"bin": binf.read_bytes(), "syms": syms, "tracker_o": object_size(trk),
            "undefined": [u for u in undef if u != "U"]}


class CETracker:
    """The tracker running on the emulated eZ80 (one Bench per process)."""

    def __init__(self, opt: str = "-Oz", bench=None):
        from ce.tools.bench import Bench
        self.info = build(opt)
        self.bench = bench or Bench()
        self.bench.reset()
        self.bench.write_ram(ORG, self.info["bin"])
        self.sym = self.info["syms"]

    def _call(self, name: str, args: list[int], max_cycles: int = 2_000_000_000):
        return self.bench.call(self.sym[name], args, flush_cache=False, max_cycles=max_cycles)

    def sizes(self) -> tuple[int, int]:
        r = self._call("trk_drv_sizes", []).ret
        return r & 0xFFF, r >> 12

    def reset(self) -> None:
        self._call("trk_drv_reset", [STATE])

    def step(self, tok: int) -> tuple[int, bytes]:
        r = self._call("trk_drv_step", [STATE, tok, FEAT1])
        return r.cycles, self.bench.read_ram(FEAT1, FEAT_BYTES)

    def run(self, toks: np.ndarray, store: bool = True) -> tuple[int, bytes]:
        """Step a stream (state carries over between chunks). Returns (cycles, features)."""
        cycles, raw = 0, bytearray()
        toks = np.asarray(toks, dtype="<u2")
        for a in range(0, len(toks), CHUNK):
            part = toks[a:a + CHUNK]
            self.bench.write_ram(TOKS, part.tobytes())
            fn = "trk_drv_run" if store else "trk_drv_run_last"
            r = self._call(fn, [STATE, TOKS, len(part), FEATS if store else FEAT1])
            cycles += r.cycles
            if store:
                raw += self.bench.read_ram(FEATS, FEAT_BYTES * len(part))
        return cycles, bytes(raw)


def decode(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """trk_feat records -> (cats, bags) like bind.batch."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, FEAT_BYTES)
    cats = a[:, :F.NCAT].copy()
    hint = np.zeros(len(a), dtype=np.uint64)
    for i in range(5):
        hint |= a[:, F.NCAT + i].astype(np.uint64) << np.uint64(8 * i)
    hint &= np.uint64((1 << 38) - 1)
    var = np.zeros(len(a), dtype=np.uint64)
    for i in range(4):
        var |= a[:, F.NCAT + 5 + i].astype(np.uint64) << np.uint64(8 * i)
    return cats, hint | (var << np.uint64(38))


def check_exact(ce: CETracker, stream: np.ndarray) -> str | None:
    ce.reset()
    _, raw = ce.run(stream)
    c_cats, c_bags = decode(raw)
    h_cats, h_bags = bind.batch(stream)
    diff = np.flatnonzero((c_cats != h_cats).any(axis=1) | (c_bags != h_bags))
    if len(diff):
        i = int(diff[0])
        return f"{len(diff)} tokens differ, first {i}: eZ80 {c_cats[i]} {c_bags[i]:#x} " \
               f"host {h_cats[i]} {h_bags[i]:#x}"
    return None


def stress_stream() -> np.ndarray:
    """A program built to hit the slowest paths: 16 list names sharing a prefix (hint over
    every entry, full bucket recomputation on DelVar), 37 unresolved labels with one first
    character (range checks when each resolves), label hints, 40 nested blocks, a long Menu(,
    then <eos> and a <bos> reset."""
    from tivars import TIProgram

    from eval.tokens import split_tokens
    names = F.NAME_CHARS
    lines = [f"{{1}}→ʟABCD{c}" for c in names[:16]]
    lines += ["Disp ʟABCD", "DelVar ʟABCDA", "DelVar ʟABCDP"]
    lines += [f"Goto A{c}" for c in names] + ["Lbl ", "Lbl A"]
    lines += [f"Lbl A{c}" for c in names] + ["Goto ", "Goto A"]
    lines += ["While 1"] * 40 + ["End"] * 40
    lines += ['Menu("T"' + "".join(f',"{c}",{c}{c}' for c in names[:7]) + ")"]
    lines += ['"' + "ABC:Lbl Goto " * 10 + '→Str1', "Disp Str1+Str2"]
    prog = TIProgram(name="X")
    prog.load_string("\n".join(lines))
    index = {h: i for i, h in enumerate(spec_vocab())}
    body = [index.get(t.hex().upper(), 3) for t in split_tokens(prog.data)]
    return np.array([1] + body + [2, 1, 11], dtype=np.uint16)


def spec_vocab() -> list[str]:
    return json.loads((ROOT / "corpus" / "vocab.json").read_text())


def step_cycles(ce: CETracker, stream: np.ndarray) -> np.ndarray:
    ce.reset()
    return np.array([ce.step(int(t))[0] for t in stream], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exact", type=int, default=200, help="val programs checked bit-exact")
    p.add_argument("--steps", type=int, default=12, help="val programs timed per step")
    p.add_argument("--opt", default="-Oz")
    p.add_argument("--json", help="also write the numbers here")
    a = p.parse_args(argv)
    ce = CETracker(a.opt)
    st, ft = ce.sizes()
    sz = ce.info["tracker_o"]
    print(f"CE build {a.opt}: tracker.o code {sz['text']} B, rodata {sz['rodata']} B, "
          f"data {sz['data']} B, bss {sz['bss']} B; undefined symbols "
          f"{ce.info['undefined'] or 'none'}; sizeof(trk_state) {st} B, trk_feat {ft} B")
    val = np.fromfile(ROOT / "data" / "tok" / "val.bin", dtype="<u2")
    progs = difftest.split_programs(val)
    exact = np.concatenate(progs[:a.exact])
    err = check_exact(ce, exact)
    print(f"bit-exact vs host on {min(a.exact, len(progs))} val programs "
          f"({len(exact):,} tokens): {'OK' if err is None else 'MISMATCH ' + err}")
    ce.reset()
    cyc, _ = ce.run(exact, store=False)
    mean = cyc / len(exact)
    print(f"stream mean: {mean:,.0f} cycles/token = {100 * mean / V1_CYCLES_PER_TOKEN:.4f}% of "
          f"v1's {V1_CYCLES_PER_TOKEN / 1e6:.1f} M cycles/token")
    timed = np.concatenate(progs[:a.steps] + [stress_stream()])
    err = err or check_exact(ce, timed)
    per = step_cycles(ce, timed)
    cats, _ = bind.batch(timed)
    print(f"per-step calls on {a.steps} programs + a stress program ({len(timed):,} tokens): "
          f"mean {per.mean():,.0f}, "
          f"median {np.median(per):,.0f}, p99 {np.percentile(per, 99):,.0f}, max {per.max():,} "
          f"cycles (max = {100 * per.max() / V1_CYCLES_PER_TOKEN:.4f}% of v1)")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "opt": a.opt, "state_bytes": st, "feat_bytes": ft, "object": sz,
            "undefined": ce.info["undefined"], "exact_tokens": int(len(exact)),
            "exact_error": err, "stream_mean": mean, "step_mean": float(per.mean()),
            "step_p99": float(np.percentile(per, 99)), "step_max": int(per.max()),
            "v1_cycles_per_token": V1_CYCLES_PER_TOKEN}, indent=2) + "\n")
    names = ["START", "EXPR", "STRING", "STORE", "LBL", "REF", "LIST", "PRGM", "DELVAR"]
    for ctx, name in enumerate(names):
        sel = per[cats[:, 0] == ctx]
        if len(sel):
            print(f"  after-token ctx {name:7s} n={len(sel):6d} mean {sel.mean():7,.0f} "
                  f"max {sel.max():7,}")
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
