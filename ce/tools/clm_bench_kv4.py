"""E20 spike (PLAN-E20.md): emulator stages and measurements for the 4-bit K/V builds.

    python -m ce.tools.clm_bench_kv4 stage "E20 M0" -- python -m ce.tools.clm_bench ...
    python -m ce.tools.clm_bench_kv4 tests [--jobs 4]            # emu pytest shards
    python -m ce.tools.clm_bench_kv4 sequences --out runs/e20/m1.json      # M1
    python -m ce.tools.clm_bench_kv4 programs --seeds 1-8,20 --out runs/e20/m2.json
    python -m ce.tools.clm_bench_kv4 profile --out runs/e20/m3.json        # M3
    python -m ce.tools.clm_bench_kv4 memory --out runs/e20/m4.json         # M4 (no emulator)

Scheduling (PLAN-E20 Resources, R4 and revision 2.1 D): every emulator stage first checks the
cross-experiment lock without blocking (as lab.e15.require_emulator_free) and stops if it is
held, then takes lab.emulator_lock.emulator_stage at once with a log callback that raises on
any "waiting for the emulator" line, so E20 never queues behind another experiment. Emulator
commands run as child processes inside the stage with the lock's descriptor (pass_fds). The
emulator is driven only through ce/tools/bench.py's Bench (via EmuCore), as the ce tests do.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from lab import emulator_lock as EL

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "runs" / "e20" / "stages.log"


class EmulatorBusy(RuntimeError):
    pass


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def require_emulator_free(who: str) -> None:
    """Stop (raise) if another process holds the emulator lock; never wait."""
    if EL.held():
        return
    EL.LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(EL.LOCK, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            holder = f.read().strip() or "another process"
            _log(f"{who}: the emulator lock is held by {holder}; stopping")
            raise EmulatorBusy(f"{who}: the emulator lock is held by {holder} (a benchmark "
                               "stops rather than waits)") from None
        fcntl.flock(f, fcntl.LOCK_UN)


def _never_wait(line: str) -> None:
    if line.startswith("waiting for the emulator"):
        _log(f"stopping instead of waiting: {line}")
        raise EmulatorBusy(line)
    _log(line)


@contextlib.contextmanager
def e20_stage(who: str):
    require_emulator_free(who)
    with EL.emulator_stage(who, log=_never_wait):
        _log(f"{who}: emulator lock held")
        t0 = time.time()
        try:
            yield
        finally:
            _log(f"{who}: released after {time.time() - t0:.0f} s")


def build_all() -> None:
    """Boot (once, cached) and build every core once, before any parallel CEmu process."""
    from ce.tools.bench import Bench
    from ce.tools.clm_kv4 import core_paths
    Bench()
    for tag in ("", "w", "wk4", "w8k4"):
        core_paths(tag)


def run_children(cmds: list[list[str]], logs: list[Path] | None = None,
                 jobs: int = 4) -> list[int]:
    """Run the commands inside the held stage (the lock's descriptor passed), at most `jobs` at
    a time (PLAN-E20: up to 4 CEmu processes), reading the prebuilt cores (build_all first);
    returns their exit codes in order."""
    from ce.tools.clm_kv4 import PREBUILT_ENV
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **{PREBUILT_ENV: "1"})
    codes: list = [None] * len(cmds)
    running: dict = {}
    nxt = 0
    try:
        while nxt < len(cmds) or running:
            while nxt < len(cmds) and len(running) < jobs:
                out = open(logs[nxt], "w") if logs else None
                running[nxt] = (subprocess.Popen(cmds[nxt], cwd=ROOT, env=env,
                                                 pass_fds=EL.held_fds(), stdout=out,
                                                 stderr=subprocess.STDOUT if out else None), out)
                nxt += 1
            for i, (p, out) in list(running.items()):
                if p.poll() is not None:
                    codes[i] = p.returncode
                    if out:
                        out.close()
                    del running[i]
            time.sleep(0.5)
    finally:                     # never leave children driving CEmu with the lock's descriptor
        for p, out in running.values():
            p.terminate()
        for p, out in running.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            if out:
                out.close()
    return [int(c) for c in codes]


# ---- emulator test shards ----------------------------------------------------------------------

PY = sys.executable
# shard: (test files, the `core` backend). Tests are selected by node id from a collection, by
# their `core` parameter (the first in every id), not with -k: -k also matches marker names, and
# "ram" is inside "parametrize".
SHARDS = {
    "w8k4-emu": (["ce/tests/test_calclm_w8k4.py"], "emu"),
    "w8k4-emu-hz": (["ce/tests/test_calclm_w8k4_hz.py"], "emu"),
    "w8k4-emuflash": (["ce/tests/test_calclm_w8k4.py"], "emuflash"),
    "w8k4-emuflash-hz": (["ce/tests/test_calclm_w8k4_hz.py"], "emuflash"),
    "w8k4-emuflash-ram": (["ce/tests/test_calclm_w8k4.py"], "emuflash-ram"),
    "wk4-emu": (["ce/tests/test_calclm_wk4.py", "ce/tests/test_calclm_wk4_hz.py"], "emu"),
    "wk4-emuflash": (["ce/tests/test_calclm_wk4.py", "ce/tests/test_calclm_wk4_hz.py"],
                     "emuflash"),
}
BACKEND_NAMES = ("host", "host-ram", "emu", "emuflash", "emuflash-ram")


def backend_of(node_id: str) -> str | None:
    """The `core` parameter of a collected test id (the longest backend name its parameters start
    with), or None for a test without one."""
    if "[" not in node_id:
        return None
    params = node_id[node_id.index("[") + 1:-1]
    hits = [b for b in BACKEND_NAMES if params == b or params.startswith(b + "-")]
    return max(hits, key=len) if hits else None


def select(files: list[str], backend: str) -> list[str]:
    res = subprocess.run([PY, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *files], cwd=ROOT, capture_output=True, text=True, check=True)
    ids = [ln.strip() for ln in res.stdout.splitlines() if "::" in ln]
    return [i for i in ids if backend_of(i) == backend]


def cmd_stage(a) -> int:
    with e20_stage(a.who):
        build_all()
        return max(run_children([a.cmd]))


def cmd_tests(a) -> int:
    names = a.shards or list(SHARDS)
    out = ROOT / "runs" / "e20" / "tests"
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    with e20_stage("E20 emulator tests " + ",".join(names)):
        build_all()
        chosen = {n: select(*SHARDS[n]) for n in names}
        for n, ids in chosen.items():
            _log(f"shard {n}: {len(ids)} tests selected")
            (out / f"{n}.ids").write_text("\n".join(ids) + "\n")
        cmds = [[PY, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *chosen[n],
                 *(["-m", "not slow"] if a.fast else []),
                 f"--junitxml={out / (n + '.xml')}"] for n in names]
        codes = run_children(cmds, [out / f"{n}.log" for n in names], a.jobs)
        for i, n in enumerate(names):    # every selected test reported (passed or skipped)
            got = sum(int(s.get("tests", 0)) for s in
                      ET.parse(out / f"{n}.xml").getroot().iter("testsuite"))
            if got != len(chosen[n]) and not a.fast:
                _log(f"shard {n}: {got} results for {len(chosen[n])} selected tests")
                codes[i] = codes[i] or 1
        for n, c in zip(names, codes):
            _log(f"shard {n}: exit {c}")
            rc = max(rc, c)
    return rc


# ---- measurements: jobs in parallel children (ce/tools/e20_measure.py) -------------------------

def m1_jobs() -> list[tuple]:
    jobs = []
    for cfg in ("L4i8", "L4k4", "W8k4A"):
        jobs += [(cfg, hz, seq) for hz in (True, False) for seq in ("A", "B")]
    for cfg in ("v2", "rc2", "W8k4B"):
        jobs += [(cfg, True, seq) for seq in ("A", "B")]
    return jobs + [("W8s11k4A", True, "A")]


def run_jobs(kind: str, jobs: list[tuple], out: Path, n: int) -> list[dict]:
    parts = ROOT / "runs" / "e20" / "parts" / kind
    parts.mkdir(parents=True, exist_ok=True)
    names = ["_".join(str(x) for x in j) for j in jobs]
    cmds = [[PY, "-m", "ce.tools.e20_measure", kind, str(parts / f"{nm}.json"), json.dumps(j)]
            for nm, j in zip(names, jobs)]
    with e20_stage(f"E20 {kind} ({len(jobs)} jobs, {n} processes)"):
        build_all()
        codes = run_children(cmds, [parts / f"{nm}.log" for nm in names], n)
    bad = [nm for nm, c in zip(names, codes) if c]
    if bad:
        raise SystemExit(f"{kind}: jobs failed: {bad} (logs in {parts})")
    recs = [json.loads((parts / f"{nm}.json").read_text()) for nm in names]
    out.write_text(json.dumps({"kind": kind, "lcd_dma": True, "flash": True,
                               "flush_cache": False, "jobs": recs}, indent=1) + "\n")
    _log(f"{kind}: wrote {out}")
    return recs


def cmd_measure(a) -> int:
    if a.what == "sequences":
        run_jobs("m1", m1_jobs(), Path(a.out), a.jobs)
    elif a.what == "programs":
        seeds = []
        for part in a.seeds.split(","):
            lo, _, hi = part.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
        jobs = [(cfg, s) for cfg in ("rc2", "W8k4A", "W8k4B") for s in seeds]
        run_jobs("m2", jobs, Path(a.out), a.jobs)
    elif a.what == "profile":
        jobs = [(c,) for c in ("L4i8", "L4k4", "W8k4A", "W8k4B")]
        run_jobs("m3", jobs, Path(a.out), a.jobs)
        ld2 = [(c, on) for c in ("v2", "rc2", "W8k4A", "W8k4B") for on in (True, False)]
        run_jobs("m3ld2", ld2, Path(a.out).with_name("m3_ld2.json"), a.jobs)
    return 0


# ---- M4: memory and storage (no emulator) ------------------------------------------------------

BSS_LIMIT = 0xD13FD8 - 0xD052C6        # 60,690 B
CLMTMP_LIMIT = 65505
LOAD_ADDR = 0xD1A87F


def _map(path: Path) -> dict:
    text = path.read_text()
    bss = int(text.split("___bss_len = (___bss_high - ___bss_low)")[0].split()[-1], 16)
    data_high = int(text.split("___data_high = .")[0].split()[-1], 16)
    return {"bss": bss, "image": data_high - LOAD_ADDR}


def _stage(export: Path, out: Path) -> dict:
    from ce.tools.stage_export import stage
    data = stage(export, out)
    files = {p.name: p.stat().st_size for p in sorted(out.glob("*.8xv"))}
    return {"data_bytes": data, "file_bytes": files}


def cmd_memory(a) -> int:
    from ce.calclm.tools.build_flash import appvar_8xv
    from ce.calclm.tools.make_kv4 import payload
    from ce.calclm.tools.make_ld2 import payload as ld2_payload
    from ce.tools.clm import CALCLM, Export, act_bytes
    from ce.tools.clm_kv4 import vc_bytes
    env = dict(os.environ, PATH=f"{Path.home() / 'CEdev' / 'bin'}:{os.environ['PATH']}")
    for args in (["SHAPE=w8k4"], ["SHAPE=w"]):
        subprocess.run(["make", *args], cwd=CALCLM, env=env, check=True, capture_output=True)
    build = ROOT / "ce" / "build" / "e20"
    w8, rc2 = ROOT / "runs/e16/export/W8-s10", ROOT / "runs/e10/export/L4-s23"
    st_w8, st_rc2 = _stage(w8, build / "W8-s10"), _stage(rc2, build / "rc2")
    kv4 = payload(w8, ROOT / "runs/e19/kv4/W8-s10.json")
    (build / "W8-s10" / "CLMKV4.8xv").write_bytes(appvar_8xv("CLMKV4", kv4))
    ld2 = ld2_payload(rc2, ROOT / "ce/tests/fixtures/ld2_offsets_L4-s23.json")
    (build / "rc2" / "CLMLD2.8xv").write_bytes(appvar_8xv("CLMLD2", ld2))
    prog = {t: (CALCLM / "bin" / t / "CALCLM.8xp").stat().st_size for t in ("w8k4", "w")}
    code = {t: (CALCLM / "bin" / t / "CLMCODE.8xv").stat().st_size for t in ("w8k4", "w")}
    maps = {t: _map(CALCLM / "bin" / t / "CALCLM.map") for t in ("w8k4", "w")}
    files_w8 = {**{n: s for n, s in st_w8["file_bytes"].items()},
                "CLMKV4.8xv": (build / "W8-s10" / "CLMKV4.8xv").stat().st_size,
                "CLMCODE.8xv": code["w8k4"], "CALCLM.8xp": prog["w8k4"]}
    files_rc2 = {**{n: s for n, s in st_rc2["file_bytes"].items()},
                 "CLMLD2.8xv": (build / "rc2" / "CLMLD2.8xv").stat().st_size,
                 "CLMCODE.8xv": code["w"], "CALCLM.8xp": prog["w"]}
    ld2_ws = 2652
    tmp_b, tmp_a = vc_bytes(8, 256, True) + ld2_ws, vc_bytes(8, 256, False) + ld2_ws
    ex = Export.load(w8)
    rec = {
        "bss": {"w8k4": maps["w8k4"]["bss"], "w_rc2": maps["w"]["bss"], "limit": BSS_LIMIT,
                "k_cache": 8 * 256 * 19},
        "clmtmp": {"placement_B": tmp_b, "placement_A": tmp_a, "limit": CLMTMP_LIMIT,
                   "parts_B": {"packed_v": 8 * 256 * 16, "scratch": 256 * 67,
                               "ram_tables_with_slack": 8192 + 512, "ld2_workspace": ld2_ws}},
        "program_image": {"w8k4": maps["w8k4"]["image"], "w_rc2": maps["w"]["image"]},
        "user_ram_needed": {"placement_B": maps["w8k4"]["image"] + tmp_b + 4096,
                            "placement_A": maps["w8k4"]["image"] + tmp_a + 4096,
                            "w_rc2": maps["w"]["image"] + 49152 + ld2_ws + 4096,
                            "nominal": 150000, "device_free_2026_10_15": 150888,
                            "note": "plus the C libraries, as README's W row"},
        "vram_act_bytes": {"w8k4": act_bytes(ex, 3),
                           "w_rc2": act_bytes(Export.load(rc2), 3), "limit": 0xD52A00 - 0xD4F300},
        "appvars": {"model_w8": len(ex.appvars), "cap_spike_header": 24, "cap_today": 16,
                    "clmkv4_payload": len(kv4)},
        "archive_files": {"w8k4": files_w8, "w8k4_total": sum(files_w8.values()),
                          "rc2": files_rc2, "rc2_total": sum(files_rc2.values()),
                          "rc2_readme_total": 1046973, "model_file_limit": 1200000,
                          "w8_model_data_bytes": sum(v for k, v in st_w8["data_bytes"].items()
                                                     if k.startswith("CLM") and k != "CLMVOC.8xv")},
    }
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("stage", help="run one command inside an E20 emulator stage")
    s.add_argument("who")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    t = sub.add_parser("tests", help="the emulator test shards, in parallel inside one stage")
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--fast", action="store_true", help="deselect the slow tests")
    t.add_argument("shards", nargs="*", help=f"some of {', '.join(SHARDS)} (default: all)")
    for name in ("sequences", "programs", "profile"):
        m = sub.add_parser(name)
        m.add_argument("--out", required=True)
        m.add_argument("--jobs", type=int, default=4)
        if name == "programs":
            m.add_argument("--seeds", default="1-8,20")
    mem = sub.add_parser("memory", help="M4: BSS, CLMTMP, user RAM, AppVars, archive")
    mem.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.what in ("sequences", "programs", "profile"):
        return cmd_measure(a)
    if a.what == "memory":
        return cmd_memory(a)
    if a.what == "tests" and set(a.shards) - set(SHARDS):
        ap.error(f"unknown shards {sorted(set(a.shards) - set(SHARDS))}")
    if a.what == "stage":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        return cmd_stage(a)
    return cmd_tests(a)


if __name__ == "__main__":
    raise SystemExit(main())
