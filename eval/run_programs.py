"""Run TI-BASIC programs on the emulated TI-84 Plus CE (CEmu, the author's OS 5.8 ROM) and report
how many run: OK, ERR:<NAME>, TIMEOUT, REJECTED. See eval/EVAL.md.

  uv run python -m eval.run_programs DIR_OF_8XP --name myrun
  uv run python -m eval.run_programs --ids gen.jsonl --vocab corpus/vocab.json --name gen
  uv run python -m eval.run_programs --ids data/tok/val.bin --vocab corpus/vocab.json --name val
  uv run python -m eval.run_programs --corpus data/extract/programs.jsonl --sample 400 --name base
  uv run python -m eval.run_programs --handmade --name smoke
  uv run python -m eval.run_programs DIR --library DIR --name suite   # programs call each other

Writes eval/runs/<name>/: results.jsonl (one record per program, sorted by index),
summary.json, summary.md, shots/ (first --shots programs), inputs/ (generated .8xp), config.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from . import driver, handmade, inputs, summary
from . import emu as E

RUNS = Path(__file__).resolve().parent / "runs"
_ENV = None
_CFG = None
_YIELDS = False


def _init_worker(library: str | None, cfg: driver.Config, yields: bool = False,
                 prefill_root: str | None = None) -> None:
    global _ENV, _CFG, _YIELDS
    _ENV = E.prepare(Path(library) if library else None)
    _CFG = replace(cfg, preloaded=_ENV.preloaded)
    _YIELDS = yields
    driver.PREFILL_ROOT = Path(prefill_root) if prefill_root else None


def _harness_error(spec, why: str) -> dict:
    res = driver._base(spec)
    res.update(outcome="HARNESS_ERROR", label="HARNESS_ERROR", reject_reason=why)
    return res


def _work(job: tuple) -> dict:
    spec, shot = job
    try:
        return driver.run_one(_ENV.emu, spec, _ENV.clean_state, _CFG, shot=shot, yields=_YIELDS)
    except Exception as exc:  # noqa: BLE001 - one bad program must not stop the run
        return _harness_error(spec, f"{type(exc).__name__}: {exc}")


def _shot(out: Path, spec, n_shots: int) -> Path | None:
    if spec.index >= n_shots or spec.reject_reason:
        return None
    return out / "shots" / f"{spec.index:06d}-{spec.name}.png"


def run_specs(specs: list, out: Path, workers: int = 1, cfg: driver.Config | None = None,
              shots: int = 10, library: Path | None = None, title: str = "run",
              work=_work, job_timeout_s: float | None = None, yields: bool = False) -> list[dict]:
    """Run every spec (in parallel worker processes, one emulator each) and write the run
    directory. Returns the result records sorted by index. A worker process that dies or
    hangs costs only its program (HARNESS_ERROR); the pool replaces the process. `yields`
    turns on driver's yield capture (records gain `yields`; config.json says so)."""
    cfg = cfg or driver.Config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lib = str(library) if library else None
    # prefill: the workers' cached states go in one per-run dir, removed below (pool workers
    # are terminated, so their own exit handlers never run)
    if cfg.prefill:
        E.CACHE.mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(prefix="prefill-run-", dir=E.CACHE) if cfg.prefill else None
    init = (lib, cfg, yields) if root is None else (lib, cfg, yields, root)
    try:
        _init_worker(*init)      # builds the cached states before workers start
        jobs = [(s, _shot(out, s, shots)) for s in specs]
        timeout = job_timeout_s or cfg.wall_limit_s + 60
        t0 = time.perf_counter()
        results: list[dict] = []
        if workers <= 1:
            for i, job in enumerate(jobs):
                results.append(work(job))
                _progress(i + 1, len(jobs), t0)
        else:
            ctx = mp.get_context("forkserver")
            with ctx.Pool(workers, initializer=_init_worker, initargs=init) as pool:
                pending = [(job, pool.apply_async(work, (job,))) for job in jobs]
                for i, (job, ar) in enumerate(pending):   # FIFO: job i has started by now
                    try:
                        results.append(ar.get(timeout=timeout))
                    except mp.TimeoutError:
                        results.append(_harness_error(job[0], f"worker lost or no result in "
                                                              f"{timeout:.0f} s"))
                    _progress(i + 1, len(jobs), t0)
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
            driver.PREFILL_ROOT = None
    wall = time.perf_counter() - t0
    return _write_run(out, results, wall, workers, title, cfg, lib, yields, len(specs))


def _write_run(out: Path, results: list[dict], wall: float, workers: int, title: str,
               cfg: driver.Config, lib: str | None, yields: bool, n: int) -> list[dict]:
    """A run directory's files (results.jsonl, summary.json, summary.md, config.json); returns
    the records sorted by index."""
    results.sort(key=lambda r: r["index"])
    with open(out / "results.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summary.summarize(results, wall_s=wall, workers=max(1, workers))
    (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    (out / "summary.md").write_text(summary.to_markdown(s, title))
    meta = {"title": title, "workers": workers, "library": lib, "n": n,
            "config": _config_meta(cfg), "rom_sha256_16": _ENV.emu.digest}
    if yields:
        meta["yields"] = True
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    return results


def _work_job(job: tuple) -> dict:
    """_work for EmuPool: the job carries its own driver settings, yield capture and the
    batch's shared prefilled state (driver.run_one's `prefilled`)."""
    spec, shot, cfg, yields, prefilled = job
    try:
        return driver.run_one(_ENV.emu, spec, _ENV.clean_state,
                              replace(cfg, preloaded=_ENV.preloaded), shot=shot, yields=yields,
                              prefilled=prefilled)
    except Exception as exc:  # noqa: BLE001 - one bad program must not stop the run
        return _harness_error(spec, f"{type(exc).__name__}: {exc}")


def _build_prefilled(job: tuple) -> str:
    """An EmuPool build job: the clean state with these prefill pairs, saved at path."""
    pairs, path = job
    return str(driver.save_prefilled(_ENV.emu, _ENV.clean_state, pairs, Path(path)))


class EmuPool:
    """One set of emulator worker processes kept alive across many run_specs-style batches
    (a fresh pool per batch costs more than a 25-program batch takes to run).

    submit() is run_specs split in two: it queues the batch's programs and returns a Batch at
    once; Batch.result() returns the records and writes the run directory run_specs writes
    (results.jsonl, summary.json, summary.md, config.json; `workers` there is the number
    given to submit). Every run is driver.run_one from the clean state as in run_specs, so
    the records are the ones run_specs gives, wall_s and summary timings aside.

    Results are collected strictly in submission order, whichever batch asks: when the pool
    waits on job i, every earlier job has returned, so job i has started (run_specs' FIFO
    rule), and a job with no result within cfg.wall_limit_s + 60 s (job_timeout_s) is a
    HARNESS_ERROR. A worker that dies is replaced by the pool. After such a timeout every
    worker is replaced and the jobs still out are sent again (run_specs' pool, and a stuck
    worker with it, ended with its batch; a run starts from a saved state, so a rerun gives
    the same record). With workers <= 1 every program runs in this process at submit(), as
    run_specs does.

    Prefill (task suite v2's harness): run_specs' workers each build the prefilled state
    (driver.prefilled_state, about 0.8 s of link transfer) for every batch they take. Here
    one build job per distinct prefill is queued at submit(); runs go to the workers at the
    next result(), once every queued build is done (so the FIFO rule above holds), and load
    that state (driver.run_one's `prefilled`). A run whose shared state is missing builds
    its own as before. The state is the clean state with the same variables stored over the
    link, whichever process stores them. It is deleted once its last run is collected. Each
    state is about 5 MB under eval/runs/_cache/prefill-run-*/ (an E9 discovery queues 636 at
    once: about 3.3 GB until its batches are collected)."""

    BUILD_TIMEOUT_S = 120.0

    def __init__(self, workers: int = 8, library: Path | None = None, work=_work_job):
        self.workers = workers
        self.lib = str(library) if library else None
        self.work = work
        E.CACHE.mkdir(parents=True, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="prefill-run-", dir=E.CACHE)   # removed by close()
        self._init = (self.lib, driver.Config(), False, self.root)
        self._pool = None
        self._closed = False
        self._jobs: list = []        # per job: [job, timeout s, async result or None]
        self._sent = 0               # jobs 0..sent-1 are with the workers
        self._got: list[dict] = []   # records of jobs 0..len-1, in submission order
        self._shared: dict = {}      # prefill pairs -> its shared state (being) built
        self._last_use: dict = {}    # prefill pairs -> the last job index that loads it
        self._builds: list = []      # build jobs not yet waited for
        self._n_built = 0
        self._last_done = time.perf_counter()
        try:
            _init_worker(*self._init)      # builds the cached states before workers start
            if workers > 1:
                self._start()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> EmuPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _start(self) -> None:
        ctx = mp.get_context("forkserver")
        self._pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=self._init)

    def _stop(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def close(self) -> None:
        """Stop the workers (as leaving run_specs' pool does) and remove the prefill files."""
        self._stop()
        self._closed = True
        shutil.rmtree(self.root, ignore_errors=True)
        driver.PREFILL_ROOT = None

    def submit(self, specs: list, out: Path, workers: int | None = None,
               cfg: driver.Config | None = None, shots: int = 10, title: str = "run",
               job_timeout_s: float | None = None, yields: bool = False) -> Batch:
        """Queue run_specs(specs, out, workers, cfg, shots, title=..., job_timeout_s=...,
        yields=...) on this pool (workers: only what config.json and the summary record;
        default the pool's)."""
        if self._closed:
            raise RuntimeError("EmuPool is closed")
        t0 = time.perf_counter()
        cfg = cfg or driver.Config()
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        timeout = job_timeout_s or cfg.wall_limit_s + 60
        first = len(self._jobs)
        shared = self._prefilled(cfg.prefill) if specs else None
        for s in specs:
            job = (s, _shot(out, s, shots), cfg, yields, shared)
            self._jobs.append([job, timeout, None])
            if self._pool is None:
                self._got.append(self.work(job))
                self._sent += 1
        if shared is not None:
            self._last_use[cfg.prefill] = len(self._jobs) - 1
        return Batch(self, range(first, len(self._jobs)), out,
                     self.workers if workers is None else workers, cfg, shots, title, yields, t0)

    def _prefilled(self, pairs) -> Path | None:
        """The shared state for these prefill pairs: its build job is queued now."""
        if pairs is None or self._pool is None:
            return None
        if pairs not in self._shared:
            path = Path(self.root) / f"shared-{self._n_built:05d}.img"
            self._n_built += 1
            self._shared[pairs] = path
            self._build(pairs, path)
        return self._shared[pairs]

    def _build(self, pairs, path: Path) -> None:
        self._builds.append(self._pool.apply_async(_build_prefilled, ((pairs, str(path)),)))

    def submit_token_ids(self, seqs, vocab: Path, out: Path, workers: int | None = None,
                         budget_ms: int = 30_000, game: bool = False,
                         input_text: str | None = None,
                         input_values: tuple[str, ...] | None = None, decoy: str | None = None,
                         prefill: tuple[tuple[str, str], ...] | None = None, **kw) -> Batch:
        """run_token_ids on this pool (submit's Batch)."""
        specs, cfg = token_id_specs(seqs, vocab, out, budget_ms, game, input_text,
                                    input_values, decoy, prefill)
        return self.submit(specs, out, workers=workers, cfg=cfg, **kw)

    def _dispatch(self) -> None:
        """Wait for the queued builds, then send every queued run to the workers, in order. A
        build not done BUILD_TIMEOUT_S after the one before it (no progress; an E9 discovery
        queues hundreds): every worker is replaced, and runs build their own."""
        if self._sent == len(self._jobs):
            return
        stuck = False
        for b in self._builds:
            b.wait(self.BUILD_TIMEOUT_S)
            if not b.ready():
                stuck = True
                break
        self._builds = []
        if stuck:
            self._restart(len(self._got), rebuild=False)   # every uncollected run again
        while self._sent < len(self._jobs):
            entry = self._jobs[self._sent]
            entry[2] = self._pool.apply_async(self.work, (entry[0],))
            self._sent += 1

    def _restart(self, first: int, rebuild: bool = True) -> None:
        """New workers; jobs first..sent-1 (and, with rebuild, every missing shared state)
        are sent again by the next _dispatch."""
        self._stop()
        self._start()
        self._builds = []
        if rebuild:
            for pairs, path in self._shared.items():
                if not path.exists():
                    self._build(pairs, path)
        self._sent = min(self._sent, first)

    def _record(self, i: int) -> dict:
        if len(self._got) <= i and self._closed:
            raise RuntimeError("EmuPool is closed; its uncollected batches are lost")
        while len(self._got) <= i:      # FIFO: every earlier job has returned
            self._dispatch()
            k = len(self._got)
            job, timeout, ar = self._jobs[k]
            try:
                rec = ar.get(timeout=timeout)
            except mp.TimeoutError:
                rec = _harness_error(job[0], f"worker lost or no result in {timeout:.0f} s")
                self._restart(k + 1)
            self._got.append(rec)
            pairs = job[2].prefill
            if job[4] is not None and self._last_use.get(pairs) == k:
                self._shared.pop(pairs).unlink(missing_ok=True)   # its last run is in
                del self._last_use[pairs]
        return self._got[i]


class Batch:
    """One EmuPool.submit(): result() waits for its records and writes its run directory."""

    def __init__(self, pool: EmuPool, jobs: range, out: Path, workers: int,
                 cfg: driver.Config, shots: int, title: str, yields: bool, t_submit: float):
        self.pool, self.jobs, self.out, self.workers = pool, jobs, out, workers
        self.cfg, self.title, self.yields = cfg, title, yields
        self.t_submit = t_submit
        self._records: list[dict] | None = None

    def result(self) -> list[dict]:
        """The records sorted by index (run_specs' return). Summary wall time: from this
        batch's submission, or the previous collected batch's end if later, to its end."""
        if self._records is None:
            t0 = max(self.t_submit, self.pool._last_done)
            results = []
            for k, i in enumerate(self.jobs):
                results.append(self.pool._record(i))
                _progress(k + 1, len(self.jobs), t0)
            wall = time.perf_counter() - t0
            self.pool._last_done = time.perf_counter()
            self._records = _write_run(self.out, results, wall, self.workers, self.title,
                                       self.cfg, self.pool.lib, self.yields, len(self.jobs))
            for i in self.jobs:          # the batch holds its records now; the pool lets go
                self.pool._got[i] = self.pool._jobs[i] = None
        return self._records


def _config_meta(cfg: driver.Config) -> dict:
    """config.json's "config": the driver settings; input_values, decoy and prefill only when
    set (a default run's config.json is the one it always was)."""
    return {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(cfg).items()
            if not (k in ("input_values", "decoy", "prefill") and v is None)}


def _progress(done: int, total: int, t0: float) -> None:
    if done == total or done % max(1, total // 20) == 0:
        rate = 60 * done / max(1e-9, time.perf_counter() - t0)
        print(f"  {done}/{total} programs, {rate:.0f}/min", file=sys.stderr, flush=True)


def run_dir(directory: Path, out: Path, **kw) -> list[dict]:
    return run_specs(inputs.from_dir(directory, Path(out) / "inputs"), out, **kw)


def run_token_ids(seqs, vocab: Path, out: Path, workers: int = 1, budget_ms: int = 30_000,
                  game: bool = False, input_text: str | None = None,
                  input_values: tuple[str, ...] | None = None, decoy: str | None = None,
                  prefill: tuple[tuple[str, str], ...] | None = None, **kw) -> list[dict]:
    """API for the model: token-id sequences (SPEC.md vocab) -> result records. game: feed
    getKey loops driver.GAME_KEYS with key probes (records gain `key_probes`). input_text:
    the value the harness types at every Input/Prompt (driver.Config.input_text).
    input_values: one value per successive Input/Prompt wait (driver.Config.input_values;
    records gain `extra_inputs`). decoy, prefill: task suite v2's hardened harness
    (driver.Config.decoy and .prefill; records gain `decoy` and `prefill`)."""
    specs, cfg = token_id_specs(seqs, vocab, out, budget_ms, game, input_text, input_values,
                                decoy, prefill)
    return run_specs(specs, out, workers=workers, cfg=cfg, **kw)


def token_id_specs(seqs, vocab: Path, out: Path, budget_ms: int = 30_000, game: bool = False,
                   input_text: str | None = None, input_values: tuple[str, ...] | None = None,
                   decoy: str | None = None, prefill: tuple[tuple[str, str], ...] | None = None
                   ) -> tuple[list, driver.Config]:
    """run_token_ids' programs (written to OUT/inputs/) and driver settings."""
    specs = inputs.from_token_ids(seqs, vocab, Path(out) / "inputs")
    cfg = driver.Config(budget_ms=budget_ms)
    if game:
        cfg = replace(cfg, feed_keys=driver.GAME_KEYS, key_probe=True)
    if input_text is not None:
        cfg = replace(cfg, input_text=input_text)
    if input_values is not None:
        cfg = replace(cfg, input_values=tuple(input_values))
    if decoy is not None:
        cfg = replace(cfg, decoy=decoy)
    if prefill is not None:
        cfg = replace(cfg, prefill=tuple(map(tuple, prefill)))
    return specs, cfg


def _read_ids(path: Path) -> list[list[int]]:
    if path.suffix == ".bin":
        return inputs.read_bin(path)
    text = Path(path).read_text()
    if path.suffix == ".json":
        return json.loads(text)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r["ids"] if isinstance(r, dict) else r for r in rows]


def _specs_from_args(a, out: Path) -> tuple[list, str]:
    work = out / "inputs"
    if a.handmade:
        items = [(c.name, c.body()) for c in handmade.CASES]
        return inputs.from_bodies(items, work, source="handmade"), "hand-made programs"
    if a.ids:
        if not a.vocab:
            raise SystemExit("--ids needs --vocab")
        return inputs.from_token_ids(_read_ids(Path(a.ids)), Path(a.vocab), work), str(a.ids)
    if a.corpus:
        specs = inputs.from_corpus(Path(a.corpus), a.sample, a.seed, work)
        return specs, f"corpus sample n={a.sample} seed={a.seed}"
    if a.dir:
        return inputs.from_dir(Path(a.dir), work), str(a.dir)
    raise SystemExit("give a directory of .8xp files, --ids, --corpus or --handmade")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir", nargs="?", help="directory of .8xp programs")
    p.add_argument("--ids", help="token-id sequences: .jsonl (list or {ids: [...]}), .json, "
                                  "or a corpus .bin (uint16, <bos> ... <eos>)")
    p.add_argument("--vocab", help="vocab.json for --ids (corpus/vocab.json's format)")
    p.add_argument("--corpus", help="corpus data/extract/programs.jsonl to sample from")
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--handmade", action="store_true", help="the built-in hand-made test set")
    p.add_argument("--library", help="send every .8xp here to Archive first (program suites)")
    p.add_argument("--name", help="run name: output goes to eval/runs/NAME/")
    p.add_argument("--out", help="output directory (overrides --name)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--budget-ms", type=int, default=30_000, help="emulated ms per program")
    p.add_argument("--shots", type=int, default=20, help="screenshots for the first N programs")
    p.add_argument("--input-text", default="1", help="answer typed at every Input/Prompt")
    a = p.parse_args(argv)
    if not E.rom_available():
        print(f"ROM not found: {E.ROM} (set CALCLM_CEMU_ROOT or CALCLM_ROM)", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else RUNS / (a.name or time.strftime("run-%Y%m%d-%H%M%S"))
    specs, title = _specs_from_args(a, out)
    print(f"{len(specs)} programs -> {out}", file=sys.stderr)
    cfg = driver.Config(budget_ms=a.budget_ms, input_text=a.input_text)
    run_specs(specs, out, workers=a.workers, cfg=cfg,
              shots=a.shots, library=Path(a.library) if a.library else None, title=title)
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
