"""Run one TI-BASIC program on the emulated TI-84 Plus CE and classify the outcome.

Every program starts from the same saved state. It is sent over the emulated link, launched
from the home screen (Clear, prgm, name, Enter) and driven until it returns, errors or runs
out of emulated time:

  Input / Prompt          type INPUT_TEXT ("1") then ENTER; with input_values, the k-th wait
                          gets input_values[k], waits past the tuple the last value (counted
                          in the record's extra_inputs, a key only such runs have); with
                          decoy also set, waits past the tuple get the decoy instead
  Pause (text or graph)   ENTER (also Input with no variable on the graph screen)
  Menu(                   first visit of a given Menu( : option 1; later visits: the last
                          option (Up wraps to it, then ENTER), usually BACK / QUIT
  getKey loop             when the parser keeps cycling over code that contains getKey,
                          one physical key every GETKEY_EVERY_MS from FEED_KEYS
  budget spent            stop answering; within GRACE_MS it either finishes (OK/ERR count),
                          settles into a prompt (TIMEOUT waiting for that prompt) or keeps
                          running (TIMEOUT getkey or computing); then ON (ERR:BREAK)

Outcomes: OK, ERR:<NAME>, TIMEOUT (timeout_state: input, menu, pause, graph_wait, getkey,
computing), REJECTED (tivars or the OS refused the file).

Prefill (opt-in, `prefill`): before the program is sent, real variables (A-Z, θ) get the given
values over the link, the way a calculator in use holds old values; the program's bytes, Ans and
the home screen are untouched. Runs with `decoy` or `prefill` record them under the same keys.

Yield capture (opt-in, `yields=True`): the record gains `yields`, one entry per wait the harness
answers, [kind, emulated ms, graph seen so far, non-blank home-screen lines], read just before
the answer; kind is input, menu, pause, graph_wait or getkey. Nothing else in the record changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import osram
from . import tokens as T

IDLE_BUSY = 0.4      # CPU busy fraction below which the OS is waiting for a key
REAL_VARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZθ")
GAME_KEYS = ("Right", "Down", "Left", "Up", "Clear")   # key probes: arrows, then a quit key


@dataclass(frozen=True)
class Config:
    budget_ms: int = 30_000          # emulated time from launch
    step_ms: int = 50
    quiet_ms: int = 300              # idle this long before answering a prompt
    end_quiet_ms: int = 100          # back at home and idle this long = finished
    getkey_every_ms: int = 500
    grace_ms: int = 3000             # after the budget: no input, wait to see what it waits for
    input_text: str = "1"
    input_values: tuple[str, ...] | None = None   # per-wait answers instead of input_text
    feed_keys: tuple = ("Enter", "Right", "Down", "2nd", "Left", "Up", "Clear")
    max_actions: int = 400
    key_probe: bool = False          # record, per fed getKey key, whether the screen changed
    probe_ms: int = 300              # how long after a probed press to look at the screen
    locate_errors: bool = True       # after ERR:, take 2:Goto to find the error position
    wall_limit_s: float = 180.0
    preloaded: frozenset = frozenset()
    decoy: str | None = None         # the answer at waits past input_values (needs input_values)
    prefill: tuple[tuple[str, str], ...] | None = None   # (real variable, value) before the run

    def __post_init__(self):
        if self.input_values is not None and not (
                self.input_values and all(isinstance(v, str) for v in self.input_values)):
            raise ValueError("input_values must hold at least one string (or be None)")
        if self.decoy is not None and not (isinstance(self.decoy, str) and self.decoy
                                           and self.input_values is not None):
            raise ValueError("decoy must be a non-empty string, and needs input_values")
        if self.prefill is not None:
            names = [p[0] if isinstance(p, tuple) and len(p) == 2 else None
                     for p in self.prefill]
            if not (self.prefill and all(n in REAL_VARS for n in names)
                    and len(set(names)) == len(names)
                    and all(isinstance(p[1], str) and p[1] for p in self.prefill)):
                raise ValueError("prefill must be (variable, value) string pairs naming "
                                 "distinct real variables A-Z or θ")


class _Clock:
    """Counts emulated milliseconds by wrapping this Emu instance's run()."""

    def __init__(self, emu):
        self.ms = 0
        orig = emu.run

        def run(ms: int) -> None:
            self.ms += ms
            orig(ms)
        emu.run = run


def _clock(emu) -> _Clock:
    if not hasattr(emu, "_eval_clock"):
        emu._eval_clock = _Clock(emu)
    return emu._eval_clock


def _base(spec) -> dict:
    f = T.features(spec.body)
    return {"index": spec.index, "name": spec.name, "orig_name": spec.orig_name,
            "source": spec.source, "n_bytes": len(spec.body), "n_tokens": f.pop("n_tokens"),
            "static": f, "meta": spec.meta, "outcome": None, "label": None, "error": None,
            "timeout_state": None, "reject_reason": None, "emu_ms": 0, "wall_s": 0.0,
            "graph": False, "actions": {}, "last_action": None, "ms_since_action": None,
            "error_offset": None, "error_line_no": None, "error_line": None,
            "errno_at_end": None,
            "home_text": [], "lcd_text": [], "screenshot": None}


def _reject(res: dict, reason: str) -> dict:
    res.update(outcome="REJECTED", label="REJECTED", reject_reason=reason)
    return res


class _Run:
    def __init__(self, emu, spec, cfg: Config, yields: bool = False):
        self.emu, self.spec, self.cfg = emu, spec, cfg
        self.yields: list | None = [] if yields else None
        self.clock = _clock(emu)
        self.t0 = self.clock.ms
        self.actions: dict[str, int] = {}
        self.last_action: str | None = None
        self.last_action_ms = 0
        self.menu_visits: dict[tuple, int] = {}
        self.feed_i = 0
        self.probes: list | None = [] if cfg.key_probe else None
        self.graph = False
        self.has_getkey = T.features(spec.body)["getkey"] or bool(cfg.preloaded)
        self.inputs_typed = 0
        self.extra_inputs = 0

    @property
    def elapsed(self) -> int:
        return self.clock.ms - self.t0

    def act(self, kind: str) -> None:
        self.actions[kind] = self.actions.get(kind, 0) + 1
        self.last_action, self.last_action_ms = kind, self.elapsed

    def inject(self, *keys: str) -> None:
        """Hand the OS keycodes (never lost; each waits until the OS took the previous)."""
        from .emu import cemu  # noqa: PLC0415
        for k in keys:
            try:
                self.emu.inject_key(k)
            except cemu().EmuError:
                self.act("lost_key")
                return

    def snap(self, kind: str) -> None:
        """Yield capture: the home screen at a wait, before the harness answers it."""
        if self.yields is None:
            return
        try:
            txt = self.emu.home_text()
        except Exception:  # noqa: BLE001 - capture must never change the run's outcome
            txt = []
        self.yields.append([kind, self.elapsed, bool(self.graph), [x for x in txt if x.strip()]])

    def answer(self, kind: str) -> None:
        self.snap(kind)
        if kind == "input":
            from tools.cemu.keys import text_keys  # noqa: PLC0415 - path set up by eval.emu
            self.inject(*text_keys(self.typed_value()), "Enter")
        elif kind == "menu":
            where = osram.read_pcs(self.emu)[:2]
            n = self.menu_visits.get(where, 0)
            self.menu_visits[where] = n + 1
            self.inject("1") if n == 0 else self.inject("Up", "Enter")
        else:
            self.inject("Enter")
        self.act(kind)

    def typed_value(self) -> str:
        """What this Input/Prompt wait is answered with: input_text, or with input_values the
        k-th value (past the tuple, counted in extra_inputs: the decoy if set, else the last
        value)."""
        vals = self.cfg.input_values
        if vals is None:
            return self.cfg.input_text
        k = self.inputs_typed
        self.inputs_typed += 1
        if k >= len(vals):
            self.extra_inputs += 1
            if self.cfg.decoy is not None:
                return self.cfg.decoy
        return vals[min(k, len(vals) - 1)]

    def in_getkey_loop(self, samples: int = 20, gap_ms: int = 5) -> bool:
        """Sample the parser's position; True if the code it cycles over contains getKey."""
        spans: dict[tuple[int, int], list[int]] = {}
        for _ in range(samples):
            beg, cur, end = osram.read_pcs(self.emu)
            spans.setdefault((beg, end), []).append(cur)
            self.emu.run(gap_ms)
        for (beg, end), curs in spans.items():
            size = end - beg + 1
            if not 0 < size <= T.MAX_PROGRAM:
                continue
            lo, hi = min(curs) - beg - 4, max(curs) - beg + 4
            body = self.emu.read(beg, size)
            if any(tok == T.GETKEY and lo <= off <= hi for off, tok in T.token_offsets(body)):
                return True
        return False

    def feed_key(self) -> None:
        self.snap("getkey")
        from tools.cemu.keys import PHYSICAL  # noqa: PLC0415 - path set up by eval.emu
        key = self.cfg.feed_keys[self.feed_i % len(self.cfg.feed_keys)]
        self.feed_i += 1
        before = self.screen_sig() if self.probes is not None else None
        self.emu.press(*PHYSICAL[key])
        self.act("getkey")
        if self.probes is not None:
            self.emu.run(self.cfg.probe_ms)
            self.probes.append([key, self.screen_sig() != before, self.elapsed])

    def screen_sig(self):
        """What a key probe compares: the LCD's lit cells (graph and text alike)."""
        try:
            return tuple(self.emu.screen())
        except Exception:  # noqa: BLE001 - a probe must never change the run's outcome
            return None

    def drive(self) -> tuple[str, osram.OsState]:
        """Poll until the program finishes, errors or the budget is spent. Returns
        ("OK" | "ERR" | "TIMEOUT:<state>", last state)."""
        cfg, emu = self.cfg, self.emu
        quiet = busy_run = home_quiet = 0
        started = False
        deadline = None
        prev_sig = None
        wall0 = time.perf_counter()
        while True:
            busy = emu.busy_fraction(cfg.step_ms)
            st = osram.read_state(emu)
            self.graph |= st.plot
            if st.error_shown:
                return "ERR", st
            idle = busy < IDLE_BUSY
            if st.prog:
                started, home_quiet = True, 0
            elif idle and not st.key_ready:
                home_quiet += cfg.step_ms
                if home_quiet >= cfg.end_quiet_ms and (started or self.elapsed >= 300):
                    return "OK", st
            sig = (st.prog, st.cx, st.menu, st.plot, st.key_ready)
            # a wait counts only once the same screen state has been idle for quiet_ms with
            # no key of ours pending (Menu( can sit on an injected key for over a second)
            quiet = quiet + cfg.step_ms if idle and sig == prev_sig and not st.key_ready else 0
            prev_sig = sig
            busy_run = 0 if idle else busy_run + cfg.step_ms
            if deadline is None and (
                    self.elapsed >= cfg.budget_ms or sum(self.actions.values()) >= cfg.max_actions
                    or time.perf_counter() - wall0 > cfg.wall_limit_s):
                deadline = self.elapsed + cfg.grace_ms   # no more input; let it settle
            if deadline is not None:
                if st.prog and quiet >= cfg.quiet_ms:
                    return "TIMEOUT:" + st.wait_kind(), st
                if self.elapsed >= deadline:
                    return "TIMEOUT:" + ("getkey" if self.in_getkey_loop() else "computing"), st
                continue
            if st.prog and quiet >= cfg.quiet_ms and not st.key_ready:
                self.answer(st.wait_kind())   # never while our last key is still pending
                quiet = 0
            elif (st.prog and self.has_getkey and busy_run >= cfg.getkey_every_ms
                  and self.in_getkey_loop()):
                self.feed_key()
                busy_run = 0


def locate_error(emu, body: bytes, step_ms: int = 50, max_ms: int = 30_000) -> int | None:
    """Byte offset of the error in `body`: choose 2:Goto on the ERR: menu and read the program
    editor's cursor once the editor has stopped moving it (our key taken, CPU idle, cursor
    unchanged for step_ms). On a long program the editor walks the cursor down a few dozen
    lines per emulated second, busy all the way. None when Goto is not offered, the editor
    has not settled within max_ms, or the error is in another program."""
    if not emu.read(osram.ERR_NO, 1)[0] & osram.ERR_GOTO:
        return None
    emu.inject_key("2")
    last = None
    for _ in range(max_ms // step_ms):
        idle = emu.busy_fraction(step_ms) < IDLE_BUSY
        pending = osram.read_state(emu).key_ready
        top = emu.u24(osram.EDIT_TOP)
        off = emu.u24(osram.EDIT_CURSOR) - top
        if idle and not pending and off == last:
            break
        last = None if pending else off
    else:
        return None
    if not 0 <= off <= len(body) or emu.read(top, off) != body[:off]:
        return None
    return off


def stored(emu, name: str) -> bool:
    """A program variable with this name is in the VAT (θ is stored as 0x5B)."""
    raw = name.replace("θ", "[").encode("latin-1")
    return any(v["type"] in (5, 6) and v["name"].rstrip(b"\0") == raw for v in emu.vat())


def prefill(emu, pairs: tuple[tuple[str, str], ...]) -> None:
    """Store each (real variable, value) in one link transfer, then read every one back."""
    import tempfile  # noqa: PLC0415

    from .emu import cemu  # noqa: PLC0415
    mod = cemu()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (name, text) in enumerate(pairs):
            _, raw = mod.name_bytes(name)
            path = Path(tmp) / f"v{i}.8xn"
            path.write_bytes(mod.var_file(mod.TYPE_REAL, raw[:1],
                                          mod.encode_real(float(text.replace("⁻", "-")))))
            paths.append(path)
        emu.send(*paths)
    for name, text in pairs:
        got, want = emu.get_real(name), float(text.replace("⁻", "-"))
        if got is None or abs(got - want) > 1e-12 * max(1.0, abs(want)):
            raise RuntimeError(f"prefill: {name} holds {got}, not {text}")


_PREFILLED: dict = {}   # this process's last (clean state, prefill) and the state saved after it
PREFILL_ROOT: Path | None = None   # where cache dirs go (run_specs sets a per-run dir it removes)


def prefilled_state(emu, clean_state: Path, pairs: tuple[tuple[str, str], ...]) -> Path:
    """The clean state with `pairs` stored (prefill), saved once and reused while the same
    pairs come in a row (a task suite runs every program of one tuple together). The link
    transfer costs about 0.8 s of wall time; loading the saved state costs what loading the
    clean state does. Each process writes only its own fresh directory (PREFILL_ROOT, else
    eval/runs/_cache: ROM-derived data), so processes never share a file."""
    import tempfile  # noqa: PLC0415

    key = (str(clean_state), tuple(pairs))
    path = _PREFILLED.get("path")
    if _PREFILLED.get("key") != key or path is None or not path.exists():
        _PREFILLED.pop("key", None)
        if path is None or not path.parent.exists():
            import atexit  # noqa: PLC0415
            import shutil  # noqa: PLC0415

            from .emu import CACHE  # noqa: PLC0415
            root = PREFILL_ROOT or CACHE
            root.mkdir(parents=True, exist_ok=True)
            d = Path(tempfile.mkdtemp(prefix="prefill-", dir=root))
            atexit.register(shutil.rmtree, d, ignore_errors=True)
            path = d / "state.img"
        save_prefilled(emu, clean_state, pairs, path)
        _PREFILLED.update(key=key, path=path)
    return path


def save_prefilled(emu, clean_state: Path, pairs: tuple[tuple[str, str], ...],
                   path: Path) -> Path:
    """The clean state with `pairs` stored (prefill), saved at `path` (written whole under a
    temporary name, then renamed)."""
    import os  # noqa: PLC0415

    path = Path(path)
    emu.load_state(clean_state)
    prefill(emu, pairs)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    emu.save_state(tmp)
    tmp.replace(path)
    return path


def run_one(emu, spec, clean_state: Path | None, cfg: Config, shot: Path | None = None,
            yields: bool = False, prefilled: Path | None = None) -> dict:
    """Run one ProgramSpec from `clean_state`; returns the result record (with `yields` when
    asked: see the module docstring). With cfg.prefill, `prefilled` may name a state that
    save_prefilled wrote for those pairs (eval.run_programs.EmuPool shares one per batch);
    it is loaded when it exists, else this process builds its own (prefilled_state)."""
    res = _base(spec)
    if yields:
        res["yields"] = []
    if spec.reject_reason:
        return _reject(res, spec.reject_reason)
    wall0 = time.perf_counter()
    if cfg.prefill is None:
        emu.load_state(clean_state)
    elif prefilled is not None and Path(prefilled).exists():
        emu.load_state(prefilled)
    else:
        emu.load_state(prefilled_state(emu, clean_state, cfg.prefill))
    if spec.name not in cfg.preloaded:
        from .emu import cemu  # noqa: PLC0415
        try:
            emu.send(spec.path)
        except cemu().EmuError as exc:
            return _reject(res, f"link: {exc}")
        if not stored(emu, spec.name):
            return _reject(res, "the OS did not store the program")
    run = _Run(emu, spec, cfg, yields)
    emu.run_program(spec.name, wait=False)
    status, st = run.drive()
    if yields:
        res["yields"] = run.yields
    if run.probes is not None:
        res["key_probes"] = run.probes
    if cfg.input_values is not None:
        res["extra_inputs"] = run.extra_inputs
    if cfg.decoy is not None:
        res["decoy"] = cfg.decoy
    if cfg.prefill is not None:
        res["prefill"] = dict(cfg.prefill)
    res.update(emu_ms=run.elapsed, graph=run.graph, actions=run.actions,
               last_action=run.last_action,
               ms_since_action=None if run.last_action is None
               else run.elapsed - run.last_action_ms)
    res["home_text"] = emu.home_text()
    res["lcd_text"] = emu.screen()
    if shot is not None:
        Path(shot).parent.mkdir(parents=True, exist_ok=True)
        res["screenshot"] = str(emu.screenshot(shot))
    if status == "OK":
        # errNo set at a normal end: an error the OS handled itself (e.g. NONREAL ANS while
        # graphing), kept for transparency
        res.update(outcome="OK", label="OK", errno_at_end=st.error)
    elif status == "ERR":
        res.update(outcome="ERR", label=f"ERR:{st.error}", error=st.error)
        off = locate_error(emu, spec.body) if cfg.locate_errors else None
        if off is not None:
            line_no, line = T.line_at(spec.body, off)
            res.update(error_offset=off, error_line_no=line_no, error_line=line)
    else:
        res.update(outcome="TIMEOUT", label="TIMEOUT", timeout_state=status.split(":", 1)[1])
        emu.on_break()
        emu.run(300)
    res["wall_s"] = round(time.perf_counter() - wall0, 3)
    return res
