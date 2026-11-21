"""The cross-experiment emulator lock (PLAN-E11 "Order with E13 and E14": no emulator overlap).

The judge's wall-clock limit makes harness errors depend on host load, so no two experiments'
emulator stages (program verification, scoring or judging: lab.tasks score2 and sample2, lab.score,
the task-pool builders) may run at the same time. The signal is an advisory fcntl.flock on one
file, LOCK below, shared by every runner that drives the emulator:

    from lab.emulator_lock import emulator_stage
    with emulator_stage('E13 verify pool', log=_log):
        ...                                   # every emulator command of the stage

A runner that finds the lock held waits, polling every POLL_S seconds, and logs who holds it (the
holder writes "<who> (pid N, since HH:MM:SS)" into the file). The flock is released when the
holding process exits, so a killed runner never leaves a stale lock. The lock is per process and
re-entrant: a nested emulator_stage in the process that holds it does not wait for itself.

Emulator commands started inside the stage should get the lock's descriptor
(subprocess.run(..., pass_fds=held_fds())): the flock then lasts until the last of them exits, so a
runner killed mid-stage does not free the emulator while its children still drive it.

Only runners that take this lock are kept apart; E13's and E14's runners must take it around
their emulator stages too.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / 'runs' / 'locks' / 'emulator.lock'
POLL_S = 60.0
_HELD: list[str] = []       # what this process holds the lock for (re-entrancy)
_FDS: list[int] = []        # the held lock file's descriptor


def held() -> bool:
    """This process holds the emulator lock."""
    return bool(_HELD)


def held_fds() -> tuple[int, ...]:
    """The held lock's descriptor (for subprocess pass_fds), or () outside a stage."""
    return tuple(_FDS)


def _holder(f) -> str:
    f.seek(0)
    return f.read().strip() or 'another process'


@contextlib.contextmanager
def emulator_stage(who: str, log=print, poll_s: float | None = None, lock: Path | None = None):
    """Hold the emulator lock for one emulator stage, waiting (polling) while another process
    holds it. who names the stage in the lock file and in other runners' wait lines."""
    if _HELD:
        yield
        return
    path = Path(lock or LOCK)
    poll = POLL_S if poll_s is None else poll_s
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as f:
        t0, last = time.time(), None
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                holder = _holder(f)
                if holder != last:
                    log(f'waiting for the emulator: {holder} holds {path} (polling every '
                        f'{poll:g} s)')
                    last = holder
                time.sleep(poll)
        if last is not None:
            log(f'emulator lock {path} acquired after {time.time() - t0:.0f} s of waiting')
        f.seek(0)
        f.truncate()
        f.write(f'{who} (pid {os.getpid()}, since {time.strftime("%H:%M:%S")})\n')
        f.flush()
        _HELD.append(who)
        _FDS.append(f.fileno())
        try:
            yield
        finally:
            _HELD.clear()
            _FDS.clear()
            f.seek(0)
            f.truncate()
            f.flush()
