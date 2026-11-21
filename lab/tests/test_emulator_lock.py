"""The cross-experiment emulator lock (lab/emulator_lock.py): another process's emulator stage is
waited for (polling, with a log line), never overlapped."""
import subprocess
import sys
import time
from pathlib import Path

from lab import emulator_lock as EL

HOLD = ('import fcntl, sys, time\n'
        'f = open(sys.argv[1], "a+")\n'
        'fcntl.flock(f, fcntl.LOCK_EX)\n'
        'f.seek(0); f.truncate(); f.write("E13 verify (pid x)\\n"); f.flush()\n'
        'print("held", flush=True)\n'
        'time.sleep(float(sys.argv[2]))\n')


def holder(lock: Path, seconds: float):
    lock.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([sys.executable, '-c', HOLD, str(lock), str(seconds)],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'held'
    return p


def test_the_shared_lock_is_the_main_checkouts_runs_locks_file():
    assert EL.LOCK == Path(__file__).resolve().parents[2] / 'runs' / 'locks' / 'emulator.lock'
    assert EL.POLL_S == 60


def test_a_free_lock_is_taken_at_once_and_names_its_holder(tmp_path):
    lock = tmp_path / 'locks' / 'emulator.lock'
    lines = []
    with EL.emulator_stage('E11 L8-s12 emulator', log=lines.append, lock=lock):
        assert EL.held()
        assert lock.read_text().startswith('E11 L8-s12 emulator (pid ')
    assert not EL.held() and lines == [] and lock.read_text() == ''


def test_another_processs_stage_is_waited_for_with_a_log_line(tmp_path):
    lock = tmp_path / 'locks' / 'emulator.lock'
    p = holder(lock, 1.0)
    lines, t0 = [], time.time()
    try:
        with EL.emulator_stage('E11 check 5', log=lines.append, poll_s=.05, lock=lock):
            waited = time.time() - t0                     # the other stage had ended
    finally:
        p.kill()
        p.wait()
    assert waited >= .8
    assert lines[0].startswith('waiting for the emulator: E13 verify (pid x) holds')
    assert 'acquired after' in lines[-1]
    assert sum(x.startswith('waiting') for x in lines) == 1      # once per holder, not per poll


def test_the_lock_is_re_entrant_within_a_process(tmp_path):
    lock = tmp_path / 'emulator.lock'
    with EL.emulator_stage('outer', lock=lock):
        with EL.emulator_stage('inner', lock=lock):
            assert EL.held()
        assert EL.held()
    assert not EL.held()


def test_a_second_process_waits_while_this_one_holds_it(tmp_path):
    lock = tmp_path / 'emulator.lock'
    probe = ('import fcntl, sys\n'
             'f = open(sys.argv[1], "a+")\n'
             'try:\n'
             '    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); print("free")\n'
             'except BlockingIOError:\n'
             '    print("busy")\n')
    with EL.emulator_stage('E11', lock=lock):
        out = subprocess.run([sys.executable, '-c', probe, str(lock)], capture_output=True,
                             text=True).stdout.strip()
    assert out == 'busy'
    out = subprocess.run([sys.executable, '-c', probe, str(lock)], capture_output=True,
                         text=True).stdout.strip()
    assert out == 'free'


def test_a_child_started_in_the_stage_keeps_the_lock_until_it_exits(tmp_path):
    """A runner killed mid-stage releases its own descriptor, but the emulator child it started
    (given the lock's descriptor) keeps the stage locked until it exits too."""
    lock = tmp_path / 'emulator.lock'
    probe = ('import fcntl, sys\n'
             'f = open(sys.argv[1], "a+")\n'
             'try:\n'
             '    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); print("free")\n'
             'except BlockingIOError:\n'
             '    print("busy")\n')
    state = lambda: subprocess.run([sys.executable, '-c', probe, str(lock)],      # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    with EL.emulator_stage('E11', lock=lock):
        assert len(EL.held_fds()) == 1
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5)'],
                                 pass_fds=EL.held_fds())
    assert EL.held_fds() == ()
    try:
        assert state() == 'busy'                     # the stage's child still runs
    finally:
        child.wait()
    assert state() == 'free'
