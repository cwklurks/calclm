"""Scale generation behavior, with generation and emulator calls mocked."""
import json

import pytest

from lab import synth


@pytest.fixture
def scale_run(tmp_path, monkeypatch):
    from lab import score

    seeds = tmp_path / 'seeds.jsonl'
    seeds.write_text(json.dumps({'topic': 'geometry', 'token_count': 10,
                                 'text': 'Prompt A\nDisp A*3', 'train_index': 17}) + '\n')
    def no_real_generation(*args, **kwargs):
        raise AssertionError('generation must be mocked')
    monkeypatch.setattr(synth, 'generate', no_real_generation)
    judgings = []

    def emulator(seqs, out, **kw):
        judgings.append((len(seqs), kw['input_text'], kw['yields']))
        return [{'index': i, 'outcome': 'OK', 'home_text': [
                    '6' if kw['input_text'] == '2' else '21'], 'actions': {'input': 1}}
                for i in range(len(seqs))]

    monkeypatch.setattr(score, 'run_emulator', emulator)
    return dict(seeds_path=seeds, out=tmp_path / 'out', data=tmp_path,
                refs=[], generator='codex', parallel=1, judge_every=2,
                target_kept=100, max_attempts=5), judgings


def replies(monkeypatch, programs):
    stream = iter(programs)
    calls = []

    def generate(prompt, **kw):
        calls.append((prompt, kw))
        return {'reply': f'```\n{next(stream)}\n```', 'reason': None, 'wall_seconds': 0.01}

    monkeypatch.setattr(synth, 'generate', generate)
    return calls


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_computation_keys_ignore_labels_and_variable_names_but_keep_formulas():
    original = 'Disp "AREA"\nPrompt A,B\nA*B/2→C\nDisp "RESULT",C'
    renamed = 'Disp "TRIANGLE"\nInput "X",X\nPrompt Y\nX*Y/2→Z\nOutput(3,5,Z)'
    assert synth.computation_key(original) == synth.computation_key(renamed)
    assert synth.computation_key(original) != synth.computation_key(renamed.replace('/2', '/3'))
    assert synth.computation_key('Disp A+B') != synth.computation_key('Disp A*B')
    assert synth.computation_key('Disp A,A') != synth.computation_key('Disp A,B')
    assert synth.computation_key('Disp sin(A)') != synth.computation_key('Disp cos(A)')
    assert synth.computation_key('Disp θ*2') == synth.computation_key('Disp Z*2')
    assert synth.computation_key('Disp A+1\nDisp A*2') != synth.computation_key('Disp A*2\nDisp A+1')


def test_computation_keys_handle_strings_separators_and_nested_output_arguments():
    assert synth.computation_key('Disp "A: B, 42→C":Disp A*2') == synth.computation_key('Disp X*2')
    assert synth.computation_key('Disp "UNCLOSED\nDisp A*2') == synth.computation_key('Disp X*2')
    assert synth.computation_key('Output(1,2,max(A,B))') == synth.computation_key('Disp max(X,Y)')
    assert synth.computation_key('Disp "ONLY A TITLE"') == '[]'


def test_seed_cycles_cover_every_seed_before_reuse_and_diversity_prompt_is_bounded():
    seeds = [{'topic': t, 'token_count': n, 'text': 'Prompt A\nDisp A*3'}
             for t in ('geometry', 'other') for n in (40, 200, 500)]
    plan = synth.pick_seeds(seeds, 15)
    assert set(plan[:6]) == set(plan[6:12]) == set(range(6))
    assert plan == synth.pick_seeds(seeds, 15)
    prompt = synth.make_prompt(seeds[1], 'variation', [f'TASK {i}' for i in range(45)])
    assert 'Topic: geometry' in prompt
    assert 'TASK 44' in prompt and 'TASK 5' in prompt and 'TASK 4"' not in prompt
    assert 'not in' in prompt.lower()
    assert synth.program_title('Disp "CIRCLE AREA"\nDisp A*2') == 'CIRCLE AREA'
    assert synth.program_title('Disp A\nDisp "SECOND TITLE') == 'SECOND TITLE'
    assert synth.program_title('Input "PROMPT",A\nDisp A*2') is None


def test_scale_judges_batches_and_stops_at_maximum_with_model_and_progress(scale_run, monkeypatch):
    kw, judgings = scale_run
    calls = replies(monkeypatch, [f'Prompt A\nDisp A*{n}' for n in (3, 5, 11, 13, 17)])
    result = synth.run_scale(**kw)
    assert result['attempts'] == result['generated'] == len(calls) == 5
    assert result['stop_reason'] == 'max_attempts'
    assert result['model'] == 'gpt-6.1-sol'
    assert judgings == [(2, '2', True), (2, '7', True)] * 2 + [(1, '2', True), (1, '7', True)]
    progress = read_rows(kw['out'] / 'progress.jsonl')
    assert [p['generated'] for p in progress] == [2, 4, 5]
    assert all('keep_rate' in p and 'top_keys' in p for p in progress)
    attempts = read_rows(kw['out'] / 'attempts.jsonl')
    assert [r['mode'] for r in attempts] == ['sibling', 'variation', 'sibling', 'variation', 'sibling']
    assert all(r['model'] == 'gpt-6.1-sol' for r in attempts)
    assert synth.run_scale(**kw) == result
    assert len(calls) == 5


def test_scale_caps_formula_keys_and_prompts_from_its_own_kept_titles(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(judge_every=1, max_attempts=4)
    calls = replies(monkeypatch, [
        'Disp "TRIANGLE AREA"\nInput "BASE",A\nDisp A*3',
        'Disp "ALPHA"\nPrompt X\nOutput(2,3,X*3)',
        'ClrHome\nInput "HEIGHT",θ\nDisp "RESULT",θ*3',
        'Disp "CUBE VOLUME"\nPrompt R\nDisp R^3',
    ])
    result = synth.run_scale(**kw)
    assert result['kept'] == 3
    attempts = read_rows(kw['out'] / 'attempts.jsonl')
    assert [r['reason'] for r in attempts] == [None, None, 'per_key_cap', None]
    assert sorted(result['key_counts'].values()) == [1, 2]
    assert json.loads((kw['out'] / 'key_counts.json').read_text()) == result['key_counts']
    assert 'TRIANGLE AREA' not in calls[0][0]
    assert 'TRIANGLE AREA' in calls[1][0] and 'ALPHA' in calls[2][0]
    assert 'RESULT' not in calls[3][0]  # The capped program was never kept.


def test_scale_stops_exactly_at_target_and_does_not_generate_another_batch(scale_run, monkeypatch):
    kw, judgings = scale_run
    kw.update(target_kept=1, max_attempts=20)
    calls = replies(monkeypatch, ['Prompt A\nDisp A*3', 'Prompt X\nDisp X^3'])
    result = synth.run_scale(**kw)
    assert result['kept'] == 1 and result['stop_reason'] == 'target_kept'
    assert len(calls) == result['attempts'] == 2
    assert len(read_rows(kw['out'] / 'kept.jsonl')) == 1
    assert read_rows(kw['out'] / 'attempts.jsonl')[1]['reason'] == 'target_reached'


def test_parallel_calls_have_intents_before_submission_and_obey_batch_barriers(scale_run, monkeypatch):
    import threading

    kw, judgings = scale_run
    kw.update(parallel=2, max_attempts=4)
    barrier, lock = threading.Barrier(2), threading.Lock()
    prompts = []

    def generate(prompt, **opts):
        with lock:
            i = len(prompts)
            prompts.append(prompt)
        assert len(list((kw['out'] / 'attempts').glob('*.json'))) >= i + 1
        barrier.wait(timeout=3)
        return {'reply': f'```\nDisp "TASK {i}"\nPrompt A\nDisp A*{i + 3}\n```',
                'reason': None, 'wall_seconds': 0.01}

    monkeypatch.setattr(synth, 'generate', generate)
    result = synth.run_scale(**kw)
    assert len(prompts) == result['attempts'] == 4
    assert result['kept'] == 4
    assert len(judgings) == 4
    assert all('Tasks already kept' not in p for p in prompts[:2])
    assert all('TASK 0' in p and 'TASK 1' in p for p in prompts[2:])


def test_parallel_resume_never_repeats_intents_and_recovers_exports(scale_run, monkeypatch):
    import re
    import threading
    from collections import Counter

    kw, judgings = scale_run
    kw.update(parallel=3, max_attempts=6, judge_every=3)
    kw['seeds_path'].write_text(''.join(json.dumps(dict(
        topic='geometry', token_count=10, text=f'Disp "SEED {n}"')) + '\n' for n in range(6)))
    failure_seed = synth.pick_seeds([dict(topic='geometry', token_count=10)] * 6, 6)[1]
    barrier = threading.Barrier(3)
    seen, phase = [], ['interrupt']

    def generate(prompt, **opts):
        n = int(re.search(r'<seed>\nDisp "SEED (\d+)"', prompt)[1])
        seen.append(n)
        if phase[0] == 'interrupt':
            barrier.wait(timeout=3)
            if n == failure_seed:
                raise RuntimeError('simulated worker interruption')
        return dict(reply=f'```\nPrompt A\nDisp A*{n + 20}\n```', reason=None, wall_seconds=0.01)

    monkeypatch.setattr(synth, 'generate', generate)
    with pytest.raises(RuntimeError, match='interruption'):
        synth.run_scale(**kw)
    checkpoints = [json.loads(p.read_text()) for p in (kw['out'] / 'attempts').glob('*.json')]
    assert Counter(r['reason'] for r in checkpoints) == {'pending_judge': 2, 'interrupted_call': 1}
    phase[0] = 'resume'
    result = synth.run_scale(**kw)
    assert len(seen) == len(set(seen)) == result['attempts'] == 6
    assert result['generated'] == 5
    assert result['rejections']['interrupted_call'] == 1
    saved_judgings = list(judgings)
    saved_kept = (kw['out'] / 'kept.jsonl').read_text()
    for name in ('kept.jsonl', 'attempts.jsonl', 'progress.jsonl'):
        (kw['out'] / name).write_text('')
    (kw['out'] / 'scale_summary.json').unlink()
    assert synth.run_scale(**kw) == result
    assert (kw['out'] / 'kept.jsonl').read_text() == saved_kept
    assert len(read_rows(kw['out'] / 'progress.jsonl')) == 2
    assert judgings == saved_judgings and len(seen) == 6


@pytest.mark.parametrize('reasons', [
    ['quota_limit'] * 3,
    ['quota_limit', 'quota_limit', 'generation_error', 'quota_limit', 'quota_limit', 'quota_limit'],
])
def test_scale_stops_on_three_consecutive_quota_errors_and_resumes_without_calls(
        scale_run, monkeypatch, reasons):
    kw, judgings = scale_run
    kw.update(max_attempts=25, judge_every=10)
    remaining = iter(reasons)
    calls = []

    def generate(*args, **opts):
        calls.append(1)
        return dict(reply='', reason=next(remaining), wall_seconds=0.01)

    monkeypatch.setattr(synth, 'generate', generate)
    result = synth.run_scale(**kw)
    assert result['attempts'] == len(calls) == len(reasons)
    assert result['stop_reason'] == 'repeated_quota_limit'
    assert result['generated'] == result['kept'] == 0
    assert judgings == []
    (kw['out'] / 'scale_summary.json').unlink()
    assert synth.run_scale(**kw) == result
    assert len(calls) == len(reasons)


def test_cli_defaults_to_scale_and_exposes_all_controls(monkeypatch, capsys):
    from pathlib import Path

    calls = []
    monkeypatch.setattr(synth, 'run_pilot', lambda **kw: pytest.fail('unexpected pilot call'))
    monkeypatch.setattr(synth, 'run_scale', lambda **kw: calls.append(kw) or {})
    synth.main([])
    defaults = calls.pop()
    assert {k: defaults[k] for k in ('parallel', 'judge_every', 'per_key',
                                   'target_kept', 'max_attempts', 'generator')} == dict(
        parallel=6, judge_every=600, per_key=2, target_kept=10_000,
        max_attempts=25_000, generator='codex')
    assert defaults['out'] == Path(synth.__file__).resolve().parents[1] / 'data/synth/scale'
    synth.main(['--parallel', '2', '--judge-every', '4', '--per-key', '1',
                '--target-kept', '8', '--max-attempts', '12', '--out', '/tmp/scale-test'])
    custom = calls.pop()
    assert [custom[k] for k in ('parallel', 'judge_every', 'per_key', 'target_kept',
                               'max_attempts')] == [2, 4, 1, 8, 12]
    assert custom['out'] == Path('/tmp/scale-test')


def test_judge_every_counts_generated_replies_and_flushes_final_partial_batch(scale_run, monkeypatch):
    kw, judgings = scale_run
    results = iter([
        dict(reply='', reason='generation_error'),
        dict(reply='invalid unfenced reply', reason=None),
        dict(reply='```\nPrompt A\nDisp A*3\n```', reason=None),
        dict(reply='', reason='timeout'),
        dict(reply='```\nPrompt X\nDisp X^3\n```', reason=None),
    ])
    monkeypatch.setattr(synth, 'generate', lambda *a, **kw: next(results))
    result = synth.run_scale(**kw)
    assert result['attempts'] == 5 and result['generated'] == 3
    progress = read_rows(kw['out'] / 'progress.jsonl')
    assert [r['generated'] for r in progress] == [2, 3]
    assert [r['attempts'] for r in progress] == [3, 5]
    assert judgings == [(1, '2', True), (1, '7', True)] * 2


def test_parallel_quota_stop_drains_paid_calls_and_stays_stopped_after_late_success(scale_run, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    kw, _ = scale_run
    kw.update(parallel=4, judge_every=10, max_attempts=50)
    release, lock, calls = threading.Event(), threading.Lock(), []

    def generate(*args, **opts):
        with lock:
            i = len(calls)
            calls.append(i)
        if i < 3:
            return dict(reply='', reason='quota_limit')
        assert release.wait(timeout=5)
        return dict(reply='```\nPrompt A\nDisp A*3\n```', reason=None)

    monkeypatch.setattr(synth, 'generate', generate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(synth.run_scale, **kw)
        try:
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                rows = [json.loads(p.read_text()) for p in (kw['out'] / 'attempts').glob('*.json')]
                if sum(r['reason'] == 'quota_limit' for r in rows) == 3:
                    break
                release.wait(0.01)
            else:
                pytest.fail('quota results were not checkpointed')
        finally:
            release.set()
        result = future.result(timeout=5)
    assert 3 <= len(calls) <= 6  # Three quota results, at most parallel-1 in flight.
    assert result['attempts'] == len(calls)
    assert result['stop_reason'] == 'repeated_quota_limit'
    (kw['out'] / 'scale_summary.json').unlink()
    assert synth.run_scale(**kw) == result
    assert len(calls) == result['attempts']


def test_scale_preserves_math_and_train_val_and_kept_duplicate_filters(scale_run, monkeypatch):
    kw, _ = scale_run
    programs = ['Prompt A\nDisp A*3', 'Input "X=",X\nDisp X^2', 'Disp 42',
                'Prompt A,B\nDisp (A+B)^3', 'Prompt A,B\nDisp (A+B)^3']
    _, val_ids = synth.tokenize_reply(f'```\n{programs[0]}\n```')
    _, train_ids = synth.tokenize_reply(f'```\n{programs[1]}\n```')
    kw.update(refs=[('train', val_ids), ('val', val_ids), ('train', train_ids)], per_key=10)
    replies(monkeypatch, programs)
    result = synth.run_scale(**kw)
    assert result['kept'] == 1
    assert [r['reason'] for r in read_rows(kw['out'] / 'attempts.jsonl')] == [
        'duplicate_val', 'duplicate_train', 'math_tier', None, 'duplicate_synth']


@pytest.mark.parametrize('indices', [[], [0, 0], [0, 1]])
def test_bad_judge_indices_abort_batch_and_resume_without_generation(scale_run, monkeypatch, indices):
    from lab import score

    kw, _ = scale_run
    kw.update(max_attempts=1)
    calls = replies(monkeypatch, ['Prompt A\nDisp A*3'])
    good_emulator = score.run_emulator

    def bad_emulator(seqs, out, **opts):
        row = good_emulator(seqs, out, **opts)[0]
        return [dict(row, index=i) for i in indices]

    monkeypatch.setattr(score, 'run_emulator', bad_emulator)
    with pytest.raises(ValueError, match='emulator result'):
        synth.run_scale(**kw)
    assert not list((kw['out'] / 'batches').glob('*.json'))
    monkeypatch.setattr(score, 'run_emulator', good_emulator)
    assert synth.run_scale(**kw)['kept'] == 1
    assert len(calls) == 1


def test_changed_resume_settings_are_rejected_before_generation(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(max_attempts=1)
    calls = replies(monkeypatch, ['Prompt A\nDisp A*3'])
    synth.run_scale(**kw)
    with pytest.raises(ValueError, match='resume configuration differs'):
        synth.run_scale(**dict(kw, per_key=3))
    assert len(calls) == 1


def test_glm_generator_runs_opencode_with_glm_5_3(monkeypatch):
    """The GLM generator runs GLM-5.3 through OpenCode Go."""
    import subprocess
    seen = {}

    def run(cmd, **kw):
        seen['cmd'] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(
            {'type': 'text', 'part': {'text': '```\nDisp 1\n```'}}) + '\n', '')
    monkeypatch.setattr(subprocess, 'run', run)
    r = synth.generate('p', timeout=5, generator='glm')
    assert seen['cmd'][:2] == ['opencode', 'run']
    assert seen['cmd'][seen['cmd'].index('-m') + 1] == 'opencode-go/glm-5.3'
    assert r['reason'] is None and 'Disp 1' in r['reply']


def test_a_quota_stop_on_one_generator_resumes_on_another(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(max_attempts=6, judge_every=1, generator='glm')

    def quota(*args, **opts):
        return dict(reply='', reason='quota_limit', wall_seconds=0.01)
    monkeypatch.setattr(synth, 'generate', quota)
    first = synth.run_scale(**kw)
    assert first['stop_reason'] == 'repeated_quota_limit' and first['kept'] == 0
    (kw['out'] / 'scale_summary.json').unlink()
    calls = replies(monkeypatch, ['Prompt A\nDisp A*3', 'Prompt B\nDisp B+5', 'Prompt C\nDisp C/2'])
    second = synth.run_scale(**dict(kw, generator='codex'))
    assert len(calls) == 3 and second['kept'] >= 1
    rows = [json.loads(p.read_text()) for p in sorted((kw['out'] / 'attempts').glob('*.json'))]
    assert {r['model'] for r in rows} == {'opencode-go/glm-5.3', 'gpt-6.1-sol'}
    history = json.loads((kw['out'] / 'generators.json').read_text())
    assert [h['model'] for h in history] == ['opencode-go/glm-5.3', 'gpt-6.1-sol']


def test_opencode_v2_runs_from_its_isolated_directory_without_dir(monkeypatch):
    """opencode v2 (2.0.32) has no --dir; the call's cwd is the empty temp directory."""
    import subprocess
    seen = {}

    def run(cmd, **kw):
        seen['cmd'], seen['cwd'] = cmd, kw.get('cwd')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', run)
    synth.generate('p', timeout=5, generator='glm')
    assert '--dir' not in seen['cmd'] and seen['cwd'] and 'calclm-synth-' in seen['cwd']


def test_an_expired_or_invalid_key_stops_like_a_quota(monkeypatch):
    """2026-10-17: Go's key had expired; every call failed at once and must not burn attempts."""
    import subprocess
    event = {'type': 'error', 'error': {'type': 'provider.auth',
                                        'message': 'This API key has expired. Create a new API key'}}
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, json.dumps(event) + '\n', ''))
    assert synth.generate('p', timeout=5, generator='glm')['reason'] == 'quota_limit'


def test_a_successful_reply_is_not_a_quota_even_if_its_log_contains_429(monkeypatch):
    """2026-10-17: codex logs 'tokens used 14,429', and \\b429\\b flagged a good reply."""
    import subprocess
    from pathlib import Path
    reply = '```text\nDisp 1\n```'

    def run(cmd, **kw):
        Path(cmd[cmd.index('-o') + 1]).write_text(reply)
        return subprocess.CompletedProcess(cmd, 0, reply, 'codex\n' + reply + '\ntokens used\n14,429\n')
    monkeypatch.setattr(subprocess, 'run', run)
    r = synth.generate('p', timeout=5, generator='codex')
    assert r['reply'] == reply and r['reason'] is None


def test_the_target_can_change_on_resume_and_the_run_stops_at_the_new_one(scale_run, monkeypatch):
    """A resumed run can take a lower target and stops there, without starting over."""
    kw, _ = scale_run
    kw.update(target_kept=1, max_attempts=20)
    replies(monkeypatch, ['Prompt A\nDisp A*3', 'Prompt X\nDisp X^3'])
    assert synth.run_scale(**kw)['kept'] == 1
    (kw['out'] / 'scale_summary.json').unlink(missing_ok=True)
    calls = replies(monkeypatch, ['Prompt B\nDisp B-7', 'Prompt C\nDisp C^2+1'])
    result = synth.run_scale(**dict(kw, target_kept=2))
    assert result['kept'] == 2 and result['stop_reason'] == 'target_kept'
    assert json.loads((kw['out'] / 'scale_config.json').read_text())['target_kept'] == 2
    assert calls


def test_topic_steering_can_resume_a_legacy_run_without_repeating_calls(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(max_attempts=6, target_kept=1)
    replies(monkeypatch, ['Disp "CIRCLE AREA"\nPrompt A\nDisp A*3',
                          'Prompt B\nDisp B^3'])
    assert synth.run_scale(**kw)['kept'] == 1
    journal = kw['out'] / 'attempts'
    before = {p.name: p.read_bytes() for p in journal.glob('*.json')}
    config_path = kw['out'] / 'scale_config.json'
    config = json.loads(config_path.read_text())
    config.pop('steer_topics', None)  # Actual pre-steering run format.
    config_path.write_text(json.dumps(config))
    (kw['out'] / 'scale_summary.json').unlink()
    calls = replies(monkeypatch, ['Disp "NEW TASK"\nPrompt X\nDisp X+11',
                                 'Prompt Y\nDisp Y/13'])
    result = synth.run_scale(**dict(kw, steer_topics=True, target_kept=2))
    assert result['kept'] == 2 and len(calls) == 2
    assert all((journal / name).read_bytes() == value for name, value in before.items())
    attempts = read_rows(kw['out'] / 'attempts.jsonl')
    assert 'subtopic' not in attempts[0]
    assert attempts[2]['subtopic'] != attempts[3]['subtopic']
    assert attempts[2]['subtopic'] in calls[0][0]
    assert attempts[2]['topic'] in calls[0][0]
    assert attempts[2]['seed_topic'] == 'geometry'
    assert 'subtopic' in read_rows(kw['out'] / 'kept.jsonl')[-1]
    manifest = json.loads((kw['out'] / 'topic_steering.json').read_text())
    assert 150 <= len(manifest['catalog']) <= 200
    synth.run_scale(**dict(kw, steer_topics=True, target_kept=2))
    assert len(calls) == 2


def test_steering_resume_keeps_allocations_after_an_interrupted_call(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(steer_topics=True, max_attempts=4, judge_every=1)
    calls = []

    def generate(prompt, **opts):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return dict(reply='```\nDisp "COMPUTED"\nPrompt A\nDisp A*3\n```', reason=None)

    monkeypatch.setattr(synth, 'generate', generate)
    with pytest.raises(RuntimeError, match='interrupted'):
        synth.run_scale(**kw)
    later = replies(monkeypatch, ['Prompt A\nDisp A+11', 'Prompt A\nDisp A^3+1'])
    result = synth.run_scale(**kw)
    assert result['attempts'] == 4 and len(later) == 2
    rows = read_rows(kw['out'] / 'attempts.jsonl')
    assert rows[1]['reason'] == 'interrupted_call'
    assert len({r['subtopic'] for r in rows}) == 4


@pytest.mark.parametrize('text, shifted', [
    ('Prompt A\nabs(A)+1→A\nDisp A*3', True),
    ('Prompt A\n1/(2+abs(A))→P\nDisp P', True),
    ('Prompt N\n1+int(abs(N))→N\nDisp N!', True),
    ('Input "RATE SEED",R\nDisp R*2', True),
    ('Prompt A\nabs(A)→A\nDisp √(A)', False),
    ('Prompt A\nmax(1,int(abs(A)))→A\nDisp A!', False),
    ('Prompt N\nint(log(abs(N)))+1→D\nDisp D', False),
    ('Prompt V,W\nabs(V)+abs(W)→V\nDisp V', False),
    ('Prompt N\nabs(int(N))+1→N\nDisp N^2', True),      # final audit: nested parentheses
    ('Prompt N\nabs(iPart(N))+2→N\nDisp N^2', True),
])
def test_input_shift_flags_a_constant_added_to_an_input(text, shifted):
    """Audit 2026-10-18: steered programs replaced inputs by |x|+1 to pass the emulator check."""
    assert synth.shifts_input(text) is shifted


def test_a_shifted_reply_is_rejected_before_judging(scale_run, monkeypatch):
    kw, judgings = scale_run
    kw.update(target_kept=1, max_attempts=1)
    replies(monkeypatch, ['Prompt A\nabs(A)+1→A\nDisp A*3'])
    result = synth.run_scale(**kw)
    assert result['kept'] == 0 and judgings == []
    assert read_rows(kw['out'] / 'attempts.jsonl')[0]['reason'] == 'input_shift'


def test_resume_drops_kept_shifted_programs_and_keeps_new_ones(scale_run, monkeypatch):
    kw, _ = scale_run
    kw.update(target_kept=1, max_attempts=20)
    monkeypatch.setattr(synth, 'shifts_input', lambda text: False)   # the run before the fix
    replies(monkeypatch, ['Prompt A\nabs(A)+1→A\nDisp A*3', 'Prompt X\nDisp X^3'])
    assert synth.run_scale(**kw)['kept'] == 1
    monkeypatch.undo()                     # also undoes the fixture's stubs; restore them
    (kw['out'] / 'scale_summary.json').unlink(missing_ok=True)
    from lab import score
    monkeypatch.setattr(score, 'run_emulator', lambda seqs, out, **k: [
        {'index': i, 'outcome': 'OK', 'home_text': ['6' if k['input_text'] == '2' else '21'],
         'actions': {'input': 1}} for i in range(len(seqs))])
    replies(monkeypatch, ['Prompt B\nDisp B*3', 'Prompt C\nDisp C^2'])   # one judge batch
    result = synth.run_scale(**kw)
    assert result['kept'] == 1
    assert [r['text'] for r in read_rows(kw['out'] / 'kept.jsonl')] == ['Prompt B\nDisp B*3']
    assert read_rows(kw['out'] / 'attempts.jsonl')[0]['reason'] == 'input_shift'


def test_the_steering_prompt_forbids_changing_inputs():
    s = {'topic': 'physics', 'text': 'Prompt M\nDisp M*9.8'}
    prompt = synth.make_prompt(s, 'sibling', subtopic={'topic': 'physics', 'label': 'weight'})
    assert 'derive valid values' not in prompt
    assert "Never change or replace the user's inputs" in synth.make_prompt(s, 'variation')
    assert "Never change or replace the user's inputs" in prompt


def test_resume_filters_unjudged_replies_saved_by_older_code(scale_run, monkeypatch):
    """Review of 5114f57: a run killed before judging leaves pending_judge rows."""
    kw, _ = scale_run
    kw.update(target_kept=5, max_attempts=1)
    monkeypatch.setattr(synth, 'shifts_input', lambda text: False)   # the code before the fix
    replies(monkeypatch, ['Prompt A\nabs(A)+1→A\nDisp A*3'])
    synth.run_scale(**kw)
    for path in (kw['out'] / 'batches').glob('*.json'):
        path.unlink()                       # as if killed before its batch was judged
    (kw['out'] / 'scale_summary.json').unlink()
    journal = kw['out'] / 'attempts' / '000000.json'
    row = json.loads(journal.read_text())
    journal.write_text(json.dumps({k: v for k, v in row.items()
                                   if k not in ('runs', 'classification')}
                                  | {'reason': 'pending_judge'}))
    monkeypatch.undo()                      # also undoes the fixture's stubs; restore the judge
    from lab import score
    judged = []
    monkeypatch.setattr(score, 'run_emulator', lambda seqs, out, **k: judged.append(seqs) or [
        {'index': i, 'outcome': 'OK', 'home_text': ['6' if k['input_text'] == '2' else '21'],
         'actions': {'input': 1}} for i in range(len(seqs))])
    assert synth.run_scale(**kw)['kept'] == 0 and judged == []
    assert read_rows(kw['out'] / 'attempts.jsonl')[0]['reason'] == 'input_shift'
