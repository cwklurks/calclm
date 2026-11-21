"""Synthetic pilot behavior; generation and emulator are external test boundaries."""
from collections import Counter

from lab import synth


def test_seed_selection_is_balanced_reproducible_and_includes_other():
    seeds = [{"topic": t, "token_count": n, "text": "seed"}
             for t in ("algebra", "geometry", "other")
             for n in (30, 150, 450) for _ in range(8)]
    selected = synth.pick_seeds(seeds, 18, seed=1234)
    assert selected == synth.pick_seeds(seeds, 18, seed=1234)
    assert len(set(selected)) == 18
    assert Counter(seeds[i]["topic"] for i in selected) == {"algebra": 6, "geometry": 6, "other": 6}
    assert Counter(synth.length_bin(seeds[i]["token_count"]) for i in selected) == {0: 6, 1: 6, 2: 6}


def test_generation_uses_one_bounded_deepseek_call_and_recognizes_quota(monkeypatch, tmp_path):
    import json
    import subprocess

    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"type": "text", "part": {
            "text": '```tibasic\nPrompt A\nDisp A*3\n```'}}), '')

    monkeypatch.setattr(subprocess, 'run', run)
    result = synth.generate('private seed prompt', timeout=12)
    assert result['reply'] == '```tibasic\nPrompt A\nDisp A*3\n```'
    assert result['reason'] is None
    assert len(calls) == 1
    assert calls[0][0][:4] == ['opencode', 'run', '-m', 'opencode-go/deepseek-v4.1-flash']
    assert calls[0][1]['timeout'] == 12
    assert '--title' in calls[0][0]  # avoids OpenCode's automatic title-model request
    assert '--dir' not in calls[0][0]                 # opencode v2: cwd is the isolated dir
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, json.dumps({'type': 'error', 'error': {'message': 'Usage limit exceeded'}}), ''))
    assert synth.generate('prompt', timeout=12)['reason'] == 'quota_limit'


def test_bad_reply_unknown_vocab_and_excess_length_are_rejected():
    import pytest
    from corpus.tokenize import Vocab

    assert synth.tokenize_reply('```tibasic\nPrompt A\nDisp A*3\n```')[1][0] == 1
    with pytest.raises(ValueError, match='code_block'):
        synth.tokenize_reply('Prompt A\nDisp A*3')
    with pytest.raises(ValueError, match='out_of_vocab'):
        synth.tokenize_reply('```\nPrompt A\nDisp A*3\n```', vocab=Vocab(['<pad>', '<bos>', '<eos>', '<unk>', '41']))
    with pytest.raises(ValueError, match='too_long'):
        synth.tokenize_reply('```\nPrompt A\nDisp A*3\n```', max_tokens=2)


def test_dedup_checks_val_before_train_and_kept_and_matches_near_programs():
    from corpus.tokenize import Vocab

    v = Vocab.load()
    _, ids = synth.tokenize_reply('```\nPrompt A,B\nDisp A*3+B*4\nStop\n```')
    _, near = synth.tokenize_reply('```\nPrompt A,B\nDisp A*3+B*4\nStop\nDisp 4\n```')
    d = synth.Deduper([('train', ids), ('val', ids)], vocab=v)
    assert d.check(ids)['reason'] == 'duplicate_val'
    assert d.check(near)['reason'] == 'near_duplicate_val'
    d = synth.Deduper([], vocab=v)
    assert d.check(ids) is None
    d.add(ids)
    assert d.check(ids)['reason'] == 'duplicate_synth'
    assert d.check(near)['reason'] == 'near_duplicate_synth'


def test_pilot_stops_after_repeated_limits_and_resumes_without_repeating_calls(tmp_path, monkeypatch):
    import json
    import subprocess

    seeds = tmp_path / 'seeds.jsonl'
    seeds.write_text(json.dumps({'topic': 'algebra', 'token_count': 8, 'text': 'Prompt A\nDisp A*3'}) + '\n')
    calls = []

    def limited(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, '', 'Rate limit exceeded')

    monkeypatch.setattr(subprocess, 'run', limited)
    kw = dict(seeds_path=seeds, out=tmp_path / 'out', data=tmp_path, count=200, refs=[])
    result = synth.run_pilot(**kw)
    assert result['attempts'] == 3 and result['stop_reason'] == 'repeated_quota_limit'
    assert result['kept'] == 0 and len(calls) == 3
    assert synth.run_pilot(**kw)['attempts'] == 3 and len(calls) == 3
    rows = [json.loads(x) for x in (tmp_path / 'out/pilot_attempts.jsonl').read_text().splitlines()]
    assert [r['reason'] for r in rows] == ['quota_limit'] * 3
    assert (tmp_path / 'out/pilot.jsonl').read_text() == ''


def test_pilot_keeps_only_math_and_logs_judgments_and_dedup(tmp_path, monkeypatch):
    import json
    import subprocess
    from eval import run_programs

    seeds = tmp_path / 'seeds.jsonl'
    seeds.write_text(json.dumps({'topic': 'algebra', 'token_count': 8, 'text': 'Prompt A\nDisp A*3'}) + '\n')
    replies = iter(['```\nPrompt A\nDisp A*3\n```', '```\nPrompt A\nDisp A*3\n```',
                    '```\nDisp 42\n```'])
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0,
        json.dumps({'type': 'text', 'part': {'text': next(replies)}}), ''))
    typed = []

    def emulator(seqs, vocab, out, **kw):
        typed.append((kw['input_text'], kw['yields']))
        return [{'index': i, 'outcome': 'OK', 'home_text': [str(6 if kw['input_text'] == '2' else 21)],
                 'actions': {'input': int(i < 2)}} for i, _ in enumerate(seqs)]

    monkeypatch.setattr(run_programs, 'run_token_ids', emulator)
    result = synth.run_pilot(seeds_path=seeds, out=tmp_path / 'out', data=tmp_path, count=3, refs=[])
    assert result['kept'] == 1
    assert typed == [('2', True), ('7', True)]
    attempts = [json.loads(x) for x in (tmp_path / 'out/pilot_attempts.jsonl').read_text().splitlines()]
    assert [x['reason'] for x in attempts] == [None, 'duplicate_synth', 'math_tier']
    assert attempts[2]['classification']['takes_input'] is False
    kept = json.loads((tmp_path / 'out/pilot.jsonl').read_text())
    assert kept['seed_index'] == 0 and kept['mode'] in ('sibling', 'variation')
    assert kept['ids'][0] == 1 and kept['ids'][-1] == 2


def test_corpus_unknown_tokens_do_not_abort_filter_or_join_known_tokens():
    from corpus.tokenize import Vocab

    _, ids = synth.tokenize_reply('```\nPrompt A,B\nDisp A*3+B*4\nStop\n```')
    d = synth.Deduper([('train', [*ids[:-1], 3, 2])], vocab=Vocab.load())
    assert d.check(ids)['reason'] == 'near_duplicate_train'


def test_resume_after_output_checkpoint_preserves_kept_programs(tmp_path, monkeypatch):
    import json
    import subprocess
    from eval import run_programs

    seeds = tmp_path / 'seeds.jsonl'
    seeds.write_text(json.dumps({'topic': 'algebra', 'token_count': 8, 'text': 'seed'}) + '\n')
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0,
        json.dumps({'type': 'text', 'part': {'text': '```\nPrompt A\nDisp A*3\n```'}}), ''))
    monkeypatch.setattr(run_programs, 'run_token_ids', lambda seqs, vocab, out, **kw: [
        {'index': 0, 'outcome': 'OK', 'home_text': ['6' if kw['input_text'] == '2' else '21'],
         'actions': {'input': 1}}])
    kw = dict(seeds_path=seeds, out=tmp_path / 'out', data=tmp_path, count=1, refs=[])
    assert synth.run_pilot(**kw)['kept'] == 1
    (tmp_path / 'out/pilot_summary.json').unlink()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('extra call')))
    assert synth.run_pilot(**kw)['kept'] == 1


def test_codex_generator_calls_gpt_6_1_sol_read_only_and_reads_its_last_message(monkeypatch):
    """The codex generator runs GPT-6.1-Sol read-only and reads its last message."""
    import subprocess
    from pathlib import Path
    from lab import synth
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
        Path(cmd[cmd.index("-o") + 1]).write_text("```\nDisp 1\n```")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    r = synth.generate("write a program", timeout=5, generator="codex")
    cmd = seen["cmd"]
    assert cmd[:2] == ["codex", "exec"] and cmd[cmd.index("-m") + 1] == "gpt-6.1-sol"
    assert cmd[cmd.index("-s") + 1] == "read-only" and "--ephemeral" in cmd
    assert cmd[cmd.index("-C") + 1] == seen["cwd"]
    assert r["reply"] == "```\nDisp 1\n```" and r["reason"] is None


def test_codex_generator_detects_usage_limits(monkeypatch):
    import subprocess
    from lab import synth
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "ERROR: You've hit your usage limit."))
    assert synth.generate("p", timeout=5, generator="codex")["reason"] == "quota_limit"


def test_a_timed_out_codex_call_is_a_timeout_not_a_crash(monkeypatch):
    import subprocess
    from lab import synth

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 5)
    monkeypatch.setattr(subprocess, "run", slow)
    assert synth.generate("p", timeout=5, generator="codex")["reason"] == "timeout"


def test_pi_unicode_minus_and_soft_hyphens_tokenize_like_their_ascii_spellings():
    """2026-10-17: 1,019 of the first 1,024 tokenization rejects in the 3,000 run were the
    pi character (U+03C0), which tivars does not accept; it spells the 0xAC token `pi`.
    A few more were U+2212 MINUS SIGN and soft hyphens from the model's typography."""
    ids = lambda body: synth.tokenize_reply(f'```\n{body}\n```')[1]
    assert ids('Prompt R\nDisp πR²') == ids('Prompt R\nDisp piR²')
    assert ids('Prompt A,B\nDisp A−B') == ids('Prompt A,B\nDisp A-B')
    assert ids('Prompt A\nDis­p A') == ids('Prompt A\nDisp A')
