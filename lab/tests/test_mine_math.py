"""Mining uses synthetic programs and paired yield records, never corpus fixtures."""
import json

import pytest

from lab import mine_math as M
from lab.tests.test_function import NL, Q, rec, tok
from lab.tests.test_mathtier import _run
from model.tokens import EOS


def test_mines_passing_trimmed_program_with_global_index_and_provenance(tmp_path):
    # The empty middle sample is not sent to the emulator. Record order is shuffled.
    geometry = tok(Q, *'TRIANGLE', Q, NL, 'DC', 'A', NL, 'DE', 'A', '82', '3') + [EOS]
    samples = [geometry, [1], tok('DE', '6') + [EOS]]
    a = [rec('OK', home=['6'], actions={'input': 1}, index=0),
         rec('OK', home=['6'], index=1)]
    b = [rec('OK', home=['21'], actions={'input': 1}, index=0),
         rec('OK', home=['6'], index=1)]
    da, db = _run(tmp_path / 'a', a[::-1], '2'), _run(tmp_path / 'b', b, '7')
    provenance = [{'source': 'ticalc', 'file': 'triangle.zip', 'name': 'TRI'}] + [{}, {}]
    rows = M.mine_samples(samples, da, db, provenance, start_index=500)
    assert len(rows) == 1
    row = rows[0]
    assert row['train_index'] == 500
    assert row['token_count'] == len(geometry) - 2
    assert row['source'] == 'ticalc' and row['file'] == 'triangle.zip'
    assert row['outcome'] == {'2': 'OK', '7': 'OK'}
    assert row['visible_results'] == {'2': ['6'], '7': ['21']}
    assert row['topic'] == 'geometry'
    assert 'TRIANGLE' in row['text'] and '<bos>' not in row['text']
    assert json.loads(json.dumps(row)) == row


@pytest.mark.parametrize('label,prompt', [
    ('geometry', 'CIRCLE RADIUS'), ('algebra', 'QUADRATIC EQUATION'),
    ('physics', 'VELOCITY'), ('finance', 'COMPOUND INTEREST'),
    ('statistics', 'STANDARD DEVIATION'), ('conversion', 'CELSIUS TO FAHRENHEIT'),
    ('number theory', 'PRIME FACTOR'), ('chemistry', 'MOLAR MASS'),
    ('number theory', 'PRIMES TO'),
    ('other', 'HELLO'), ('other', 'APRIL'),
])
def test_topics_use_words_in_strings_not_variable_names(label, prompt):
    ids = tok(Q, *[c if c != ' ' else '29' for c in prompt], Q, NL, 'DC', 'A')
    assert M.topic(ids) == label
    assert M.topic(tok('D5', 'A', 'D5', 'I', 'L')) == 'other'


def test_mining_uses_trimmed_text_and_rejects_wrong_run_settings(tmp_path):
    head = tok('DC', 'A', NL, 'DE', 'A', '82', '3', NL)
    samples = [head + tok(Q, *'CIRCLE', Q)[1:] + [EOS]]
    a = [rec('TIMEOUT', home=['6'], actions={'input': 1}, index=0)]
    b = [rec('OK', home=['21'], actions={'input': 1}, index=0)]
    da, db = _run(tmp_path / 'a', a, '2', cap=len(head)), _run(tmp_path / 'b', b, '7', cap=len(head))
    rows = M.mine_samples(samples, da, db, [{}], cap=len(head))
    assert rows[0]['token_count'] == len(head) - 2  # BOS and trailing newline removed.
    assert 'CIRCLE' not in rows[0]['text'] and rows[0]['topic'] == 'other'
    with pytest.raises(ValueError, match='input_text'):
        M.mine_samples(samples, db, da, [{}], cap=len(head))
    with pytest.raises(ValueError, match='max_len'):
        M.mine_samples(samples, da, db, [{}])


def test_cli_keeps_global_indices_across_shards_and_writes_numeric_summary(tmp_path):
    shards, runs = tmp_path / 'shards', tmp_path / 'runs'
    shards.mkdir()
    samples = [tok('DC', 'A', NL, 'DE', 'A', '82', '3') + [EOS],
               tok('DC', 'B', NL, 'DE', 'B', '82', '3') + [EOS]]
    for i, sample in enumerate(samples):
        (shards / f'{i:03d}.samples.jsonl').write_text(json.dumps({'ids': sample}) + '\n')
        _run(runs / f'{i:03d}' / '2',
             [rec('OK', home=['6'], actions={'input': 1}, index=0)], '2')
        _run(runs / f'{i:03d}' / '7',
             [rec('OK', home=['21'], actions={'input': 1}, index=0)], '7')
    meta = tmp_path / 'programs.jsonl'
    meta.write_text('\n'.join(json.dumps(x) for x in [
        {'split': 'train', 'offset': len(samples[0]), 'n_tokens': len(samples[1]) - 2, 'source': 'github'},
        {'split': 'val', 'offset': 0, 'n_tokens': 10, 'source': 'excluded'},
        {'split': 'train', 'offset': 0, 'n_tokens': len(samples[0]) - 2, 'source': 'ticalc'},
    ]))
    seeds, summary = tmp_path / 'seeds.jsonl', tmp_path / 'summary.json'
    M.main(['--samples-dir', str(shards), '--runs-dir', str(runs), '--programs', str(meta),
            '--out', str(seeds), '--summary', str(summary)])
    rows = [json.loads(x) for x in seeds.read_text().splitlines()]
    assert [r['train_index'] for r in rows] == [0, 1]
    assert [r['source'] for r in rows] == ['ticalc', 'github']
    stats = json.loads(summary.read_text())
    assert stats['n_samples'] == stats['n_run'] == stats['n_math'] == 2
    assert stats['math'] == 100.0
    assert stats['topics'] == {'other': 2}
    assert stats['lengths']['1-40'] == 2
    assert 'text' not in summary.read_text()


def test_miner_refuses_duplicate_result_indices(tmp_path):
    samples = [tok('DC', 'A') + [EOS]] * 2
    records = [rec('OK', home=['6'], actions={'input': 1}, index=0)] * 2
    da, db = _run(tmp_path / 'a', records, '2'), _run(tmp_path / 'b', records, '7')
    with pytest.raises(ValueError, match='indices'):
        M.mine_samples(samples, da, db, [{}, {}])


def test_summary_includes_empty_shards_without_inventing_a_rate():
    metrics = [{'n_samples': 2, 'n_run': 0, 'n_math': 0, **dict.fromkeys(M.FLAGS)}]
    stats = M.summarize([], metrics)
    assert stats['n_empty'] == 2 and stats['n_math'] == 0
    assert stats['math'] is None
    assert stats['flag_counts'] == dict.fromkeys(M.FLAGS, 0)
