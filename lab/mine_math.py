"""Mine paired math-tier passes into a private JSONL seed file.

Indices are zero-based train program indices, including samples empty after trimming.
Token counts exclude BOS/EOS and describe the trimmed program that was actually judged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from corpus.tokenize import render_ids
from lab.function import aligned, string_literals
from lab.mathtier import FLAGS, classify, math_metrics, response_lines
from lab.score import read_samples
from model.tokens import BOS, EOS


# Most specific domains precede broad geometry/algebra terms to break equal-score ties.
TOPIC_WORDS = {
    'chemistry': r'mol(?:e|es|ar|arity|ecular)?|atomic|avogadro|stoich\w*|chem\w*|isotope|element',
    'finance': r'interest|apr|loan|mortgage|payment|principal|finance|annuity|investment|profit|tax',
    'statistics': r'stat\w*|standard deviation|variance|median|mean|regression|probability|binomial|normal distribution',
    'number theory': r'primes?|gcd|lcm|divisor|integer|number theory|modulo|fibonacci',
    'conversion': r'convert\w*|conversion|celsius|fahrenheit|kelvin|inches|feet|miles|kilometers|pounds|gallons',
    'geometry': r'triang\w*|circle|radius|diameter|area|volume|perimeter|pythag\w*|trapez\w*|rectangle|sphere|cylinder|polygon|geometry|angle|trig\w*|sine|cosine|tangent',
    'physics': r'phys\w*|velocity|acceleration|force|energy|kinetic|mass|projectile|momentum|voltage|current|resistance|gravity|distance|speed|watt|joule',
    'algebra': r'algebra|equation|quadratic|polynom\w*|discriminant|slope|linear|factor\w*|roots?|vertex|intercept|matrix|system|logarithm',
}


def topic(ids) -> str:
    """Rough deterministic label from literal/prompt keywords; no formula correctness claim."""
    text = ' '.join(string_literals(ids)).lower()
    counts = {key: len(re.findall(r'\b(?:' + words + r')\b', text))
              for key, words in TOPIC_WORDS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else 'other'


def mine_samples(samples, dir_a: Path, dir_b: Path, provenance, start_index=0, cap=1024):
    """Return seed rows from one shard; provenance follows this shard's sample order."""
    math_metrics(samples, dir_a, dir_b, cap)  # Validate cap, yield capture and typed inputs.
    if len(provenance) != len(samples):
        raise ValueError('provenance must have one row per sample')
    a, b = aligned(samples, dir_a, cap), aligned(samples, dir_b, cap)
    for directory, records in ((dir_a, a), (dir_b, b)):
        indices = sorted(row[2]['index'] for row in records.values())
        if indices != list(range(len(records))):
            raise ValueError(f'{directory}: result indices must be unique and contiguous from zero')
    rows = []
    for i in sorted(a):
        _, ids, ra = a[i]
        rb = b[i][2]
        if not classify(ids, ra, rb)['math']:
            continue
        meta = provenance[i]
        rows.append({
            'train_index': start_index + i,
            'token_count': sum(t not in (BOS, EOS) for t in ids),
            'text': render_ids(ids),
            **{k: meta.get(k) for k in ('source', 'sources', 'file', 'member', 'name', 'sha1')},
            'outcome': {'2': ra['outcome'], '7': rb['outcome']},
            'visible_results': {'2': sorted(response_lines(ra, '2', ids)),
                                '7': sorted(response_lines(rb, '7', ids))},
            'topic': topic(ids),
        })
    return rows


LENGTH_BINS = ((1, 40), (41, 100), (101, 300), (301, 600), (601, 1022), (1023, 10**9))


def summarize(rows, metrics):
    """Numeric summary only; safe to use when writing the committed report."""
    n_run = sum(m['n_run'] for m in metrics)
    counts = {key: sum(round(m[key] * m['n_run'] / 100) for m in metrics if m['n_run'])
              for key in FLAGS}
    return {
        'n_samples': sum(m['n_samples'] for m in metrics),
        'n_run': n_run, 'n_empty': sum(m['n_samples'] - m['n_run'] for m in metrics),
        'n_math': len(rows),
        **{key: 100 * count / n_run if n_run else None for key, count in counts.items()},
        'flag_counts': counts,
        'topics': dict(sorted(Counter(row['topic'] for row in rows).items())),
        'lengths': {(f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'):
                    sum(lo <= row['token_count'] <= hi for row in rows)
                   for lo, hi in LENGTH_BINS},
        'shards': metrics,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--samples-dir', required=True, type=Path)
    ap.add_argument('--runs-dir', required=True, type=Path,
                    help='each shard has subdirectories 2/ and 7/ with lab.score outputs')
    ap.add_argument('--programs', required=True, type=Path, help='data/tok/programs.jsonl')
    ap.add_argument('--out', required=True, type=Path, help='private seeds.jsonl; never commit')
    ap.add_argument('--summary', required=True, type=Path, help='numeric summary JSON')
    ap.add_argument('--cap', default=1024, type=int)
    args = ap.parse_args(argv)
    provenance = sorted((json.loads(line) for line in args.programs.read_text().splitlines()
                         if line.strip()), key=lambda row: (row['split'], row['offset']))
    provenance = [row for row in provenance if row['split'] == 'train']
    paths = sorted(args.samples_dir.glob('*.samples.jsonl'))
    if not paths:
        raise ValueError('no train sample shards')
    rows, metrics, start, offset = [], [], 0, 0
    for path in paths:
        samples = read_samples(path)
        meta = provenance[start:start + len(samples)]
        if len(meta) != len(samples):
            raise ValueError(f'{path}: too few train provenance rows')
        for sample, record in zip(samples, meta):
            n_tokens = sum(t not in (BOS, EOS) for t in sample)
            if record['offset'] != offset or record['n_tokens'] != n_tokens:
                raise ValueError(f'{path}: train metadata offset/length mismatch at {start}')
            offset += len(sample)
        name = path.name.removesuffix('.samples.jsonl')
        da, db = args.runs_dir / name / '2', args.runs_dir / name / '7'
        mined = mine_samples(samples, da, db, meta, start, args.cap)
        m = math_metrics(samples, da, db, args.cap)
        m.update(shard=name, n_samples=len(samples))
        rows.extend(mined)
        metrics.append(m)
        start += len(samples)
    if start != len(provenance):
        raise ValueError(f'{start} samples for {len(provenance)} train provenance rows')
    stats = summarize(rows, metrics)
    if stats['flag_counts']['math'] != len(rows):
        raise ValueError('math pass count does not match seed count')
    # Do not replace a completed seed file until all shards and metadata validate.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temp.replace(args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'shards'}, indent=2))


if __name__ == '__main__':
    main()
