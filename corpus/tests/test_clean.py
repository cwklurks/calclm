"""corpus/clean.py (PLAN-E22 Sections 3.1, 3.3, 3.5): the scrubber, X0-X3, the lexicon and the
leakage re-check. Every string here is made up (no corpus text, D6)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from corpus import clean as C

ROOT = Path(__file__).resolve().parents[2]
VOCAB = json.loads((ROOT / 'corpus' / 'vocab.json').read_text())
BOS, EOS = 1, 2


def ids(text: str) -> list[int]:
    from lab.tasks import text_ids
    return text_ids(text)


def shown(toks) -> str:
    disp = C.display_table(VOCAB)
    return ''.join(disp[t] for t in toks)


def scrub(text: str, lexicon=()) -> str:
    s = C.Scrubber(VOCAB, lexicon=frozenset(lexicon))
    out, _ = s.scrub(ids(text))
    return shown(out)


# ---------------------------------------------------------------- lexing

def test_string_ranges_end_at_quote_store_newline_and_program_end():
    t = ids('Disp "AB"\n"CD"→Str1\nDisp "EF\nDisp "GH')
    disp = C.display_table(VOCAB)
    got = [''.join(disp[x] for x in t[a:b]) for a, b in C.string_ranges(t, VOCAB)]
    assert got == ['AB', 'CD', 'EF', 'GH']


def test_display_folds_lowercase_letters_to_uppercase():
    t = ids('Disp "made by qx"')
    a, b = C.string_ranges(t, VOCAB)[0]
    assert C.Scrubber(VOCAB).text(t[a:b])[0] == 'MADE BY QX'


# ---------------------------------------------------------------- rules R1-R5

@pytest.mark.parametrize('src,want', [
    ('Disp "GAME BY ZORBLAX"', 'Disp "GAME "'),
    ('Disp "made by qxwell"', 'Disp "MADE "'),
    ('Disp "BY: QUENTWORTH V2"', 'Disp ""'),
    ('Disp "AUTHOR ZORB"', 'Disp ""'),
    ('Disp "V1 CREDITS QX AND ZY"', 'Disp "V1 "'),
    ('Disp "THANKS TO QWERTZ"', 'Disp ""'),
    ('Disp "SPECIAL THANKS QWERTZ"', 'Disp ""'),
    ('Disp "GREETZ ZORB"', 'Disp ""'),
    ('Disp "COPYRIGHT 2004 QX"', 'Disp ""'),
    ('Disp "GAME (C) 2004"', 'Disp "GAME "'),
    ('Disp "GAME COPR. QX"', 'Disp "GAME "'),
])
def test_r1_deletes_from_the_marker_to_the_end_of_the_string(src, want):
    assert scrub(src) == shown(ids(want))


@pytest.mark.parametrize('src', [
    'Disp "DIVIDED BY X"', 'Disp "MULTIPLIED BY A"', 'Disp "SORTED BY SIZE"',
    'Disp "INCREASED BY N"', 'Disp "STEP BY STEP"', 'Disp "TIMES BY B"',
    'Disp "DIVIDE BY 0"', 'Disp "BYTES"', 'Disp "GOOD BYE"', 'Disp "SIN(C)"',
    'Disp "F(C)=2"', 'Disp "A AND B"',
])
def test_math_text_is_not_edited(src):
    assert scrub(src) == shown(ids(src))


def test_r2_r3_r4_delete_only_the_address_or_handle():
    assert scrub('Disp "MAIL QX@EXAMPLE.COM NOW"') == shown(ids('Disp "MAIL  NOW"'))
    assert scrub('Disp "SEE WWW.QXZORB.NET OK"') == shown(ids('Disp "SEE  OK"'))
    assert scrub('Disp "SEE HTTP://QX.ZORB/A OK"') == shown(ids('Disp "SEE  OK"'))
    assert scrub('Disp "AT QXZORB.ORG"') == shown(ids('Disp "AT "'))
    assert scrub('Disp "PING @QXZ HI"') == shown(ids('Disp "PING  HI"'))
    assert scrub('Disp "X.5"') == shown(ids('Disp "X.5"'))


def test_r5_lexicon_names_whole_words_anywhere_in_a_string():
    lex = {'ZORBLAX'}
    assert scrub('Disp "HI ZORBLAX OK"', lex) == shown(ids('Disp "HI  OK"'))
    assert scrub('Disp "ZORBLAXES"', lex) == shown(ids('Disp "ZORBLAXES"'))


def test_rule_counts_and_order():
    s = C.Scrubber(VOCAB, lexicon=frozenset({'QUUX'}))
    _, info = s.scrub(ids('Disp "QUUX AT Q@EXAMPLE.COM BY ZORB"\nDisp "WWW.A.COM"\nDisp "OK"'))
    assert info['strings'] == 3 and info['edited_strings'] == 2
    assert info['rule_strings'] == {'R1': 1, 'R2': 1, 'R3': 1, 'R4': 0, 'R5': 1}
    assert info['tokens_removed'] > 0


# ---------------------------------------------------------------- what is never edited

def test_code_outside_strings_is_never_edited():
    src = 'B*Y→Z\nDisp B,Y\nprgmZORBLAX\nDisp "OK"'
    s = C.Scrubber(VOCAB, lexicon=frozenset({'ZORBLAX', 'Z'}))
    out, info = s.scrub(ids(src))
    assert out == ids(src)
    assert info['code_names'] == 1          # counted, not renamed


def test_an_emptied_string_keeps_its_quotes_and_lines():
    t = ids('Disp "BY ZORB"\nDisp "X"')
    out, _ = C.Scrubber(VOCAB).scrub(t)
    nl = VOCAB.index('3F')
    assert shown(out) == shown(ids('Disp ""\nDisp "X"'))
    assert out.count(nl) == t.count(nl)


def test_bos_eos_are_kept_and_strings_end_with_the_program():
    t = [BOS, *ids('Disp "BY ZORB'), EOS]
    out, _ = C.Scrubber(VOCAB).scrub(t)
    assert out[0] == BOS and out[-1] == EOS and shown(out[1:-1]) == shown(ids('Disp "'))


def test_idempotent():
    s = C.Scrubber(VOCAB, lexicon=frozenset({'QUUX'}))
    for src in ('Disp "QUUX BY ZORB"', 'Disp "A@EXAMPLE.ORG WWW.Q.NET @QX"', 'Disp "(C) Q"'):
        once, _ = s.scrub(ids(src))
        twice, info = s.scrub(once)
        assert twice == once and info['edited_strings'] == 0


def _suite_programs():
    from eval import handmade
    from lab import tasks_v2 as V2
    from lab import tasks_v3 as V3
    from lab.tasks import program_ids
    for task in V2.SUITE_V2:
        yield task.opening_ids()
        for x in (task.gold, *task.mutants, *(b for _, b in V2.LOOPHOLES.get(task.id, ())),
                  *(b for _, b in V3.V3_LOOPHOLES.get(task.id, ()))):
            yield program_ids(task, x)
    for case in handmade.CASES:
        yield ids(case.src if hasattr(case, 'src') else case[1])


def test_no_edit_to_any_suite_opening_gold_mutant_or_handmade_program():
    s = C.Scrubber(VOCAB)
    progs = list(_suite_programs())
    assert len(progs) > 100
    for p in progs:
        assert s.scrub(list(p))[0] == list(p)


# ---------------------------------------------------------------- the lexicon

def test_lexicon_words_after_markers_min_three_letters_minus_stop():
    texts = ['GAME BY QWOX ZARNIK', 'AUTHOR: JO PLUMTREX', 'MADE BY THE TEAM',
             'DIVIDED BY XYLOMB', 'THANKS TO QWOX, ZAP']
    lex = C.build_lexicon(texts, owners=['zibbo-dev', 'Q'], stop={'THE', 'TEAM', 'DEV'})
    assert lex == {'QWOX', 'ZARNIK', 'PLUMTREX', 'ZAP', 'ZIBBO'}


def test_frequent_non_credit_words_join_the_stop_list():
    progs = [['HELLO QWOX'] for _ in range(5)] + [['BY QWOX'], ['BY ZARNIK']]
    assert C.frequent_words(progs, threshold=5) == {'HELLO', 'QWOX'}
    assert C.frequent_words(progs, threshold=6) == set()


def test_github_owner():
    assert C.github_owner('github', 'ab__cd/x/y.8xp') == 'ab'
    assert C.github_owner('github', '_tarballs/ab__cd.tar.gz') == 'ab'
    assert C.github_owner('ticalc', 'ab__cd/x') is None


# ---------------------------------------------------------------- X0-X3

def prog(key, toks, seed=False):
    a = np.asarray(toks, dtype=np.uint16)
    return {'key': key, 'ids': a, 'seed': seed, 'sha1': hashlib.sha1(a.tobytes()).hexdigest()}


def body(rng, n, lo=40, hi=700):
    return list(rng.integers(lo, hi, size=n))


def test_x0_exact_keeps_the_seed_else_the_first():
    a = [5, 6, 7, 8]
    kept, dropped = C.x0([prog(0, a), prog(1, a, seed=True), prog(2, [9, 9])])
    assert [p['key'] for p in kept] == [1, 2] and dropped == [0]
    kept, dropped = C.x0([prog(0, a), prog(1, a)])
    assert [p['key'] for p in kept] == [0] and dropped == [1]


def test_signatures_and_normalized_key_equal_build_py():
    from corpus import build as B
    from corpus.tokenize import Vocab
    v = Vocab(VOCAB)
    rng = np.random.default_rng(1)
    progs = [prog(i, body(rng, 30 + 10 * i, 4, len(VOCAB))) for i in range(6)]
    uniq = [{'bytes': v.decode(p['ids'].tolist())} for p in progs]
    assert np.array_equal(C.signatures([C.codes(p['ids'], VOCAB) for p in progs]),
                          B.minhash_signatures(uniq))
    for p, u in zip(progs, uniq):
        assert C.normalized_key(p['ids'], VOCAB) == B.normalized_key(u['bytes'])


def test_x1_keeps_the_longest_seed_else_the_longest():
    rng = np.random.default_rng(2)
    base = body(rng, 400)
    near = base[:-2]
    other = body(rng, 300)
    kept, dropped = C.x1([prog(0, base), prog(1, near, seed=True), prog(2, other)], VOCAB)
    assert sorted(p['key'] for p in kept) == [1, 2] and dropped == [0]
    kept, dropped = C.x1([prog(0, near), prog(1, base), prog(2, other)], VOCAB)
    assert sorted(p['key'] for p in kept) == [1, 2] and dropped == [0]


def test_x2_drops_a_contained_program_never_a_seed():
    rng = np.random.default_rng(3)
    big = body(rng, 600)
    small = big[100:300]
    kept, dropped = C.x2([prog(0, small), prog(1, big)])
    assert [p['key'] for p in kept] == [1] and dropped == [0]
    kept, dropped = C.x2([prog(0, small, seed=True), prog(1, big)])
    assert sorted(p['key'] for p in kept) == [0, 1] and dropped == []


def _with_lines(rng, n_lines, width=12):
    nl = VOCAB.index('3F')
    out = []
    for _ in range(n_lines):
        out += body(rng, width) + [nl]
    return out


def test_x3_caps_shared_blocks_never_cuts_a_line_never_touches_a_seed():
    rng = np.random.default_rng(4)
    nl = VOCAB.index('3F')
    block = _with_lines(rng, 8)                       # shared by 7 programs
    progs = [prog(i, _with_lines(rng, 20 + i) + block + _with_lines(rng, 20 + i),
                  seed=(i == 6)) for i in range(7)]
    out, stats = C.x3(progs, K=4)
    by = {p['key']: p for p in out}
    # seed first, then longest: 6 (seed), 5, 4, 3 keep the block; 2, 1, 0 lose its lines
    for k in (6, 5, 4, 3):
        assert np.array_equal(by[k]['ids'], progs[k]['ids'])
    for k in (0, 1, 2):
        before, after = progs[k]['ids'].tolist(), by[k]['ids'].tolist()
        assert len(after) < len(before)
        lines_b = _split_lines(before, nl)
        lines_a = _split_lines(after, nl)
        assert all(line in lines_b for line in lines_a)     # whole lines only
    assert stats['touched'] == 3 and stats['gutted'] == 0
    out2, stats2 = C.x3(progs, K=4)
    assert all(np.array_equal(a['ids'], b['ids']) for a, b in zip(out, out2))


def _split_lines(t, nl):
    lines, cur = [], []
    for x in t:
        cur.append(x)
        if x == nl:
            lines.append(tuple(cur))
            cur = []
    if cur:
        lines.append(tuple(cur))
    return lines


def test_x3_gutting_rule_and_filler_never_excised():
    rng = np.random.default_rng(5)
    block = _with_lines(rng, 10)
    progs = [prog(i, block + _with_lines(rng, 1 + i)) for i in range(6)]
    out, stats = C.x3(progs, K=4)
    assert stats['gutted'] == 2 and len(out) == 4
    filler = [VOCAB.index('29')] * 200 + [VOCAB.index('3F')]
    progs = [prog(i, filler + _with_lines(rng, 5)) for i in range(8)]
    out, stats = C.x3(progs, K=4)
    assert stats['touched'] == 0 and len(out) == 8


def test_informative_windows():
    a = np.array([7] * 40, dtype=np.uint16)
    assert not C.informative(a, 32).any()
    b = np.array([1, 2, 3, 4, 5] + [7] * 35, dtype=np.uint16)
    inf = C.informative(b, 32)
    assert inf[0] and not inf[-1] and len(inf) == 9


# ---------------------------------------------------------------- leakage

def test_leakage_drops_direct_links_only_never_a_transitive_group():
    rng = np.random.default_rng(6)
    v = body(rng, 300)
    t_same = prog(0, v)                                 # same body as the valc program
    t_mate = prog(1, v[:-3] + body(rng, 3))             # a MinHash mate
    t_cont = prog(2, body(rng, 500) + v + body(rng, 500))  # contains it
    t_far = prog(3, body(rng, 400))
    t_trans = prog(4, t_far['ids'].tolist()[:-2])       # linked to t_far only
    links = C.leakage_links([t_same, t_mate, t_cont, t_far, t_trans], [prog('v0', v)], VOCAB)
    assert set(links) == {0, 1, 2}
    assert 'sha1' in links[0] and 'contains' in links[2]


def test_leakage_stops_above_100_drops():
    with pytest.raises(C.LeakageStop):
        C.check_leakage_count(101)
    C.check_leakage_count(100)


# ---------------------------------------------------------------- isolation

def test_the_scrubber_never_opens_the_sealed_labels():
    src = (ROOT / 'corpus' / 'clean.py').read_text()
    assert 'sealed' not in src


def test_no_hosted_model_client():
    tree = ast.parse((ROOT / 'corpus' / 'clean.py').read_text())
    names = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import,))
             for a in n.names} | {(n.module or '').split('.')[0] for n in ast.walk(tree)
                                  if isinstance(n, ast.ImportFrom)}
    assert not names & {'anthropic', 'openai', 'requests', 'httpx', 'urllib', 'http',
                        'google', 'cohere', 'mistralai', 'socket'}
