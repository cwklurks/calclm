"""Topic allocation tests; no provider or emulator calls."""
from lab import synth
from lab.synth_topics import TopicSteering, build_catalog


def test_catalog_uses_school_topics_and_useful_mined_titles():
    catalog = build_catalog([
        {'topic': 'geometry', 'text': 'Disp "TORUS VOLUME"'},
        {'topic': 'geometry', 'text': 'Disp "ENTER NUMBER"'},
        {'topic': 'geometry', 'text': 'Disp "********"'},
        {'topic': 'algebra', 'text': 'Disp "FACTORING TOOK"'},
        {'topic': 'algebra', 'text': 'Disp "TI-84 INTEGER"'},
    ])
    labels = [row['label'] for row in catalog]
    assert 150 <= len(labels) <= 200
    assert len(labels) == len(set(labels))
    assert 'TORUS VOLUME' in labels
    assert 'ENTER NUMBER' not in labels and '********' not in labels
    assert 'FACTORING TOOK' not in labels and 'TI-84 INTEGER' not in labels


def test_scheduler_rotates_without_waiting_for_judging_and_uses_legacy_coverage():
    catalog = [{'topic': 'geometry', 'label': t}
               for t in ('circle area', 'triangle area', 'sphere volume')]
    legacy = [{'title': 'CIRCLE AREA'}, {'title': 'CIRCLE AREA'}]
    steering = TopicSteering(catalog, legacy)
    assigned = [steering.assign()['label'] for _ in range(6)]
    assert assigned == ['triangle area', 'sphere volume'] * 3
    # Failed topics still yield turns to the rest of the catalog.
    assert 'circle area' in [steering.assign()['label'] for _ in range(12)]


def test_scheduler_resume_reconstructs_allocations_and_only_kept_titles_count():
    catalog = [{'topic': 'geometry', 'label': t} for t in ('circle area', 'sphere volume')]
    rows = [{'subtopic': 'circle area', 'reason': 'math_tier', 'title': 'BAD'},
            {'subtopic': 'sphere volume', 'reason': None, 'title': 'SPHERE VOLUME'}]
    steering = TopicSteering(catalog, [rows[1]], rows)
    assert steering.covered_titles('circle area') == []
    assert steering.covered_titles('sphere volume') == ['SPHERE VOLUME']
    assert steering.assign()['label'] == 'circle area'
    steering.keep({'subtopic': 'circle area', 'title': 'AREA OF CIRCLE'})
    assert steering.covered_titles('circle area') == ['AREA OF CIRCLE']


def test_assigned_topic_overrides_seed_task_in_both_modes():
    seed = {'topic': 'algebra', 'text': 'Disp "QUADRATIC"'}
    for mode in ('sibling', 'variation'):
        prompt = synth.make_prompt(seed, mode, ['OLD TASK'],
                                   subtopic={'topic': 'physics', 'label': 'spring energy'})
        assert 'Assigned subtopic: spring energy' in prompt
        assert 'Topic: physics' in prompt
        assert 'OLD TASK' in prompt
        assert 'SAME task' not in prompt and 'same topic' not in prompt
        assert 'style' in prompt and '0<P<1' in prompt
