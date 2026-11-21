import json
from pathlib import Path

import pytest

from eval.gen_intref import generate_jsonl

EXPORT = Path(__file__).resolve().parents[2] / "runs" / "v0" / "export"
pytestmark = pytest.mark.skipif(not (EXPORT / "model.json").exists(), reason="no v0 export")


def test_writes_one_record_per_seed_and_is_deterministic(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    generate_jsonl(EXPORT, a, seeds=[5, 6], temp=0.8, top_k=20, max_new=6, workers=2)
    generate_jsonl(EXPORT, b, seeds=[5, 6], temp=0.8, top_k=20, max_new=6, workers=1)
    recs = [json.loads(line) for line in a.read_text().splitlines()]
    assert [r["seed"] for r in recs] == [5, 6]
    # <bos> (id 1) followed by at most max_new sampled tokens
    assert all(r["ids"][0] == 1 and 1 < len(r["ids"]) <= 7 for r in recs)
    assert a.read_text() == b.read_text()


V2 = EXPORT.parents[1] / "v2" / "export"


@pytest.mark.skipif(not (V2 / "model.json").exists(), reason="no v2 export")
def test_cap_and_masks_reproduce_intref_past_the_window(tmp_path):
    """v2's acceptance setting: a format 2 export, masks on, a cap past the 256-token window."""
    from model.intmask import IntMask
    from model.intref import IntModel, SamplerConfig, generate

    out = tmp_path / "v2.jsonl"
    generate_jsonl(V2, out, seeds=[3, 4], temp=0.8, top_k=20, max_new=299, workers=2, cap=300,
                   masks=True)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    im = IntModel.load(V2)
    for r in recs:
        want = generate(im, [], 299, SamplerConfig(205, 20, False, r["seed"]), cap=300,
                        mask=IntMask(im.vocab))
        assert r["ids"] == want and r["cap"] == 300 and r["masks"] is True
    assert max(len(r["ids"]) for r in recs) > 256 or all(r["ids"][-1] == 2 for r in recs)
