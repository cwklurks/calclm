"""The harness's typed input value (driver.Config.input_text): lab.score --input-text and
run_programs.run_token_ids pass it to the driver when given; off, every call is the one it was
before (the driver's own default "1")."""
import json

from eval import driver, run_programs
from model.tokens import BOS, EOS


def test_run_token_ids_puts_input_text_on_the_config_only_when_given(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(run_programs.inputs, "from_token_ids", lambda *a, **k: [])
    monkeypatch.setattr(run_programs, "run_specs", lambda specs, out, cfg, **kw: seen.setdefault(
        "cfg", cfg) and [])
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000,
                               input_text="7")
    assert seen["cfg"].input_text == "7" and seen["cfg"].budget_ms == 5000
    seen.clear()
    run_programs.run_token_ids([[1, 2]], tmp_path / "v.json", tmp_path, budget_ms=5000)
    assert seen["cfg"] == driver.Config(budget_ms=5000)


def test_lab_score_passes_input_text_through(monkeypatch, tmp_path):
    from lab import score
    seen = {}

    def fake(programs, vocab, out, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr("eval.run_programs.run_token_ids", fake)
    score.run_emulator([[1, 5, 2]], tmp_path, input_text="2", yields=True)
    assert seen["input_text"] == "2" and seen["yields"] is True
    seen.clear()
    score.run_emulator([[1, 5, 2]], tmp_path)
    assert "input_text" not in seen


def test_cli_input_text_flag_reaches_score_and_score_json(tmp_path, monkeypatch):
    from lab import score as Sc
    seen = []

    def fake_score(samples, out, vocab, train, workers, budget_ms, max_len=256, **kw):
        seen.append(kw)
        return {"n_samples": len(samples)}

    monkeypatch.setattr(Sc, "score", fake_score)
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"ids": [BOS, 5, EOS]}) + "\n")
    Sc.main(["--samples", str(path), "--out", str(tmp_path), "--input-text", "7"])
    assert json.loads((tmp_path / "score.json").read_text())["input_text"] == "7"
    Sc.main(["--samples", str(path), "--out", str(tmp_path)])
    assert seen == [{"input_text": "7"}, {}]
    assert "input_text" not in json.loads((tmp_path / "score.json").read_text())
