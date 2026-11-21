import json

import numpy as np
from tivars import TIProgram

from model.data import load_split
from model.make_proxy import build_proxy
from model.tokens import BOS, EOS


def _write_prog(path, name, text):
    prog = TIProgram(name=name)
    prog.load_string(text)
    prog.save(str(path))


def test_build_proxy_dedups_and_splits_by_program(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    texts = [f'Disp "P{i}"\nPause ' for i in range(12)]
    for i, text in enumerate(texts):
        _write_prog(src / f"P{i}.8xp", f"P{i}", text)
    _write_prog(src / "DUP.8xp", "DUP", texts[0])  # identical body: dropped
    out = tmp_path / "proxy"
    build_proxy(sorted(src.glob("*.8xp")), out, val_frac=0.25)

    vocab = json.loads((out / "vocab.json").read_text())
    meta = json.loads((out / "tok" / "meta.json").read_text())
    train, val = load_split(out / "tok" / "train.bin"), load_split(out / "tok" / "val.bin")
    assert meta["vocab_size"] == len(vocab)
    n_progs = int((train == BOS).sum() + (val == BOS).sum())
    assert n_progs == 12 == meta["n_train_programs"] + meta["n_val_programs"]
    assert (val == BOS).sum() >= 1 and (train == BOS).sum() >= 1
    for split in (train, val):
        assert split[0] == BOS and split[-1] == EOS
        assert int(split.max()) < len(vocab)
    assert np.array_equal(train, load_split(out / "tok" / "train.bin"))


def test_the_source_folder_is_required(monkeypatch, capsys):
    import pytest

    from model import make_proxy
    monkeypatch.setattr("sys.argv", ["make_proxy", "--out", "x"])
    with pytest.raises(SystemExit):
        make_proxy.main()
    assert "--src" in capsys.readouterr().err
    assert not hasattr(make_proxy, "DEFAULT_SRC")
