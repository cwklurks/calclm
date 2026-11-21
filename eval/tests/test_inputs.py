"""Input adapters: .8xp directory, token-id sequences + vocab, corpus JSONL sample."""
import json

from tivars import TIProgram

from eval import inputs as I


def write_8xp(path, name, src):
    p = TIProgram(name=name)
    p.load_string(src)
    p.save(str(path))
    return p.data


def test_from_dir_reads_programs_in_name_order(tmp_path):
    a = write_8xp(tmp_path / "B.8xp", "BPROG", "Disp 2\n")
    write_8xp(tmp_path / "A.8xp", "APROG", "Disp 1\n")
    (tmp_path / "notes.txt").write_text("x")
    specs = I.from_dir(tmp_path, tmp_path / "work")
    assert [s.name for s in specs] == ["APROG", "BPROG"]
    assert [s.index for s in specs] == [0, 1]
    assert specs[1].body == a and specs[1].reject_reason is None
    assert specs[1].path.exists()


def test_from_dir_garbage_file_is_rejected(tmp_path):
    (tmp_path / "BAD.8xp").write_bytes(b"not a ti file at all")
    [s] = I.from_dir(tmp_path, tmp_path / "work")
    assert s.reject_reason and s.path is None


def test_invalid_name_is_renamed(tmp_path):
    body = bytes.fromhex("DE313F")
    [s] = I.from_bodies([("bad name", body)], tmp_path)
    assert s.name == I.FALLBACK_NAME and s.orig_name == "bad name"
    assert TIProgram.open(str(s.path)).data == body


def test_from_token_ids(tmp_path):
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE", "31", "3F"]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    specs = I.from_token_ids([[1, 4, 5, 2], [1, 4, 3, 2]], tmp_path / "vocab.json", tmp_path / "w")
    assert specs[0].body == bytes.fromhex("DE31") and specs[0].reject_reason is None
    assert TIProgram.open(str(specs[0].path)).data == bytes.fromhex("DE31")
    assert specs[0].name == "G0000000" and specs[1].name == "G0000001"
    assert specs[1].reject_reason.startswith("tokens:")


def test_empty_program_rejected(tmp_path):
    [s] = I.from_bodies([("EMPTY", b"")], tmp_path)
    assert s.reject_reason == "empty program"


def test_corpus_sample_is_deterministic_and_filtered(tmp_path):
    rows = []
    for i in range(50):
        rows.append({"source": "s", "file": f"f{i}.zip", "member": f"f{i}.zip/P{i}.8xp",
                     "name": f"P{i}", "status": "kept" if i % 5 else "asm",
                     "tags": ["ice_source"] if i == 1 else [], "body": "DE313F"})
    path = tmp_path / "programs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    a = I.from_corpus(path, 10, seed=3, workdir=tmp_path / "a")
    b = I.from_corpus(path, 10, seed=3, workdir=tmp_path / "b")
    assert [s.source for s in a] == [s.source for s in b]
    assert len(a) == 10
    assert all(s.meta["status"] == "kept" and "ice_source" not in s.meta["tags"] for s in a)


def test_read_bin_splits_programs(tmp_path):
    import numpy as np
    ids = np.array([1, 4, 5, 2, 1, 6, 2, 1, 4, 2], dtype="<u2")
    (tmp_path / "val.bin").write_bytes(ids.tobytes())
    assert I.read_bin(tmp_path / "val.bin") == [[1, 4, 5, 2], [1, 6, 2], [1, 4, 2]]


def test_rejected_token_ids_keep_known_tokens_for_statistics(tmp_path):
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "DE", "31", "3F", "AD"]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    [s] = I.from_token_ids([[1, 7, 6, 4, 3, 2]], tmp_path / "vocab.json", tmp_path / "w")
    assert s.reject_reason.startswith("tokens:") and s.path is None
    assert s.body == bytes.fromhex("AD3FDE")          # <unk> dropped, rest kept
