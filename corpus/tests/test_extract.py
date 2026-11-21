import io
import tarfile
import zipfile
from collections import Counter

from tivars.models import TI_84PCE
from tivars.types import TIProgram

from corpus.build import normalized_key
from corpus.extract import Budget, classify, walk_blob
from corpus.safepath import safe_relpath
from corpus.tokenize import body_to_8xp


def enc(text: str) -> bytes:
    return TIProgram.encode(text, model=TI_84PCE)


def test_asm_headers_are_filtered():
    for head in ("BB6D", "EF7B", "EF7A", "EF6D", "BB6C", "EF69"):
        assert classify(bytes.fromhex(head) + b"\x00" * 50)["status"] == "asm"
    # after cosmetic leading ':' and empty lines
    assert classify(bytes.fromhex("3E3FBB6C") + b"\x30" * 50)["status"] == "asm"


def test_trivial_and_kept():
    assert classify(enc("Disp 1"))["status"] == "trivial"
    r = classify(enc('ClrHome\nFor(I,1,10)\nDisp I\nEnd\nDisp "DONE"'))
    assert r["status"] == "kept" and r["tags"] == []


def test_undecodable_bodies_are_dropped():
    assert classify(b"\x00\x26" * 40)["status"] == "undecodable"


def test_tags():
    rest = enc("GAME\nFor(A,1,10)\nEnd\nPause")
    assert "ice_source" in classify(bytes.fromhex("062C07") + rest)["tags"]
    assert "ice_source" in classify(bytes.fromhex("2C") + rest)["tags"]
    assert "axe_source" in classify(enc(".GAME\nClrDraw\nDispGraph\nPause 100"))["tags"]
    assert "dcs_header" in classify(enc(':DCS\n"0FF0"\nClrHome\nDisp 123'))["tags"]
    assert "asm_call" in classify(enc("ClrHome\nAsm(prgmLIB)\nDisp 1"))["tags"]
    assert "lib_calls" in classify(enc('ClrHome\nreal(0,1,2)\nDisp 1'))["tags"]


def test_ti83_hex_asm_is_filtered():
    body = enc("ClrHome\nDisp 1\nEnd\n0000\nEnd")
    assert classify(body)["status"] == "asm_hex83"


def test_safe_relpath_blocks_traversal():
    assert safe_relpath("../../etc/passwd") == "etc/passwd"
    assert safe_relpath("/abs/x.8xp") == "abs/x.8xp"
    assert safe_relpath("..") is None
    assert safe_relpath("a/.hidden") == "a/hidden"


def test_nested_zip_with_traversal_names_is_read_in_memory(tmp_path):
    body = enc('ClrHome\nFor(I,1,10)\nDisp I\nEnd')
    raw = body_to_8xp(body, tmp_path / "P.8xp", name="P").read_bytes()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("../../evil/P.8xp", raw)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("inner.zip", inner.getvalue())
    found, events = [], Counter()
    walk_blob(outer.getvalue(), "top.zip", 0, Budget(), found, events)
    assert [(f[1], f[3]) for f in found] == [("P", body)]
    assert not (tmp_path / "evil").exists()


def test_normalized_key_ignores_cosmetic_edits():
    a = enc('Disp "HI"\nIf A=1\nThen\nDisp (A+1)\nEnd')
    b = enc('Disp "HI\nIf A=1\nThen\nDisp (A+1\nEnd')
    assert normalized_key(a) == normalized_key(b)
    assert normalized_key(a) != normalized_key(enc('Disp "HO"\nIf A=1\nThen\nDisp (A+1)\nEnd'))


def test_tar_gz_members_are_streamed(tmp_path):
    body = enc('ClrHome\nFor(I,1,10)\nDisp I\nEnd')
    raw = body_to_8xp(body, tmp_path / "P.8xp", name="P").read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, payload in (("repo-abc/progs/P.8xp", raw), ("repo-abc/README.md", b"hi")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    found, events = [], Counter()
    walk_blob(buf.getvalue(), "repo.tar.gz", 0, Budget(), found, events)
    assert [(f[1], f[3]) for f in found] == [("P", body)]
