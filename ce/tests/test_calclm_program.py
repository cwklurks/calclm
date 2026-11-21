"""The calculator program's build inputs: the CLMVOC token table (built from each export's own
model.json vocab) and the CE toolchain build of CALCLM.8xp. The program's OS layer (fileioc,
graphx, keypadc) only runs on real hardware; these tests cover what can be checked here."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ce.tools.clm import CALCLM, EXPORTS

sys.path.insert(0, str(CALCLM / "tools"))
import make_vocab  # noqa: E402

AVAILABLE = [k for k, p in EXPORTS.items() if (p / "model.json").exists()]
pytestmark = pytest.mark.skipif(not AVAILABLE, reason="no trained export under runs/")


@pytest.mark.parametrize("name", AVAILABLE)
def test_clmvoc_matches_the_export_vocab(name):
    meta = json.loads((EXPORTS[name] / "model.json").read_text())
    data, glyphs, _ = make_vocab.build(EXPORTS[name])
    assert len(data) <= 65505 and len(glyphs) <= 128
    assert int.from_bytes(data[4:6], "little") == len(meta["vocab"])
    assert int.from_bytes(data[6:9], "little") == meta["blob_bytes"]
    recs = make_vocab.decode(data)
    for i, h in enumerate(meta["vocab"]):
        tok, disp = recs[i]
        if i < 4:
            assert tok == b"" and disp == b""
            continue
        assert tok == bytes.fromhex(h)
        assert disp, f"token {h} has no display text"
        assert all(b == 0x0A or 0x20 <= b < 0x7F or 0x80 <= b < 0x80 + len(glyphs) for b in disp)
    assert recs[meta["vocab"].index("3F")][1] == b"\n"          # newline token


def test_clmvoc_text_reads_like_ti_basic():
    data, glyphs, _ = make_vocab.build(EXPORTS[AVAILABLE[0]])
    recs = make_vocab.decode(data)
    vocab = json.loads((EXPORTS[AVAILABLE[0]] / "model.json").read_text())["vocab"]
    by_hex = {h: recs[i][1] for i, h in enumerate(vocab)}
    assert by_hex["DE"] == b"Disp "
    assert by_hex["04"] == bytes([0x80 + glyphs.index("→")])     # store arrow glyph
    assert by_hex["CE"] == b"If "


@pytest.mark.skipif(not (Path.home() / "CEdev" / "bin" / "cedev-config").exists(),
                    reason="needs ~/CEdev")
def test_program_builds_and_fits():
    env = dict(os.environ, PATH=f"{Path.home() / 'CEdev' / 'bin'}:{os.environ['PATH']}")
    subprocess.run(["make"], cwd=CALCLM, env=env, check=True, capture_output=True)
    prog = CALCLM / "bin" / "CALCLM.8xp"
    assert prog.exists() and prog.stat().st_size <= 60000
    mp = (CALCLM / "bin" / "CALCLM.map").read_text()
    bss = int(mp.split("___bss_len = (___bss_high - ___bss_low)")[0].split()[-1], 16)
    assert bss <= 0xD13FD8 - 0xD052C6
    # LD2 must not add its 2,652-byte workspace to the program's BSS.
    # v2.1: 60,325 plus one byte of linker alignment; v2.4 adds the string automaton (2 bytes),
    # v2.5 clm_feed's guard (2 bytes)
    assert bss <= 60330


def _map_value(mp: str, name: str) -> int:
    import re
    return int(re.search(r"0x([0-9a-f]+)\s+" + re.escape(name) + r" = ", mp).group(1), 16)


@pytest.mark.skipif(not (Path.home() / "CEdev" / "bin" / "cedev-config").exists(),
                    reason="needs ~/CEdev")
def test_w_program_builds_and_fits():
    """The shipped W build (v2.5, PHASE3-topic-menu.md 6): CALCLM.8xp in archive, the BSS
    region, and the expanded image in user RAM within 65,505 bytes (the largest variable), in
    case the OS counted the decompressed copy as one."""
    env = dict(os.environ, PATH=f"{Path.home() / 'CEdev' / 'bin'}:{os.environ['PATH']}")
    subprocess.run(["make", "SHAPE=w"], cwd=CALCLM, env=env, check=True, capture_output=True)
    prog = CALCLM / "bin" / "w" / "CALCLM.8xp"
    assert prog.exists() and prog.stat().st_size <= 60000
    mp = (CALCLM / "bin" / "w" / "CALCLM.map").read_text()
    bss = _map_value(mp, "___bss_len")
    assert bss <= 0xD13FD8 - 0xD052C6 and bss <= 60330
    image = _map_value(mp, "___data_high") - 0xD1A87F      # from userMem: header .. data
    assert image <= 65505, image


def test_8xv_writer_matches_tivars(tmp_path):
    import build_flash
    from tivars.types import TIAppVar
    data = bytes(range(256)) * 3
    var = TIAppVar(name="CLMCODE")
    var.data = data
    var.save(str(tmp_path / "a.8xv"))
    ours = build_flash.appvar_8xv("CLMCODE", data, b"Created with tivars_lib_py v1.1.1")
    assert ours == (tmp_path / "a.8xv").read_bytes()


def test_deployed_clmcode_is_the_tested_payload():
    """bin/CLMCODE.8xv carries exactly the payload the emulator tests attach (clmcode.bin),
    built from the same sources as core.o (same build id)."""
    from ce.tools.clm import OUT
    av = (CALCLM / "bin" / "CLMCODE.8xv").read_bytes()
    payload = (OUT / "clmcode.bin").read_bytes()
    size = int.from_bytes(av[72:74], "little")
    assert av[74:74 + size] == payload
    code_id = int((OUT / "obj" / "code_id").read_text().strip(), 16)
    assert int.from_bytes(payload[9:13], "little") == code_id


def test_mask_ids_header_is_current():
    """core/clm_mask_ids.h = gen_mask_ids.py on the vocab the tracker's class table was built
    from (tracker/trk_tables.h, checked against corpus/vocab.json by tracker/tests)."""
    import re

    import gen_mask_ids
    vocab = json.loads((CALCLM.parents[1] / "corpus" / "vocab.json").read_text())
    header = (CALCLM / "core" / "clm_mask_ids.h").read_text()
    assert header == gen_mask_ids.render(vocab), "run: uv run python ce/calclm/tools/gen_mask_ids.py"
    tables = (CALCLM.parents[1] / "tracker" / "trk_tables.h").read_text()
    trk_vocab = int(re.search(r"#define TRK_VOCAB (\d+)", tables).group(1))
    assert f"#define MASK_VOCAB {trk_vocab}\n" in header


@pytest.mark.skipif(not (CALCLM.parents[1] / "runs" / "v2" / "export" / "model.json").exists(),
                    reason="runs/v2/export not present")
def test_v2_export_uses_the_tracker_vocab():
    """The masks are compiled for corpus/vocab.json; the v2 export must use the same ids."""
    root = CALCLM.parents[1]
    meta = json.loads((root / "runs" / "v2" / "export" / "model.json").read_text())
    assert meta["vocab"] == json.loads((root / "corpus" / "vocab.json").read_text())
    data, _, _ = make_vocab.build(root / "runs" / "v2" / "export")
    assert int.from_bytes(data[6:9], "little") == meta["blob_bytes"]


@pytest.mark.parametrize("name", [*AVAILABLE, "v2"])
def test_clmvoc_hash_is_the_tracker_vocab_hash_only_for_its_vocab(name):
    """src/model.c allows the masks only when the FNV-1a hash of CLMVOC's token bytes equals
    MASK_VOCAB_HASH: true for v1 and v2 (corpus/vocab.json), false for v0 (vocab 718)."""
    import re

    import gen_mask_ids
    root = CALCLM.parents[1]
    path = EXPORTS.get(name, root / "runs" / "v2" / "export")
    if not (path / "model.json").exists():
        pytest.skip(f"{path} not present")
    data, _, _ = make_vocab.build(path)
    recs = make_vocab.decode(data)
    got = gen_mask_ids.vocab_hash(recs[i][0] for i in range(len(recs)))
    header = (CALCLM / "core" / "clm_mask_ids.h").read_text()
    want = int(re.search(r"#define MASK_VOCAB_HASH (0x[0-9a-f]+)UL", header).group(1), 16)
    corpus = json.loads((root / "corpus" / "vocab.json").read_text())
    same = json.loads((path / "model.json").read_text())["vocab"] == corpus
    assert (got == want) == same


def test_the_program_turns_horizons_on_after_every_load():
    """v2.1 (PLAN-v2.1.md E1 passed, lab/E12.md): clm_load turns ALiBi horizons off, so
    model_prepare, which loads before every generation, turns them on for format 2 models."""
    import re
    src = (Path(__file__).parents[1] / "calclm" / "src" / "model.c").read_text()
    body = re.search(r"int model_prepare\(.*?\n}\n", src, re.S).group(0)
    load = body.index("clm_load(")
    assert body.index("decoder(", load) > load


# ---- v2.5 topics (PHASE3-topic-menu.md 2d) -----------------------------------------------------

def test_topics_header_is_current():
    """src/topics.h = gen_topics.py on model/topics.py and the committed measurement fixture."""
    import gen_topics

    from model import topics as TP
    header = (CALCLM / "src" / "topics.h").read_text()
    assert header == gen_topics.render(TP.load_fixture()), \
        "run: uv run python ce/calclm/tools/gen_topics.py"


def test_topics_header_is_bound_to_the_mask_vocab_and_the_measured_model():
    import re

    from model import topics as TP
    header = (CALCLM / "src" / "topics.h").read_text()
    masks = (CALCLM / "core" / "clm_mask_ids.h").read_text()
    th = re.search(r"#define TOPIC_VOCAB_HASH (0x[0-9a-f]+)UL", header).group(1)
    mh = re.search(r"#define MASK_VOCAB_HASH (0x[0-9a-f]+)UL", masks).group(1)
    assert int(th, 16) == int(mh, 16)
    fx = TP.load_fixture()
    sha = re.search(r'#define TOPIC_EXPORT_SHA "((?:\\x[0-9a-f]{2}){6})"', header).group(1)
    assert sha.replace("\\x", "") == fx["export_sha256"][:12]
    import hashlib
    fit = TP.FIT
    assert hashlib.sha256(fit.read_bytes()).hexdigest() == fx["offsets_sha256"]
    E = re.search(r"static const uint32_t topic_ld2_E\[16\] = \{([^}]*)\}", header).group(1)
    assert [int(x) for x in E.replace(",", " ").split()] == json.loads(fit.read_text())["E"]
    assert json.loads(fit.read_text())["export_sha256"] == fx["export_sha256"]


def test_gen_topics_binds_the_fixture_it_loads():
    """render(fx, fixture) names that fixture in the header and takes the LD2 fit whose sha256 the
    fixture records (review L3): v2.5's fixture still renders v2.5's binding."""
    import re

    import gen_topics

    from model import topics as TP
    v25 = CALCLM.parents[1] / "ce" / "tests" / "fixtures" / "topics_v2.5.json"
    header = gen_topics.render(TP.load_fixture(v25), v25)
    assert "topics_v2.5.json" in header and "topics_v2.6.json" not in header
    fit = CALCLM.parents[1] / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"
    E = re.search(r"static const uint32_t topic_ld2_E\[16\] = \{([^}]*)\}", header).group(1)
    assert [int(x) for x in E.replace(",", " ").split()] == json.loads(fit.read_text())["E"]
    assert '#define TOPIC_EXPORT_SHA "\\xfb\\x4b\\xc7\\x74' in header


def test_v26_ships_e24s_l4c_s11_and_its_fit():
    """v2.6-rc2 (docs/agents/release-v2.6-rc2.md): the menu is measured on, and topics.h bound
    to, E24's L4c-s11 (the menu sweep's pick; not E22's L4c-s11, hence the e24 path and the
    sha) and its own LD2 fit."""
    from ce.tools.clm import ROOT, W_V26_EXPORT
    from model import topics as TP
    assert W_V26_EXPORT == ROOT / "runs" / "e24" / "export" / "L4c-s11"
    assert TP.FIXTURE.name == "topics_v2.6.json"
    assert TP.FIT.name == f"ld2_offsets_{W_V26_EXPORT.name}.json" == "ld2_offsets_L4c-s11.json"
    fx = TP.load_fixture()
    assert fx["export_sha256"].startswith("e68eca508872b7a5")
    assert json.loads(TP.FIT.read_text())["export_sha256"] == fx["export_sha256"]
    if (W_V26_EXPORT / "model.json").exists():
        assert json.loads((W_V26_EXPORT / "model.json").read_text())["sha256"] == \
            fx["export_sha256"]


def test_topic_fixture_holds_numbers_only():
    """The committed fixture carries counts and hashes, never program text or token ids."""
    from model import topics as TP
    fx = TP.load_fixture()
    assert set(fx["topics"]) == {t.key for t in TP.CANDIDATES}
    for t in fx["topics"].values():
        assert set(t) == {"task", "k", "passes", "labelled", "v3", "harness_errors",
                          "wilson95", "mean_tokens", "p90_tokens", "at_cap"}
        assert t["k"] == 100 and 0 <= t["passes"] <= 100
    assert fx["seeds"] == [600001, 600100]
