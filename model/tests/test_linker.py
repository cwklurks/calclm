"""Linker decoder (PLAN-v2.1.md E2): the port of odec2's rules 1-6 onto intref with the shipped
masks, the integer stop hazard (rule 7) and its offset fit.

Only the development checkpoint runs/v2/export (S0-s10) is used here (PLAN-v2.1.md, Common setup)."""
import hashlib
import json
from fractions import Fraction
from math import exp
from pathlib import Path

import numpy as np
import pytest

from model import linker
from model.intops import xorshift32
from model.intref import IntModel, SamplerConfig, choose, generate as intref_generate
from model.tokens import BOS, EOS

ROOT = Path(__file__).resolve().parents[2]
V2_EXPORT = ROOT / "runs" / "v2" / "export"
VAL = ROOT / "data" / "tok" / "val.bin"
PARITY = Path(__file__).parent / "linker_parity_s10.json"   # not under data/ (git-ignored)


def _tracker_available() -> bool:
    try:
        from tracker import bind
        bind.lib()
        return True
    except Exception:  # noqa: BLE001 - the C tracker needs gcc
        return False


needs_s10 = pytest.mark.skipif(not (_tracker_available() and (V2_EXPORT / "model.bin").exists()),
                               reason="runs/v2/export or the C tracker not present")


@pytest.fixture(scope="module")
def s10():
    return IntModel.load(V2_EXPORT)


@pytest.fixture(scope="module")
def ids(s10):
    return linker.vocab_ids(s10.vocab)


def _sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<u2").tobytes()).hexdigest()


# ---- constants ---------------------------------------------------------------------------------

@needs_s10
def test_vocab_ids_are_the_scratch_constants(ids):
    """odec.py hard-codes LBL 55, GOTO 48, MENU 112, NL 4, COLON 15, odec2 Stop 138 / Return 150."""
    assert (ids.lbl, ids.goto, ids.menu, ids.nl, ids.colon, ids.stop, ids.ret) == (
        55, 48, 112, 4, 15, 138, 150)
    hexes = ["CE", "D1", "D2", "D3", "DA", "DB", "D9", "D5", "E6", "AD", "DC", "DD"]
    assert ids.exit == frozenset(ids.vocab.index(h) for h in hexes)
    assert ids.char["A"] == ids.vocab.index("41") and ids.char["θ"] == ids.vocab.index("5B")


def test_arms():
    R = linker.Rules
    assert linker.ARMS == {"M": R(), "OLD": R(old=True), "HZ": R(hazard=True),
                           "LD": R(old=True, hazard=True, r1=True, r2=True),
                           "LDs": R(old=True, soft_bind=True, hazard=True, r1=True, r2=True),
                           "LD2": R(old=True, hazard=True, r1=True, r2=True, hazard_full=True)}


# ---- rule 7 arithmetic -------------------------------------------------------------------------

def test_hazard16_formula():
    for w_e, W, b in [(1000, 11000, 1.0), (65535, 300000, -2.5), (7, 1_300_000, 8.0),
                      (40000, 40001, -8.0)]:
        E = linker.offsets_to_E([b])[0]
        want = Fraction(65536 * w_e * E, w_e * E + (W - w_e) * 65536)
        assert linker.hazard16(w_e, W, E) == int(want)            # floor, exact
    # with b = 0 the hazard is the sampler's own top-k <eos> probability, floored to 1/65536
    assert linker.hazard16(1000, 11000, 65536) == 65536 * 1000 // 11000


def test_hazard16_bounds():
    E = linker.offsets_to_E([3.0])[0]
    assert linker.hazard16(0, 123456, E) == 0                       # <eos> not in the top-k
    assert linker.hazard16(0, 0, E) == 0                            # zero denominator
    assert linker.hazard16(5000, 5000, E) == 65536                  # <eos> is all the weight
    assert linker.hazard16(5000, 5000, 22) == 65536                 # ... whatever the offset
    assert linker.hazard16(1, 20 * 65535, 22) == 0
    hs = [linker.hazard16(3000, 90000, e) for e in linker.offsets_to_E([-8, -2, 0, 2, 8])]
    assert hs == sorted(hs) and hs[-1] < 65536


def test_offsets_to_E():
    assert linker.offsets_to_E([0.0, 1.0, -8.0, 8.0, -0.7]) == [
        65536, round(65536 * exp(1)), 22, round(65536 * exp(8)), round(65536 * exp(-0.7))]
    assert all(isinstance(e, int) for e in linker.offsets_to_E([0.3]))


def test_hazard_draw_is_one_xorshift_and_compares_the_high_half():
    s = 12345
    u = xorshift32(s)
    assert linker.hazard_draw(s, u >> 16) == (False, u)           # (u >> 16) < h16 is strict
    assert linker.hazard_draw(s, (u >> 16) + 1) == (True, u)
    assert linker.hazard_draw(s, 0) == (False, u)
    assert linker.hazard_draw(s, 65536) == (True, u)


@needs_s10
def test_hazard_cell(ids):
    cell = lambda *a: linker.hazard_cell(ids, *a)                  # (pending, depth, prev, pos)
    assert cell(True, 3, ids.stop, 0) == 2 * 4 + 0                  # pending wins
    assert cell(False, 1, ids.stop, 300) == 3 * 4 + 1               # inside a block
    assert cell(False, 0, ids.stop, 600) == 0 * 4 + 2               # right after Stop
    assert cell(False, 0, ids.ret, 1023) == 0 * 4 + 2               # right after Return
    assert cell(False, 0, ids.lbl, 255) == 1 * 4 + 0
    assert cell(False, 0, ids.lbl, 256) == 1 * 4 + 1
    assert cell(False, 0, ids.lbl, 511) == 1 * 4 + 1
    assert cell(False, 0, ids.lbl, 512) == 1 * 4 + 2
    assert cell(False, 0, ids.lbl, 1024) == 1 * 4 + 3


@needs_s10
def test_topk_weights_give_choose_s_draw(s10):
    """rule 7's w_e and W come from the sampler's own top-k weights: drawing from them with
    choose's arithmetic (xorshift32, state % W, cumulative) picks choose's token."""
    rng = np.random.default_rng(5)
    v = len(s10.vocab)
    for trial in range(200):
        logits = rng.integers(-3000, 3000, size=v).astype(np.int64)
        if trial % 3 == 0:
            logits[rng.integers(4, v, size=30)] = logits.max() - rng.integers(0, 40, size=30)
        banned = rng.random(v) < 0.2
        cfg = SamplerConfig(int(rng.integers(100, 400)), int(rng.integers(1, 65)), False, 1)
        state = int(rng.integers(1, 2**32))
        cand, w = linker.topk_weights(s10, logits, cfg, banned)
        nxt = xorshift32(state)
        target, acc = nxt % sum(w), 0
        for tok, wi in zip(cand, w):
            acc += wi
            if acc > target:
                break
        assert choose(s10, logits, cfg, state, banned) == (tok, nxt)


# ---- the offset fit ----------------------------------------------------------------------------

def _synthetic(rng, n, b):
    w_e = rng.integers(1, 65536, size=n)
    W = w_e + rng.integers(0, 20 * 65535, size=n)
    z = w_e * np.exp(b)
    return w_e, W, z / (z + (W - w_e))


def test_solve_offset_recovers_a_known_offset():
    rng = np.random.default_rng(0)
    for b in (-3.0, -0.4, 0.0, 1.3, 4.0):
        w_e, W, h = _synthetic(rng, 5000, b)
        assert linker.solve_offset(w_e, W, float(h.sum())) == pytest.approx(b, abs=1e-7)
    w_e, W, h = _synthetic(rng, 200_000, 1.3)                       # from Bernoulli draws
    y = rng.random(len(h)) < h
    assert linker.solve_offset(w_e, W, float(y.sum())) == pytest.approx(1.3, abs=0.05)


def test_solve_offset_edges_and_clipping():
    w_e = np.array([0, 100, 50, 7])
    W = np.array([500, 100, 1000, 9000])                            # row 1: h = 1 at any b
    assert linker.solve_offset(w_e, W, 0.5) == -8.0                  # below the h = 1 floor
    assert linker.solve_offset(w_e, W, 3.5) == 8.0                   # above what e^8 reaches
    b = linker.solve_offset(w_e, W, 1.5)
    hs = [0, 1, 50 * exp(b) / (50 * exp(b) + 950), 7 * exp(b) / (7 * exp(b) + 8993)]
    assert sum(hs) == pytest.approx(1.5, abs=1e-9)


def test_solve_cells_marks_empty_and_eos_free_cells():
    rng = np.random.default_rng(1)
    w_e, W, h = _synthetic(rng, 3000, 0.7)
    y = (rng.random(3000) < h).astype(np.int64)
    cell = np.where(np.arange(3000) < 1500, 5, 9)
    y[cell == 9] = 0
    rows = np.stack([cell, w_e, W, y], 1)
    b, info = linker.solve_cells(rows)
    assert len(b) == 16 and info["status"][5] == "fit" and b[5] == pytest.approx(0.7, abs=0.2)
    assert b[9] == 0.0 and info["status"][9] == "no_eos" and info["n"][9] == 1500
    assert b[0] == 0.0 and info["status"][0] == "empty" and info["n"][0] == 0
    assert info["eos"][5] == int(y[:1500].sum())


def test_train_programs_selection(tmp_path):
    progs = [[BOS, 5, 6, EOS], [BOS, 7, EOS], [BOS, 8, 9], [BOS, 10, 11, 12, EOS], [BOS, 13, EOS]]
    np.asarray([t for p in progs for t in p], dtype="<u2").tofile(tmp_path / "train.bin")
    got = linker.train_programs(tmp_path, 2)
    kept = [p for p in progs if p[-1] == EOS]
    idx = sorted(np.random.default_rng(0).choice(len(kept), 2, replace=False))
    assert [list(p) for p in got] == [kept[i] for i in idx]
    assert linker.programs_sha256(got) == hashlib.sha256(
        b"".join(np.asarray(kept[i], dtype="<u2").tobytes() for i in idx)).hexdigest()


@needs_s10
@pytest.mark.skipif(not VAL.exists(), reason="data/tok/val.bin not present")
def test_fit_rows_teacher_force_short_programs(s10):
    from model.data import load_split, program_starts
    s = load_split(VAL)
    st = program_starts(s)
    progs = [s[a:b] for a, b in zip(st, np.append(st[1:], len(s)))]
    progs = [p for p in progs if p[-1] == EOS and 20 < len(p) < 60][:2]
    rows = linker.fit_rows(s10, progs, workers=1)
    assert len(rows) == sum(len(p) - 1 for p in progs)
    cell, w_e, W, y = rows.T
    assert y.sum() == 2 and y[len(progs[0]) - 2] == 1 and y[-1] == 1
    assert ((0 <= cell) & (cell < 16)).all() and (W > 0).all() and ((0 <= w_e) & (w_e <= W)).all()
    assert w_e[len(progs[0]) - 2] > 0                       # <eos> is in the top-k where it occurs


# ---- rule 1: bind and soft bind ----------------------------------------------------------------

def _tracker(ids, hexes):
    sp = linker.XT(ids)
    sp.step(BOS)
    for h in hexes:
        sp.step(ids.vocab.index(h))
    return sp


def _bans(ids, rules, sp, logits):
    b = np.zeros(len(ids.vocab), dtype=bool)
    P = sorted(sp.unres)
    out = linker.rule_bans(rules, ids, sp, P, 1000, 1, b, logits)
    return b, out


@needs_s10
def test_bind_restricts_a_lbl_name_to_pending_names(ids):
    old = linker.ARMS["OLD"]
    logits = np.arange(len(ids.vocab), dtype=np.int64)
    pre = ["D7", "41", "42", "3F", "D6"]                   # Goto AB / Lbl
    for extra, allowed in [([], [ids.char["A"]]), (["41"], [ids.char["B"]]),
                           (["41", "42"], [ids.nl, ids.colon])]:
        sp = _tracker(ids, pre + extra)
        assert sp.name.kind == "lbl" and sp.unres == {"AB"}
        b, out = _bans(ids, old, sp, logits)
        assert list(np.flatnonzero(~b)) == sorted(allowed) and out is logits


@needs_s10
def test_soft_bind_boosts_the_pending_continuations_instead(ids):
    lds = linker.ARMS["LDs"]
    logits = np.arange(len(ids.vocab), dtype=np.int64)
    sp = _tracker(ids, ["D7", "41", "42", "3F", "D6", "41"])  # Goto AB / Lbl A
    b, out = _bans(ids, lds, sp, logits)
    assert not b.any()
    boost = np.zeros(len(logits), dtype=np.int64)
    boost[ids.char["B"]] = 512
    assert np.array_equal(out, logits + boost) and logits[ids.char["B"]] == ids.char["B"]
    sp = _tracker(ids, ["D7", "41", "42", "3F", "D6", "41", "42"])
    b, out = _bans(ids, lds, sp, logits)
    assert not b.any() and list(np.flatnonzero(out != logits)) == sorted([ids.nl, ids.colon])


@needs_s10
def test_soft_bind_still_bans_redefinition_while_references_are_pending(ids):
    lds, old = linker.ARMS["LDs"], linker.ARMS["OLD"]
    logits = np.zeros(len(ids.vocab), dtype=np.int64)
    pre = ["D6", "43", "44", "3F", "D7", "41", "42", "3F", "D6"]   # Lbl CD / Goto AB / Lbl
    sp = _tracker(ids, pre)
    assert sp.unres == {"AB"} and sp.defined == {"CD"}
    b, out = _bans(ids, lds, sp, logits)                           # boost A, nothing banned
    assert not b.any() and list(np.flatnonzero(out)) == [ids.char["A"]]
    sp = _tracker(ids, pre + ["43"])                               # Lbl C: CD is defined
    b, out = _bans(ids, lds, sp, logits)
    assert list(np.flatnonzero(b)) == [ids.char["D"]] and not out.any()
    sp = _tracker(ids, pre + ["43", "44"])                         # Lbl CD
    b, _ = _bans(ids, lds, sp, logits)
    assert list(np.flatnonzero(b)) == sorted([ids.nl, ids.colon, EOS])
    b, _ = _bans(ids, old, sp, logits)                             # hard bind: no pending fits
    assert not b.any()


@needs_s10
def test_no_redefinition_is_a_ban_in_both_bind_arms(ids):
    logits = np.zeros(len(ids.vocab), dtype=np.int64)
    sp = _tracker(ids, ["D6", "41", "42", "3F", "D6", "41"])  # Lbl AB / Lbl A, nothing pending
    assert not sp.unres and sp.defined == {"AB"}
    for arm in ("OLD", "LDs"):
        b, out = _bans(ids, linker.ARMS[arm], sp, logits)
        assert list(np.flatnonzero(b)) == [ids.char["B"]] and out is logits
    sp = _tracker(ids, ["D6", "41", "42", "3F", "D6", "41", "42"])  # Lbl AB / Lbl AB
    b, _ = _bans(ids, linker.ARMS["LDs"], sp, logits)
    assert list(np.flatnonzero(b)) == sorted([ids.nl, ids.colon, EOS])
    b, _ = _bans(ids, linker.ARMS["M"], sp, logits)
    assert not b.any()


# ---- generation --------------------------------------------------------------------------------

@needs_s10
@pytest.mark.parametrize("seed", [1, 2, 3, 0])
def test_arm_M_is_intref_generate_with_masks(s10, seed):
    from model.intmask import IntMask
    want = intref_generate(s10, [], 127, SamplerConfig(205, 20, False, seed), cap=128,
                           mask=IntMask(s10.vocab))
    got = linker.generate(s10, linker.ARMS["M"], seed, cap=128)
    assert got["ids"] == want and not any(got["stats"].values())


@needs_s10
@pytest.mark.slow
def test_arm_M_is_intref_generate_with_masks_to_the_cap(s10):
    from model.intmask import IntMask
    want = intref_generate(s10, [], 1023, SamplerConfig(205, 20, False, 4), cap=1024,
                           mask=IntMask(s10.vocab))
    assert linker.generate(s10, linker.ARMS["M"], 4)["ids"] == want


class _Fixed:
    """A stand-in model whose logits never change: <eos> on top, then A, B and 0."""

    def __init__(self, s10):
        self.vocab, self.expw = s10.vocab, s10.expw
        self.logits = np.full(len(s10.vocab), -10_000, dtype=np.int64)
        for h, lg in (("41", -100), ("42", -200), ("30", -300)):
            self.logits[s10.vocab.index(h)] = lg
        self.logits[EOS] = 0

    def new_cache(self):
        return None

    def step(self, tok, pos, cache):
        return self.logits.copy()


def _hand_hz(fake, ids, seed, E, cap):
    """Rule 7 alone, written out: one xorshift32 per hazard step, before choose's own."""
    from model.intmask import IntMask
    cfg = SamplerConfig(205, 20, False, seed)
    mask = IntMask(fake.vocab)
    mask.reset()
    mask.step(BOS)
    toks, state, draws = [BOS], seed, 0
    while len(toks) < cap:
        b = mask.banned()
        if len(toks) > 1 and not b[EOS]:
            cand, w = linker.topk_weights(fake, fake.logits, cfg, b)
            h = linker.hazard16(w[cand.index(EOS)], sum(w), E[linker.hazard_cell(
                ids, False, 0, toks[-1], len(toks) - 1)])
            state, draws = xorshift32(state), draws + 1
            if (state >> 16) < h:
                toks.append(EOS)
                break
        b[EOS] = True
        tok, state = choose(fake, fake.logits, cfg, state, b)
        toks.append(tok)
        mask.step(tok)
    return toks, draws


@needs_s10
def test_hazard_steps_draw_once_more_than_plain_sampling(s10, ids):
    fake = _Fixed(s10)
    E = linker.offsets_to_E([-4.5] * 16)
    stops = 0
    for seed in range(1, 13):
        want, draws = _hand_hz(fake, ids, seed, E, 64)
        got = linker.generate(fake, linker.ARMS["HZ"], seed, E=E, cap=64)
        assert got["ids"] == want
        assert got["stats"]["hazard_draws"] == draws == len(want) - 2   # none at position 0
        assert got["stats"]["hazard_stop"] == (want[-1] == EOS)
        stops += want[-1] == EOS
    assert 0 < stops < 12                                   # both outcomes are exercised


@needs_s10
def test_hazard_arm_needs_offsets(s10):
    with pytest.raises(ValueError, match="offsets"):
        linker.generate(s10, linker.ARMS["LD"], 1, cap=16)


@needs_s10
def test_same_seed_same_output(s10):
    E = linker.offsets_to_E([0.5, 1.0, 1.5, 2.0, 0.0, 1.0, 1.0, 1.0, -1, -1, -1, -1, 0, 0, 0, 0])
    for arm in ("LD", "LDs"):
        a = linker.generate(s10, linker.ARMS[arm], 7, E=E, cap=160)
        b = linker.generate(s10, linker.ARMS[arm], 7, E=E, cap=160)
        assert a == b


# ---- port parity with odec2 (preregistered port check) ----------------------------------------

@needs_s10
@pytest.mark.slow
@pytest.mark.parametrize("name,rules", [
    ("old_r1_r2", linker.Rules(old=True, r1=True, r2=True, row_bos_fix=False)),
    ("old", linker.Rules(old=True, row_bos_fix=False))])
def test_port_reproduces_odec2_on_s10(name, rules):
    """Hazard off and Row's <bos> fix off: odec2's token sequences on s10, seeds 1-50 (old, R1,
    R2) and 1-20 (old). A missing fixture fails the test (it is not skipped)."""
    ref = json.loads(PARITY.read_text())[name]
    assert ref["rules"] == {"old": True, "hazard": False, "r1": rules.r1, "r2": rules.r2}
    seeds = sorted(int(s) for s in ref["seeds"])
    assert len(seeds) == {"old_r1_r2": 50, "old": 20}[name]
    for r in linker.sample_records(V2_EXPORT, rules, seeds, workers=12):
        want = ref["seeds"][str(r["seed"])]
        assert (len(r["ids"]), _sha(r["ids"])) == (want["len"], want["sha256"]), r["seed"]
        assert {k: r["stats"][k] for k in want["stats"]} == want["stats"], r["seed"]


# ---- CLI ---------------------------------------------------------------------------------------

@needs_s10
def test_cli_sample_writes_records_in_seed_order(tmp_path):
    out = tmp_path / "s.jsonl"
    linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "OLD", "--n", "3", "--seed0", "5",
                 "--cap", "48", "--out", str(out), "--workers", "2"])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["seed"] for r in recs] == [5, 6, 7]
    assert list(recs[0])[:6] == ["seed", "temp", "top_k", "cap", "masks", "arm"]
    assert recs[0]["temp"] == 0.8 and recs[0]["top_k"] == 20 and recs[0]["cap"] == 48
    assert recs[0]["masks"] is True and recs[0]["arm"] == "OLD" and recs[0]["horizons"] is False
    assert recs[0]["offsets_sha256"] is None
    assert recs[0]["ids"][0] == BOS and len(recs[0]["ids"]) <= 48 and "forced_lbl" in recs[0]["stats"]


@needs_s10
def test_cli_hazard_arm_reads_E_from_the_offsets_file(tmp_path, s10):
    offs = tmp_path / "offs.json"
    E = linker.offsets_to_E([1.0] * 16)
    offs.write_text(json.dumps({"b": [1.0] * 16, "E": E, "export": str(V2_EXPORT)}))
    out = tmp_path / "s.jsonl"
    linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "LD", "--offsets", str(offs),
                 "--n", "2", "--cap", "40", "--out", str(out), "--workers", "2"])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    sha = hashlib.sha256(offs.read_bytes()).hexdigest()
    for r in recs:
        assert r["ids"] == linker.generate(s10, linker.ARMS["LD"], r["seed"], E=E, cap=40)["ids"]
        assert r["offsets_sha256"] == sha
    with pytest.raises(SystemExit):
        linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "HZ", "--n", "1",
                     "--out", str(out), "--workers", "1"])


@needs_s10
def test_cli_fit_writes_offsets_and_cells(tmp_path, s10):
    idx = {h: i for i, h in enumerate(s10.vocab)}
    A, B, NL, STOP = idx["41"], idx["42"], idx["3F"], idx["D9"]
    progs = [[BOS, A, NL, B, EOS], [BOS, B, NL, STOP, EOS], [BOS, A, B, NL, A, NL, STOP, EOS]]
    np.asarray([t for p in progs for t in p], dtype="<u2").tofile(tmp_path / "train.bin")
    out = tmp_path / "offs.json"
    linker.main(["fit", "--export", str(V2_EXPORT), "--data", str(tmp_path), "--n-programs", "2",
                 "--out", str(out), "--workers", "1"])
    r = json.loads(out.read_text())
    assert len(r["b"]) == len(r["E"]) == 16 and r["E"] == linker.offsets_to_E(r["b"])
    assert r["n_programs"] == 2 and sum(r["cells"]["eos"]) == 2
    assert sum(r["cells"]["n"]) == sum(len(p) - 1 for p in linker.train_programs(tmp_path, 2))
    assert r["programs_sha256"] == linker.programs_sha256(linker.train_programs(tmp_path, 2))
    assert {r["cells"]["status"][c] for c in range(16) if r["cells"]["n"][c] == 0} <= {"empty"}


@needs_s10
def test_cli_horizons_flag_reaches_the_model(tmp_path):
    out = tmp_path / "h.jsonl"
    linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "M", "--horizons", "--n", "1",
                 "--cap", "24", "--out", str(out), "--workers", "1"])
    assert json.loads(out.read_text())["horizons"] is True
    assert linker.load_model(V2_EXPORT, horizons=True).horizons is not None


@needs_s10
def test_fit_skips_positions_where_the_masks_leave_only_eos(s10):
    """Reading a custom list the tracker sees as unassigned (`Disp ʟAB`): M bans every id and
    falls back to <eos> alone, so h = 1 whatever b. The sampler never gets there (M bans the ʟ),
    and such positions would pin the fit at b = -8; they get cell -1 and are left out."""
    idx = {h: i for i, h in enumerate(s10.vocab)}
    prog = [BOS, idx["DE"], idx["EB"], idx["41"], idx["42"], idx["3F"], idx["41"], EOS]
    rows = linker.fit_rows(s10, [prog], workers=1)
    assert list(rows[:, 0] == -1) == [False, False, True, True, True, False, False]
    b, info = linker.solve_cells(rows)
    assert sum(info["n"]) == 4 and sum(info["eos"]) == 1


class _Script:
    """A stand-in model that writes `script` (then B forever): the scripted token is the only
    one with any top-k weight, so sampling follows it whatever the seed."""

    def __init__(self, s10, script):
        self.vocab, self.expw, self.script = s10.vocab, s10.expw, script
        self.other = s10.vocab.index("42")

    def new_cache(self):
        return None

    def step(self, tok, pos, cache):
        lg = np.full(len(self.vocab), -10_000, dtype=np.int64)
        lg[self.script[pos + 1] if pos + 1 < len(self.script) else self.other] = 0
        return lg


def test_all_arms_fix_row_bos():
    assert all(r.row_bos_fix for r in linker.ARMS.values())


@needs_s10
def test_row_bos_fix_sees_a_first_line_lbl(ids):
    lbl, a, nl, goto = ids.lbl, ids.char["A"], ids.nl, ids.goto
    prog = [lbl, a, nl, goto, a, nl]                                # Lbl A / Goto A
    for fix in (True, False):
        row = linker.Row(ids)
        for t in prog if fix else [BOS, *prog]:
            row.step(t)
        if fix:
            assert row.defined == {"A"} and row.pending == [] and row.closure()
        else:                                                       # odec2: statement 1 unseen
            assert row.defined == set() and row.pending == ["A"]


@needs_s10
def test_row_bos_fix_lets_r1_end_after_a_loop_back_to_a_first_line_lbl(s10, ids):
    getkey = ids.vocab.index("AD")                                  # an exit token for rule 4
    script = [BOS, ids.lbl, ids.char["A"], ids.nl, getkey, ids.nl, ids.goto, ids.char["A"], ids.nl]
    fake = _Script(s10, script)
    on = linker.generate(fake, linker.Rules(old=True, r1=True, r2=True), 3, cap=40)
    assert on["ids"] == script + [EOS] and on["stats"]["r1"] == 1
    off = linker.generate(fake, linker.Rules(old=True, r1=True, r2=True, row_bos_fix=False), 3,
                          cap=40)
    assert off["ids"][:len(script) + 1] == script + [ids.lbl]       # R2 forces a Lbl instead
    assert off["stats"]["r1"] == 0 and off["stats"]["r2"] >= 1


# ---- rule 7 against an independent loop (model/tests/linker_ref.py) -------------------------------

HOT = {"block": {12, 13, 14}, "pending": {8, 9, 10}, "afterstop": {0, 1, 2}, "clean_b0": {4},
       "clean_b1": {5}, "clean_b2": {6}}


def compare_job(a) -> tuple:
    """(conf, arm, seed, cap) -> (a, same ids, reference draws, port draws, reference stops)."""
    from model.tests import linker_ref
    conf, arm, seed, cap = a
    E = linker_ref.hot_E(HOT[conf])
    ref = linker_ref.gen_ref(seed, E=E, cap=cap, **linker_ref.ARMS[arm])
    port = linker.generate(linker_ref.init(), linker.ARMS[arm], seed, E=E, cap=cap)
    return (a, ref["ids"] == port["ids"], ref["stats"]["draws"], port["stats"]["hazard_draws"],
            ref["stats"]["hazard_stop"])


def _check(results):
    for a, same, d_ref, d_port, _ in results:
        assert same and d_ref == d_port, a
    assert sum(r[4] for r in results) > 0 and sum(r[2] for r in results) > 0


@needs_s10
def test_hazard_cells_match_an_independent_loop_fast():
    _check([compare_job(a) for a in (("pending", "HZ", 1, 300), ("block", "LD", 1, 300),
                                      ("clean_b0", "LD", 2, 300), ("clean_b1", "HZ", 2, 300))])


@needs_s10
def test_hazard_bins_switch_at_positions_256_and_512(s10):
    """<eos> always in the top-k, +8 only in clean bin 1 (or 2): the stop lands exactly at
    position 256 (or 512), i.e. ids of length 258 (514)."""
    from model.tests.linker_ref import hot_E
    fake = _Fixed(s10)
    for hot, cap, n, seeds in ((5, 300, 258, (1, 2, 3)), (6, 600, 514, (1,))):
        for seed in seeds:
            out = linker.generate(fake, linker.ARMS["HZ"], seed, E=hot_E({hot}), cap=cap)
            assert len(out["ids"]) == n and out["ids"][-1] == EOS, (hot, seed)


@needs_s10
@pytest.mark.slow
def test_hazard_cells_match_an_independent_loop():
    """+8 in one state (or bin) and -8 elsewhere, so a wrong cell flips a stop; HZ, LD, LDs."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(c, arm, seed, 600 if c == "clean_b2" else 300) for c in HOT for arm in ("HZ", "LD")
            for seed in (1, 2, 3)]
    jobs += [(c, "LDs", seed, 300) for c in ("pending", "clean_b0") for seed in (1, 2, 3)]
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        _check(list(pool.map(compare_job, jobs, chunksize=1)))


# ---- fit rows: cells and alignment on a crafted program -----------------------------------------

def _crafted(ids):
    h = lambda *xs: [ids.vocab.index(x) for x in xs]            # noqa: E731
    head = [BOS, *h("D9", "3F", "D3", "49", "2B", "31", "2B", "32", "3F", "DE", "49", "3F", "D4",
                   "3F", "D7", "41", "3F")]   # Stop / For(I,1,2 / Disp I / End / Goto A
    return head + h("DE", "31", "3F") * 81 + h("D6", "41", "3F") + [EOS]   # Disp 1 ... / Lbl A


@needs_s10
def test_fit_rows_cells_and_alignment_on_a_crafted_program(s10, ids):
    from model.tests import linker_ref
    prog = _crafted(ids)
    rows = linker.fit_rows(s10, [prog], workers=1)
    assert len(rows) == len(prog) - 1                               # row t predicts token t+1
    cell, y = rows[:, 0], rows[:, 3]
    assert cell[0] == 4                                             # after <bos>: clean, bin 0
    assert cell[1] == 0                                             # after Stop
    assert cell[2] == 4                                             # after its newline
    assert cell[3] == 12 and cell[12] == 12                         # For( ... inside the loop
    assert cell[13] == 4                                            # End closes it
    assert cell[16] == 4 and cell[17] == 8                          # Goto A is pending at its \n
    assert cell[255] == 8 and cell[256] == 9                        # the 255/256 bin boundary
    assert cell[-1] == 5 and list(np.flatnonzero(y)) == [len(prog) - 2]
    assert np.array_equal(rows, linker_ref.fit_rows_ref(prog))


# ---- rule 3, rule 2's fallback and the hazard under an <eos> ban (scripted models) --------------

@needs_s10
def test_rule3_threshold_is_D_plus_3_plus_body(s10, ids):
    """Goto A pends A (D = 1 + 3 + 1 + 24 = 29). The next Goto's name may only be A or a defined
    name once R <= D + 27: at cap 60 R is 55 there, at cap 61 R is 56, at cap 62 R is 57."""
    a, b = ids.char["A"], ids.char["B"]
    fake = _Script(s10, [BOS, ids.goto, a, ids.nl, ids.goto, b, ids.nl])
    for cap, want in ((60, a), (61, a), (62, b)):
        assert linker.generate(fake, linker.Rules(old=True), 1, cap=cap)["ids"][5] == want, cap


class _EosOnlyAt:
    """IntMask, except that with `at` tokens fed it bans every id but <eos>."""

    def __init__(self, vocab, at):
        from model.intmask import IntMask
        self.m, self.at, self.fed = IntMask(vocab), at, 0

    def reset(self):
        self.m.reset()
        self.fed = 0

    def step(self, tok):
        self.m.step(tok)
        self.fed += 1

    def banned(self):
        b = self.m.banned()
        if self.fed == self.at:
            b[:] = True
            b[EOS] = False
        return b


@needs_s10
def test_rule2_turns_a_drawn_eos_into_lbl(s10, ids, monkeypatch):
    """With a reference pending, <eos> is drawn only when nothing else is allowed; it becomes
    Lbl at a statement start and newline + Lbl elsewhere (rule 1 then binds the name)."""
    a, one = ids.char["A"], ids.vocab.index("31")
    fake = _Script(s10, [BOS, ids.goto, a, ids.nl, ids.vocab.index("DE"), one, ids.nl])
    for at, want in ((4, [ids.lbl, a]), (6, [ids.nl, ids.lbl, a])):
        monkeypatch.setattr(linker, "IntMask", lambda v, level=1, at=at: _EosOnlyAt(v, at))
        out = linker.generate(fake, linker.Rules(old=True), 1, cap=40)
        assert out["ids"][at:at + len(want)] == want and out["stats"]["eos_sub"] == 1, at


@needs_s10
def test_hazard_is_not_drawn_where_the_masks_ban_eos(s10, ids):
    """{1}->ʟAB / Disp ʟAB: inside the read of ʟAB, M bans <eos> until the name is complete."""
    from model.intmask import IntMask
    h = lambda *xs: [ids.vocab.index(x) for x in xs]            # noqa: E731
    script = [BOS, *h("08", "31", "09", "04", "EB", "41", "42", "3F", "DE", "EB", "41", "42", "3F")]
    out = linker.generate(_Script(s10, script), linker.ARMS["HZ"], 1,
                          E=linker.offsets_to_E([0.0] * 16), cap=len(script) + 3)
    assert out["ids"][:len(script)] == script
    mask, free, banned = IntMask(s10.vocab), 0, 0
    mask.reset()
    for i, t in enumerate(out["ids"][:-1]):
        mask.step(t)
        if i >= 1:                                      # no hazard at position 0
            eos_banned = bool(mask.banned()[EOS])
            free, banned = free + (not eos_banned), banned + eos_banned
    assert banned >= 2 and out["stats"]["hazard_draws"] == free


@needs_s10
def test_cli_refuses_offsets_fitted_on_another_export(tmp_path, monkeypatch):
    offs, out = tmp_path / "offs.json", tmp_path / "s.jsonl"
    base = {"b": [0.0] * 16, "E": [65536] * 16}
    for export in ("runs/lab/S0-s3/export", None):
        offs.write_text(json.dumps({**base, "export": export} if export else base))
        with pytest.raises(SystemExit, match="export"):
            linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "LD", "--offsets", str(offs),
                         "--n", "1", "--cap", "8", "--out", str(out), "--workers", "1"])
    assert not out.exists()
    # the same export written relative to the repo root, read from there
    monkeypatch.chdir(ROOT)
    offs.write_text(json.dumps({**base, "export": "runs/v2/export"}))
    linker.main(["sample", "--export", str(V2_EXPORT), "--arm", "LD", "--offsets", str(offs),
                 "--horizons", "--n", "1", "--cap", "8", "--out", str(out), "--workers", "1"])
    assert json.loads(out.read_text())["horizons"] is True


# ---- E2b: rule 7 on the full T 1.0 distribution (PLAN-v2.2-E2b.md) ------------------------------

@needs_s10
def test_full_hazard_weights_are_T1_over_every_allowed_id(s10):
    from model.intref import ranked_candidates
    rng = np.random.default_rng(7)
    v = len(s10.vocab)
    for _ in range(50):
        logits = rng.integers(-3000, 3000, size=v).astype(np.int64)
        banned = rng.random(v) < 0.3
        cfg = SamplerConfig(205, 20, False, 1)
        cand, w = linker.hazard_weights(s10, logits, cfg, banned, full=True)
        assert cand == ranked_candidates(logits, v, banned)          # every allowed id
        assert w == linker.topk_weights(s10, logits, SamplerConfig(256, v, False, 1), banned)[1]
        assert linker.hazard_weights(s10, logits, cfg, banned, full=False) == \
            linker.topk_weights(s10, logits, cfg, banned)             # E2's rule 7, unchanged


@needs_s10
def test_fit_rows_use_the_full_weights_when_asked(s10):
    idx = {h: i for i, h in enumerate(s10.vocab)}
    prog = [BOS, idx["41"], idx["3F"], idx["D9"], EOS]
    top = linker.fit_rows(s10, [prog], workers=1)
    full = linker.fit_rows(s10, [prog], workers=1, full=True)
    assert (top[:, [0, 3]] == full[:, [0, 3]]).all()                  # same cells and targets
    assert (full[:, 2] >= top[:, 2]).all() and (full[:, 2] > top[:, 2]).any()   # W over all ids


@needs_s10
def test_cli_fit_and_sample_keep_the_hazard_mode_and_horizons_together(tmp_path, s10):
    idx = {h: i for i, h in enumerate(s10.vocab)}
    progs = [[BOS, idx["41"], idx["3F"], idx["42"], EOS], [BOS, idx["42"], idx["3F"], idx["D9"], EOS]]
    np.asarray([t for p in progs for t in p], dtype="<u2").tofile(tmp_path / "train.bin")
    full, top = tmp_path / "full.json", tmp_path / "top.json"
    linker.main(["fit", "--export", str(V2_EXPORT), "--data", str(tmp_path), "--n-programs", "2",
                 "--out", str(full), "--workers", "1", "--hazard-full", "--horizons"])
    linker.main(["fit", "--export", str(V2_EXPORT), "--data", str(tmp_path), "--n-programs", "2",
                 "--out", str(top), "--workers", "1"])
    f = json.loads(full.read_text())
    assert f["hazard_full"] is True and f["horizons"] is True and f["temperature_q8"] == 256
    assert json.loads(top.read_text())["hazard_full"] is False
    out = tmp_path / "s.jsonl"
    args = ["sample", "--export", str(V2_EXPORT), "--n", "1", "--cap", "24", "--out", str(out),
            "--workers", "1"]
    linker.main([*args, "--arm", "LD2", "--offsets", str(full), "--horizons"])
    for bad in (["--arm", "LD2", "--offsets", str(top), "--horizons"],    # top-k offsets
                ["--arm", "LD2", "--offsets", str(full)],                 # fitted with horizons
                ["--arm", "LD", "--offsets", str(full), "--horizons"]):   # full offsets for LD
        with pytest.raises(SystemExit):
            linker.main([*args, *bad])


@pytest.mark.slow
@needs_s10
def test_ld_still_reproduces_e2s_s10_samples():
    """PLAN-v2.2-E2b.md check: with hazard_full off, LD's samples are E2's, byte for byte."""
    ref = Path(__file__).resolve().parents[2] / "runs" / "e12" / "S0-s10" / "LD" / "samples.jsonl"
    off = ref.parents[1] / "offsets.json"
    if not ref.exists() or not off.exists():
        pytest.skip("needs E2's s10 LD run")
    E = [int(x) for x in json.loads(off.read_text())["E"]]
    recs = [json.loads(x) for x in ref.read_text().splitlines()[:10]]
    model = IntModel.load(V2_EXPORT)
    for r in recs:
        assert linker.generate(model, linker.ARMS["LD"], r["seed"], E=E, cap=1024) == \
            {"ids": r["ids"], "stats": r["stats"]}


@needs_s10
def test_ld2_draws_rule_7_from_the_full_weights_and_ld_does_not(s10, monkeypatch):
    seen = []
    real = linker.hazard_weights

    def spy(model, logits, cfg, banned, full):
        seen.append(full)
        return real(model, logits, cfg, banned, full)
    monkeypatch.setattr(linker, "hazard_weights", spy)
    E = linker.offsets_to_E([0.0] * 16)
    linker.generate(s10, linker.ARMS["LD2"], 3, E=E, cap=60)
    assert seen and all(seen)
    seen.clear()
    linker.generate(s10, linker.ARMS["LD"], 3, E=E, cap=60)
    assert seen and not any(seen)


@needs_s10
def test_ld2_counts_draws_and_stops_per_cell_and_ld_keeps_its_stats(s10):
    E = linker.offsets_to_E([2.0] * 16)
    st = linker.generate(s10, linker.ARMS["LD2"], 5, E=E, cap=200)["stats"]
    assert len(st["draws_c"]) == len(st["stops_c"]) == linker.N_CELLS
    assert sum(st["draws_c"]) == st["hazard_draws"] and sum(st["stops_c"]) == st["hazard_stop"]
    ld = linker.generate(s10, linker.ARMS["LD"], 5, E=E, cap=200)["stats"]
    assert set(ld) == {"forced_lbl", "hazard_stop", "r1", "r2", "eos_sub", "hazard_draws"}


# ---- v2.4 mask level (docs/agents/phase1-safety-masks.md) -------------------------------------

@needs_s10
def test_mask_level_reaches_intmask_and_level_1_is_the_default(s10, monkeypatch):
    seen, real = [], linker.IntMask
    monkeypatch.setattr(linker, "IntMask", lambda vocab, level=1: seen.append(level) or real(
        vocab, level))
    linker.generate(s10, linker.ARMS["OLD"], 1, cap=20)
    linker.generate(s10, linker.ARMS["OLD"], 1, cap=20, mask_level=2)
    assert seen == [1, 2]


@needs_s10
def test_level_2_samples_hold_no_always_banned_token(s10):
    from model import safety as S
    idx = {h: i for i, h in enumerate(s10.vocab)}
    always = {idx[h] for h in S.ALWAYS_HEX}
    for seed in range(1, 9):
        out = linker.generate(s10, linker.ARMS["OLD"], seed, cap=300, mask_level=2)
        assert not always & set(out["ids"]), seed


@needs_s10
def test_cli_mask_level(tmp_path, s10):
    out = tmp_path / "s.jsonl"
    base = ["sample", "--export", str(V2_EXPORT), "--arm", "OLD", "--n", "2", "--cap", "40",
            "--workers", "1"]
    linker.main(base + ["--out", str(out), "--mask-level", "2"])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    for r in recs:
        assert r["mask_level"] == 2
        assert r["ids"] == linker.generate(s10, linker.ARMS["OLD"], r["seed"], cap=40,
                                           mask_level=2)["ids"]
    linker.main(base + ["--out", str(out)])
    assert all("mask_level" not in json.loads(x) for x in out.read_text().splitlines())
