"""lab.e14 (PLAN-E14.md): LD2 from a task opening (generate_prompted, a copy of
model.linker.generate's loop behind prompt feeding) and the `sample` command whose output
`lab.tasks score2` judges unchanged.

Checks against saved runs read runs/ and dist/ (git-ignored, read only; CALCLM_RUNS /
CALCLM_DIST point a worktree at the main checkout's) and skip without them."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from lab import e14
from lab import tasks as T
from lab import tasks_v2 as V
from model import linker
from model.intops import xorshift32
from model.intref import IntModel, SamplerConfig, choose
from model.tokens import BOS, EOS, PAD

ROOT = T.ROOT
RUNS = Path(os.environ.get("CALCLM_RUNS", ROOT / "runs"))
DIST = Path(os.environ.get("CALCLM_DIST", ROOT / "dist"))
S10 = RUNS / "v2" / "export"                                  # S0-s10 (= runs/lab/S0-s10/export)
L4 = RUNS / "e10" / "export"
L4_FIT = ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_L4-s23.json"
RC2 = DIST / "CalcLM-v2.3-rc2-reference.json"
E2B = RUNS / "e2b" / "S0-s10"
E10_L4_S10 = RUNS / "e10" / "s10" / "L4" / "tasks"
SLOW_WORKERS = 8                    # PLAN-E14 checks: at most 8 workers


def _tracker_available() -> bool:
    try:
        from tracker import bind
        bind.lib()
        return True
    except Exception:  # noqa: BLE001 - the C tracker needs gcc
        return False


have_tracker = pytest.mark.skipif(not _tracker_available(), reason="the C tracker needs gcc")
needs_s10 = pytest.mark.skipif(not (_tracker_available() and (S10 / "model.bin").exists()),
                               reason="runs/v2/export or the C tracker not present")


@pytest.fixture(scope="module")
def s10():
    return IntModel.load(S10)


@pytest.fixture(scope="module")
def s10h():
    return IntModel.load(S10, horizons=True)


@pytest.fixture(scope="module")
def ids(s10):
    return linker.vocab_ids(s10.vocab)


def ids_sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<u2").tobytes()).hexdigest()


def hw_E():
    return [int(x) for x in json.loads(
        (ROOT / "ce" / "tests" / "fixtures" / "ld2_offsets_full.json").read_text())["E"]]


# ---- the copy ------------------------------------------------------------------------------------

def _loop(fn) -> list[str]:
    src = inspect.getsource(fn).splitlines()
    start = next(i for i, x in enumerate(src) if x.strip() == "queued: list[int] = []")
    return [x for x in src[start:] if x.strip()]


def test_the_loop_is_model_linker_generates_unchanged():
    assert _loop(e14.generate_prompted) == _loop(linker.generate)


# ---- check (3): the empty prompt is unprompted LD2 ----------------------------------------------------

@needs_s10
@pytest.mark.parametrize("arm", ["LD2", "LD", "HZ", "M"])
def test_empty_prompt_is_linker_generate(s10h, arm):
    for seed in (1, 2, 3, 0):
        want = linker.generate(s10h, linker.ARMS[arm], seed, E=hw_E(), cap=160)
        assert e14.generate_prompted(s10h, linker.ARMS[arm], seed, hw_E(), [], cap=160) == want


@pytest.mark.skipif(not (RC2.exists() and (L4 / "L4-s23" / "model.bin").exists()),
                    reason="needs dist/CalcLM-v2.3-rc2-reference.json and the L4-s23 export")
def test_empty_prompt_gives_the_v23_rc2_reference_seed_20():
    ref = json.loads(RC2.read_text())
    assert (ref["seed"], ref["cap"], ref["ld2"], ref["horizons"]) == (20, 1024, True, True)
    assert ref["offsets_sha256"] == hashlib.sha256(L4_FIT.read_bytes()).hexdigest()
    model = IntModel.load(L4 / "L4-s23", horizons=True)
    out = e14.generate_prompted(model, linker.ARMS["LD2"], 20,
                                json.loads(L4_FIT.read_text())["E"], [])
    assert ids_sha(out["ids"]) == ref["ids_sha256"] == ids_sha(ref["ids"])


def _e2b_job(a):
    seeds, E = a
    model = IntModel.load(RUNS / "lab" / "S0-s10" / "export", horizons=True)
    return [e14.generate_prompted(model, linker.ARMS["LD2"], s, E, []) for s in seeds]


@pytest.mark.slow
@pytest.mark.skipif(not (E2B / "LD2H" / "samples.jsonl").exists(), reason="needs E2b's s10 run")
@pytest.mark.parametrize("part", range(2))      # 2 x 500 seeds: each under the 600 s limit
def test_empty_prompt_reproduces_e2bs_1000_s10_ld2h_samples(part):
    """E2b's s10 LD2H samples (model.linker.generate, LD2, horizons, its offsets_full.json):
    ids and stats for all 1,000 seeds (500 * part + 1 .. 500 * part + 500)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    offs = E2B / "offsets_full.json"
    E = [int(x) for x in json.loads(offs.read_text())["E"]]
    recs = [json.loads(x) for x in (E2B / "LD2H" / "samples.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in recs] == list(range(1, 1001))
    sha = hashlib.sha256(offs.read_bytes()).hexdigest()
    assert all(r["arm"] == "LD2" and r["horizons"] and r["offsets_sha256"] == sha for r in recs)
    recs = recs[500 * part:500 * (part + 1)]
    chunks = [list(range(a, a + 25)) for a in range(500 * part + 1, 500 * part + 501, 25)]
    with ProcessPoolExecutor(SLOW_WORKERS, mp_context=mp.get_context("spawn")) as pool:
        outs = [o for part in pool.map(_e2b_job, [(c, E) for c in chunks]) for o in part]
    bad = [r["seed"] for r, o in zip(recs, outs, strict=True)
           if o != {"ids": r["ids"], "stats": r["stats"]}]
    assert bad == []


# ---- check (4): prompted M is v2.1 decoding from the opening --------------------------------------

def _m_job(job):
    task_id, seed = job
    return e14.generate_prompted(T._model, linker.ARMS["M"], seed, None,
                                 V.TASKS_V2[task_id].opening_ids())["ids"]


@pytest.mark.slow
@pytest.mark.skipif(not (E10_L4_S10 / "samples.jsonl").exists(),
                    reason="needs E10's L4-s10 dev task samples")
@pytest.mark.parametrize("part", range(4))      # 4 x 500 jobs: each under the 600 s limit
def test_prompted_m_reproduces_e10s_l4_s10_dev_samples_row_for_row(part):
    """Arm M (masks only) fed each opening: E10's lab.tasks sample rows (intref.generate with the
    opening as prompt, horizons, k 50) for all 2,000 dev jobs (tasks 10*part .. 10*part+9)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    meta = json.loads((E10_L4_S10 / "sample.json").read_text())
    assert meta["export"] == "runs/e10/export/L4-s10" and meta["k"] == 50
    rows = [json.loads(x) for x in (E10_L4_S10 / "samples.jsonl").read_text().splitlines()]
    jobs = [(t.id, s) for t in T.SUITE for s in range(1, 51)]
    assert [(r["task"], r["seed"]) for r in rows] == jobs
    jobs, rows = jobs[500 * part:500 * (part + 1)], rows[500 * part:500 * (part + 1)]
    with ProcessPoolExecutor(SLOW_WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=T._load, initargs=(str(L4 / "L4-s10"),)) as pool:
        outs = list(pool.map(_m_job, jobs, chunksize=4))
    bad = [j for j, r, o in zip(jobs, rows, outs, strict=True) if r["ids"] != o]
    assert bad == []


# ---- check (5): the state after the prompt ---------------------------------------------------------

class _NoModel:
    """Logits are not part of check (5): a stand-in whose step costs nothing."""

    def __init__(self, vocab):
        self.vocab = vocab

    def new_cache(self):
        return None

    def step(self, tok, pos, cache):
        return np.zeros(len(self.vocab), dtype=np.int64)


def _xt(sp):
    d = {k: v for k, v in vars(sp).items()}
    d["name"] = None if sp.name is None else vars(sp.name)      # _Name has no __eq__
    return d


def _mask(m):
    return (m.state.raw, m.prev, m.ctx, m.bag, m.list_read, tuple(m.base))


@have_tracker
def test_state_after_every_opening_is_fresh_objects_stepped_through_it():
    """XT, Row and IntMask after feed_prompt equal fresh ones stepped through <bos> + opening
    (Row without <bos>), for all 60 openings: nothing pending, no closure point."""
    from model.intmask import IntMask
    vocab = T._vocab().entries
    ids_ = linker.vocab_ids(vocab)
    for t in V.SUITE_V2:
        toks = [BOS, *t.opening_ids()]
        f = e14.feed_prompt(_NoModel(vocab), linker.ARMS["LD2"], toks[1:])
        mask, sp, row = IntMask(vocab), linker.XT(ids_), linker.Row(ids_)
        mask.reset()
        for tok in toks:
            mask.step(tok)
            sp.step(tok)
        for tok in toks[1:]:
            row.step(tok)
        assert f.tokens == toks, t.id
        assert _mask(f.mask) == _mask(mask) and _xt(f.sp) == _xt(sp), t.id
        assert {k: v for k, v in vars(f.row).items() if k != "ids"} == \
            {k: v for k, v in vars(row).items() if k != "ids"}, t.id
        assert not f.sp.unres and not f.row.pending and not f.row.closure(), t.id
    assert len(V.SUITE_V2) == 60


@have_tracker
def test_dev_openings_are_lab_tasks_openings():
    assert all(V.TASKS_V2[t.id].opening_ids() == t.opening_ids() for t in T.SUITE)


class _Forced:
    """The real model, its logits replaced by a one-hot on `script`'s next token while the
    script lasts (so sampling writes it)."""

    def __init__(self, model, script):
        self.m, self.vocab, self.expw, self.script = model, model.vocab, model.expw, script

    def new_cache(self):
        return self.m.new_cache()

    def step(self, tok, pos, cache):
        real = self.m.step(tok, pos, cache)
        if pos + 1 < len(self.script):
            lg = np.full(len(self.vocab), -10_000, dtype=np.int64)
            lg[self.script[pos + 1]] = 0
            return lg
        return real


PROMPT_SRC = ('ClrHome\nDisp "A:B"\n{1,2}→ʟAB\nLbl Q\nPrompt A\nWhile A>0\nIf A>3\nGoto R\n'
              'Disp ʟAB\n')


def _snapshots(monkeypatch):
    """One snapshot per decision, keyed by R: masks, XT, P, D, bans, logits and Row."""
    snaps, cur = {}, {}
    real_rb, real_closure = linker.rule_bans, linker.Row.closure
    from model.intmask import IntMask

    class SpyMask(IntMask):
        def banned(self):
            before = self.list_read
            b = super().banned()
            cur["mask"] = (*_mask(self), before, b.copy())
            return b

    def rb(rules, ids_, sp, P, R, D, b, logits, body=linker.BODY):
        cur["n"] = R
        snap = {k: v for k, v in _xt(sp).items() if k != "events"}
        cur["rb"] = (copy.deepcopy(snap), list(P), D, b.copy(), np.asarray(logits).copy())
        return real_rb(rules, ids_, sp, P, R, D, b, logits, body)

    def closure(row):
        cur["row"] = copy.deepcopy({k: v for k, v in vars(row).items() if k != "ids"})
        snaps[cur["n"]] = dict(cur)
        return real_closure(row)
    for mod in (linker, e14):
        monkeypatch.setattr(mod, "IntMask", SpyMask)
        monkeypatch.setattr(mod, "rule_bans", rb)
    monkeypatch.setattr(linker.Row, "closure", closure)
    return snaps


@needs_s10
@pytest.mark.parametrize("arm", ["LD2", "LD", "M"])
def test_state_after_the_prompt_equals_state_after_sampling_it(s10, monkeypatch, arm):
    """At the first decision after the prompt: the masks, XT (labels, blocks, exit ledger, goto
    context), Row, P / R / D and the model's logits are those of linker.generate after it
    sampled the same tokens (a prompt with a defined label, a pending one, a block and a list)."""
    prefix = [BOS, *T.text_ids(PROMPT_SRC)]
    cap = len(prefix) + 40                              # R stays above the deadline's D
    model, E = _Forced(s10, prefix), linker.offsets_to_E([-8.0] * 16)
    snaps = _snapshots(monkeypatch)
    sampled = linker.generate(model, linker.ARMS[arm], 5, E=E, cap=cap)
    assert sampled["ids"][:len(prefix)] == prefix      # sampling did write the prompt
    want = snaps[cap - len(prefix)]
    snaps.clear()
    prompted = e14.generate_prompted(model, linker.ARMS[arm], 5, E, prefix[1:], cap=cap)
    assert prompted["ids"][:len(prefix)] == prefix and max(snaps) == cap - len(prefix)
    got = snaps[cap - len(prefix)]
    assert got["mask"][:7] == want["mask"][:7] and np.array_equal(got["mask"][7], want["mask"][7])
    (gs, gP, gD, gb, gl), (ws, wP, wD, wb, wl) = got["rb"], want["rb"]
    assert gs == ws and gP == wP == ["R"] and gD == wD
    assert np.array_equal(gb, wb) and np.array_equal(gl, wl)
    assert gs["defined"] == {"Q"} and len(gs["blocks"]) == 1 and gs["goto_ctx"][1] == 1
    assert got["row"] == want["row"] and got["row"]["pending"] == ["R"]


# ---- the prompt draws nothing; positions and labels count it ----------------------------------

@needs_s10
def test_the_prompt_draws_nothing_and_rule_7_uses_absolute_positions(s10, ids):
    """HZ (rule 7 alone) after a prompt: a hand loop starting from the seed's own state at the
    first decided token, with the hazard cell's position counting the prompt."""
    from model.intmask import IntMask
    fake = _Fixed(s10)
    prefix = [BOS, *T.text_ids("ClrHome\nDisp 1\n" * 40)]
    assert 150 < len(prefix) < 256
    E = linker.offsets_to_E([-8.0] * 5 + [3.0] + [-8.0] * 10)       # clean, bin 1 is hot
    late = 0
    for seed in range(1, 9):
        cfg = SamplerConfig(205, 20, False, seed)
        mask = IntMask(fake.vocab)
        mask.reset()
        for t in prefix:
            mask.step(t)
        toks, state, draws = list(prefix), seed, 0
        while len(toks) < 400:
            b = mask.banned()
            if not b[EOS]:
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
        got = e14.generate_prompted(fake, linker.ARMS["HZ"], seed, E, prefix[1:], cap=400)
        assert got["ids"] == toks and got["stats"]["hazard_draws"] == draws, seed
        late += toks[-1] == EOS and len(toks) - 2 >= 256
    assert late > 0


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


@needs_s10
def test_prompted_ld2_honours_the_prompts_labels(s10h, ids):
    """A prompt that defines Lbl Q and leaves Goto R pending: LD2 never ends while R is
    pending, defines R once and never defines Q again."""
    prompt = T.text_ids("ClrHome\nLbl Q\nDisp 1\nGoto R\n")

    class Count(linker.XT):
        def reset(self):
            super().reset()
            self.lbls = []

        def _register_label(self, kind, name):
            if kind == "lbl":
                self.lbls.append(name)
            super()._register_label(kind, name)
    ended = 0
    for seed in range(1, 9):
        out = e14.generate_prompted(s10h, linker.ARMS["LD2"], seed, hw_E(), prompt,
                                    cap=len(prompt) + 121)
        assert out["ids"][:len(prompt) + 1] == [BOS, *prompt]
        sp = Count(ids)
        for t in out["ids"]:
            if t != EOS:
                sp.step(t)
        assert sp.lbls.count("Q") == 1 and sp.lbls.count("R") == 1 and not sp.unres, seed
        ended += out["ids"][-1] == EOS
    assert ended > 0


@pytest.mark.parametrize("bad", [[BOS, 5], [5, EOS], [PAD], [5, 3], [-1], [5, 812]])
def test_the_prompt_holds_program_tokens_only(bad):
    with pytest.raises(ValueError, match="prompt"):
        e14.check_prompt(bad, 1024, 812)


def test_the_prompt_leaves_room_under_the_cap():
    with pytest.raises(ValueError, match="cap"):
        e14.check_prompt([5, 6, 7], 4)
    assert e14.check_prompt([5, 6, 7], 5) == [5, 6, 7]


# ---- the offsets ---------------------------------------------------------------------------------

def fake_export(tmp_path) -> Path:
    ex = tmp_path / "ex"
    ex.mkdir(exist_ok=True)
    (ex / "model.bin").write_bytes(b"model")
    return ex


def fit_file(tmp_path, ex, **kw) -> Path:
    off = {"b": [0.0] * 16, "E": list(range(16)), "export": str(Path(ex).resolve()),
           "hazard_full": True, "horizons": True, **kw}
    p = tmp_path / "fit.json"
    p.write_text(json.dumps({k: v for k, v in off.items() if v is not None}))
    return p


def test_load_offsets_is_the_record_sample_json_keeps(tmp_path):
    ex = fake_export(tmp_path)
    f = fit_file(tmp_path, ex, export_sha256=hashlib.sha256(b"model").hexdigest())
    assert e14.load_offsets(f, ex) == {
        "decoder": "LD2", "offsets": str(f),
        "offsets_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "rules": {"old": True, "soft_bind": False, "row_bos_fix": True, "hazard": True,
                  "r1": True, "r2": True, "hazard_full": True}, "body": 24, "E": list(range(16))}


@pytest.mark.parametrize("bad", [{"export": "/elsewhere"}, {"export": None},
                                 {"hazard_full": False}, {"horizons": False},
                                 {"export_sha256": "0" * 64}, {"E": [1] * 15}])
def test_load_offsets_refuses_anything_but_this_exports_ld2_fit(tmp_path, bad):
    ex = fake_export(tmp_path)
    with pytest.raises(ValueError):
        e14.load_offsets(fit_file(tmp_path, ex, **bad), ex)


# ---- the sample command --------------------------------------------------------------------------

def _fake_draw(calls, gold=False):
    def draw(export, jobs, E, workers, mask_level=1):
        calls.append((export, jobs, E, workers) + ((mask_level,) if mask_level != 1 else ()))
        return [{"ids": T.program_ids(V.TASKS_V2[t], V.TASKS_V2[t].gold) if gold else
                 [BOS, *V.TASKS_V2[t].opening_ids(), 5, EOS],
                 "stats": {"hazard_stop": s % 2}} for t, s in jobs]
    return draw


DEV_KEYS = ["export", "k", "seed0", "n", "suite_sha256", "temp", "top_k", "cap", "masks",
            "horizons", "export_sha256"]                      # lab.tasks.sample's, in order
HELDOUT_KEYS = ["export", "k", "seed0", "n", "suite_sha256", "split", "tasks", "temp", "top_k",
                "cap", "masks", "horizons", "export_sha256"]  # lab.tasks_v2.sample2's
LD2_KEYS = ["decoder", "offsets", "offsets_sha256", "rules", "body", "E"]


def test_sample_dev_writes_lab_tasks_rows_and_meta_and_score2_judges_them(tmp_path, monkeypatch):
    from lab.tests.test_tasks_v2 import fake_runner
    ex = fake_export(tmp_path)
    f = fit_file(tmp_path, ex)
    calls = []
    monkeypatch.setattr(e14, "draw", _fake_draw(calls, gold=True))
    out = tmp_path / "s"
    assert e14.main(["sample", "--export", str(ex), "--offsets", str(f), "--split", "dev",
                     "--k", "2", "--seed0", "3", "--workers", "4", "--out", str(out)]) == 0
    jobs = [(t.id, s) for t in T.SUITE for s in (3, 4)]
    assert calls == [(ex, jobs, list(range(16)), 4)]
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    assert [list(r) for r in rows] == [["task", "seed", "ids", "stats"]] * 80
    assert [(r["task"], r["seed"], r["stats"]["hazard_stop"]) for r in rows] == \
        [(t, s, s % 2) for t, s in jobs]
    meta = json.loads((out / "sample.json").read_text())
    assert list(meta) == DEV_KEYS + LD2_KEYS
    assert meta["suite_sha256"] == T.suite_sha256(T.SUITE) == V.V1_SUITE_SHA256
    assert (meta["k"], meta["seed0"], meta["n"], meta["temp"], meta["top_k"], meta["cap"],
            meta["masks"], meta["horizons"]) == (2, 3, 80, 0.8, 20, 1024, True, True)
    assert meta["export_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert {k: meta[k] for k in LD2_KEYS} == e14.load_offsets(f, ex)
    m = V.score2(out, runner=fake_runner([]))
    assert m["source"] == "v1" and m["n_samples"] == 80 and m["n_pass"] == 80


def test_sample_heldout_writes_sample2s_meta_and_score2_judges_it(tmp_path, monkeypatch):
    from lab.tests.test_tasks_v2 import fake_runner
    ex = fake_export(tmp_path)
    monkeypatch.setattr(e14, "draw", _fake_draw([], gold=True))
    e14.sample(ex, fit_file(tmp_path, ex), "heldout", k=1, seed0=1, out=tmp_path / "h",
               workers=1)
    meta = json.loads((tmp_path / "h" / "sample.json").read_text())
    assert list(meta) == HELDOUT_KEYS + LD2_KEYS
    assert meta["suite_sha256"] == V.SUITE_V2_SHA256 and meta["split"] == "heldout"
    assert meta["tasks"] == list(V.HELDOUT_IDS)
    m = V.score2(tmp_path / "h", runner=fake_runner([]))
    assert m["source"] == "v2" and m["task_correct_heldout"] == 100.0


def test_sample_checks_the_offsets_before_drawing(tmp_path, monkeypatch, capsys):
    ex = fake_export(tmp_path)
    monkeypatch.setattr(e14, "draw", lambda *a: pytest.fail("drew"))
    f = fit_file(tmp_path, ex, horizons=False)
    assert e14.main(["sample", "--export", str(ex), "--offsets", str(f), "--split", "dev",
                     "--k", "1", "--out", str(tmp_path / "o")]) == 2
    assert "horizons" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_draw_loads_the_export_with_horizons_and_samples_ld2_from_each_opening(monkeypatch):
    import model.intref as IR
    loaded, calls = [], []
    monkeypatch.setattr(T, "_model", None)          # restored after the test
    monkeypatch.setattr(e14, "_E", None)

    class FakeModel:
        vocab = T._vocab().entries
    monkeypatch.setattr(IR.IntModel, "load", classmethod(
        lambda cls, d, horizons=False: loaded.append((d, horizons)) or FakeModel()))

    def fake_gen(model, rules, seed, E, prompt, **kw):
        calls.append((type(model), rules, seed, E, prompt, kw))
        return {"ids": [BOS, *prompt, EOS], "stats": {}}
    monkeypatch.setattr(e14, "generate_prompted", fake_gen)
    jobs = [("nt_gcd", 4), (V.HELDOUT_IDS[0], 9)]
    out = e14.draw("EX", jobs, [3] * 16, workers=1)
    assert loaded == [("EX", True)]
    assert calls == [(FakeModel, linker.ARMS["LD2"], s, [3] * 16, V.TASKS_V2[t].opening_ids(),
                      {"mask_level": 1}) for t, s in jobs]
    assert out == [{"ids": [BOS, *V.TASKS_V2[t].opening_ids(), EOS], "stats": {}}
                   for t, _ in jobs]


@pytest.mark.slow
@pytest.mark.skipif(not (L4 / "L4-s23" / "model.bin").exists(), reason="needs L4-s23")
def test_sample_on_l4_s23_with_workers_is_generate_prompted(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    jobs = [("alg_slope", 1), ("nt_digit_sum", 2), (V.HELDOUT_IDS[3], 1)]
    E = json.loads(L4_FIT.read_text())["E"]
    got = e14.draw(L4 / "L4-s23", jobs, E, workers=3)
    model = IntModel.load(L4 / "L4-s23", horizons=True)
    for (t, s), g in zip(jobs, got, strict=True):
        assert g == e14.generate_prompted(model, linker.ARMS["LD2"], s, E,
                                          V.TASKS_V2[t].opening_ids())


# ---- v2.4 mask level (docs/agents/phase1-safety-masks.md) -----------------------------------------

@needs_s10
def test_generate_prompted_passes_the_mask_level_to_intmask(s10h, monkeypatch):
    seen, real = [], e14.IntMask
    monkeypatch.setattr(e14, "IntMask", lambda vocab, level=1: seen.append(level) or real(vocab,
                                                                                         level))
    prompt = V.TASKS_V2["nt_gcd"].opening_ids()
    e14.generate_prompted(s10h, linker.ARMS["LD2"], 1, hw_E(), prompt, cap=60)
    e14.generate_prompted(s10h, linker.ARMS["LD2"], 1, hw_E(), prompt, cap=60, mask_level=2)
    assert seen == [1, 2]


@needs_s10
def test_empty_prompt_is_linker_generate_at_mask_level_2(s10h):
    for seed in (1, 2):
        want = linker.generate(s10h, linker.ARMS["LD2"], seed, E=hw_E(), cap=160, mask_level=2)
        assert e14.generate_prompted(s10h, linker.ARMS["LD2"], seed, hw_E(), [], cap=160,
                                     mask_level=2) == want


def test_sample_mask_level_reaches_draw_and_sample_json(tmp_path, monkeypatch):
    ex = fake_export(tmp_path)
    f = fit_file(tmp_path, ex)
    calls = []
    monkeypatch.setattr(e14, "draw", _fake_draw(calls))
    args = ["sample", "--export", str(ex), "--offsets", str(f), "--split", "dev", "--k", "1",
            "--workers", "1"]
    assert e14.main(args + ["--out", str(tmp_path / "a"), "--mask-level", "2"]) == 0
    assert e14.main(args + ["--out", str(tmp_path / "b")]) == 0
    assert [c[4:] for c in calls] == [(2,), ()]
    meta2 = json.loads((tmp_path / "a" / "sample.json").read_text())
    meta1 = json.loads((tmp_path / "b" / "sample.json").read_text())
    assert meta2["mask_level"] == 2 and list(meta2) == DEV_KEYS + LD2_KEYS + ["mask_level"]
    assert list(meta1) == DEV_KEYS + LD2_KEYS          # level 1: v2.3's sample.json unchanged


# ---- v2.5 topic flags (PHASE3-topic-menu.md 4a): --tasks, --cap, --seed-mix -----------------------

def _kw_draw(calls):
    def draw(export, jobs, E, workers, **kw):
        calls.append((jobs, kw))
        return [{"ids": [BOS, *V.TASKS_V2[t].opening_ids(), 5, EOS], "stats": {}}
                for t, _ in jobs]
    return draw


TOPIC_TASKS = ["geo_cylinder_volume", "ho_chem_heat"]


def test_sample_tasks_cap_and_seed_mix_reach_draw_and_sample_json(tmp_path, monkeypatch):
    from lab.tests.test_tasks_v2 import fake_runner
    ex = fake_export(tmp_path)
    f = fit_file(tmp_path, ex)
    calls = []
    monkeypatch.setattr(e14, "draw", _kw_draw(calls))
    out = tmp_path / "t"
    assert e14.main(["sample", "--export", str(ex), "--offsets", str(f), "--tasks",
                     ",".join(TOPIC_TASKS), "--k", "2", "--seed0", "600001", "--workers", "1",
                     "--cap", "256", "--seed-mix", "fmix32", "--mask-level", "2",
                     "--out", str(out)]) == 0
    jobs = [(t, s) for t in TOPIC_TASKS for s in (600001, 600002)]
    assert calls == [(jobs, {"mask_level": 2, "cap": 256, "seed_mix": "fmix32"})]
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    assert [(r["task"], r["seed"]) for r in rows] == jobs           # the raw seeds
    meta = json.loads((out / "sample.json").read_text())
    assert meta["suite_sha256"] == V.SUITE_V2_SHA256 and meta["tasks"] == TOPIC_TASKS
    assert (meta["cap"], meta["seed_mix"], meta["mask_level"], meta["n"]) == (256, "fmix32", 2, 4)
    assert meta["split"] == "tasks"
    m = V.score2(out, runner=fake_runner([]))                       # judged as v2 samples
    assert m["source"] == "v2" and m["n_samples"] == 4


def test_sample_defaults_keep_e14s_sample_json(tmp_path, monkeypatch):
    """Without the new flags, draw is called and sample.json is written exactly as before."""
    ex = fake_export(tmp_path)
    calls = []
    monkeypatch.setattr(e14, "draw", _kw_draw(calls))
    assert e14.main(["sample", "--export", str(ex), "--offsets", str(fit_file(tmp_path, ex)),
                     "--split", "heldout", "--k", "1", "--workers", "1",
                     "--out", str(tmp_path / "d")]) == 0
    assert calls[0][1] == {"mask_level": 1}
    meta = json.loads((tmp_path / "d" / "sample.json").read_text())
    assert list(meta) == HELDOUT_KEYS + LD2_KEYS and meta["cap"] == 1024


def test_sample_needs_a_split_or_tasks_and_known_task_ids(tmp_path, monkeypatch, capsys):
    ex = fake_export(tmp_path)
    monkeypatch.setattr(e14, "draw", lambda *a, **k: pytest.fail("drew"))
    base = ["sample", "--export", str(ex), "--offsets", str(fit_file(tmp_path, ex)), "--k", "1",
            "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        e14.main(base)
    with pytest.raises(SystemExit):
        e14.main(base + ["--split", "dev", "--tasks", "geo_circle_area"])
    assert e14.main(base + ["--tasks", "geo_circle_area,nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_draw_mixes_the_seed_and_passes_the_cap(monkeypatch):
    from model.topics import mix_seed
    calls = []
    monkeypatch.setattr(T, "_model", "M")
    monkeypatch.setattr(T, "_load", lambda export: None)

    def fake_gen(model, rules, seed, E, prompt, **kw):
        calls.append((seed, kw))
        return {"ids": [BOS, EOS], "stats": {}}
    monkeypatch.setattr(e14, "generate_prompted", fake_gen)
    jobs = [("geo_circle_area", 1), ("geo_circle_area", 600001)]
    e14.draw("EX", jobs, [3] * 16, workers=1, mask_level=2, cap=256, seed_mix="fmix32")
    assert calls == [(mix_seed(1), {"mask_level": 2, "cap": 256}),
                     (mix_seed(600001), {"mask_level": 2, "cap": 256})]
    calls.clear()
    e14.draw("EX", jobs, [3] * 16, workers=1)
    assert calls == [(1, {"mask_level": 1}), (600001, {"mask_level": 1})]
    with pytest.raises(ValueError):
        e14.draw("EX", jobs, [3] * 16, workers=1, seed_mix="sha")


RR = RUNS / "release_review" / "prompted_dev_k50"


@pytest.mark.slow
@pytest.mark.skipif(not ((RR / "samples.jsonl").exists()
                         and (L4 / "L4-s23" / "model.bin").exists()),
                    reason="needs runs/release_review and the L4-s23 export")
def test_default_flags_reproduce_the_release_reviews_samples_id_for_id(monkeypatch):
    """A slice of rc2's release review (prompted dev, k 50, seeds 500001-) drawn again with
    the defaults (cap 1024, no seed mix, level 1): ids and stats row for row."""
    monkeypatch.chdir(ROOT)
    rows = [json.loads(x) for x in (RR / "samples.jsonl").read_text().splitlines()]
    pick = [rows[i] for i in (0, 1, 777, 1999)]
    E = json.loads(L4_FIT.read_text())["E"]
    got = e14.draw(L4 / "L4-s23", [(r["task"], r["seed"]) for r in pick], E, workers=4)
    assert [g["ids"] for g in got] == [r["ids"] for r in pick]
    assert [g["stats"] for g in got] == [r["stats"] for r in pick]
