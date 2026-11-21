// Unit tests of the demo engine (docs/agents/phase6-browser-demo.md A2-A5): run with
// `node --test demo/tests/` after building (demo/README.md).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const H = require("./helpers.js");

const T = H.topicsHeader();
let eng;
test.before(async () => { eng = await H.engine(); });

function runAll(e, opts) {
  const info = e.start(opts);
  const stream = [];
  for (let s; (s = e.step()); ) stream.push(s);
  return { info, stream, ids: e.ids() };
}

test("the module loads the deployed model with masks, LD2 and the topics", () => {
  assert.equal(eng.masksOk, true);
  assert.equal(eng.ld2Ok, true);
  assert.equal(eng.topicsReason, "");
  assert.equal(eng.exportSha256.slice(0, 16), "e68eca508872b7a5");   // v2.6-rc2: E24 L4c-s11
});

test("the topic table is topics.h's", () => {
  assert.equal(eng.topics.length, T.n);
  assert.deepEqual(eng.topics.map((t) => t.label), T.labels);
  assert.deepEqual(eng.topics.map((t) => t.check), T.checks);
  assert.deepEqual(eng.topics.map((t) => t.rate), T.rates);
  assert.deepEqual(eng.topics.map((t) => t.time), T.times);
  assert.deepEqual(eng.topics.map((t) => t.opening), T.openings);
});

test("clm_mix_seed is fmix32", () => {
  // model.topics.mix_seed: MurmurHash3's finalizer, 0 -> 0
  const fmix32 = (h) => {
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x85ebca6b) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 0xc2b2ae35) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
  };
  for (const s of [0, 1, 2, 3, 48, 600001, 0xffffffff]) assert.equal(eng.mixSeed(s), fmix32(s));
  assert.equal(eng.mixSeed(0), 0);
});

test("a topic at level 2 runs at the calculator's topic settings", () => {
  for (let i = 0; i < T.n; i++) {
    const info = eng.start({ topic: i, seed: 7, level: 2 });
    assert.deepEqual(
      [info.seed, info.temperatureQ8, info.topK, info.cap, info.masks, info.ld2, info.nPrompt],
      [eng.mixSeed(7), T.temp, T.topK, T.cap, T.masks, true, T.openings[i].length]);
    assert.equal(info.device, true);
  }
});

test("Random runs at the calculator's defaults; level 3 is the Masks row off", () => {
  const r2 = eng.start({ topic: null, seed: 7, level: 2 });
  assert.deepEqual([r2.seed, r2.temperatureQ8, r2.topK, r2.cap, r2.masks, r2.ld2, r2.nPrompt],
                   [7, 205, 20, 1024, 2, true, 0]);
  assert.equal(r2.device, true);
  const r3 = eng.start({ topic: null, seed: 7, level: 3 });
  assert.deepEqual([r3.masks, r3.ld2, r3.cap, r3.device], [3, false, 1024, true]);
  const r1 = eng.start({ topic: null, seed: 7, level: 1 });
  assert.deepEqual([r1.masks, r1.ld2, r1.device], [1, true, false]);
});

test("a topic at levels 1 and 3 changes only the masks and LD2, and is not the calculator's", () => {
  const t1 = eng.start({ topic: 0, seed: 7, level: 1 });
  assert.deepEqual([t1.seed, t1.cap, t1.masks, t1.ld2, t1.device],
                   [eng.mixSeed(7), 256, 1, true, false]);
  const t3 = eng.start({ topic: 0, seed: 7, level: 3 });
  assert.deepEqual([t3.seed, t3.cap, t3.masks, t3.ld2, t3.device],
                   [eng.mixSeed(7), 256, 3, false, false]);
});

test("bad arguments are refused", () => {
  assert.throws(() => eng.start({ topic: T.n, seed: 1, level: 2 }));
  assert.throws(() => eng.start({ topic: 0, seed: 1, level: 4 }));
  assert.throws(() => eng.start({ topic: 0, seed: -1, level: 2 }));
  assert.throws(() => eng.start({ topic: 0, seed: 2 ** 32, level: 2 }));
});

test("a topic streams its opening first, then writes, and stops by the 256-token cap", () => {
  const q = eng.topics.findIndex((t) => t.key === "QUAD");               // QUAD: long programs
  assert.notEqual(q, -1);
  const { stream, ids } = runAll(eng, { topic: q, seed: 3, level: 2 });
  const n = T.openings[q].length;
  assert.deepEqual(stream.slice(0, n).map((s) => s.token), T.openings[q]);
  assert.ok(stream.slice(0, n).every((s) => s.prompt));
  assert.ok(stream.slice(n).every((s) => !s.prompt));
  assert.deepEqual(ids, [1, ...stream.map((s) => s.token)]);
  assert.ok(ids.length <= 256);
  assert.equal(eng.endedBy, ids[ids.length - 1] === 2 ? "eos" : "cap");
});

test("the hardware-check sheets' programs are reproduced (A2)", (t) => {
  const lines = H.sheetLines("e68eca508872b7a5");
  if (lines === null) return t.skip("no hw/checks in this tree");
  assert.equal(lines.length, 2, "v2.6-rc2 Random (level 2) and CYL lines");
  for (const l of lines) {
    const topic = l.topic ? eng.topics.findIndex((t) => t.key === l.topic) : null;
    assert.notEqual(topic, -1, l.topic);
    const level = Number(l.masks || 1);
    const { ids } = runAll(eng, { topic, seed: Number(l.seed), level });
    const got = H.sheetNumbers(eng, ids);
    for (const k of ["tokens", "lines"]) assert.equal(got[k], Number(l[k]), `${l.sheet} ${k}`);
    for (const k of ["ids_sha256", "text_sha256", "body_sha256"]) {
      assert.equal(got[k], l[k], `${l.sheet} seed ${l.seed} ${k}`);
    }
  }
});

test("another LD2 fit turns the topics off, as on the calculator (A4)", async () => {
  const CalcLM = H.loadScripts();
  const m = { ...globalThis.CALCLM_MODEL };
  const ld2 = CalcLM.base64ToBytes(m.ld2);
  const dv = new DataView(ld2.buffer);
  dv.setUint32(15 + 4 * 5, dv.getUint32(15 + 4 * 5, true) + 1, true);   // E[5] + 1
  let h = 0x811c9dc5;                                                    // FNV-1a of 0..78
  for (let i = 0; i < 79; i++) h = Math.imul(h ^ ld2[i], 0x01000193) >>> 0;
  dv.setUint32(79, h, true);
  m.ld2 = Buffer.from(ld2).toString("base64");
  const other = await CalcLM.create(CalcLM.base64ToBytes(globalThis.CALCLM_WASM_B64), m);
  assert.equal(other.ld2Ok, true);
  assert.equal(other.topicsReason, "Topics off: CLMLD2 not the measured fit");
  assert.throws(() => other.start({ topic: 0, seed: 1, level: 2 }));
  const r = runAll(other, { topic: null, seed: 1, level: 2 });
  assert.ok(r.ids.length > 1);
});

test("the topic keys follow the calculator's menu order", () => {
  // v2.6-rc2's measured menu (topics_v2.6.json, section 1b's rule): eight main entries, then
  // SPEED and C2F as "rarely"; none dropped
  assert.deepEqual(eng.topics.map((t) => t.key),
                   ["CYL", "CIRC", "HYP", "F2C", "OHM", "QUAD", "SPH", "HEAT", "SPEED", "C2F"]);
});
