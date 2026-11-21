// CalcLM browser demo engine (docs/agents/phase6-browser-demo.md): runs the calculator's own C
// inference core, compiled to WebAssembly (demo/tools/build_wasm.sh), on the model package
// (demo/tools/build_model.py). No network: the module and the model come in as bytes.
//
//   const eng = await CalcLM.create(CalcLM.base64ToBytes(CALCLM_WASM_B64), CALCLM_MODEL);
//   eng.start({ topic: 0, seed: 1, level: 2 });   // topic: index into eng.topics, or null
//   for (let s; (s = eng.step()); ) show(s.token);
//   eng.render(eng.ids().slice(1));               // the program as TI-BASIC text
//
// A classic script: in the browser it sets globalThis.CalcLM, in Node module.exports.
(function (root) {
  "use strict";

  const BOS = 1, EOS = 2, SPECIALS = 4, CLM_DONE = -1;

  function base64ToBytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function hexToBytes(h) {
    const out = new Uint8Array(h.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = parseInt(h.substr(2 * i, 2), 16);
    return out;
  }

  // Is this run what the calculator does? Random at level 2 (Masks on) or 3 (Masks off), and
  // a topic at level 2. Level 1 is v2.3's masks (rc2's hardware-check sheet).
  function onDevice(topic, level) {
    return topic === null ? level >= 2 : level === 2;
  }

  class Engine {
    constructor(exports, model) {
      this.x = exports;
      this.model = model;
      this.exportSha256 = model.export_sha256;
      this.run = null;
      this.endedBy = null;
      this._load();
      const n = this.x.demo_topic_n();
      this.topics = [];
      for (let i = 0; i < n; i++) {
        const label = this._str(this.x.demo_topic_label(i));
        this.topics.push({
          index: i, key: (model.topic_keys || {})[label] || null, label,
          check: this._str(this.x.demo_topic_check(i)),
          rate: this._str(this.x.demo_topic_rate(i)),
          time: this._str(this.x.demo_topic_time(i)),
          opening: this._u16(this.x.demo_topic_ids(i), this.x.demo_topic_len(i)),
        });
      }
      this.random = { label: "Random program", rate: this._str(this.x.demo_topic_rate(-1)),
                      time: this._str(this.x.demo_topic_time(-1)) };
    }

    _mem() { return new Uint8Array(this.x.memory.buffer); }   // fresh: memory can grow

    _copy(bytes) {
      const p = this.x.demo_alloc(bytes.length);
      if (!p) throw new Error("out of WebAssembly memory");
      this._mem().set(bytes, p);
      return p;
    }

    _u32s(values) {
      const b = new Uint8Array(4 * values.length);
      const dv = new DataView(b.buffer);
      values.forEach((v, i) => dv.setUint32(4 * i, v, true));
      return this._copy(b);
    }

    _str(p) {
      const m = this._mem();
      let e = p;
      while (m[e]) e++;
      return new TextDecoder().decode(m.subarray(p, e));
    }

    _u16(p, n) { return Array.from(new Uint16Array(this.x.memory.buffer, p, n)); }

    _load() {
      const m = this.model;
      const avs = m.appvars.map(base64ToBytes);
      const ptrs = avs.map((a) => this._copy(a));
      const pData = this._u32s(ptrs), pSize = this._u32s(avs.map((a) => a.length));
      // CLMVOC's records: per id a byte count, then the token's bytes (specials: none)
      const voc = [];
      for (const h of m.vocab_hex) voc.push(h.length / 2, ...hexToBytes(h));
      const pVoc = this._copy(Uint8Array.from(voc));
      const ld2 = m.ld2 ? base64ToBytes(m.ld2) : new Uint8Array(0);
      const pLd2 = this._copy(ld2.length ? ld2 : new Uint8Array(1));
      const pSha = this._copy(hexToBytes(m.export_sha256.slice(0, 12)));
      const rc = this.x.demo_load(avs.length, pData, pSize, pVoc, voc.length, pLd2, ld2.length,
                                  pSha);
      if (rc !== 0) throw new Error(`the calculator core refused the model (clm_load ${rc})`);
      this.masksOk = !!this.x.demo_masks_ok();
      this.ld2Ok = !!this.x.demo_ld2_ok();
      this.topicsReason = this.x.demo_topics_reason() === 0 ? ""
        : this._str(this.x.demo_reason_text());
    }

    mixSeed(seed) { return this.x.demo_mix_seed(seed >>> 0) >>> 0; }

    // topic: an index into this.topics, or null for Random. seed: 0 .. 2^32-1 (0 is the
    // core's default seed). level: 1, 2 or 3. Returns the run's settings as the core got them.
    start({ topic = null, seed, level = 2 }) {
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
        throw new RangeError("seed must be an integer from 0 to 4294967295");
      }
      if (![1, 2, 3].includes(level)) throw new RangeError("mask level must be 1, 2 or 3");
      if (topic !== null && !(Number.isInteger(topic) && topic >= 0 && topic < this.topics.length)) {
        throw new RangeError(`topic must be null or 0..${this.topics.length - 1}`);
      }
      if (topic !== null && this.topicsReason) throw new Error(this.topicsReason);
      const rc = this.x.demo_start(topic === null ? -1 : topic, seed, level);
      if (rc !== 0) throw new Error(`the calculator core refused the settings (clm_reset ${rc})`);
      this.endedBy = null;
      this._fed = 0;
      this.run = {
        topic, level, userSeed: seed, device: onDevice(topic, level),
        seed: this.x.demo_run_seed() >>> 0, temperatureQ8: this.x.demo_run_temp(),
        topK: this.x.demo_run_top_k(), cap: this.x.demo_run_cap(),
        masks: this.x.demo_run_masks(), ld2: !!this.x.demo_run_ld2(),
        nPrompt: this.x.demo_run_prompt(),
      };
      return { ...this.run };
    }

    // The next token: { token, prompt } (prompt: a token of the topic's opening, fed rather
    // than written), or null once the program ended (this.endedBy: "eos" or "cap").
    step() {
      if (!this.run || this.endedBy) return null;
      const t = this.x.demo_step();
      if (t === CLM_DONE) {
        this.endedBy = "cap";
        return null;
      }
      if (t < 0) throw new Error(`the calculator core refused an opening token (clm_feed ${t})`);
      const prompt = this._fed < this.run.nPrompt;
      if (prompt) this._fed++;
      if (t === EOS) this.endedBy = "eos";
      return { token: t, prompt };
    }

    // the tokens so far, <bos> first (the core's token buffer)
    ids() { return this._u16(this.x.demo_tokens(), this.x.demo_n_tok()); }

    // model.intref.render: each token's text (tivars), specials dropped, trailing newlines off.
    // render decodes the whole byte string at once and falls back to "<undecodable: ...> HEX"
    // when tivars raises; every vocab id is one valid TI token, so that fallback never fires
    // for the deployed vocab and this table join is the same text (tested on the grid).
    render(ids) {
      let s = "";
      for (const t of ids) if (t >= SPECIALS) s += this.model.vocab_text[t];
      return s.replace(/\n+$/, "");
    }

    // prgmGEN's body: the token bytes after <bos>, specials dropped (main.c save())
    body(ids) {
      const parts = [];
      for (const t of ids) if (t >= SPECIALS) parts.push(...hexToBytes(this.model.vocab_hex[t]));
      return Uint8Array.from(parts);
    }
  }

  async function create(wasmBytes, model) {
    if (!model || model.format !== 1) throw new Error("not a CalcLM demo model package");
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new Engine(instance.exports, model);
  }

  const api = { create, base64ToBytes, onDevice, BOS, EOS };
  root.CalcLM = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
