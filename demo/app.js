// The demo page (docs/agents/phase6-browser-demo.md): the calculator's start screen as a form,
// and the program streamed token by token from the engine (calclm.js). No network calls.
(function () {
  "use strict";

  const CALC_TOK_PER_S = 0.674;   // phase 1's CEmu rate at level 2 (PHASE3-topic-menu.md 10)
  const CALC_SEED_MAX = 999999999; // main.c: the Seed row takes at most nine digits
  const $ = (id) => document.getElementById(id);
  const ui = {
    model: $("model"), menu: $("menu"), reason: $("reason"), seed: $("seed"), dice: $("dice"),
    pace: $("pace"), device: $("device"), go: $("go"), stop: $("stop"), retry: $("retry"),
    title: $("title"), count: $("count"), barFed: $("bar-fed"), barWritten: $("bar-written"),
    program: $("program"), status: $("status"), check: $("check"), facts: $("facts"),
    copy: $("copy"), form: $("controls"),
  };
  let eng = null;
  let timer = null;
  let running = false;
  let gen = 0;                     // the current run; a stale finish() must not touch the page

  function fail(msg) {
    ui.model.textContent = msg;
    ui.model.classList.add("off");
  }

  // ---- the menu ---------------------------------------------------------------------------

  function menuEntry(value, key, label, rate, time, disabled) {
    const l = document.createElement("label");
    if (disabled) l.className = "off";
    const input = document.createElement("input");
    Object.assign(input, { type: "radio", name: "topic", value, disabled });
    const cells = [key, label, rate, time].map((text, i) => {
      const s = document.createElement("span");
      s.className = ["key", "label", "rate", "time"][i];
      s.textContent = text;
      return s;
    });
    l.append(input, ...cells);
    return l;
  }

  function buildMenu() {
    const off = !!eng.topicsReason;
    eng.topics.forEach((t, i) => {
      ui.menu.append(menuEntry(String(i), i < 9 ? String(i + 1) : "", t.label, t.rate, t.time, off));
    });
    ui.menu.append(menuEntry("random", "0", eng.random.label, eng.random.rate, eng.random.time, false));
    const first = ui.menu.querySelector(off ? "input[value=random]" : "input");
    first.checked = true;
    if (off) {
      ui.reason.textContent = eng.topicsReason;
      ui.reason.hidden = false;
    }
  }

  function selection() {
    const v = ui.form.querySelector("input[name=topic]:checked").value;
    const topic = v === "random" ? null : Number(v);
    const level = Number(ui.form.querySelector("input[name=level]:checked").value);
    const text = ui.seed.value.trim();
    const seed = /^\d{1,10}$/.test(text) ? Number(text) : NaN;   // digits only, as on the keypad
    return { topic, level, seed };
  }

  function validSeed(s) { return Number.isInteger(s) && s >= 0 && s <= 0xffffffff; }

  function describe() {
    const { topic, level, seed } = selection();
    const ok = validSeed(seed);
    ui.seed.setAttribute("aria-invalid", ok ? "false" : "true");
    ui.go.disabled = running || !ok;
    const on = CalcLM.onDevice(topic, level) && !(seed > CALC_SEED_MAX);
    ui.device.className = "device " + (on ? "on" : "off");
    if (seed > CALC_SEED_MAX) {
      ui.device.textContent = `Seed above ${CALC_SEED_MAX.toLocaleString("en")}: the calculator's Seed row cannot hold it, so this run is not one the calculator can make.`;
    } else if (topic === null) {
      ui.device.textContent = level === 1
        ? "Not a v2.5 setting: v2.3's masks with LD2, as in rc2's hardware-check sheet. Length 1,024 tokens, T 0.8, top-k 20."
        : `As on the calculator: Random, Masks ${level === 2 ? "on" : "off"}, length 1,024 tokens, T 0.8, top-k 20, the seed as typed.`;
    } else {
      ui.device.textContent = on
        ? "As on the calculator: the topic's fixed settings (cap 256 tokens, T 0.8, top-k 20, masks level 2, LD2) and the mixed seed."
        : `Not something the calculator does: it always runs a topic at level 2. This run uses level ${level}${level === 3 ? " without LD2" : " with LD2"}, with the topic's seed mix and cap.`;
    }
  }

  // ---- the program --------------------------------------------------------------------------

  // the program as the editor lists it: ':' at each line start; the opening dim
  function paint(ids, nPrompt, done) {
    const join = (a) => a.filter((t) => t >= 4).map((t) => eng.model.vocab_text[t]).join("");
    let opening = join(ids.slice(1, 1 + nPrompt));
    let written = join(ids.slice(1 + nPrompt));
    if (done) {                                   // model.intref.render strips trailing newlines
      written = written.replace(/\n+$/, "");
      if (!written) opening = opening.replace(/\n+$/, "");
    }
    const colons = (s) => s.replace(/\n/g, "\n:");
    const a = document.createElement("span");
    a.className = "opening";
    a.textContent = ":" + colons(opening);
    const b = document.createElement("span");
    b.textContent = colons(written);
    ui.program.replaceChildren(a, b);
    ui.program.scrollTop = ui.program.scrollHeight;
  }

  function progress(ids, run) {
    const n = ids.length - 1;                    // tokens after <bos>, the calculator's count
    const fed = Math.min(n, run.nPrompt);
    ui.count.textContent = `${n} / ${run.cap - 1} tok (cap ${run.cap} with <bos>)`;
    ui.barFed.style.width = `${(100 * fed) / (run.cap - 1)}%`;
    ui.barWritten.style.width = `${(100 * (n - fed)) / (run.cap - 1)}%`;
    if (!eng.endedBy && running) {
      ui.status.textContent = n < run.nPrompt
        ? `reading the topic ${n}/${run.nPrompt}`
        : `writing... ${n} tok`;
    }
  }

  function fact(dl, k, v) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }

  async function finish(run, stopped, message) {
    const mine = gen;
    running = false;
    clearTimeout(timer);
    const ids = eng.ids();
    paint(ids, run.nPrompt, !stopped);
    progress(ids, run);
    const n = ids.length - 1;
    const minutes = n / CALC_TOK_PER_S / 60;
    ui.status.textContent = message ? `${message} (stopped: ${n} tok)` : stopped ? `stopped: ${n} tok`
      : `done: ${n} tok` + (eng.endedBy === "cap" ? " (the cap)" : "");
    ui.check.textContent = run.topic !== null && !stopped ? eng.topics[run.topic].check : "";
    ui.facts.replaceChildren();
    fact(ui.facts, "On the calculator", `about ${minutes < 1 ? "<1" : Math.round(minutes)} min at ${CALC_TOK_PER_S} tok/s`);
    fact(ui.facts, "Decoder seed", `${run.seed}${run.topic !== null ? ` (fmix32 of ${run.userSeed})` : ""}`);
    fact(ui.facts, "Settings", `masks level ${run.masks}, LD2 ${run.ld2 ? "on" : "off"}, T q8 ${run.temperatureQ8}, top-k ${run.topK}, cap ${run.cap}`);
    ui.stop.disabled = true;
    ui.retry.disabled = false;
    ui.copy.disabled = false;
    describe();
    if (!stopped && typeof crypto !== "undefined" && crypto.subtle) {
      const b = new Uint8Array(2 * ids.length);
      ids.forEach((t, i) => { b[2 * i] = t & 255; b[2 * i + 1] = t >> 8; });
      try {
        const h = new Uint8Array(await crypto.subtle.digest("SHA-256", b));
        if (gen !== mine) return;                 // a new run started meanwhile
        fact(ui.facts, "ids sha256", Array.from(h, (x) => x.toString(16).padStart(2, "0")).join(""));
      } catch (e) { /* no digest outside a secure context; the rest stands */ }
    }
  }

  function start(sel) {
    let run;
    try {
      run = eng.start(sel);
    } catch (e) {
      ui.status.textContent = String(e.message || e);
      return;
    }
    gen++;
    running = true;
    ui.go.disabled = true;
    ui.stop.disabled = false;
    ui.retry.disabled = true;
    ui.copy.disabled = true;
    ui.check.textContent = "";
    ui.facts.replaceChildren();
    const label = run.topic === null ? eng.random.label : eng.topics[run.topic].label;
    ui.title.textContent = `${label}  seed ${run.userSeed}  level ${run.level}`;
    const delay = Number(ui.pace.value);
    const tick = () => {
      if (!running) return;
      const t0 = performance.now();
      let s;
      try {
        do { s = eng.step(); } while (s && delay === 0 && performance.now() - t0 < 12);
      } catch (e) {
        finish(run, true, String(e.message || e));
        return;
      }
      const ids = eng.ids();
      if (!s) { finish(run, false); return; }
      paint(ids, run.nPrompt, false);
      progress(ids, run);
      timer = setTimeout(tick, delay);
    };
    progress(eng.ids(), run);
    tick();
  }

  // ---- wiring -------------------------------------------------------------------------------

  ui.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!running && eng) start(selection());
  });
  ui.form.addEventListener("change", describe);
  ui.seed.addEventListener("input", describe);
  ui.stop.addEventListener("click", () => {
    if (running && eng.run) finish(eng.run, true);
  });
  ui.retry.addEventListener("click", () => {
    if (running || !eng.run) return;
    const next = (eng.run.userSeed + 1) >>> 0;     // the calculator's [2nd]: seed + 1
    ui.seed.value = String(next);
    start({ topic: eng.run.topic, seed: next, level: eng.run.level });
  });
  ui.dice.addEventListener("click", () => {
    ui.seed.value = String(Date.now() % 1000000);    // main.c: rtc_Time() % 1000000
    describe();
  });
  ui.copy.addEventListener("click", () => {
    if (!eng.run || !navigator.clipboard) return;
    navigator.clipboard.writeText(eng.render(eng.ids().slice(1))).catch(() => {});
  });

  async function init() {
    if (!globalThis.CALCLM_WASM_B64 || !globalThis.CALCLM_MODEL) {
      fail("The build is missing: run demo/tools/build_wasm.sh and python -m demo.tools.build_model (see demo/README.md), then reload.");
      return;
    }
    try {
      eng = await CalcLM.create(CalcLM.base64ToBytes(globalThis.CALCLM_WASM_B64), globalThis.CALCLM_MODEL);
    } catch (e) {
      fail(`The model did not load: ${e.message || e}`);
      return;
    }
    ui.model.textContent = `Model ${eng.model.name} (export ${eng.exportSha256.slice(0, 12)}...), ` +
      `W build, LD2 ${eng.ld2Ok ? "on" : "unavailable"}, masks ${eng.masksOk ? "available" : "unavailable"}.`;
    buildMenu();
    describe();
  }

  init();
})();
