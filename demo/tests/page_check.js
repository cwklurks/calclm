#!/usr/bin/env node
// A8 of docs/agents/phase6-browser-demo.md: open demo/index.html from file:// in a local
// headless Chromium (no network; the DevTools connection is a localhost pipe of this test, not
// of the page), write Cylinder volume at seed 1, level 2, and compare what the page shows with
// the v2.6-rc2 hardware-check sheet: "done: 103 tok", the check line and the ids sha256.
//
//   node demo/tests/page_check.js [path-to-chrome]   (or CALCLM_DEMO_CHROME)
// Exits 0 on success, 1 on a mismatch, 2 when no headless Chromium is found (skip).
"use strict";
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { pathToFileURL } = require("node:url");
const H = require("./helpers.js");

function findChrome() {
  const given = process.argv[2] || process.env.CALCLM_DEMO_CHROME;
  if (given) return given;
  const base = path.join(os.homedir(), ".cache", "ms-playwright");
  if (!fs.existsSync(base)) return null;
  for (const d of fs.readdirSync(base).sort().reverse()) {
    for (const rel of ["chrome-headless-shell-linux64/chrome-headless-shell",
                       "chrome-linux64/chrome"]) {
      const p = path.join(base, d, rel);
      if (fs.existsSync(p)) return p;
    }
  }
  return null;
}

function launch(chrome) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "calclm-demo-"));
  // --no-sandbox: the test's own browser opens only this repository's page from disk (the page
  // makes no network calls); Chromium's sandbox needs user namespaces that sandboxed shells
  // and some distributions do not allow.
  const proc = spawn(chrome, ["--headless", "--disable-gpu", "--no-sandbox", "--no-first-run",
                              "--no-default-browser-check", "--remote-debugging-port=0",
                              `--user-data-dir=${profile}`, "about:blank"],
                     { stdio: ["ignore", "ignore", "pipe"] });
  const ws = new Promise((resolve, reject) => {
    let err = "";
    proc.stderr.on("data", (d) => {
      err += d;
      const m = err.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) resolve(m[1]);
    });
    proc.on("exit", (c) => reject(new Error(`chrome exited ${c}: ${err}`)));
  });
  return { proc, profile, ws };
}

function cdp(url) {
  const sock = new WebSocket(url);
  let id = 0;
  const pending = new Map();
  const events = [];
  sock.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method) events.push(msg);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
    }
  };
  const open = new Promise((r) => { sock.onopen = r; });
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const m = { id: ++id, method, params };
    if (sessionId) m.sessionId = sessionId;
    pending.set(m.id, { resolve, reject });
    sock.send(JSON.stringify(m));
  });
  return { open, send, events, close: () => sock.close() };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const chrome = findChrome();
  if (!chrome) {
    console.log("SKIP: no headless Chromium (set CALCLM_DEMO_CHROME)");
    process.exitCode = 2;
    return;
  }
  const sheets = H.sheetLines("e68eca508872b7a5");
  if (sheets === null) {
    console.log("SKIP: no hw/checks in this tree (the hardware-check sheet is the reference)");
    process.exitCode = 2;
    return;
  }
  const sheet = sheets.find((l) => l.topic === "CYL");
  // CALCLM_DEMO_URL: the page served elsewhere (e.g. python -m http.server on localhost)
  const page = process.env.CALCLM_DEMO_URL || pathToFileURL(path.join(H.DEMO, "index.html")).href;
  const { proc, profile, ws } = launch(chrome);
  const c = cdp(await ws);
  try {
    await c.open;
    // attach to a blank tab first, so every request of the page is recorded (Network domain)
    const { targetId } = await c.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await c.send("Target.attachToTarget", { targetId, flatten: true });
    await c.send("Network.enable", {}, sessionId);
    await c.send("Page.navigate", { url: page }, sessionId);
    const ev = async (expr) => (await c.send("Runtime.evaluate",
      { expression: expr, returnByValue: true, awaitPromise: true }, sessionId)).result.value;
    let ready = false;
    for (let i = 0; i < 100 && !ready; i++) {
      await sleep(100);
      ready = await ev("!!document.querySelector('#menu input') && !document.getElementById('go').disabled");
    }
    if (!ready) throw new Error("the page did not load the model: " + await ev("document.getElementById('model').textContent"));
    await ev(`(() => {
      document.querySelector('#menu input[value="0"]').checked = true;
      document.querySelector('input[name=level][value="2"]').checked = true;
      document.getElementById('seed').value = '${sheet.seed}';
      document.getElementById('pace').value = '0';
      document.getElementById('controls').dispatchEvent(new Event('change'));
      document.getElementById('go').click();
    })()`);
    let got = null;
    for (let i = 0; i < 300; i++) {
      await sleep(100);
      got = await ev(`(() => {
        const facts = [...document.querySelectorAll('#facts dt')].map((dt) => [dt.textContent, dt.nextElementSibling.textContent]);
        return { status: document.getElementById('status').textContent,
                 check: document.getElementById('check').textContent,
                 device: document.getElementById('device').className,
                 program: document.getElementById('program').textContent,
                 facts: Object.fromEntries(facts) };
      })()`);
      if (got.status.startsWith("done") && got.facts["ids sha256"]) break;
    }
    const eng = await H.engine();
    const want = {
      status: `done: ${sheet.tokens} tok`,
      check: eng.topics[0].check,
      ids: sheet.ids_sha256,
    };
    const shown = got.program.replace(/^:/, "").replace(/\n:/g, "\n");
    eng.start({ topic: 0, seed: Number(sheet.seed), level: 2 });
    while (eng.step()) { /* to the end */ }
    const problems = [];
    if (got.status !== want.status) problems.push(`status ${JSON.stringify(got.status)}`);
    if (got.check !== want.check) problems.push(`check ${JSON.stringify(got.check)}`);
    if (got.facts["ids sha256"] !== want.ids) problems.push(`ids sha256 ${got.facts["ids sha256"]}`);
    if (shown !== eng.render(eng.ids().slice(1))) problems.push("program text differs from render");
    if (!got.device.includes("on")) problems.push("not marked as the calculator's settings");
    // every request the page made is one of its own files (the scripts, the stylesheet)
    const base = page.slice(0, page.lastIndexOf("/") + 1);
    const urls = c.events.filter((e) => e.method === "Network.requestWillBeSent" && e.sessionId === sessionId)
      .map((e) => e.params.request.url);
    const foreign = urls.filter((u) => !u.startsWith(base));
    if (!urls.some((u) => u.endsWith("build/calclm-model.js"))) problems.push("requests were not recorded");
    if (foreign.length) problems.push(`requests outside the demo folder: ${foreign.join(", ")}`);
    console.log(problems.length ? `FAIL: ${problems.join("; ")}`
      : `page OK (${page.split(":")[0]}): ${got.status}, ${got.check}, ids sha256 ${want.ids.slice(0, 16)}..., ` +
        `${urls.length} requests, all to the demo's own files`);
    process.exitCode = problems.length ? 1 : 0;
    if (process.env.CALCLM_DEMO_SHOT) {        // a PNG of the page, for a look at the layout
      await c.send("Emulation.setDeviceMetricsOverride",
                   { width: 1200, height: 900, deviceScaleFactor: 1, mobile: false }, sessionId);
      const { data } = await c.send("Page.captureScreenshot", { format: "png" }, sessionId);
      fs.writeFileSync(process.env.CALCLM_DEMO_SHOT, Buffer.from(data, "base64"));
    }
  } finally {
    c.close();
    proc.kill();
    await sleep(200);
    fs.rmSync(profile, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
