// Shared by the Node checks: load the engine with the built module and model package (the
// same two script files index.html loads), parse ce/calclm/src/topics.h and the
// hardware-check sheets.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const DEMO = path.resolve(__dirname, "..");
const ROOT = path.resolve(DEMO, "..");
const BUILD = path.join(DEMO, "build");

function loadScripts() {
  for (const f of ["calclm-core.js", "calclm-model.js"]) {
    const p = path.join(BUILD, f);
    if (!fs.existsSync(p)) {
      throw new Error(`${p} is missing: run demo/tools/build_wasm.sh and ` +
                      "python -m demo.tools.build_model first (demo/README.md)");
    }
    require(p);
  }
  return require(path.join(DEMO, "calclm.js"));
}

async function engine(model) {
  const CalcLM = loadScripts();
  return CalcLM.create(CalcLM.base64ToBytes(globalThis.CALCLM_WASM_B64),
                       model || globalThis.CALCLM_MODEL);
}

// topics.h: the arrays the calculator program is built from
function topicsHeader() {
  const h = fs.readFileSync(path.join(ROOT, "ce/calclm/src/topics.h"), "utf8");
  const nums = (name) => h.match(new RegExp(`${name}\\[\\d+\\] = \\{([^}]*)\\}`))[1]
    .split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const strs = (name) => [...h.match(new RegExp(`${name}\\[TOPIC_N\\] = \\{([^}]*)\\}`))[1]
    .matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]);
  const ids = nums("topic_ids"), start = nums("topic_start");
  const define = (name) => Number(h.match(new RegExp(`#define ${name} (\\d+)`))[1]);
  return {
    n: define("TOPIC_N"), cap: define("TOPIC_CAP"), temp: define("TOPIC_TEMP_Q8"),
    topK: define("TOPIC_TOP_K"), masks: define("TOPIC_MASKS"),
    labels: strs("topic_label"), checks: strs("topic_check"), rates: strs("topic_rate"),
    times: strs("topic_time"),
    openings: start.slice(0, -1).map((s, i) => ids.slice(s, start[i + 1])),
  };
}

// <!-- hardware-check seed=... --> lines of every hw/checks/*/CHECK.md for one export, or null
// when this tree has no hw/checks (the public export leaves the hardware-check records out)
function sheetLines(exportSha16) {
  const dir = path.join(ROOT, "hw/checks");
  if (!fs.existsSync(dir)) return null;
  const out = [];
  for (const d of fs.readdirSync(dir).sort()) {
    const f = path.join(dir, d, "CHECK.md");
    if (!fs.existsSync(f)) continue;
    for (const m of fs.readFileSync(f, "utf8").matchAll(/<!-- hardware-check ([^>]*) -->/g)) {
      const kv = Object.fromEntries(m[1].trim().split(/\s+/).map((s) => s.split("=")));
      if (kv.export_sha256 === exportSha16) out.push({ sheet: d, ...kv });
    }
  }
  return out;
}

const sha256 = (buf) => crypto.createHash("sha256").update(buf).digest("hex");

function idsBytes(ids) {
  const b = Buffer.alloc(2 * ids.length);
  ids.forEach((t, i) => b.writeUInt16LE(t, 2 * i));
  return b;
}

// the hardware-check sheet's numbers for one generated program (ce/tools/hw_check.py)
function sheetNumbers(eng, ids) {
  const text = eng.render(ids.slice(1));
  return {
    tokens: ids.length - 1,
    lines: text.split("\n").length,
    ids_sha256: sha256(idsBytes(ids)),
    text_sha256: sha256(Buffer.from(text, "utf8")),
    body_sha256: sha256(Buffer.from(eng.body(ids))),
  };
}

module.exports = { DEMO, ROOT, BUILD, engine, loadScripts, topicsHeader, sheetLines,
                   sheetNumbers, sha256 };
