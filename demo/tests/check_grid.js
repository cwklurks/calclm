#!/usr/bin/env node
// A1 of docs/agents/phase6-browser-demo.md: run every case of the Python reference grid
// (demo/build/reference.json, written by `python -m demo.tools.reference`) through the
// WebAssembly module and compare the token ids and the rendered text exactly. Writes a summary
// (counts only, no program text) to demo/build/grid.json; exits 1 on any mismatch.
//
//   node demo/tests/check_grid.js [reference.json]
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const H = require("./helpers.js");

function firstDiff(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

async function main() {
  const refPath = process.argv[2] || path.join(H.BUILD, "reference.json");
  const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
  const eng = await H.engine();
  if (ref.export_sha256 !== eng.exportSha256) {
    throw new Error(`reference is for export ${ref.export_sha256}, the model package for ` +
                    eng.exportSha256);
  }
  const t0 = Date.now();
  const rows = [];
  for (const c of ref.cases) {
    const topic = c.topic === null ? null : eng.topics.findIndex((t) => t.key === c.topic);
    if (topic === -1) throw new Error(`no topic ${c.topic} in the module`);
    eng.start({ topic, seed: c.seed, level: c.level });
    while (eng.step()) { /* stream to the end */ }
    const ids = eng.ids();
    const d = firstDiff(ids, c.ids);
    const text = eng.render(ids.slice(1));
    const ok = d === -1 && text === c.text;
    rows.push({ topic: c.topic, seed: c.seed, level: c.level, tokens: ids.length - 1,
                ref_tokens: c.ids.length - 1, ids_equal: d === -1, text_equal: text === c.text,
                first_diff: d, ok });
    if (!ok) {
      console.log(`MISMATCH ${c.topic || "Random"} seed ${c.seed} level ${c.level}: ` +
                  `first differing position ${d} (module ${ids[d]}, reference ${c.ids[d]}), ` +
                  `text equal ${text === c.text}`);
    }
  }
  const match = rows.filter((r) => r.ok).length;
  const summary = {
    export_sha256: eng.exportSha256, cases: rows.length, match,
    topic_cases: rows.filter((r) => r.topic !== null).length,
    random_cases: rows.filter((r) => r.topic === null).length,
    tokens: rows.reduce((s, r) => s + r.tokens, 0), seconds: (Date.now() - t0) / 1000, rows,
  };
  fs.writeFileSync(path.join(H.BUILD, "grid.json"), JSON.stringify(summary, null, 1) + "\n");
  console.log(`${match}/${rows.length} cases byte-identical (${summary.topic_cases} topic, ` +
              `${summary.random_cases} Random; ${summary.tokens} tokens in ` +
              `${summary.seconds.toFixed(1)} s)`);
  process.exitCode = rows.length > 0 && match === rows.length ? 0 : 1;   // an empty grid proves nothing
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
