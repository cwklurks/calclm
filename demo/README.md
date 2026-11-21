# CalcLM in the browser (local demo)

> Not runnable from the public repository: it needs a model export, which is not published.
> Only `demo/tools/build_wasm.sh` works without it.

A static page that runs the deployed CalcLM model (v2.6-rc2: E24's L4c-s11, W build, LD2 decoder; E10's
L4-s23 until v2.5-rc1) on your
computer and writes the same TI-BASIC program the TI-84 Plus CE writes for the same topic, seed
and mask level, token for token. It runs the calculator program's own C inference core
(`ce/calclm/core`, `tracker/tracker.c`) and the pure part of its start screen (`src/menu.c`,
`src/topics.h`), compiled to WebAssembly. Plan and results:
docs/agents/phase6-browser-demo.md (in the private research record).

**Local only.** `build/` holds the model weights (`calclm-model.js`) and, after the check,
generated programs (`reference.json`). It is git-ignored. Do not commit, upload or host it.

## Build

From the repository root; needs clang with the wasm32 target and `wasm-ld` (LLVM), the project
Python environment and, for the checks, Node 22 or later. No emscripten, no npm packages.

```sh
bash demo/tools/build_wasm.sh                                   # build/calclm.wasm, calclm-core.js
uv run python -m demo.tools.build_model                         # build/calclm-model.js (weights)
```

`build_model` looks for `runs/e24/export/L4c-s11` in this checkout, then in the main checkout
when run from a git worktree; `--export DIR` or `CALCLM_DEMO_EXPORT` picks another. The LD2 fit
is `ce/tests/fixtures/ld2_offsets_L4c-s11.json` (`--fit`, `model.topics.FIT`); the build refuses a fit made for
another export.

## Open

Either open `demo/index.html` directly (file://), or serve the folder:

```sh
cd demo && python -m http.server 8000 --bind 127.0.0.1    # then http://127.0.0.1:8000/
```

Checked from file:// in headless Chromium (the full run, `tests/page_check.js`) and headless
Firefox (the page and module load). The page loads its scripts with `<script src>` (browsers
block `fetch` on file://), makes no
network requests (its Content-Security-Policy forbids them) and uses no CDN.

## Use

- **Topic**: one of the ten menu entries (with the calculator's measured rate and time) or
  Random program.
- **Seed**: digits, 0 to 4,294,967,295. The calculator's Seed row holds at most 999,999,999, so
  the page marks larger seeds as not the calculator's. A topic mixes the seed with fmix32 first,
  as the calculator does.
- **Mask level**: 2 is the calculator (Masks on). 3 is the calculator's Masks row off (Random
  only on the device: safety rules, no grammar masks, no LD2). 1 is v2.3's masks, as in rc2's
  hardware-check sheet. The page says when a combination is not something the calculator does
  (a topic at level 1 or 3).
- **Pace**: as fast as it runs, readable, or the calculator's speed (0.67 tokens/s).

While it runs, the topic's opening is shown dim ("reading the topic n/N"), then the model's
tokens. The count is tokens after `<bos>`, the calculator's "done: N tok"; topics stop at 256
tokens with `<bos>`, Random at 1,024. At the end the page shows the topic's check line, the
decoder seed and settings, and the sha256 of the ids, comparable with the hardware-check sheets'
`ids_sha256`. "Another try" is the calculator's `[2nd]`: same topic, seed + 1.

## Check

```sh
node --test demo/tests/engine.test.js            # settings, topic table, sheets, topic gate
uv run python -m demo.tools.reference            # build/reference.json (~2.5 min)
node demo/tests/check_grid.js                    # 12 per topic + 12 (v2.6-rc2: 132), id for id and text
node demo/tests/page_check.js                    # the page from file:// in headless Chromium
uv run python -m pytest demo/tests               # all of the above (grid: slow)
```

`check_grid.js` compares the module with the Python reference that the hardware-check sheets use
(`ce/tools/hw_check.py`) on every topic x seeds {1, 2, 3, 48} x mask levels {1, 2, 3} and Random
at the same seeds and levels. `engine.test.js` also reproduces the sheet lines for this
export committed in `hw/checks/` (v2.6-rc2: Random seed 23 at level 2, Cylinder volume seed 1). `page_check.js` uses a Chromium from `~/.cache/ms-playwright` or `CALCLM_DEMO_CHROME`,
and skips without one.
