# M5 evaluation harness: do generated TI-BASIC programs run on the real TI-OS?

`eval/run_programs.py` runs TI-BASIC programs on an emulated TI-84 Plus CE (CEmu v2.0 core,
the author's own ROM dump, OS 5.8.0.0022), drives them through their prompts, and classifies
each one as OK, ERR:<NAME>, TIMEOUT or REJECTED. Programs come in as a directory of .8xp
files, as token-id sequences plus `corpus/vocab.json` (the model's output format), or as a
sample of the corpus. Native code is out of scope: `Asm(` is just another error (OS 5.8
answers ERR:INVALID).

## Method

1. **Input to .8xp.** Each program becomes a `ProgramSpec` (`eval/inputs.py`). Token ids are
   mapped through vocab.json: `<bos>`/`<pad>` are skipped, decoding stops at `<eos>`, and
   `<unk>` or an out-of-range id makes the program REJECTED. The resulting file is written
   with tivars `TIProgram`. Generated programs are named `G0000000`, `G0000001`, ...
   Corpus and .8xp programs keep their own names (θ is supported). A name the home screen
   cannot type is replaced by `ZZEVAL`.
2. **Clean state every time.** Before each program the emulator loads one saved state: the
   freshly booted calculator (RAM cleared, OS defaults: MathPrint, Radian, Func, L1-L6
   empty). Anything a program deletes, archives or changes is gone for the next program.
   With `--library DIR`, every .8xp in DIR is first sent to Archive, and that state is
   cached and used as the clean state. This mode is for program suites whose programs call
   each other.
3. **Send and launch.** The program is sent over the emulated USB link to RAM. The harness
   then checks that it is in the VAT; if not, the outcome is REJECTED. It is launched like
   the CEmu autotester does: Clear, then the `prgm` keycode, the name and Enter.
4. **Drive** (`eval/driver.py`). Every 50 ms of emulated time the harness reads the OS
   state from RAM (`eval/osram.py`, equates from CEmu's `ti84pce.inc`; the behaviour was
   checked on this ROM):
   - bit 1 of flags+8 is set while a BASIC program runs;
   - `cxCurApp` (0xD007E0) is 0x50 during Input/Prompt and 0x52 on the ERR: screen;
   - `menuCurrent` (0xD00824) is 0x14 for a program's `Menu(`;
   - `plotDisp` means the graph screen is shown;
   - `errNo` (0xD008DF) holds the error number;
   - the CPU busy fraction is about 0.17 when the OS waits for a key and 1.0 while a
     program computes.
5. **Classify** (definitions below), record everything, then stop a TIMEOUT program with ON.

### Input-driving policy

The harness answers a wait only after the screen state has been idle and unchanged for
300 ms, and never while a key it sent is still unread. `Menu(` can sit on an injected key
for over a second. Answering twice in that window once sent an Enter into a later ERR:
menu, so this rule is load-bearing. Answers are injected as OS keycodes, so none is lost.

| Program is waiting at | Harness does |
|---|---|
| `Input` / `Prompt` (any variable type) | types `1`, then ENTER (`--input-text` changes it) |
| `Pause`, `Pause` on the graph screen, `Input` with no variable | ENTER |
| `Menu(` | first visit of that `Menu(` (keyed by the parser position): option 1. Every later visit: the last option (Up wraps to it, then ENTER), which is usually BACK or QUIT. |
| `getKey` loop | when the program has been busy for 500 ms and sampling the parser's position (`curPC`) shows it is cycling over code that contains a `getKey` token, it presses one physical key from `Enter, Right, Down, 2nd, Left, Up, Clear` (rotating) |
| budget spent (default 30 s emulated) | it stops answering. Within 3 s more the program either ends (counted as OK or ERR), settles into a prompt (TIMEOUT waiting for that prompt) or keeps running (TIMEOUT `getkey` or `computing`). Then ON is pressed. |

## Outcome definitions

| Outcome | Meaning |
|---|---|
| OK | the program returned to the OS (prog flag clear, CPU idle 100 ms, no ERR: screen) within the budget. This includes ending on the graph screen. |
| ERR:<NAME> | the OS showed its ERR: screen. NAME comes from `errNo & 0x7F`, using the OS 5.8 table in `osram.ERRORS` (SYNTAX, DOMAIN, UNDEFINED, DIM MISMATCH, INVALID DIM, MEMORY, INVALID, ...). The harness then takes 2:Goto, waits until the editor has stopped moving the cursor (CPU idle, cursor unchanged for 50 ms; at most 30 s emulated) and reads it to record `error_offset`, `error_line_no` and `error_line`. |
| TIMEOUT | still running when the budget ran out. `timeout_state` is one of `input`, `menu`, `pause`, `graph_wait` (waiting for the user), `getkey` (polling keys) or `computing` |
| REJECTED | tivars could not build or read the file, `<unk>`/bad token ids, an empty or oversized body, or the OS did not store it |
| HARNESS_ERROR | an exception, a dead worker or no result within `wall_limit_s + 60` s. Reported separately and never seen in the runs below. |

The OS sometimes sets `errNo` for an error it handles itself, for example NONREAL ANS while
graphing `√(X)`, or Xmax=Xmin at a Prompt. That is not an error outcome. An OK program's
`errno_at_end` keeps it for transparency.

"Run without ERR" = OK + TIMEOUT, matching SPEC M5's "% that run without ERR:".
"Tokenize" = not REJECTED.

### Per-program record (`results.jsonl`)

| Field(s) | Contents |
|---|---|
| `index`, `name`, `orig_name`, `source`, `meta` | identity and corpus metadata |
| `n_bytes`, `n_tokens` | size |
| `static` | flags from the tokens: `getkey`, `input`, `pause`, `menu`, `prgm_call`, `asm`, `draw`, plus `unknown_tokens` |
| `outcome`, `label`, `error`, `timeout_state`, `reject_reason` | the result |
| `error_offset`, `error_line_no`, `error_line`, `errno_at_end` | error detail |
| `emu_ms` | emulated ms from launch to outcome |
| `wall_s` | wall time |
| `graph` | the graph screen was displayed at any poll |
| `actions` | counts of each kind of answer the harness gave |
| `last_action`, `ms_since_action` | so errors right after a driven answer can be discounted |
| `home_text` | the OS textShadow: final home-screen text |
| `lcd_text` | the LCD read by template OCR; shows the ERR: screen |
| `screenshot` | PNG path, for the first `--shots` programs |

`summary.json` / `summary.md` hold:

- counts and % per outcome;
- the error-name histogram;
- the timeout-state histogram;
- tokenize % and run-without-ERR %;
- a count of errors that came within 2 s of a driven Input answer;
- outcome % within static subsets: `self_contained` (no `prgm` call, no Asm, no corpus
  `lib_calls` tag), `calls_prgm`, `uses_asm`, `uses_getkey`, `uses_input`, `uses_menu`,
  `uses_draw`, and `le_256_tokens` (fits CalcLM's context);
- throughput.

## Commands

```sh
uv run pytest eval/tests -q                       # 146 tests, ~30 s (emulator tests skip without the ROM)
uv run python -m eval.run_programs --handmade --name handmade
uv run python -m eval.run_programs SUITE_DIR \
    --library SUITE_DIR --name suite --budget-ms 60000   # programs that call each other
uv run python -m eval.run_programs --corpus data/extract/programs.jsonl --sample 400 --seed 0 --name corpus-baseline-400
uv run python -m eval.run_programs --corpus data/extract/programs.jsonl --sample 2000 --seed 1 --name corpus-baseline-2000
uv run python -m eval.run_programs --ids data/tok/val.bin --vocab corpus/vocab.json --name val-tokens
# generated programs: a .jsonl of id lists (or {"ids": [...]}) / .json / .bin
uv run python -m eval.run_programs --ids gen.jsonl --vocab corpus/vocab.json --name gen-v0 --workers 24
```

Common options:

- `--workers N` (default: half the cores);
- `--budget-ms` (default 30000, emulated);
- `--shots N` (screenshots of the first N);
- `--input-text`;
- `--out DIR` instead of `eval/runs/NAME/`.

Python API:

- `run_programs.run_token_ids(seqs, vocab_path, out_dir, workers=, budget_ms=)` returns the
  records;
- `run_programs.run_specs(...)`;
- `driver.run_one(emu, spec, clean_state, Config())` runs a single program.

The emulator comes from the CEmu harness checkout that `CALCLM_CEMU_ROOT` names (its
`tools/cemu`; the ROM is its `rom/ti84pce.rom`, or the file `CALCLM_ROM` names). Without
either, the emulator tests skip. That checkout is used read-only: it is imported through
`sys.path` with bytecode writing off, and its state directory is redirected. The booted state, the OCR glyph
templates and library states contain ROM-derived data. They live in `eval/runs/_cache/`,
which is git-ignored with the rest of `eval/runs/`. The first run boots the ROM once
(about 3 s).

## Results

### Hand-made set (`eval/handmade.py`, 34 programs)

Every case gets its expected outcome:

- OK: valid code, loops, `Output(`, `Stop`, `Wait`, Input, Prompt, string Input, Pause,
  `Menu(` with a QUIT item, five getKey idioms, graph Pause/DispGraph/`Input` on the graph
  screen, and Xmax=Xmin at a Prompt;
- ERR: SYNTAX, DOMAIN, DIVIDE BY 0, UNDEFINED, DIM MISMATCH, INVALID DIM, DATA TYPE, LABEL,
  MEMORY, INVALID (`Asm(`), and a slow `Menu(` that ends in DIVIDE BY 0;
- TIMEOUT: `computing`, `input`, `getkey`, `menu`.

`eval/tests/test_driver.py` also checks:

- the TIMEOUT state is stable for every budget from 3000 to 3550 ms;
- error lines are located (`Disp 1/0` on line 2, and so on), also deep in long programs
  (line 200, and line 160 after 159 wrapped lines);
- state does not leak between programs;
- θ names work.

### The author's own 47-program suite (`--library`, 60 s budget)

The suite's entry point ran with no error. The harness drove it through 4 nested menus and
56 Input answers, with correct results on screen. It ended as TIMEOUT `input`: one of its
tools loops until the user types a letter to leave, and the fixed answer `1` never does.

Run on its own, each other program is a subprogram:

- 15 print a guard that sends the user to the entry point, and are OK;
- 6 more are OK (help, theory and settings screens);
- 25 are ERR (UNDEFINED 17, DATA TYPE 7, INCREMENT 1), because they read lists and
  strings that the entry point sets up.

So standalone subprograms are not meaningful entry points. For this suite, the evidence
that the input driving works is the entry point itself.

### Baseline: real human-written programs, run alone

| Set | n | OK % | ERR % | TIMEOUT % | REJECTED % | run without ERR % |
|---|---:|---:|---:|---:|---:|---:|
| corpus sample, seed 0 | 400 | 39.5 | 50.8 | 9.8 | 0 | 49.3 |
| corpus sample, seed 1 | 2000 | 40.6 | 49.3 | 10.2 | 0 | 50.8 (95% CI ±2.2) |
| held-out val split via token ids (`data/tok/val.bin`) | 613 | 37.0 | 47.8 | 14.7 | 0.5 | 51.7 (±4.0) |
| 2000 sample, self-contained | 1291 | 55.9 | 31.5 | 12.7 | 0 | 68.6 |
| 2000 sample, ≤ 256 tokens | 694 | 47.8 | 45.5 | 6.6 | 0 | 54.5 |
| val, self-contained | 415 | 52.1 | 29.4 | 17.8 | 0.7 | 69.9 |
| val, ≤ 256 tokens | 129 | 48.1 | 41.1 | 10.9 | 0 | 58.9 |

The corpus samples are kept programs from `data/extract/programs.jsonl`, excluding the author's
programs and Axe sources. The val split's 3 REJECTED programs contain `<unk>`.

Error histogram, 2000-program sample (985 errors):

| Error | n |
|---|---:|
| UNDEFINED | 606 |
| INVALID DIM | 104 |
| INVALID | 68 |
| DATA TYPE | 49 |
| ARGUMENT | 41 |
| DOMAIN | 40 |
| DIVIDE BY 0 | 34 |
| SYNTAX | 23 |
| LABEL | 8 |
| NONREAL ANS | 5 |
| others | 1 each |

977 of the 985 have a located `error_line`. These positions predate the locator fix (see
Limitations, "Error position"): in long programs they stop short of the real error line.

TIMEOUT states, same sample:

| State | n |
|---|---:|
| getkey | 98 |
| computing | 32 |
| input | 31 |
| pause | 20 |
| menu | 18 |
| graph_wait | 4 |

How to read this yardstick:

- **About half of real programs fail when run alone.** Most of that comes from being one
  file of a multi-file package. 61% of the errors are UNDEFINED: missing subprograms
  (`calls_prgm`: 82% ERR) or missing lists, matrices, Pics and strings that another file
  creates.
- **Other causes:**
  - 83+/84+-era code that the CE OS rejects: xLIB/Celtic `real(` calls give ARGUMENT;
    `Asm(` gives INVALID;
  - genuine bugs;
  - our fixed answers: 110 of the 985 errors come within 2 s of a typed `1`, for example
    a list name was expected (DATA TYPE), or the math needed other numbers (DIVIDE BY 0,
    NONREAL ANS).
- **The fairest comparison** for single-file generated programs is the self-contained or
  ≤ 256-token rows: about 69% of self-contained and about 55-59% of short human programs
  run without ERR:.
- Generated programs should be scored with exactly the same command, budget and policy.

### Throughput

Measured on the 2000-program sample: 10,639 s of emulated time, 2000 programs.

| Workers | programs/min |
|---:|---:|
| 8 | 1208 |
| 16 | 2120 |
| 24 | 2254 (load average ~4) |
| 31 | 2270 |

- Throughput flattens past 16 workers: a few long TIMEOUT programs (up to 9 s wall) form
  the tail.
- While an unrelated compile job loaded the box (load average 20-45), 24 workers gave
  1370-1750 per minute.
- The table predates the error-locator fix. On the 400-program sample with 8 workers the fix
  took throughput from 1264 to 1188 programs/min; outcomes and `emu_ms` were identical.
- Median wall time per program is 0.32 s. Busy emulation runs about 15x real time, and key
  waits cost almost nothing.
- Results are bit-identical, labels and `emu_ms` alike, across 8, 16, 24 and 31 workers
  and across repeated runs. Each worker process owns one CEmu core (the core is a
  per-process global), so the shim is safe under multiprocessing.

## Limitations

- **One fixed answer.**
  - Input loops whose exit needs a specific value end as TIMEOUT `input` (the suite's
    entry point, "how many tosses?" validators).
  - Some ERRs are policy artifacts; `err_after_input` counts them.
  - The menu policy can pick an option that needs state, such as CONTINUE before NEW GAME.
- **getKey games** usually end as TIMEOUT `getkey`, which counts as running without error;
  the harness does not judge whether a game is playable. A getKey loop whose parser region
  does not contain the token (getKey called from a subprogram in another loop) is reported
  as `computing` and gets no keys.
- **Standalone runs** penalise human programs that belong to packages.
  `--library` exists for suites, but corpus siblings (other files from the same zip) are
  not loaded automatically.
- **Error position:** the line where the OS parser stood (2:Goto cursor).
  - On a long program the editor walks the cursor down to the error, busy all the way, at
    about 40 short lines per emulated second (line 200: 4.6 s; line 598: 14 s). Runs made
    before 2026-10-13 read the cursor a fixed 500 ms after 2:Goto, so their `error_offset`,
    `error_line_no` and `error_line` stop short: no run reported an error past line 28 (val
    #14: line 14 reported, real error on line 195). Outcomes, labels, `emu_ms` and every
    other field are unaffected. On the 400-program corpus sample, 45 of 206 positions moved
    (all forward). `tracker/relocate.py` outputs (`eval/runs/w2-*-loc/`) already have the
    corrected positions.
  - A position is left empty if the editor has not settled within 30 s emulated.
  - Errors raised by a side effect, such as a graph redraw evaluating an undefined Y= or
    plot, point at the statement that triggered it, sometimes the start of the next line.
  - No position for errors whose menu has no Goto (e.g. LABEL).
- **OS facts are tied to OS 5.8.0.0022:** flag bits, `cxCurApp` and `menuCurrent` values.
  `tests/test_driver.py` will catch a change if the ROM is replaced.
- **Screen text:**
  - `lcd_text` OCR uses the large-font templates from the CEmu harness; it
    shows � for MathPrint history and small-font text.
  - `home_text` (textShadow) is the reliable final text.
  - Graph-screen content is only in the screenshot plus the `graph` flag.
- **Budget:** 30 s of emulated time. Slow but correct computations become TIMEOUT
  `computing`; raise `--budget-ms` if that matters.
- **Feeding keys:** a physical key fed to a getKey loop is consumed there almost surely. In
  theory it could be latched and read later, for example ENTER at the home screen
  re-running the program, which would end as TIMEOUT. This was never observed.
