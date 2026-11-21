# TI-BASIC program-state tracker (CalcLM v2, workstream W2)

One C99 implementation (`tracker.c`) tracks the state of a TI-BASIC program as its tokens are
written. Training calls it through ctypes (`bind.py`), and the calculator compiles the same
file with the CE toolchain. `spec.py` states the same semantics in plain Python, and the two
are tested against each other on every train and val program. The semantics were checked
against TI-OS 5.8.0.0022 in CEmu (`probe_os.py`), not assumed.

## Files

| File | What |
|---|---|
| `tracker.h`, `tracker.c` | the tracker: `tracker_reset`, `tracker_step(state, token_id, &feat)`, `tracker_features`, `tracker_summary` |
| `trk_tables.h` | generated token-class table (781 bytes) + class codes |
| `gen_tables.py` | writes `trk_tables.h` from `corpus/vocab.json` (or an export's `model.json`) |
| `feats.py` | feature vocabulary (group sizes, value codes, bag bit layout, row ids) |
| `spec.py` | Python spec (straightforward; derives token roles itself, not from the C table) |
| `bind.py` | builds `build/libtracker.so`; `batch(stream)`, `summaries(stream)`, `Tracker` |
| `host_batch.c` | host-only stream helpers used by `bind.py` |
| `difftest.py` | T2: C vs spec on fuzz, all val and all train programs |
| `ce_drv.c`, `ce_bench.py` | T3: CE build, bit-exactness and cycles in CEmu (via `ce/tools/bench.py`, unmodified) |
| `probe_os.py` | the TI-OS behaviour the semantics rest on (166 probe programs in CEmu) |
| `validate.py`, `relocate.py` | T4: tracker vs CEmu outcomes; patient error locator (see "Harness finding") |
| `tests/` | 161 tests (constructs, C basics, tables, differential, causality, CE, validate) |

## Using it

Training (host):

```python
from tracker import bind
cats, bags = bind.batch(np.fromfile("data/tok/train.bin", dtype="<u2"))
# cats: (T, 9) uint8 categorical ids; bags: (T,) uint64 multi-hot; ~0.5 s for all of train.bin
```

**Alignment (causality).** Row `i` is the state after token `i` has been read, token `i`
included. It is computed from tokens `[program start .. i]` only, and `<bos>` resets the state,
so nothing carries over from the previous program. Feed it as an extra input at position `i`,
next to the embedding of input token `i`. Do not align it with the target `i+1`. Tests:
`tests/test_causality.py` (prefix features are identical with or without the rest of the
program, for every prefix and every possible next token; `<bos>` resets after pathological
endings; stream == programs one by one).

Calculator: `trk_state` (510 B) lives in the program's RAM, and `tracker_step(&st, tok, &f)`
runs once per sampled token. To add the feature embedding to the residual stream
incrementally, keep the previous `trk_feat`. For each of the 9 groups whose id changed,
subtract the old row and add the new one. XOR the old and new `hint`/`var` bytes, and add or
subtract one row per changed bit. Most tokens change 0-2 categorical ids and no bag bits;
`var` bits change only on assignments, and hint bits only in name contexts.

## Features (state after the token)

9 categorical groups (one row each) + a 64-bit bag = **106 embedding rows** (42 + 64).
Row id = group offset + value; bag bit b is row 42 + b (`feats.row_ids`).

| Group (rows) | Values |
|---|---|
| `ctx` (0-8) | 0 statement start, 1 inside a statement, 2 inside a string, 3 right after `→`, 4 reading a `Lbl` name, 5 reading a `Goto`/`Menu(` label name, 6 reading a custom list name, 7 reading a `prgm` name, 8 right after `DelVar` |
| `name_len` (9-11) | name characters read so far in ctx 4-7: 0, 1, 2+ (else 0) |
| `paren` (12-15) | open `(` / `NAME(` depth: 0, 1, 2, 3+ |
| `lit` (16-18) | innermost literal: 0 none, 1 `{`, 2 `[` |
| `menu` (19-22) | 0 not in `Menu(`, 1 title argument, 2 option text, 3 label argument |
| `blk_top` (23-29) | innermost open block: 0 none, 1 If-Then, 2 Else, 3 For(, 4 While, 5 Repeat, 6 deeper than 32 |
| `blk_depth` (30-34) | open blocks: 0, 1, 2, 3, 4+ |
| `if_state` (35-38) | 0 none, 1 inside an `If` (or `IS>(`/`DS<(`) condition, 2 If statement ended (Then or a one-statement body comes next), 3 inside that one-statement body |
| `unres` (39-41) | labels referenced by Goto/Menu( and not defined anywhere so far: 0, 1, 2+ |

| Bag bits (rows) | Meaning |
|---|---|
| 0-36 (42-78) | name hint: next characters (A-Z, 0-9, θ in that order) that extend the name being read toward a candidate. Candidates are unresolved labels after `Lbl`, defined labels after `Goto`/in `Menu(`, assigned custom lists after `ʟ`. Only in ctx 4-6, so these rows can be tied to the letter token embeddings |
| 37 (79) | END: the name read so far is itself a candidate |
| 38-47 (80-89) | `Str` token AA00+k is assigned (AA00 = Str1 ... AA09 = Str0) |
| 48-57 (90-99) | matrix token 5C00+k (`[A]`+k) is assigned |
| 58-63 (100-105) | a custom list whose first character has (name index + 1) mod 6 = b is assigned |

C struct: `trk_feat { uint8 cat[9]; uint8 hint[5]; uint8 var[4]; }` (18 B). `bind` packs
`bag = hint (bits 0-37) | var << 38`.

## Semantics

Statements and lexing:
- A statement starts after newline, `:`, `<bos>`, or right after a `DelVar` operand (`DelVar
  ADisp 1` and `DelVar ʟABDisp 1` chain).
- Inside a string everything is literal (`:`, `Lbl`, `(`). A string ends at `"`, at `→` (which
  is then a store), or at a newline.
- `→` closes every open paren, brace and bracket. Line ends reset all of them.
  Every token whose name ends in `(` opens a paren (161 tokens).

Blocks (static nesting):
- `For(`, `While`, `Repeat` push when they start a statement. `Then` pushes If-Then;
  `Else` turns the top If-Then into Else. `End` pops.
- Errors (in `n_bad`, a summary counter): `Else` without If-Then, `Then` not right after an If
  statement. `End` with nothing open is counted separately (`n_extra_end`).
- Unclosed blocks at the end are reported, not counted as errors: TI-OS ends such programs
  without error.

Single-line If:
- `If cond` followed by `Then` is a block. Otherwise the next statement is the body; `If A:If
  B:Disp 1` nests.
- An empty statement counts as that body: `If 0::1→A` runs `1→A`, and in `If 1:` + newline +
  `Then`, the `Then` is stray.
- `IS>(`/`DS<(` skip the next statement like `If`.
- A single-line If with nothing after it is ERR:SYNTAX, whether the condition is true or false.

Labels:
- A name is 1-2 characters from A-Z, 0-9, θ. It ends at `:`, newline or end of program, and
  inside `Menu(` also at `,` or `)`.
- Anything else makes the name malformed (`Lbl ABC`, `Lbl A ` with a space, empty name): it is
  counted in `n_bad` and defines or references nothing.
- `Lbl` and `Goto` count only at statement start; inside a statement they are bad.
  `Menu(` label arguments are arguments 2, 4, 6, ... at the `Menu(` paren depth.
- A reference is unresolved while no `Lbl` of that name exists earlier. A later `Lbl` resolves
  it: Goto scans the whole program.

Variables that can be undefined:
- Custom lists are assigned by `→ʟX`, `→ʟX(n)` (element store creates the list), `→dim(ʟX`,
  `→NAME` with 2-5 characters and no `ʟ` (stores a list), and `→A` when the statement starts
  with `{`. Also by `SetUpEditor X,ʟY`, `Input`/`Prompt`, and `Matr►list(` arguments.
- `Str0-9` are assigned by `→`, `Input` (unless the Str is the prompt of `Input Str1,A`),
  `Prompt` and `Equ►String(`.
- `[A]-[J]` are assigned by `→[A]` (but `→[A](i,j)` needs `[A]`), `→dim([A]`, `Input`/`Prompt`
  and `List►matr(`.
- `Pic`/`GDB` are assigned by `StorePic n`/`StoreGDB n` (digit or Pic token).
- `DelVar` unassigns any of these.
- Every other use is a read: `ClrList`, `SortA(`, `Fill(`, `dim(`, `Archive`/`UnArchive`,
  `RecallPic`, and plain uses. A read of an unassigned variable counts as read-before-assign
  (`n_undef`).
- Statistics variables (62xx tokens except sequence 𝑛) are undefined until any statistics
  command (`1-Var Stats`, regressions, tests) has run.
- `L1-L6`, `A-Z`, `θ`, `Ans` always exist.
- `prgm` calls are counted (`n_prgm`; a standalone run has no subprograms). A call statement
  needs a 1-8 character name followed by a separator, otherwise ERR:SYNTAX.

Limits (mirrored exactly by the spec):
- Block types are kept to depth 32; the depth itself counts to 65535.
- 16 custom list names are kept. After that, unknown lists are not flagged.
- Paren, brace and bracket counters saturate at 255; positions and counters at 65535.

State: 510 B (host and eZ80). It holds two 1406-bit label sets, first-character sets, 16×5 B
list names, packed block types and counters. There is no allocation and no writable global.
The work per token is bounded and never depends on program length; `<bos>` clears 510 B.

### Evidence from the emulator (`uv run python -m tracker.probe_os`)

| Probe | TI-OS 5.8 |
|---|---|
| read `ʟAB`, `Str1`, `[A]`, `Pic1`, `GDB1`, stat var `r` (any 62xx but 6221) unassigned | ERR:UNDEFINED |
| `5→ʟAB(1)`, `3→dim(ʟAB)`, `SetUpEditor AB`, `{1,2}→AB`, `{1,2}→A` | create the list |
| `ClrList ʟAB`, `dim(ʟAB)`, `SortA(`, `Fill(`, `UnArchive`/`Archive` of a missing var | ERR:UNDEFINED |
| `5→[A](1,1)` with no `[A]` | ERR:UNDEFINED (`→ʟAB(1)` creates) |
| `DelVar ADisp A`, `DelVar Str1Disp 5`, `DelVar ʟABDisp 5` | chain without `:` |
| `Lbl ABC`, `Lbl A ` (space), `Lbl `, `Goto ABC`, `Goto `, `Menu(... ,A )` | ERR:SYNTAX |
| `Lbl` inside a string, `Lbl` in mid-statement | not found: ERR:LABEL |
| `Lbl A` after `:` or in a single-line If body | found |
| `If 0::1→A` / `If 0`,newline,newline,`1→A` | `1→A` runs (empty statement is the body) |
| `If 1:` newline `Then` | ERR:SYNTAX |
| unclosed If-Then / While / For( / Repeat at the end (true or false) | OK |
| `End` with nothing open, `Else`/`Then` without If | ERR:SYNTAX |
| `If 1` or `If 0` (or `IS>(`) as the last statement | ERR:SYNTAX |
| `Menu(` with a missing label not chosen / chosen | OK / ERR:LABEL |
| `prgmABC Disp`, 9-character name, `prgmAB+1`, empty name, lowercase | ERR:SYNTAX (`1+prgmAB`: DATA TYPE) |
| empty `Y1` | ERR:INVALID (not tracked) |

## Acceptance results

| | Result |
|---|---|
| T1 | 510 B state, no allocation, no globals or library calls (`nm` on host and eZ80 objects), warning-free with `gcc -std=c99 -pedantic -Wall -Wextra -Wconversion -Wshadow -Werror` and with the CE flags (`-Wall -Wextra -Werror`, `-Oz` and `-O2`) |
| T2 | C == spec on every token (features) and every program (summaries): 41,389 train programs (55.9M ids), 666 val, 20,000 fuzzed programs |
| T3 | bit-exact on the eZ80 (all 666 val programs, 1.03M tokens). Cycles/token (CEmu, 48 MHz, code in RAM): `-Oz` mean 6.2K (0.014% of v1's 44.2M), p99 12K, max 41.6K (0.094%, a stress program); `-O2` mean 4.5K, max 38K (0.086%). Code 7.4 KB + 1.0 KB rodata at `-Oz` |
| T4 | below |

Host speed: `bind.batch` does all of train.bin (55.9M ids) in about 0.5 s on one core.

2026-10-14 (v2 ship): the list-name table is walked one row at a time. Before, four places
indexed or walked `list_name[0]` across the rows of `list_name[16][5]`, which is undefined
behaviour (UBSan flagged the store). Behaviour is unchanged:
- T2 is identical on every train and val program and on 20,000 fuzzed ones.
- T3 is bit-exact on all 666 val programs, with the same cycles (mean 6,183, p99 12,110, max
  41,566).
- `data/feats/tracker`'s arrays are byte-identical when regenerated; only feats.json's
  `identity["tracker.c"]` changed.

The calculator core's host build now checks the tracker with UBSan's bounds checks too.

### T4: tracker vs CEmu outcomes (`uv run python -m tracker.validate`)

| | val, cut to 256 (666) | val, whole (666) | v1 generated, cut (960) |
|---|---:|---:|---:|
| ERR:LABEL with an unresolved label (required: all) | **82/82** | **2/2** | **137/137** |
| unresolved label → ERR:LABEL | 82/258 | 2/30 | 137/331 |
| ERR:UNDEFINED with a tracker read-before-assign or prgm call | 164/164 | 216/221 | 142/143 |
| ... on the error line (read 75.6/63.3/58.0%, prgm 22.6/30.8/41.3%) | 161/164 | 208/221 | 142/143 |
| read-before-assign → ERR:UNDEFINED | 136/164 (83%) | 168/217 (77%) | 90/103 (87%) |
| ERR:SYNTAX with a structure error on the error line | 12/21 | 2/13 | 40/63 |
| structure error → ERR:SYNTAX | 14/52 | 6/62 | 44/94 |
| unclosed blocks at the end; ERR rate with vs without | 152; 52% vs 48% | 61; 51% vs 46% | 175; 45% vs 40% |

Reading the table:
- The one-directional requirement holds in every set.
- Unresolved labels and read-before-assign are often not fatal, because the Goto or the read
  is not reached, or another error comes first.
- Unclosed blocks do not predict errors, as the probes say.
- Some CEmu errors depend on which path a program takes at run time, which the tracker's
  static state does not follow; the two classes found are described below.

**Run-time-only failures.** The tracker passes these and CEmu stops on them. Two classes
account for the cases found:

- a list that is assigned only inside a branch the program does not take, then read later
  (a made-up shape: `If 0` guarding the store to the list, and a later `Disp` of an element);
- an undefined variable named inside an `expr(` string, which the tracker does not parse
  (a made-up shape: `expr("ʟTMP(1)")` in a program that never creates `ʟTMP`).

`tracker/relocate.py` re-runs the ERR programs with a locator that waits until the cursor
stops (outputs in `eval/runs/w2-*-loc/`). Outcomes were identical, and positions moved in 80 of
309 (val whole), 16 of 326 (val cut) and 11 of 396 (generated) ERR programs. T4 uses those
positions. Results quoting `error_line` (EVAL.md) are affected the same way.

## Known gaps and uncertain semantics

- Static, not path-sensitive. Goto, branches and subprogram idioms (`End` used to close a
  caller's loop) make `n_extra_end`, `depth` and read-before-assign approximate.
- Input/Prompt are assumed to receive a value of the right type. `expr(`, string contents, `Y=`
  variables (empty `Y1` is ERR:INVALID), archived variables and other programs' variables are
  not modelled. Any statistics command counts as defining every statistics variable.
- `→A` with no `ʟ` counts as a list store only when the statement starts with `{`. A list value
  that starts otherwise (`seq(...)→A`, `L1→A`) is missed.
- Lists past the 16th distinct name are "unknown" (never flagged). A `prgm` name's first
  character is not checked. 3-character names and 6-character list names count as one
  structure error each.
- Probed only on OS 5.8.0.0022 in a clean emulator state. Other OS versions may differ, for
  example in label lookup or DelVar chaining.
- The feature vocabulary is 106 rows (about 20 KB at d=192 int8). The hint rows (38) can be
  tied to the name-token embeddings, and the 6 list-bucket rows dropped, if W3/W5 need to fit
  M1's flash budget.

## Reproduce

```sh
uv run python -m tracker.gen_tables            # trk_tables.h from corpus/vocab.json
uv run pytest tracker -q                       # 161 tests, ~10 s (CE/CEmu tests skip if absent)
uv run python -m tracker.difftest              # T2 on fuzz + all val + all train (~15 s, 24 procs)
                                               # needs the corpus (not published) after the fuzz step
uv run python -m tracker.bind data/tok/train.bin [--out data/tok/train.trk]
                                               # needs the corpus (not published)
uv run python -m tracker.ce_bench --exact 666 --steps 12 [--opt=-O2]   # T3
                                               # needs the corpus (not published), the emulator
                                               # and the CE C toolchain
uv run python -m tracker.probe_os              # TI-OS evidence -> eval/runs/w2-probe/
                                               # needs the emulator
uv run python -m eval.run_programs --ids data/tok/val.bin --vocab corpus/vocab.json \
    --name w2-val-full --workers 1 --shots 0   # whole val programs (~5 min)
                                               # needs the corpus (not published) and the emulator
for s in val-trim val-full gen-v1; do uv run python -m tracker.relocate $s; done
                                               # needs the corpus (not published) and the emulator
uv run python -m tracker.validate              # T4 -> eval/runs/w2-t4/report.md
                                               # needs the emulator (the runs above) and the corpus
```
