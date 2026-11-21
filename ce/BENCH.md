# M2: ternary LUT matvec on the eZ80, cycle-exact in CEmu

Kernel for the SPEC.md contract (5 ternary weights per byte, per-group 243-entry partial-sum
tables), written in eZ80 assembly (ADL mode), plus a C reference, both run and timed inside
headless CEmu on the user's own ROM (OS 5.8.0.0022, boot 5.6.1.0006). Output is bit-identical to
`model/pack.py:lut_matvec` in every test (100 tests).

**Headline** (weights in archive flash, cold cache, LCD DMA on in steady state, 48 MHz):

| | rows loop | table build | one SPEC v0 token (43 matvecs, measured back to back) |
|---|---|---|---|
| kernel code in RAM | 10.6 to 11.0 cycles/weight | 7.5 K cycles/group (31 cycles/entry) | 37.2 M cycles = **776 ms** (1.29 tok/s) |
| kernel code in flash (position-independent, archived AppVar) | 7.3 to 7.5 cycles/weight | 4.8 K cycles/group (20 cycles/entry) | 24.7 M cycles = **514 ms** (1.95 tok/s) |
| C reference (ez80-clang -O2), code in RAM | 65 to 68 cycles/weight | 78 K cycles/group | about 6x slower |

## Harness (emulator instrumentation only)

The harness is a debugger-style driver for the emulated CPU. Apart from booting the ROM to the
home screen, it runs nothing through TI-OS, launches no programs, and touches no jailbreak or
exploit. The CEmu core is compiled from the CEmu harness checkout's `vendor/CEmu/core`
(read-only) into our own library.

Where things are: `CALCLM_CEMU_ROOT` names the harness checkout (its `tools/cemu`, its
`vendor/CEmu/core` and the ROM at `rom/ti84pce.rom`), `CALCLM_ROM` another ROM file,
`CEMU_CORE` another core folder, and `CEDEV` the CE C toolchain (default `~/CEdev`, the
toolchain's usual install folder). Without a harness or a ROM, the emulator tests skip.

1. `ce/tools/shim.c` is a copy of the CEmu harness's `tools/cemu/shim.c` with added functions:
   `shim_write_flash` (memcpy into the emulator's private flash array), `shim_write_ram` /
   `shim_read_ram`, `shim_hw_info` (ASIC revision, flash type, CPU clock, privilege and stack-limit
   registers), `shim_get_regs`, `shim_set_lcd_dma` (CEmu's optional accurate LCD DMA), and
   `shim_call(pc, sp, flush_cache, max_cycles)`. `shim_call` sets PC/SPL, ADL=1, interrupts
   off, optionally flushes the serial-flash cache, then runs 1 ms slices until the CPU executes
   HALT with interrupts disabled. It returns exact counters: active cycles (total minus halted),
   DMA stall cycles, flash cache accesses and misses.
2. `ce/tools/build.sh` builds `ce/build/libcebench.so` (objects in `ce/build/cemu-obj`).
3. `ce/tools/bench.py` (`Bench`): the first use boots the ROM with the CEmu harness's own `Emu`
   logic. That module is imported read-only, pointed at our library and at `ce/build/cemu-state`,
   and it writes no bytecode there. It shows RAM Cleared, presses Clear, and waits until idle, so
   the OS has set 48 MHz and the flash controller exactly as on the unit. That state is saved.
   Each benchmark reloads it (restoring RAM, flash and peripherals). With LCD DMA on, it then
   lets the idle OS run 20 ms, because CEmu arms LCD DMA only at the next frame start. It writes
   code, data and weights, and runs a stub: `di; ld hl,arg; push hl ...; call fn; ld (ret),hl; halt`.
   `Bench.call` subtracts the stub's own cost, measured with a function that is a bare `ret`.
4. `ce/tools/tmv.py` loads the kernels, places x / weights / tables, runs build and rows as two
   separate calls (so table-build cost is measured on its own), and reads y back.

Memory map: kernel 0xD1B000 (RAM) or 0x380000 (flash), x 0xD1F000, y 0xD1F400, tables bank
0xD3, weights 0x300000 (flash, checked erased in the booted state) or 0xD40000 (RAM). Nothing
leaves the box: the ROM is read where it is, and states plus build products go to
`ce/build/` (git-ignored via `ce/.gitignore`).

## Build and run

```sh
ce/tools/build.sh                         # CEmu core + shim -> ce/build/libcebench.so (auto on first use)
ce/kernel/build.sh asm 39 0xD1B000        # asm kernel for G=39 groups, code in RAM at 0xD1B000
ce/kernel/build.sh asm 103 0x380000 0xD1AA00 pop 0 1   # position-independent, for flash; vars in RAM
ce/kernel/build.sh c 0xD1E000             # C reference
uv run pytest ce/tests                    # 100 tests, a few seconds
uv run python -m ce.tools.run_bench       # all tables below; raw numbers in ce/build/bench.json
```

`ce/kernel/build.sh` uses the CEdev makefile's own steps: `gen_tmv.py` (or `ez80-clang -S -O2`)
-> `z80-none-elf-as -march=ez80+full` -> `z80-none-elf-ld -T ce/kernel/link.ld --defsym ORG=...`
(flat script, libcrt for the C file) -> `z80-none-elf-objcopy -O binary`, plus `nm` for entry
points. `tmv.py` calls it on demand and caches outputs in `ce/build/kernel/`.

## Kernel design (`ce/kernel/gen_tmv.py`)

C-callable (CE ABI, IX/IY preserved; interrupts saved, disabled and restored internally):

```c
void tmv_build(const int8_t *x, uint8_t *tbl);                          // x padded to 5G
void tmv_rows(const uint8_t *W, int24_t *y, unsigned n_out, const uint8_t *tbl);
```

Code is generated per group count G (fully unrolled). The model needs G=39 (n_in 192, 2.5 KB)
and G=103 (n_in 512, 4.0 KB).

* **Table layout ("b-major").** One 64 KB bank U. The entry for group g and packed byte b is
  stored at `U:b:(C0+2g)` (lo) and `+1` (hi). Its value is V = T_g[b] + 635, so 0..1270, a
  non-negative 16-bit number. The weight byte *is* the page number, so the lookup is
  `ld h,e; ld c,(hl); inc l; ld b,(hl); inc l; add ix,bc`. Rows start the 24-bit
  accumulator at -635*G, so the sum is exact and fits signed 24 bits up to n_in = 635. The
  layout occupies pages 0..242 by 2G+1 columns (G=103: 50 KB inside a 62 KB span; G=39: 19 KB).
* **Row loop.** SP streams the weights: `pop de; dec sp` yields 2 weight bytes for 2 opcode
  bytes. Total cost is 8 opcode bytes + 2 table reads per group. For odd G the body covers 2 rows,
  so pairs can straddle rows (with a 1-row tail when n_out is odd). Row count is A (inner) plus
  B' (outer). Output is int24 little-endian, sign-correct.
* **Table build.** It runs per group, in descending b. Deltas are computed once per group in
  16-bit arithmetic and pushed as -d_k, with d_k = x_k - 2(x_0+..+x_{k-1}). Each entry is then
  `ld (hl),ix; dec h; add ix,<-d_k>`: 5 bytes, about 26 cycles from RAM. The 24-bit store's third
  byte spills into the next group's column, which is written later. -d_0, -d_1, -d_2 live in BC,
  DE, SP; -d_3 and -d_4 in BC', DE' via `exx`.
* **Flash-resident variant (`--pic`).** It contains no absolute jumps into itself. Four long jumps
  go through 4-byte `jp` trampolines in RAM that the loader patches (`tmv.load_kernel`). The
  binary is byte-identical for any link address. Tests run it from 0x380000, 0x2A0000, 0x2A1234
  and 0x0C0011.
* **Limits.** x in [-127, 127] (SPEC; -128 would break the bias), n_in <= 635 (C0 = 0), n_out <=
  65280 (8-bit outer loop count). `tmv_rows` may read up to 2 bytes past the end of W.

## Correctness (`uv run pytest ce/tests`: 100 passed)

Every case is compared bit-exactly with `lut_matvec` (and `lut_matvec` with dense `w @ x`):

* random W, x for 192->192, 192->512, 512->192, 3 seeds each, weights in flash; again with
  weights in RAM; with flash-resident code; relocated flash code
* n_in not a multiple of 5: 1, 2, 4, 6, 7, 13, 191, 193, 509, 511, 513, 634, with n_out 1, 2, 7
  (odd and even G, tail rows), and a subset with flash-resident code
* extremes: all x = +/-127 with all w = +/-1 (|y| = 127 n_in, up to 80,518, beyond int16), mixed
  sign extremes, all-zero weights, zero input, n_out = 0
* more than 256 loop iterations (1027 and 600 rows), shared tables for q/k/v, table column offset
  C0 != 0, the simple `de` variant, the C reference, and interrupts restored to on
* bounds: guard bytes show writes stay inside pages 0..242 x columns C0..C0+2G of the table bank
  and inside y
* timing: a nop sled costs exactly 4 cycles/byte from RAM and 2/byte + 195 per missed line from
  flash; cycle counts do not depend on the data; 192->192 regression band (1%).

A mutation check (one packed byte changed in flash) changes exactly that row by the expected
amount. An independent code review found no critical or high issues. Its medium and low
findings have been fixed: LCD DMA warm-up, the table pointer check, more tests, and doc fixes.

## Measured cycles (CEmu, 48 MHz, cold flash cache, interrupts off)

Weights in archive flash. `rows` is the row loop; `build` is the table build for the input vector.
LCD DMA on = CEmu's accurate LCD DMA, in steady state, so the home screen's 16bpp refresh steals
RAM cycles as on hardware.

| shape | code | LCD DMA | build cycles | rows cycles | rows c/weight | rows c/packed byte | total c/weight | ms @ 48 MHz |
|---|---|---|---|---|---|---|---|---|
| 192->192 | RAM | off | 273,913 | 379,570 | 10.30 | 50.7 | 17.73 | 13.61 |
| 192->512 | RAM | off | 273,913 | 1,010,981 | 10.28 | 50.6 | 13.07 | 26.77 |
| 512->192 | RAM | off | 723,001 | 982,834 | 10.00 | 49.7 | 17.35 | 35.54 |
| 192->192 | RAM | on | 293,598 | 403,734 | 10.95 | 53.9 | 18.92 | 14.53 |
| 192->512 | RAM | on | 293,597 | 1,075,052 | 10.94 | 53.8 | 13.92 | 28.51 |
| 512->192 | RAM | on | 777,443 | 1,043,847 | 10.62 | 52.8 | 18.53 | 37.94 |
| 192->192 | flash (PIC) | off | 179,153 | 269,887 | 7.32 | 36.0 | 12.18 | 9.36 |
| 192->512 | flash (PIC) | off | 179,153 | 710,418 | 7.23 | 35.6 | 9.05 | 18.53 |
| 512->192 | flash (PIC) | off | 458,897 | 696,575 | 7.09 | 35.2 | 11.75 | 24.07 |
| 192->192 | flash (PIC) | on | 186,848 | 276,220 | 7.49 | 36.9 | 12.56 | 9.65 |
| 192->512 | flash (PIC) | on | 186,848 | 726,674 | 7.39 | 36.4 | 9.29 | 19.03 |
| 512->192 | flash (PIC) | on | 480,736 | 712,945 | 7.25 | 36.1 | 12.14 | 24.87 |

Table build per group, LCD off: 7,023 cycles from RAM (243 entries x 26 + about 650 setup) and
4,594 from flash. LCD on: 7,528 and 4,791. The 39-group build serves 256 rows for q/k/v, 192 for
o, and 1024 for w1/w3, so its share ranges from 42% (o, w2) down to 12% (w1+w3).

### Weights in flash vs RAM (LCD DMA off)

| shape | code in RAM: W flash / W RAM | code in flash: W flash / W RAM |
|---|---|---|
| 192->192 | 10.30 / 9.66 c/w (RAM 6% faster) | 7.32 / 6.51 c/w (RAM 11% faster) |
| 192->512 | 10.28 / 9.65 | 7.23 / 6.42 |
| 512->192 | 10.00 / 9.37 | 7.09 / 6.29 |

Archived weights cost little. A cache-hit flash byte (2 cycles) is cheaper than a RAM byte (4),
which nearly offsets the 195-cycle miss per 32-byte line (6.1 cycles per weight byte, 1.2 per
weight). The model's 503 KB of packed weights cannot sit in RAM anyway.

### Iterations (192->192, weights in flash, LCD DMA off)

| step | change | rows c/weight | build cycles/group |
|---|---|---|---|
| C reference | `int16 tbl[G][243]`, plain loops, ez80-clang -O2 | 62.92 | 72,161 |
| asm 1 (`variant de`, kept) | b-major tables, unrolled, `ld a,(de); inc de; ld h,a` per group | 11.18 | 7,288 |
| asm 2 | SP streams weights, `pop de; dec sp` per 2 groups | 10.30 | 7,288 |
| asm 3 | descending build, negated deltas pushed (setup 900 -> 650 cycles) | 10.30 | 7,022 |
| asm 4 | interrupt state saved/restored (safe to call from C) | 10.30 | 7,023 |
| asm 5 | same code executing from flash (cache hits 2 cycles vs RAM 4), PIC trampolines | 7.32 | 4,594 |

Gains stopped here. Cost model from CEmu's timing (RAM read 4 / write 2 cycles; every opcode
byte is a memory read): per group, 8 opcode bytes (32) + 2 table reads (8) + 1.5 weight bytes
(3) + amortized miss (6.1) = 49 cycles. Per-row overhead (accumulator init, column reset,
store, loop) adds about 58 cycles counted from the code, so the model predicts 50.6 per packed
byte against 50.7 measured. Rejected after analysis, because each trades a code byte for a
4-cycle data byte or saves under 2%:

* 16-bit accumulation via `ld bc,(ix+d)` / `add hl,bc`
* planar or stride-2 tables built with `push`, which make lookups 2 to 4 bytes longer
* row-pair interleaving
* unrolling 4, 6 or 8 rows per loop iteration (measured: 0.44%, 0.58%, 0.65% faster; would
  need more tail code)
* folding the row bias into group 0
* Gray-code build order

## Hardware revision

CEmu picks the newest ASIC revision whose boot-code range contains the ROM's boot code
(`core/bootver.c`, `core/asic.c:report_reset`). Boot 5.6.1.0006 >= 5.3.6 gives **Rev M**. The
revision alone sets `asic.serFlash` (`revision >= M`, `core/asic.c`), so this means **serial flash
behind a cache** (`core/flash.c`). The certificate marks the ROM as a Python edition, which only
Rev M boot code allows. The cache is 128 sets x 2 ways x 32-byte lines (8 KB), LRU. A read costs
2 cycles in the same line as the previous flash access, 3 in another cached line, and 197 on a
miss (the source comments "195-201, typically 196-197"). RAM reads cost 4 cycles and writes 2
(`core/mem.c`). The nop-sled test confirms the harness sees exactly these costs. This matches the
WikiTI 84PCE:Wait_States facts in FINDINGS.md. The booted OS runs at 48 MHz (`cpu_speed` 3).

For the author's TI-84 Plus CE Python edition (same OS and boot code) this is the hardware
generation being modeled. On Rev M+ hardware:

1. archived weights stream at about 8 cycles per byte, so no RAM copy is needed;
2. code executing from flash is faster than code in RAM while it stays in the 8 KB cache. That
   is the roughly 30% gain above (kernels are 2.5 KB and 4.0 KB, and streaming weights never
   evict the hot code lines: each set is 2-way and the code is touched every row).

Older parallel-flash units (Rev A to I, boot <= 5.3.5) charge the programmed wait states on every
flash byte, with no cache (CEmu: port 0x1005 + 6, 10 at reset). There the kernel belongs in RAM.

Caveats: these are CEmu's timing models, not a measurement on the device. The miss penalty
varies 195 to 201 on hardware. LCD DMA in 16bpp mode costs +7% (RAM code) or +3% (flash code)
over a token. An 8bpp display should roughly halve that (not measured), and an LCD switched off
removes it.

## Per-token projection for SPEC v0

Each token runs, per layer: one table build over x for q (192->192), k (192->32) and v
(192->32); one build for o (192->192); one for w1 and w3 (192->512 each); one 103-group build
for w2 (512->192). The head (192->1024) needs one more build. `run_bench.token_sim` runs all 43
matvecs back to back with no state reset. Weights for all 43 matrices (2,482,176 ternary
weights, 503,040 bytes) sit at distinct flash addresses; both kernels are resident; the cache and
the LCD frame phase carry over; every output is checked.

| code | LCD DMA | table builds | row loops | total cycles | ms/token | tokens/s | cycles/weight |
|---|---|---|---|---|---|---|---|
| RAM | off | 9,542,353 | 25,363,071 | 34,905,424 | 727 | 1.38 | 14.06 |
| RAM | on | 10,262,995 | 26,963,127 | 37,226,122 | 776 | 1.29 | 15.00 |
| flash (PIC) | off | 6,147,201 | 17,803,673 | 23,950,874 | 499 | 2.00 | 9.65 |
| flash (PIC) | on | 6,438,923 | 18,215,169 | 24,654,092 | 514 | 1.95 | 9.93 |

The same total built from the per-shape measurements (`projection` in `run_bench.py`) agrees
within 0.6%. Per-layer rows (flash code, LCD off): q 269,887; k 49,622; v 49,622; o 269,887; w1
710,418; w3 710,418; w2 696,575. Head 1,415,280. Builds: 179,153 (G=39) and 458,897 (G=103).
Table builds are 26 to 28% of matvec time. Attention over the KV cache, norms and requantization
are not included.

## Open problems / notes for M4

* **Table build is 26 to 28% of the matvec time** and is at the floor of this layout (store, page
  step, add per entry). It is inherent to LUT methods for the 192-row matrices (243 entries per
  group vs n_out lookups).
* **Not validated on the device.** All numbers come from CEmu's timing model. A one-off
  hardware timing run (the program launched by the owner) would confirm the flash-cache
  advantage.
* **Flash-resident kernel on the real unit.** Ship the PIC binary inside an archived AppVar.
  At each launch, find its address and write the four `jp` trampolines (16 bytes RAM, see
  `tmv.load_kernel`). Garbage collection can move the AppVar between runs. The RAM-code kernel
  needs nothing.
* **RAM.** Tables need one 64 KB-aligned bank (pages 0..242; columns C0..C0+2G), and the tbl
  argument must be exactly `bank | C0` (`tmv.table_ptr`). A natural home is 0xD40000..0xD4F2FF
  (first half of VRAM) with the LCD in 8bpp showing the second half. The KV cache (96 KB) and
  activations go in user RAM.
* **Interrupts** are off during each call, at most 45 ms (the head with code in RAM; 30 ms from
  flash), so ON is not seen mid-call.
* **Attention** int8 MACs over a 256-position cache (about 0.6 M MACs/token over 6 layers) are
  unmeasured and could rival the matvecs at long context.
* SPEC unchanged. The table layout is the kernel's choice per SPEC.
