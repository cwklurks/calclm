# CalcLM integer reference (INTREF v2.0)

`model/intref.py` (with `model/intops.py`, `model/pack.py` and, for the logit masks,
`model/intmask.py`) is the normative definition of CalcLM inference. This document restates it for the calculator implementation (M4). Where the
two disagree, the Python is right and this file has a bug. M4 must reproduce `IntModel.step`
logits bit for bit, and `generate` token for token for a given seed and sampler settings.

Everything is integer: Python ints and numpy int64 in the reference, no floats at inference
time. Floats exist only in the exporter (`model/export.py`), which bakes every constant and table.

## 1. Primitives

| Name | Definition | Notes |
|---|---|---|
| `rsr(a, s)` | `s > 0`: `(a + 2^(s-1)) >> s` (arithmetic shift, i.e. floor); `s == 0`: `a`; `s < 0`: `a << -s` | Round-half-up shift. The only rounding mode used anywhere. |
| `clamp(a, lo, hi)` | saturate | `INT24 = [-2^23, 2^23-1]`, `INT16 = [-2^15, 2^15-1]`, int8 = `[-127, 127]` |
| `bitlen(n)` | bits of `n >= 0` (`bitlen(0) = 0`) | count leading zeros |
| `isqrt(n)` | `floor(sqrt(n))` | bit-by-bit method, no multiplies; `n < 2^31` here |
| `udiv(a, b)` | `floor(a / b)`, `0 <= a < 2^32`, `0 < b` | software long division; only per vector, never per element |
| `xorshift32(x)` | `x ^= x << 13; x ^= x >> 17; x ^= x << 5` (mod 2^32) | from state 1: 270369, 67634689, 2647435461 |
| `normalize16(v)` | `c = bitlen(max|v_i|) - 15; v' = clamp(rsr(v, c), -32767, 32767)` -> `(v', c)` | `v ~ v' * 2^c`; the clamp matters: `rsr` can round the peak up to exactly +32768, which int16 cannot hold |

Pseudo-floats (scalars only, one per vector or per layer):

- **PF15** `(m, e)`, `2^14 <= m < 2^15`, value `m * 2^e`.
  `pf15(n, e)`: `sh = bitlen(n) - 15; m = rsr(n, sh); if m == 2^15: m = 2^14, sh += 1` -> `(m, e + sh)`.
  `pf_mul(a, b) = pf15(a.m * b.m, a.e + b.e)`.
  `pf_div(a, b) = pf15(udiv(a.m << 16, b.m), a.e - b.e - 16)`.
- **PF8** `(m, e)`, `128 <= m < 256` (or `m = 0` for an all-zero vector): every activation scale.
  `pf8(a)`: `m = rsr(a.m, 7), e = a.e + 7; if m == 256: (128, e + 1)`. `pf15_of_pf8(s) = (s.m << 7, s.e - 7)`.
- **PF7** per weight row: mantissa `bm` in `[64, 127]`, exponent `be` (int8), value `bm * 2^be`.
- Constants: `INV127 = pf_div((2^14, -14), pf15(127))` (computed, fits 32 bits);
  `SQRTD = (sqrtd_m, sqrtd_e)` read from the header (sqrt of the vector length `d` as PF15;
  the host computes it as `pf15(isqrt(d << 28), -14)`, which needs a 36-bit isqrt).
  For d = 192: `INV127 = (16513, -21)`, `SQRTD = (28378, -11)`.

Width rule: every per-element product is below 2^31 in magnitude (signed 32-bit), and every
scalar step stays below 2^32 unsigned. `intops.check_bits` asserts this at runtime. Two notes
for a 32-bit port: the rounding add in `rsr(P, s)` can need one more bit than `P` when `s >= 27`
(do it on the unsigned magnitude or in 33+ bits; observed shifts are 4 to 21), and values
written "16-bit" below are int16 after `normalize16` (never +32768).

## 2. Number formats in the forward pass

| Quantity | Format |
|---|---|
| residual stream `r` | int24, fixed point with `F` fractional bits (`F` in the header, chosen by the exporter with 4x headroom over a calibration peak); saturating adds |
| linear-layer inputs `x8` | int8 in [-127, 127] plus one PF8 scale `sx` per vector |
| ternary weights | packed 5 per byte (SPEC.md), per-row PF7 scale |
| q, k, v | int8 on static per-layer grids; the grid step is folded into the q/k/v row scales by the exporter, so `scale_out(..., f_out=0)` lands directly on it |
| KV cache | int8, `n_layers x ctx x head_dim` each for K and V (MQA: one KV head) = 96 KB for v0 |
| softmax weights | uint8 from a per-layer exp table, max 255 per head and query |
| FFN pre-activation `y1`, SiLU output | int16, Q8 (1/256) |
| logits | int24, Q8 (1/256 nat) |

## 3. Vector operations

**`quant_vec(v, unit)`**: int8 absmax quantization of the real vector `v * unit` (`unit` PF15).
```
M = max|v_i|;  if M == 0: return zeros, (0, 0)
v', c = normalize16(v)                                # |v'| <= 32767
sx = pf8(pf_mul(pf_mul(pf15(M), unit), INV127))        # PF8 of max|v*unit| / 127
G  = pf_div(unit, pf15_of_pf8(sx))
x8_i = clamp(rsr(v'_i * G.m, -(G.e + c)), -127, 127)   # int16 x 15-bit products
```

**`norm_quant(r, g16, eg)`**: RMSNorm with gain `g16 * 2^-eg` fused with `quant_vec`. RMSNorm
followed by absmax quantization does not depend on the residual's scale, so `F` does not appear.
```
M = max|r_i|;  if M == 0: return zeros, (0, 0)
r', _ = normalize16(r)                                # |r'| <= 32767
r8 = rsr(r', 8);  SS = sum r8_i^2                      # |r8| <= 128: 8x8 unsigned squares, SS < 2^22
b  = (30 - bitlen(SS)) >> 1;  s = isqrt(SS << 2b)      # s in [2^14, 2^15) ~ sqrt(SS) * 2^b
z_i = r'_i * g16_i                                     # |z| < 2^30
unit = pf_div(SQRTD, pf15(s));  unit.e += b - eg - 8
return quant_vec(z, unit)
```

**`scale_out(acc, bm, be, sx, f_out, lo, hi)`**: dequantize a ternary matvec result.
```
if sx.m == 0: return zeros
P_i = acc_i * bm_i * sx.m                              # |acc| < 2^16, bm < 2^7, sx.m < 2^8: |P| < 2^31
y_i = clamp(rsr(P_i, -(be_i + sx.e + f_out)), lo, hi)  # per-row shift; a negative shift is an exact
                                                       # left shift, saturated (compare before shifting)
```

**Ternary matvec**: `acc = lut_matvec(W_packed, x8)` exactly as SPEC.md and `model/pack.py`
(int8 inputs zero-padded to a multiple of 5; |acc| <= 127 * n_in < 2^16 for n_in <= 512).

## 4. One token: `step(tok, pos)`

```
# embedding (8x8 products; rows of int8 with PF8 row scales)
r_i = clamp(rsr(E8[tok][i] * Em[tok], -(Ee[tok] + F)) + rsr(P8[pos][i] * Pm[pos], -(Pe[pos] + F)), INT24)

for each layer l:
    # attention
    x8, sx = norm_quant(r, attn_g16, attn_eg)
    q8 = scale_out(lut_matvec(Wq, x8), bq, eq, sx, 0, -127, 127)   # n_heads * head_dim
    k8 = scale_out(lut_matvec(Wk, x8), bk, ek, sx, 0, -127, 127)   # head_dim
    v8 = scale_out(lut_matvec(Wv, x8), bv, ev, sx, 0, -127, 127)   # head_dim
    K[l][pos] = k8;  V[l][pos] = v8
    for each head h (q_h = q8[h*hd : (h+1)*hd]):
        s_j   = sum_d q_h[d] * K[l][j][d]            for j = 0..pos      # 8x8 signed
        idx_j = min(rsr(max_j s_j - s_j, sh_l), L_l - 1)                 # sh_l >= 0 from the header
        e_j   = LUT_l[idx_j]                                             # uint8, e = 255 at the max
        Z     = sum_j e_j                                                # 255 <= Z <= 255 * (pos+1)
        a_d   = sum_j e_j * V[l][j][d]                                   # 8x8 unsigned x signed; |a_d| <= 127 Z < 2^23 needs ctx <= 256
        zb = bitlen(Z);  RZ = udiv(2^(zb+14) + (Z >> 1), Z)             # one division per head; uint16 in (2^14, 2^15]
        o[h*hd + d] = rsr(rsr(a_d, zb - 8) * RZ, 14)                     # a_d / Z in Q8, |o| <= 32512
    o8, so = quant_vec(o, (sv_m, sv_e - 8))                              # SV is the static V grid step
    r = clamp(r + scale_out(lut_matvec(Wo, o8), bo, eo, so, F, INT24), INT24)

    # feed-forward (SwiGLU)
    x8, sx = norm_quant(r, ffn_g16, ffn_eg)
    y1 = scale_out(lut_matvec(W1, x8), b1, e1, sx, 8, INT16)            # Q8
    silu_i = 0                                   if y1_i < -2048
             y1_i                                if y1_i >= 2048
             SILU[rsr(y1_i, 2) + 512]            otherwise              # 1025-entry int16 table, step 1/64
    E3 = max_i e3_i                                                      # static per matrix
    t3_i = rsr(lut_matvec(W3, x8)_i * b3_i, E3 - e3_i)                   # |t3| < 2^23, common exponent E3
    t3', c3 = normalize16(t3)                                             # int16
    h_i = silu_i * t3'_i                                                  # |h| < 2^30
    h8, sh = quant_vec(h, (sx.m << 7, sx.e - 7 + E3 + c3 - 8))
    r = clamp(r + scale_out(lut_matvec(W2, h8), b2, e2, sh, F, INT24), INT24)

x8, sx = norm_quant(r, norm_g16, norm_eg)
logits = scale_out(lut_matvec(Wout, x8), bout, eout, sx, 8, INT24)        # Q8 nats
```

## 5. Sampling and generation

- Never emitted: `<pad>` (0), `<bos>` (1), `<unk>` (3). `<eos>` (2) ends the program.
- **Ranking**: allowed ids sorted by logit descending, ties by id ascending.
- **Greedy**: the first ranked id. The PRNG state is not advanced.
- **Sampled** (temperature `Tq = round(256 T)`, `33 <= Tq <= 65535` so `INV_T` fits uint16; `1 <= k <= 64`):
  ```
  cand = first k ranked ids;  top = logits[cand[0]]
  INV_T = udiv(32 << 16, Tq)                                   # once per generation
  w_j = EXPW[min(rsr(min(top - logits[cand_j], 2^15 - 1) * INV_T, 16), N - 1)]
  state = xorshift32(state);  target = state mod sum_j w_j     # one division per token
  return the first cand_j whose running sum of w exceeds target
  ```
  `EXPW[i] = round(65535 * exp(-i / 32))` (uint16, ends at its first 0; N = 379), i.e. one table
  step is 1/32 nat after dividing by T.
- **generate(prompt, max_new, seed, cap, mask)**: `0 <= seed < 2^32`; `1 + len(prompt) < cap`.
  `cap` counts all tokens, `<bos>` included: by default `ctx` for learned positions (which
  cannot pass it) and 1024 for relative positions (section 12). `state = seed` (0 is replaced by
  2463534242); feed `<bos>`, then the prompt tokens, at positions 0, 1, ...; then repeat:
  choose, append, stop on `<eos>` or when `cap` tokens are written, else feed the new token at
  the next position. With logit masks (section 13), the tracker reads every token fed (`<bos>`
  included, the final `<eos>` never), and before every draw the masked ids leave the ranking
  like `<pad>`/`<bos>`/`<unk>`: the candidates are the first `k` allowed ids (fewer if fewer
  are allowed), and greedy takes the best allowed id.

## 6. Export blob

`model.bin` is one little-endian blob, tensors back to back without padding, in this order;
`model.json` lists offset, dtype, shape and row size of each, plus the AppVar plan and sha256.

1. `header` int32: `magic` ("CLM0"), `version`, `dim`, `n_layers`, `n_heads`, `head_dim`,
   `hidden`, `vocab`, `ctx`, `F`, `silu_len`, `expw_len`, `norm_eg`, `lut_max`, `sqrtd_m`,
   `sqrtd_e`, then per layer
   `attn_eg`, `ffn_eg`, `attn_sh`, `lut_len`, `sv_m`, `sv_e`. `version` is the format:
   1 = learned positions (v1, as described here); 2 = relative positions (section 12): one
   more field after the per-layer ones, `pos_scheme` (1 ALiBi, 2 NoPE; `model.json`
   "tail_fields"), and no `pos_emb` tensors in step 4. Every other offset rule is the same.
2. Per layer `l`: `layers.l.attn_norm.g` int16[d]; `layers.l.attn.lut` uint8[lut_len];
   for each of `wq wk wv wo`: `.w` uint8[rows][ceil(n_in/5)] packed ternary, `.m` uint8[rows]
   (PF7 mantissa), `.e` int8[rows]; `layers.l.ffn_norm.g` int16[d]; `w1 w3 w2` likewise.
3. `norm.g` int16[d]; `output.w/.m/.e`.
4. `tok_emb` int8[vocab][d], `tok_emb.m` uint8 (PF8), `tok_emb.e` int8; same for `pos_emb` [ctx]
   (format 1 only).
5. `silu` int16[1025], `expw` uint16[N].

**AppVars** (`<= 65,505` bytes each). Each AppVar `CLMnn` holds a 7-byte header, `b"CL"`,
its index, the AppVar count, and `blob_start` (u24 little-endian), followed by the contiguous
blob slice `[blob_start, blob_end)` (at most 65,498 bytes). To read blob byte `x`, find the
AppVar with `blob_start <= x < blob_start + payload` and read `header_size + (x - blob_start)`
inside it; tensor offsets come from `model.json` (generate a C header from it). How the
exporter cuts slices (so nobody has to replay it, but for reference): walk tensors in blob
order; a tensor that does not fit in the space left but fits in an empty AppVar starts a new
AppVar; a tensor larger than an AppVar fills the space left, then continues in new AppVars,
always cut between rows (a 1-D tensor's rows are its elements). Only the embeddings are split
in v0. `python -m model.export --appvars` writes the `.8xv` files; `model.json["appvars"]`
lists each AppVar's blob range and (tensor, row range) parts.
Sizes of the final v0 export are in section 10.

## 7. Cost per token (outside the ternary matvecs)

Elementwise multiplies, v0 shape, at position `pos` (verified against an instrumented counter
in `tests/test_intref.py`):

| Kind | Count | Operand widths |
|---|---|---|
| embedding | 2d = 384 | 8 x 8 |
| RMSNorm squares | 13d = 2,496 | 8 x 8 unsigned |
| RMSNorm gain | 13d = 2,496 | int16 x int16 |
| requantization (`quant_vec`) | 13d + 6(192 + 512) = 6,720 | int16 x 15-bit |
| `scale_out` (`acc * bm * sx.m`, 2 each) | 2 x (6 x 1,152 + vocab) = 15,260 (vocab 718) | 17 x 7 x 8 |
| attention head average | 6 x 192 = 1,152 | 16 x uint16 |
| FFN `acc3 * b3`, `silu * t3'` | 2 x 6 x 512 = 6,144 | 17 x 7, int16 x int16 |
| **subtotal** | **34,652** | |
| attention scores + weighted values | 6 x 384 x (pos + 1) = 2,304 (pos + 1) | 8 x 8 |

So 37k multiplies at position 0, 624k at position 255, 331k averaged over a full 256-token
window (attention dominates late in the window). Per-vector scalar work per token: 74 udiv
(one per `quant_vec`, one more per norm, one per attention head), 13 `isqrt`, about 100 PF15
products, plus one 32-bit modulo when sampling.
For scale: the ternary matvecs cover 2.42M weights per token (vocab 718) with table adds.

## 8. Correspondence with the QAT model (`model/model.py`)

The torch model fake-quantizes the same tensors on the same grids: int8 embeddings with PF8
row scales; PF8 per-vector activation scales with `round_half_up` and clamp to 127; ternary
weights with PF7 per-row scales (folded with the static q/k/v grid steps the same way as the
export); static int8 q/k/v grids from an EMA of the absmax (momentum 0.99); exact integer
attention scores (fp32 with autocast disabled), the same rounded table index and the same uint8
exp weights (gradients use the smooth `255 exp(-delta u)` through a straight-through
estimator). Under `--amp bf16` the BitLinear accumulators are rounded to bf16 during training
(noise only; the reported QAT losses are computed in fp32 or float64). RMSNorm uses
eps = 1e-6 in QAT and none in intref. What intref adds is
integer arithmetic for the parts QAT keeps in float: RMSNorm (15-bit gain, 8-bit sum of
squares, 15-bit inverse sqrt), the int24 residual, the SiLU table (step 1/64), the per-head
reciprocal, PF15 scale products and Q8 logits. Measured agreement is in section 10.

## 9. Training data windowing

Programs are packed as `<bos> ... <eos>` streams; training windows do not stop at program
boundaries. 97% of corpus tokens are in programs longer than the 256-token context (median
program 535 tokens), so a window starts at a uniformly random offset with probability 0.75 and
at a program start (`<bos>` at position 0, the on-calculator situation) with probability 0.25.
Validation reports two losses: contiguous non-overlapping windows over the val stream
(`val_loss`, every target once) and one window per val program from its `<bos>`, masked after
its `<eos>` (`val_bos_loss`).

## 10. Measured agreement (run `runs/v0`, M1 corpus build of 13:42, vocab 718)

| Metric | Value |
|---|---|
| QAT val loss, full val set (3,628 windows, fp32) | 1.2206 nats/token |
| QAT val loss from `<bos>` (613 val programs, fp32) | 1.2395 |
| QAT val loss on 64 windows spread over val (float64) | 1.22171 |
| intref val loss on the same 64 windows (16,188 targets) | 1.22161 (-0.008%) |
| top-1 agreement QAT vs intref | 98.6% |
| logit error, mean / max | 0.053 / 1.06 nats |
| intref speed (numpy, this box) | ~175 tokens/s |

Export: 711,441 bytes in 12 AppVars = 491,106 packed ternary (2,423,424 weights) + 187,008
int8 embeddings + 23,352 row scales + 9,975 gains, tables and header; residual `F = 12`
(calibration peak 433.8).

**Format 2 (the shipped v2 model: `runs/lab/S0-s10/train/avg.pt`, G2b's S0 seed 10, exported to
`runs/v2/export`, ALiBi, vocab 781).** Every val program whole, `<bos>` to `<eos>`, through the
256-token sliding window (`python -m model.evaluate --programs 0`; QAT in float64 on the CPU):

| Metric | Value |
|---|---|
| programs / targets / longest program | 666 / 1,034,321 / 20,295 tokens |
| QAT val loss (float64; the lab's fp32 GPU figure is 1.071852) | 1.071924 nats/token |
| intref val loss on the same targets | 1.071913 (-0.001%) |
| top-1 agreement QAT vs intref | 98.95% |
| logit error, mean / max | 0.045 / 2.04 nats |

The G2a stand-in (S0 seed 1) gave -0.014% and 98.89% the same way. Export: 676,986 bytes in 11
AppVars, `F = 11` (no position table, 49,664 bytes: 256 int8 rows of 192 and their PF8 scales;
the header is 4 bytes longer).

## 11. Extra input features (v2; specified here, not yet implemented in intref.py)

A model whose config has `feat_cats` (number of values of each categorical feature) and/or
`feat_bag` (bits of one multi-hot bag, at most 64) takes per-token features from the program
state tracker (tracker/README.md) as extra input. `IntModel.load` refuses such an export until
W5 implements this section; the QAT model (`Transformer.feature_embedding`) and the GPU sampler
(`lab/sample.py`) implement it in float.

**Table.** One int8 table `feat_emb` [rows][d] with a PF8 scale per row (`feat_emb.m` uint8,
`feat_emb.e` int8), quantized exactly like `tok_emb`; `rows = sum(feat_cats) + feat_bag`. The
row of categorical feature `g` with value `v` is `row_offsets[g] + v`
(`row_offsets[g] = feat_cats[0] + ... + feat_cats[g-1]`); the row of bag bit `b` is
`bag_row0 + b` (`bag_row0 = sum(feat_cats)`). In the blob the three tensors follow `expw`, so no
v1 offset moves; `model.json["features"]` records `cats`, `bag`, `row_offsets`, `bag_row0`, `rows`.
The v1 header is unchanged (the v2 header is W5's choice).

**Alignment.** The features used at position `pos` are the tracker state after reading the
token fed at `pos` (`tracker_step(state, tok)` output; at position 0, the state after `<bos>`,
i.e. the reset state). They are an input of position `pos`, which predicts token `pos + 1`.

**Embedding** (replaces the first line of `step` in section 4):
```
term(T, row)_i = rsr(T8[row][i] * Tm[row], -(Te[row] + F))         # one 8x8 product per element
r_i = clamp( term(E, tok)_i + term(P, pos)_i
           + sum over g of           term(X, row_offsets[g] + c_g)_i
           + sum over set bag bits b of term(X, bag_row0 + b)_i,   INT24)
```
Each row's contribution is rounded on its own (`rsr`), the contributions are summed exactly
(order does not matter; at most 2 + 9 + 64 terms, each below 2^23 in magnitude, fit 32 bits), and
the sum is clamped to int24 once. The exporter calibrates `F` on windows that include their
feature rows, so the 4x residual headroom covers them. Cost: (2 + n_cats + set bag bits) x d
multiplies per token instead of 2d (lab/budget.py `feat_embed`). Because each row's term is a
fixed integer vector, an implementation may keep the feature sum between tokens and only
subtract the rows that left and add the rows that entered (tracker/README.md "Using it"): the
result is bit-identical.

**QAT correspondence.** `feat_emb` uses the same straight-through int8 grid as the token
table; QAT adds the rows in fp32 (autocast off), which equals the integer sum above up to the
per-row rounding to the F-bit grid, as for the token and position rows.

**Shuffled-feature control (G1 arm A2)** changes only which features are fed (those of another
window or sampled program, `model.data.shuffle_features`), not this computation.

## 12. Relative positions and the sliding window (v2, export format 2)

A config with `pos = "alibi"` or `"nope"` has no `pos_emb` table, and the model runs on programs
of any length with a sliding window of `ctx` tokens (the header's `ctx`, 256 for v2). Its export
is format 2 (section 6). QAT (`model/model.py`, `_windowed_attention`) and the GPU sampler
(`lab/sample.py`, ring cache) implement this section in float, intref in integers, and the
calculator core runs ALiBi (`ce/calclm/core`; NoPE is refused there).
`model.json["positions"]` records `scheme`, `window` and, for ALiBi, `alibi_exponents`.

**Embedding.** `r_i = clamp(term(E, tok)_i [+ feature rows, section 11], INT24)`: the position
row of section 4 is dropped.

**Ring K/V cache.** The caches keep `ctx` entries per layer, as in v1. The token at position
`p` writes its k8/v8 to slot `p mod ctx`. At position `p` the attention runs over the
`n = min(p + 1, ctx)` slots in use, i.e. positions `p - n + 1 .. p` (itself included).
Scores, the max, Z and the weighted sums are exact integer sums, so the slot order does not
matter. The bound |a_d| <= 127 Z < 2^23 still holds (Z <= 255 ctx).

**NoPE.** Attention is exactly section 4 over those slots.

**ALiBi** (power-of-two slopes). Head `h` of `H` has exponent `e_h = H - 1 - h`: its penalty is
`2^e_h` exp-table steps per token of distance. With `d_j = p - pos(j)` the distance of a key
(0 for the token itself, at most ctx - 1), `sh_l` the layer's table shift (header `attn_sh`)
and `lut_max` the header field (512):
```
b_j   = min(d_j << e_h, lut_max) << sh_l            # score units; shifts only
z_j   = s_j - b_j                                   # s_j = sum_d q_h[d] * K[l][j][d], as before
idx_j = min(rsr(max_j z_j - z_j, sh_l), L_l - 1)    # replaces max_j s_j - s_j
e_j   = LUT_l[idx_j];  then Z, a_d, o exactly as in section 4
```
Bounds: |s_j| <= 127^2 * 32 = 516,128, and b_j <= 512 << sh_l = 32,768 for the observed shifts
(sh_l <= 6), so z_j fits int24. b_j depends only on (d_j, h, l), so an implementation may
update it incrementally per key. The cap at `lut_max` table steps is part of the definition:
it bounds b_j, and it can change a weight compared with an uncapped penalty, for example when a
far key's score is far above the others. QAT and the sampler apply the same cap, so they compute
the same integers (scores exact in fp32, bias exact powers of two).

Implementation bounds the calculator checks at load: `lut_max <= 4096` and
`lut_max << sh_l <= 2^22` for every layer, so every z_j and every `max z - z_j` fit 24 bits
(intref checks |z_j| < 2^23).

**On the calculator** (`ce/calclm/core/clm_fwd.c`, `att_scores_alibi`): in slot order the
distance falls by one per key, except at the wrap after the current slot, so the keys split into
at most three runs where b is constant (capped) or falls by `2^(e_h + sh_l)` per key. Each run is
one call of the score kernel with a start value `ch - b` and that per-key step. A cache with
fewer than `ctx` entries (low RAM) does not slide: generation stops when it is full, a prefix
of the full output, as in format 1.

**Cost** (lab/budget.py): one embedding row per token instead of two, attention over at most
`ctx` keys at every position past `ctx`, and for ALiBi one shift-and-subtract per (layer, head,
key) (`ALIBI_KEY_CYCLES`, an estimate; measured in `ce/calclm/README.md`). RAM is v1's: the same
`ctx`-entry K and V caches.

## 13. Logit masks (v2)

`model/intmask.py` (`IntMask`) states lab/masks.py's rules (G1 arm A3, G2a's deployment
setting) for one sequence, and `ce/calclm/core/clm_mask.c` follows it rule for rule. The state
is the C tracker's (tracker/tracker.c, shared by training, the lab and the calculator): after
`tracker_reset`, the tracker reads every token fed (`tracker_step`, which also gives the
features `cat[9]`, `hint[5]`, `var[4]`; the bag is `hint (bits 0-37) | var << 38`). Each rule is
a one-token look-ahead: copy the state, step the candidate token(s) on the copy, and ask
whether a summary counter grows (`n_undef`, `n_bad` or `n_extra_end`). Before each draw the
banned ids are:

1. always: DelVar, ClrAllLists, Archive, UnArchive, ClrList;
2. End if `n_extra_end` grows (no open block); Else if `n_bad` grows (no If-Then);
3. each Str0-9 and [A]-[J] token if `n_undef` grows (read before any assignment);
4. if the last token fed is one of those: `,` and `(` if `n_undef` grows (they resolve a
   pending read: `Input Str1,` uses Str1 as a prompt, `->[A](` stores into an element);
5. lists. Outside a list name (`ctx != 6`): `ʟ` if no list is assigned (bag bits 58-63 clear)
   and stepping `ʟ`, `A`, newline grows `n_undef`. Inside a list name: once per name, at the
   first draw inside it, decide whether the name is read: step the first letter A-Z whose hint
   bit is clear, then newline (no such letter: a read). For a read, ban every name character
   (A-Z, 0-9, θ; hint bit k is character k) whose hint bit is clear, and when the END hint bit
   (37) is clear, also every id from 4 up that is not a name character, and `<eos>`;
6. if nothing is left among the ids other than `<pad>`, `<bos>`, `<unk>`: allow `<eos>`.

The token ids come from the vocab (`intmask.mask_ids`; generated into
`ce/calclm/core/clm_mask_ids.h`). The tracker's class table and these ids are compiled for one
vocab (corpus/vocab.json, 781 ids, the vocab of v1 and v2): `IntMask` refuses another vocab,
and the core refuses masks for an export of another vocab size. Tests: intmask makes lab/masks.py's
decisions on every position of 67 val programs (first 400 tokens) and 40 of S0-s1's 1024-token
masked samples; lab/sample.py's generation loop driven by intref's logits gives intref's tokens
with masks; the core makes intmask's decisions in CEmu (`ce/tests/test_calclm_v2.py`).
Cost per token: one tracker step plus about 23 look-ahead steps on a 510-byte state copy
(measured in `ce/calclm/README.md`).

**Level 2 (v2.4 safety masks; `IntMask(vocab, level=2)`, `clm_reset(..., masks=2)`,
`model/safety.py`, docs/agents/phase1-safety-masks.md).** Level 1 is the list above (v2.3).
Level 2 adds, after rule 5 and before rule 6:

- always: `prgm`, `OpenLib(`, `ExecLib` and the commands that change a setting that outlives
  the program (mode, graph format and styles, table setup, zoom, clock setters, RecallGDB,
  SetUpEditor; `safety.ALWAYS_HEX`, 103 ids);
- every vocab id on the `63xx` page (57 window, zoom-memory, table, sequence and finance
  variables) when the context is `ctx == 3` (store target, after `->`), or the tracker's
  `stmt` is the class of Input or Prompt and `ctx != 0`, or the last token fed is For(, IS>( or
  DS<(;
- inside a string (`ctx == 2`): every id from `<eos>` up whose string class is in
  `STR_BAN[str_state]`. Classes: the letters b y c o p r, any other letter (uppercase `41`-`5A`,
  lowercase `BBB0`-`BBCA`, and any token whose display text has one ASCII letter, first, such
  as y₁, Y₁, r²), `(`,
  `)`, and W for every other id. `str_state` is START (0) after any token that leaves the
  tracker outside a string or opens one, else `STR_NEXT[str_state][class(tok)]`. The automaton
  (9 states) bans Y after a word-initial B, R after COPY and `)` after `(C`, in either case.

Rule 6 then runs as before (no level 2 rule can empty the vocabulary on its own).
