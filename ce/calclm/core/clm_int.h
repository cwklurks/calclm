/* Integer primitives of INTREF.md sections 1 and 3 (model/intops.py), for the CalcLM core.
 * Every function is exact: no value is ever truncated to a narrower type than its range. */
#ifndef CLM_INT_H
#define CLM_INT_H

#include <stdint.h>

#include "clm.h"

#define CLM_I24_MIN (-8388608L)
#define CLM_I24_MAX 8388607L

/* little-endian reads from blob bytes (the eZ80 is little-endian and has no alignment rules) */
#if defined(__ez80__)
#define RD16S(p) (*(const int16_t *)(const void *)(p))
#define RD16U(p) (*(const uint16_t *)(const void *)(p))
#else
#define RD16S(p) ((int16_t)((uint16_t)((p)[0] | (p)[1] << 8)))
#define RD16U(p) ((uint16_t)((p)[0] | (p)[1] << 8))
#endif

uint8_t clm_bitlen(uint32_t n);
uint32_t clm_isqrt(uint32_t n);
int32_t clm_rsr(int32_t a, int16_t s);            /* s >= 0: round-half-up shift right */
int32_t clm_clamp(int32_t a, int32_t lo, int32_t hi);

clm_pf clm_pf15(uint32_t n, int16_t e);           /* n > 0 */
clm_pf clm_pf_mul(clm_pf a, clm_pf b);
clm_pf clm_pf_div(clm_pf a, clm_pf b);
clm_pf clm_pf8(clm_pf a);

/* normalize16 of an int32 (or int24) vector: the shift c with |v >> c| <= 32767 (c may be
 * negative) */
int8_t clm_norm_shift(const int32_t *v, uint16_t n, uint32_t *peak);
int8_t clm_norm_shift24(const clm_i24 *v, uint16_t n, uint32_t *peak);
int16_t clm_norm1(int32_t v, int8_t c);           /* one element of normalize16 */

/* quant_vec(v, unit): x8 (n entries) and its PF8 scale (m = 0 for an all-zero vector) */
clm_pf clm_quant_vec(const int32_t *v, uint16_t n, clm_pf unit, int8_t *x8);

/* norm_quant(r, g16, eg): RMSNorm + quant_vec; uses z (n int32) and tmp (3n bytes) as scratch */
clm_pf clm_norm_quant(const clm_i24 *r, uint16_t n, const uint8_t *g16, int8_t eg, clm_pf sqrtd,
                      int32_t *z, int8_t *x8, clm_i24 *tmp);

/* scale_out of rows [0, n) of a matrix: mode 0 int8 [-127, 127], 1 int16, 2 int24,
 * 3 residual add (out int24 += clamp(y, int24), saturated); `rec` is 1536 bytes of scratch */
void clm_scale_rows(const clm_lin *w, const clm_i24 *acc, uint16_t n, clm_pf sx, int8_t f_out,
                    uint8_t mode, void *out, uint8_t *rec);

/* out[i] = normalize16(x, c)[i] * w[i] (exact int32), w int16 LE bytes; returns
 * sum(rsr(x'[i], 8)^2) when want_ss. Requires c = bitlen(max|x|) - 15 (so c <= 8). */
uint32_t clm_mulnorm(const clm_i24 *x, uint16_t n, int8_t c, const uint8_t *w, int32_t *out,
                     uint8_t want_ss, clm_i24 *tmp);  /* tmp: 3n bytes when x must survive */

/* t3[i] = rsr(acc[i] * w3.m[i], E3 - w3.e[i]) as int24 (may alias acc); returns max |t3| */
uint32_t clm_t3(const clm_i24 *acc, const clm_lin *w3, int8_t E3, uint16_t n, clm_i24 *t3);

/* scale_out of one row: clamp(rsr(acc * bm * sx.m, -(be + sx.e + f_out)), lo, hi) */
int32_t clm_scale1(int32_t acc, uint8_t bm, int8_t be, clm_pf sx, int8_t f_out, int32_t lo,
                   int32_t hi);

#endif
