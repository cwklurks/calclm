/* Ternary LUT matvec kernels (ce/kernel/gen_tmv.py, M2), one pair per input width:
 *   tmv_d_*: n_in = CLM_DIM    (G = CLM_G_DIM groups)
 *   tmv_h_*: n_in = CLM_HIDDEN (G = CLM_G_HID groups)
 * build(x, tbl): x is int8 in [-127, 127] zero-padded to 5G; tbl is the 64 KB table bank.
 * rows(W, y, n_out, tbl): y[r] = sum_j w[r][j] x[j] as int24 (eZ80 asm writes 3 bytes/row).
 * On the eZ80 these are the generated assembly kernels (symbols renamed by
 * ce/calclm/tools/gen_kernels.py); on the host, core/host/tmv_host.c computes the same sums. */
#ifndef CLM_TMV_H
#define CLM_TMV_H

#include "clm.h"

void tmv_d_build(const int8_t *x, uint8_t *tbl);
void tmv_d_rows(const uint8_t *W, clm_i24 *y, unsigned n_out, const uint8_t *tbl);
void tmv_h_build(const int8_t *x, uint8_t *tbl);
void tmv_h_rows(const uint8_t *W, clm_i24 *y, unsigned n_out, const uint8_t *tbl);

#endif
