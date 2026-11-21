/* Host stand-in for the eZ80 matvec kernels (same contract as core/tmv.h, SPEC.md packing):
 * y[r] = sum_g sum_j (digit_j(W[r][g]) - 1) * x[5g + j]. The table bank is not used. */
#include "../tmv.h"

static const int8_t *xd, *xh;

static void rows(const uint8_t *W, clm_i24 *y, unsigned n_out, const int8_t *x, unsigned G) {
    for (unsigned r = 0; r < n_out; r++) {
        int32_t acc = 0;
        for (unsigned g = 0; g < G; g++) {
            unsigned b = W[r * G + g];
            for (unsigned j = 0; j < 5; j++) {
                acc += ((int32_t)(b % 3) - 1) * x[5 * g + j];
                b /= 3;
            }
        }
        y[r] = acc;
    }
}

void tmv_d_build(const int8_t *x, uint8_t *tbl) { (void)tbl; xd = x; }
void tmv_h_build(const int8_t *x, uint8_t *tbl) { (void)tbl; xh = x; }

void tmv_d_rows(const uint8_t *W, clm_i24 *y, unsigned n_out, const uint8_t *tbl) {
    (void)tbl;
    rows(W, y, n_out, xd, CLM_G_DIM);
}

void tmv_h_rows(const uint8_t *W, clm_i24 *y, unsigned n_out, const uint8_t *tbl) {
    (void)tbl;
    rows(W, y, n_out, xh, CLM_G_HID);
}
