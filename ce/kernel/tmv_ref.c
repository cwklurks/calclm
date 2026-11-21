/* Straightforward C reference of the ternary LUT matvec (SPEC.md "Ternary packing and kernel
 * contract"), compiled with ez80-clang for comparison with the hand-written assembly kernel.
 *
 * tbl is int16_t[G][243]: tbl[g][b] = sum_j (digit_j(b) - 1) * x[5g + j].
 * y[r] = sum_g tbl[g][W[r*G + g]]   (int24, the eZ80 `int`). */
#include <stdint.h>

void tmv_c_build(const int8_t *x, unsigned G, int16_t *tbl) {
    for (unsigned g = 0; g < G; g++, x += 5) {
        int x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4];
        int v4 = -x1 - x2 - x3 - x4; /* digits d1..d4 = 0 (weights -1) */
        for (int d4 = 0; d4 < 3; d4++, v4 += x4) {
            int v3 = v4;
            for (int d3 = 0; d3 < 3; d3++, v3 += x3) {
                int v2 = v3;
                for (int d2 = 0; d2 < 3; d2++, v2 += x2) {
                    int v1 = v2;
                    for (int d1 = 0; d1 < 3; d1++, v1 += x1) {
                        *tbl++ = (int16_t)(v1 - x0);
                        *tbl++ = (int16_t)v1;
                        *tbl++ = (int16_t)(v1 + x0);
                    }
                }
            }
        }
    }
}

void tmv_c_rows(const uint8_t *W, int *y, unsigned n_out, unsigned G, const int16_t *tbl) {
    for (unsigned r = 0; r < n_out; r++) {
        const int16_t *t = tbl;
        int acc = 0;
        for (unsigned g = 0; g < G; g++) {
            acc += t[*W++];
            t += 243;
        }
        *y++ = acc;
    }
}
