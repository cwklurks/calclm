/* Integer primitives (INTREF.md sections 1 and 3), exact on the eZ80 (24-bit int) and host. */
#include "clm_int.h"

static const clm_pf INV127 = {16513, -21};   /* pf_div((2^14, -14), pf15(127)) */

uint8_t clm_bitlen(uint32_t n) {
    uint8_t b = 0;
    if (n >> 16) {
        n >>= 16;
        b = 16;
    }
    while (n) {
        n >>= 1;
        b++;
    }
    return b;
}

uint32_t clm_isqrt(uint32_t n) {
    uint32_t root = 0, bit = 0;
    if (n) {
        bit = (uint32_t)1 << ((clm_bitlen(n) - 1) & ~1);
    }
    while (bit) {
        if (n >= root + bit) {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

/* floor((a + 2^(s-1)) / 2^s) without the overflow of the rounding add (INTREF.md section 1) */
int32_t clm_rsr(int32_t a, int16_t s) {
    if (s <= 0) {
        return a;
    }
    if (s >= 32) {
        return 0;       /* |a| <= 2^31: (a + 2^(s-1)) / 2^s lies in [0, 1) */
    }
    uint32_t mask = ((uint32_t)1 << s) - 1;
    return (a >> s) + (int32_t)((((uint32_t)a & mask) + ((uint32_t)1 << (s - 1))) >> s);
}

int32_t clm_clamp(int32_t a, int32_t lo, int32_t hi) {
    return a < lo ? lo : (a > hi ? hi : a);
}

/* a << k, saturated to [lo, hi] by comparing before shifting (k > 0) */
static int32_t lsat(int32_t a, int16_t k, int32_t lo, int32_t hi) {
    if (k >= 31) {
        return a > 0 ? hi : (a < 0 ? lo : 0);
    }
    if (a > (hi >> k)) {
        return hi;
    }
    if (a < (lo >> k)) {
        return lo;
    }
    return clm_clamp((int32_t)((uint32_t)a << k), lo, hi);
}

clm_pf clm_pf15(uint32_t n, int16_t e) {
    int16_t sh = (int16_t)clm_bitlen(n) - 15;
    uint32_t m;
    if (sh > 0) {
        uint32_t mask = ((uint32_t)1 << sh) - 1;
        m = (n >> sh) + (((n & mask) + ((uint32_t)1 << (sh - 1))) >> sh);
    } else {
        m = n << -sh;
    }
    if (m == 32768) {
        m = 16384;
        sh++;
    }
    clm_pf r = {(uint16_t)m, (int16_t)(e + sh)};
    return r;
}

clm_pf clm_pf_mul(clm_pf a, clm_pf b) {
    return clm_pf15((uint32_t)a.m * b.m, (int16_t)(a.e + b.e));
}

clm_pf clm_pf_div(clm_pf a, clm_pf b) {
    return clm_pf15(((uint32_t)a.m << 16) / b.m, (int16_t)(a.e - b.e - 16));
}

clm_pf clm_pf8(clm_pf a) {
    clm_pf r = {(uint16_t)((a.m + 64u) >> 7), (int16_t)(a.e + 7)};
    if (r.m == 256) {
        r.m = 128;
        r.e++;
    }
    return r;
}

#if defined(__ez80__)
/* ez80/clm_vec.s */
void clm_a_maxabs32(const int32_t *v, unsigned n, uint32_t *peak);
unsigned clm_a_mulnorm(const clm_i24 *x, unsigned n, int c, const uint8_t *w, int32_t *out,
                       unsigned want_ss, clm_i24 *tmp);
unsigned clm_a_t3(const clm_i24 *acc, const uint8_t *bm, unsigned n, int E3, clm_i24 *t3,
                  const int8_t *be);
#endif

int8_t clm_norm_shift(const int32_t *v, uint16_t n, uint32_t *peak) {
#if defined(__ez80__)
    if (n) {
        clm_a_maxabs32(v, n, peak);
        return (int8_t)(clm_bitlen(*peak) - 15);
    }
#endif
    uint32_t mx = 0;
    for (uint16_t i = 0; i < n; i++) {
        uint32_t a = v[i] < 0 ? (uint32_t)0 - (uint32_t)v[i] : (uint32_t)v[i];
        if (a > mx) {
            mx = a;
        }
    }
    *peak = mx;
    return (int8_t)(clm_bitlen(mx) - 15);
}

#if defined(__ez80__)
unsigned clm_a_maxabs24(const clm_i24 *v, unsigned n);    /* ez80/clm_misc.s */
#endif

int8_t clm_norm_shift24(const clm_i24 *v, uint16_t n, uint32_t *peak) {
#if defined(__ez80__)
    if (n) {
        *peak = clm_a_maxabs24(v, n);
        return (int8_t)(clm_bitlen(*peak) - 15);
    }
#endif
    uint32_t mx = 0;
    for (uint16_t i = 0; i < n; i++) {
        int32_t x = v[i];
        uint32_t a = x < 0 ? (uint32_t)0 - (uint32_t)x : (uint32_t)x;
        if (a > mx) {
            mx = a;
        }
    }
    *peak = mx;
    return (int8_t)(clm_bitlen(mx) - 15);
}

int16_t clm_norm1(int32_t v, int8_t c) {
    if (c > 0) {
        return (int16_t)clm_clamp(clm_rsr(v, c), -32767, 32767);
    }
    return (int16_t)(v * ((int32_t)1 << -c));     /* |v| << -c < 2^15 by the choice of c */
}

#if defined(__ez80__)
/* ez80/clm_quant.s: the per-element pass below for c in [-15, 16], S in [17, 24], |x| <= 128 */
void clm_a_quant(const int32_t *v, unsigned n, int c, unsigned gm, int S, int8_t *x8);
#endif

clm_pf clm_quant_vec(const int32_t *v, uint16_t n, clm_pf unit, int8_t *x8) {
    uint32_t peak;
    int8_t c = clm_norm_shift(v, n, &peak);
    clm_pf sx = {0, 0};
    if (!peak) {
        for (uint16_t i = 0; i < n; i++) {
            x8[i] = 0;
        }
        return sx;
    }
    sx = clm_pf8(clm_pf_mul(clm_pf_mul(clm_pf15(peak, 0), unit), INV127));
    clm_pf s15 = {(uint16_t)(sx.m << 7), (int16_t)(sx.e - 7)};
    clm_pf g = clm_pf_div(unit, s15);
    int16_t S = (int16_t)(-(g.e + c));
#if defined(__ez80__)
    /* every |x| is bounded by the peak element's: run the assembly pass when it is <= 128 */
    if (S >= 17 && S <= 24 && c >= -15 && c <= 16 &&
        clm_rsr((int32_t)clm_norm1((int32_t)peak, c) * (int32_t)g.m, S) <= 128) {
        clm_a_quant(v, n, c, g.m, S, x8);
        return sx;
    }
#endif
    for (uint16_t i = 0; i < n; i++) {
        int32_t p = (int32_t)clm_norm1(v[i], c) * (int32_t)g.m;   /* |p| < 2^30 */
        p = S >= 0 ? clm_clamp(clm_rsr(p, S), -127, 127) : lsat(p, (int16_t)-S, -127, 127);
        x8[i] = (int8_t)p;
    }
    return sx;
}

clm_pf clm_norm_quant(const clm_i24 *r, uint16_t n, const uint8_t *g16, int8_t eg, clm_pf sqrtd,
                      int32_t *z, int8_t *x8, clm_i24 *tmp) {
    uint32_t peak, ss = 0;
    int8_t c = clm_norm_shift24(r, n, &peak);
    if (!peak) {
        clm_pf zero = {0, 0};
        for (uint16_t i = 0; i < n; i++) {
            x8[i] = 0;
        }
        return zero;
    }
    ss = clm_mulnorm(r, n, c, g16, z, 1, tmp);                /* z = r' * g16, |z| < 2^30 */
    int16_t b = (int16_t)((30 - clm_bitlen(ss)) >> 1);       /* ss <= 2^22, so b >= 3 */
    uint32_t s = clm_isqrt(ss << (2 * b));
    clm_pf unit = clm_pf_div(sqrtd, clm_pf15(s, 0));
    unit.e = (int16_t)(unit.e + b - eg - 8);
    return clm_quant_vec(z, n, unit, x8);
}

int32_t clm_scale1(int32_t acc, uint8_t bm, int8_t be, clm_pf sx, int8_t f_out, int32_t lo,
                   int32_t hi) {
    int32_t p = acc * (int32_t)((uint16_t)bm * sx.m);         /* |p| < 2^16 * 2^15 */
    int16_t s = (int16_t)(-(be + sx.e + f_out));
    return s >= 0 ? clm_clamp(clm_rsr(p, s), lo, hi) : lsat(p, (int16_t)-s, lo, hi);
}

#if defined(__ez80__)
/* ez80/clm_scale.s: scale_out rows when every row's shift S = sb - e is in [1, 24] */
void clm_a_scale(const clm_i24 *acc, const uint8_t *bm, unsigned n, unsigned sxm, int sb,
                 void *out, unsigned mode, uint8_t *rec, const int8_t *be);
#endif

void clm_scale_rows(const clm_lin *w, const clm_i24 *acc, uint16_t n, clm_pf sx, int8_t f_out,
                    uint8_t mode, void *out, uint8_t *rec) {
    static const int32_t lo[4] = {-127, -32768, CLM_I24_MIN, CLM_I24_MIN};
    static const int32_t hi[4] = {127, 32767, CLM_I24_MAX, CLM_I24_MAX};
#if defined(__ez80__)
    int16_t sb = (int16_t)(-(sx.e + f_out));
    if (sx.m && sb - w->emax >= 1 && sb - w->emin <= 24) {
        clm_a_scale(acc, w->m, n, sx.m, sb, out, mode, rec, w->e);
        return;
    }
#else
    (void)rec;
#endif
    for (uint16_t i = 0; i < n; i++) {
        int32_t y = sx.m ? clm_scale1(acc[i], w->m[i], w->e[i], sx, f_out, lo[mode], hi[mode]) : 0;
        if (mode == 0) {
            ((int8_t *)out)[i] = (int8_t)y;
        } else if (mode == 1) {
            ((int16_t *)out)[i] = (int16_t)y;
        } else if (mode == 2) {
            ((clm_i24 *)out)[i] = (clm_i24)y;
        } else {
            clm_i24 *r = (clm_i24 *)out;
            r[i] = (clm_i24)clm_clamp((int32_t)r[i] + y, CLM_I24_MIN, CLM_I24_MAX);
        }
    }
}

uint32_t clm_mulnorm(const clm_i24 *x, uint16_t n, int8_t c, const uint8_t *w, int32_t *out,
                     uint8_t want_ss, clm_i24 *tmp) {
#if defined(__ez80__)
    if (n && c >= -15 && c <= 8) {
        return clm_a_mulnorm(x, n, c, w, out, want_ss, tmp);
    }
#else
    (void)tmp;
#endif
    uint32_t ss = 0;
    for (uint16_t i = 0; i < n; i++) {
        int16_t xp = clm_norm1(x[i], c);
        if (want_ss) {
            int16_t r8 = (int16_t)(((int32_t)xp + 128) >> 8);  /* rsr(x', 8), |r8| <= 128 */
            ss += (uint32_t)((int32_t)r8 * r8);
        }
        out[i] = (int32_t)xp * (int32_t)RD16S(w + 2 * i);
    }
    return ss;
}

uint32_t clm_t3(const clm_i24 *acc, const clm_lin *w3, int8_t E3, uint16_t n, clm_i24 *t3) {
#if defined(__ez80__)
    if (n && E3 - w3->emin <= 8) {
        return clm_a_t3(acc, w3->m, n, E3, t3, w3->e);
    }
#endif
    uint32_t peak = 0;
    for (uint16_t i = 0; i < n; i++) {
        int32_t t = clm_rsr((int32_t)acc[i] * w3->m[i], (int16_t)(E3 - w3->e[i]));  /* < 2^23 */
        uint32_t a = t < 0 ? (uint32_t)0 - (uint32_t)t : (uint32_t)t;
        peak = a > peak ? a : peak;
        t3[i] = (clm_i24)t;
    }
    return peak;
}
