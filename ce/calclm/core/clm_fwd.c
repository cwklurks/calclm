/* One token through the model: INTREF.md sections 4 and 12 (model/intref.py IntModel.step). */
#include "clm_int.h"
#include "tmv.h"
#if defined(CLM_KV4)
#include "clm_kv4.h"
#endif

#define PROF(m, ph)              \
    do {                         \
        if ((m)->prof) {         \
            (m)->prof(ph);       \
        }                        \
    } while (0)

#define SILU_HALF 512
#define SILU_EDGE 2048

/* index over the n_heads * head_dim query entries: 256 for W, so 8 bits only below that (v2.2's
 * 192 keeps the byte counter it was built and measured with) */
#if CLM_QDIM > 255
typedef uint16_t clm_qidx;
#else
typedef uint8_t clm_qidx;
#endif

static const uint8_t *blob_row(const clm_t *m, uint32_t off) {
    for (uint8_t i = 0; i < m->n_seg; i++) {
        if (off >= m->seg[i].start && off < m->seg[i].end) {
            return m->seg[i].data + (off - m->seg[i].start);
        }
    }
    return 0;   /* unreachable: clm_load checked every row */
}

#if defined(__ez80__)
void clm_a_emb(const int8_t *x, unsigned n, unsigned mm, int s, clm_i24 *r, unsigned add);
#endif

#if !defined(__ez80__)
/* rsr(a, s) for |a| < 2^15 and s >= -8 (checked at load): the result fits 24 bits, so the
 * eZ80 does it in native int arithmetic (for s > 15 the result is 0) */
static int emb1(int a, int s) {
    if (s <= 0) {
        return a * (1 << -s);
    }
    return s > 15 ? 0 : (a + (1 << (s - 1))) >> s;
}
#endif

/* r = clamp(token row [+ position row], int24); format 2 (ALiBi) has no position row, and
 * one row term fits int24 on its own */
static void embed(clm_t *m, uint16_t tok, uint16_t pos) {
    const int8_t *e = (const int8_t *)blob_row(m, m->tok_off + (uint32_t)tok * CLM_DIM);
    int em = m->tok_m[tok];
    int se = -(m->tok_e[tok] + m->F);
#if defined(__ez80__)
    clm_a_emb(e, CLM_DIM, (unsigned)em, se, m->r, 0);
#else
    for (int i = 0; i < CLM_DIM; i++) {
        m->r[i] = (clm_i24)emb1(e[i] * em, se);
    }
#endif
    if (m->alibi) {
        return;
    }
    const int8_t *p = (const int8_t *)blob_row(m, m->pos_off + (uint32_t)pos * CLM_DIM);
    int pm = m->pos_m[pos];
    int sp = -(m->pos_e[pos] + m->F);
#if defined(__ez80__)
    clm_a_emb(p, CLM_DIM, (unsigned)pm, sp, m->r, 1);
#else
    for (int i = 0; i < CLM_DIM; i++) {
        int32_t v = (int32_t)m->r[i] + emb1(p[i] * pm, sp);    /* each term < 2^23 */
        m->r[i] = (clm_i24)clm_clamp(v, CLM_I24_MIN, CLM_I24_MAX);
    }
#endif
}

/* y = W x for a d-input matrix whose table was built; then per-row scale_out to int8 */
static void rows_q8(clm_t *m, const clm_lin *w, uint16_t n, clm_pf sx, int8_t *out) {
    PROF(m, CLM_PH_ROWS);
    tmv_d_rows(w->w, m->y, n, m->mem.tbl);
    PROF(m, CLM_PH_SCALE);
    clm_scale_rows(w, m->y, n, sx, 0, 0, out, (uint8_t *)m->v32);
}

/* r += scale_out(W x, ..., F, INT24) with the residual saturated to int24 */
static void residual_add(clm_t *m, const clm_lin *w, clm_pf sx) {
    PROF(m, CLM_PH_SCALE);
    clm_scale_rows(w, m->y, CLM_DIM, sx, m->F, 3, m->r, (uint8_t *)m->v32);
}

#if !defined(CLM_KV4)
/* store a freshly computed K or V row (int8 in place) in the biased cache format */
static void bias_row(uint8_t *row, uint8_t with_sum) {
    int32_t sum = 0;
    for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
        row[d] ^= 0x80;                           /* k + 128 */
        sum += row[d];
    }
    if (with_sum) {
        uint32_t nk = (uint32_t)(-128 * sum);     /* int24 LE */
        row[CLM_HEAD_DIM] = (uint8_t)nk;
        row[CLM_HEAD_DIM + 1] = (uint8_t)(nk >> 8);
        row[CLM_HEAD_DIM + 2] = (uint8_t)(nk >> 16);
    }
}
#endif

#if defined(__ez80__)
/* ez80/clm_attn.s */
int clm_a_scores(const uint8_t *qu, int ch, const uint8_t *K, unsigned n, clm_i24 *sc, int step);
unsigned clm_a_softmax(const clm_i24 *sc, unsigned n, int mx, int sh, const uint8_t *lut,
                       int lut_len, uint8_t *ew);
void clm_a_pv(const uint8_t *V, const uint8_t *ew, unsigned n, unsigned *a);
void clm_a_havg(const unsigned *a, unsigned z, unsigned k, unsigned rz, int32_t *o);
void clm_a_silu(int16_t *y, unsigned n, const uint8_t *silu);
unsigned clm_a_compact(const uint8_t *V, const uint8_t *ew, unsigned n, uint8_t *cv,
                       uint8_t *ce, unsigned cap);
#endif

/* P.V on the compacted nonzero rows once the window is long enough for the softmax weights to
 * be mostly zero (they are 60-90% zero late in the window); up to CV_CAP rows fit in y */
#define CV_MIN_KEYS 32
#define CV_CAP 96

/* Scores of n keys from K: s_j = q . k_j + c_j with c_j = ch + j * step, stored last key
 * first (sc[n - 1 - j], as clm_a_scores pushes them); returns the max. step = 0 is plain
 * attention; a run of ALiBi keys has a constant (capped) or linearly falling penalty. */
static clm_i24 att_scores(const uint8_t *qu, int32_t ch, int32_t step, const uint8_t *K,
                          uint16_t n, clm_i24 *sc) {
#if defined(__ez80__)
    return (clm_i24)clm_a_scores(qu, (int)ch, K, n, sc, (int)step);
#else
    clm_i24 mx = 0;
    for (uint16_t j = 0; j < n; j++, K += CLM_KROW, ch += step) {
        uint32_t nk = (uint32_t)K[CLM_HEAD_DIM] | (uint32_t)K[CLM_HEAD_DIM + 1] << 8 |
                      (uint32_t)K[CLM_HEAD_DIM + 2] << 16;
        int32_t s = ch + (int32_t)(nk & 0x800000 ? nk | 0xFF000000u : nk);
        for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
            s += (int32_t)qu[d] * K[d];           /* sum (q+128)(k+128) - 128 sum(k+128) + ch */
        }
        sc[n - 1 - j] = (clm_i24)s;
        mx = (j == 0 || s > mx) ? (clm_i24)s : mx;
    }
    return mx;
#endif
}

/* ALiBi scores (INTREF.md 12) of the n keys in slot order, c = this token's slot:
 * z_j = q . k_j - b(d_j), d_j = (c - j) mod CLM_CTX, b(d) = min(d << e, lut_max) << sh.
 * Along j the distance falls by one per key (up to the wrap after slot c), so the keys split
 * into at most three runs over which b is constant (d << e >= lut_max) or falls by 2^(e+sh)
 * per key; each run is one att_scores call. Runs go in slot order (clm_a_scores writes one
 * scratch entry just below its last score: the slot of the next run's first key). */
__attribute__((noinline))   /* keeps attention()'s frame (and v1's speed) as it was */
static clm_i24 att_scores_alibi(const clm_t *m, const uint8_t *qu, int32_t ch, const uint8_t *K,
                                uint16_t n, uint16_t c, uint8_t e, uint8_t sh) {
    uint16_t dcap = (uint16_t)((m->lut_max + (1u << e) - 1) >> e);   /* d >= dcap: capped */
    clm_i24 mx = 0;
    uint16_t j = 0;
    while (j < n) {
        uint16_t d = (uint16_t)(j <= c ? c - j : CLM_CTX + c - j);
        uint16_t run_end = j <= c ? (uint16_t)(c + 1) : n;     /* d falls by one up to here */
        uint16_t end = run_end;
        int32_t b, step;
        if (d >= dcap) {
            if (d - dcap + 1 < run_end - j) {
                end = (uint16_t)(j + d - dcap + 1);
            }
            b = (int32_t)m->lut_max << sh;
            step = 0;
        } else {
            b = (int32_t)d << (e + sh);
            step = (int32_t)1 << (e + sh);
        }
        clm_i24 s = att_scores(qu, ch - b, step, K + (uint32_t)j * CLM_KROW, (uint16_t)(end - j),
                               m->sc + (n - end));
        mx = (j == 0 || s > mx) ? s : mx;
        j = end;
    }
    return mx;
}

/* ew[j] = LUT[min(rsr(mx - s_j, sh), L - 1)]; returns Z = sum ew[j]. The scores sit last key
 * first (sc[n - 1 - j]). Inlined into each caller: a call costs attention_head's per-head code
 * about 4 K cycles (register allocation). */
__attribute__((always_inline))
static inline uint32_t att_softmax(const clm_layer *L, const clm_i24 *sc, uint16_t n, clm_i24 mx,
                                   uint8_t *ew) {
#if defined(__ez80__)
    if (L->sh >= 1 && L->sh <= 7 && L->lut_len <= 512) {
        return clm_a_softmax(sc, n, mx, L->sh, L->lut, L->lut_len, ew);
    }
#endif
    uint32_t z = 0;
    for (uint16_t j = 0; j < n; j++) {
        int32_t idx = clm_rsr((int32_t)mx - sc[n - 1 - j], L->sh);
        if (idx > L->lut_len - 1) {
            idx = L->lut_len - 1;
        }
        ew[j] = L->lut[idx];
        z += ew[j];
    }
    return z;
}

#if !defined(__ez80__)
/* a_d = sum_j e_j (v_jd + 128) (unsigned, < 2^24) */
static void att_pv(const uint8_t *V, const uint8_t *ew, uint16_t n, uint32_t *a) {
    for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
        uint32_t s = 0;
        for (uint16_t j = 0; j < n; j++) {
            s += (uint32_t)ew[j] * V[j * CLM_HEAD_DIM + d];
        }
        a[d] = s;
    }
}

/* o_d = rsr(rsr(a_d - 128 z, zb - 8) * RZ, 14) */
static void att_havg(const uint32_t *au, uint32_t z, int32_t *o) {
    uint8_t zb = clm_bitlen(z);                 /* z >= 255 */
    uint32_t rz = (((uint32_t)1 << (zb + 14)) + (z >> 1)) / z;
    for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
        int32_t a = (int32_t)au[d] - (int32_t)(z << 7);       /* |a| <= 127 z < 2^23 */
        int32_t as = clm_rsr(a, (int16_t)(zb - 8));          /* |as| <= 32512 */
        o[d] = clm_rsr(as * (int32_t)rz, 14);
    }
}
#endif

/* One head h over the n keys held (slot order; slot = this token's): scores (with ALiBi for
 * format 2), softmax, P.V and the head average into m->v32. A function of its own keeps its
 * frame within the eZ80's short (ix + d) reach, as the per-head loop was in v1. */
__attribute__((noinline))
static void attention_head(clm_t *m, const clm_layer *L, const uint8_t *K, const uint8_t *V,
                           const uint8_t *q, uint16_t n, uint16_t slot, uint8_t h) {
    int32_t ch = 32L * 128 * 128;
    for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
        ch -= 128 * (int32_t)q[d];
    }
    PROF(m, CLM_PH_SCORES);
    clm_i24 mx = m->alibi ? att_scores_alibi(m, q, ch, K, n, slot, (uint8_t)(CLM_HEADS - 1 - h),
                                             (uint8_t)L->sh)
                          : att_scores(q, ch, 0, K, n, m->sc);
    PROF(m, CLM_PH_SOFTMAX);
    uint32_t z = att_softmax(L, m->sc, n, mx, m->ew);
    PROF(m, CLM_PH_PV);
#if defined(__ez80__)
    unsigned *au = (unsigned *)(void *)m->y1;  /* [HEAD_DIM]; y1 is the FFN's, free here */
    unsigned nk = 0;
    if (n >= CV_MIN_KEYS) {                     /* exact: zero weights add nothing */
        uint8_t *cv = (uint8_t *)m->y, *ce = (uint8_t *)m->sc;   /* both free here */
        nk = clm_a_compact(V, m->ew, n, cv, ce, CV_CAP);
        if (nk <= CV_CAP) {
            for (unsigned j = nk; j & 7; j++) {
                ce[j] = 0;                      /* clm_a_pv works in groups of 8 keys */
            }
            clm_a_pv(cv, ce, nk, au);
        }
    }
    if (n < CV_MIN_KEYS || nk > CV_CAP) {
        for (uint16_t j = n; j & 7; j++) {
            m->ew[j] = 0;
        }
        clm_a_pv(V, m->ew, n, au);
    }
    uint8_t zb = clm_bitlen(z);
    uint32_t rz = (((uint32_t)1 << (zb + 14)) + (z >> 1)) / z;
    clm_a_havg(au, z, (unsigned)(zb - 8), (unsigned)rz, m->v32 + h * CLM_HEAD_DIM);
#else
    uint32_t au[CLM_HEAD_DIM];
    att_pv(V, m->ew, n, au);
    att_havg(au, z, m->v32 + h * CLM_HEAD_DIM);
#endif
}

/* attention_head with ALiBi horizons (PLAN-v2.1.md E1; intref IntModel(horizons=True)) when
 * they cut: head h attends only to the mk < n keys at distance d < ceil(lut_max / 2^e), whose
 * penalty is below the cap. In slot order they are one run [slot + 1 - mk, slot], or two when
 * the ring wraps inside the horizon (slot + 1 < mk; sliding cache only): [CTX - w, CTX) at
 * distances mk - 1 .. slot + 1 (w = mk - 1 - slot), then [0, slot]. Each run is one falling
 * ALiBi score run. The softmax of every run uses the max over both; Z and the P.V sums are
 * plain integer sums over the keys (nothing is rounded before the head average), so adding
 * the runs' parts gives the one-run result exactly. */
__attribute__((noinline))
static void attention_head_hz(clm_t *m, const clm_layer *L, const uint8_t *K, const uint8_t *V,
                              const uint8_t *q, unsigned mk, unsigned slot, uint8_t h) {
    uint8_t es = (uint8_t)(CLM_HEADS - 1 - h + L->sh);    /* penalty step per key: 2^(e + sh) */
    unsigned s0[2], cnt[2];
    uint8_t *wr[2];                                        /* run r's weights */
    uint8_t nr = 1;
    if (slot + 1 >= mk) {
        s0[0] = slot + 1 - mk;
        cnt[0] = mk;
    } else {
        cnt[0] = mk - 1 - slot;
        s0[0] = CLM_CTX - cnt[0];
        s0[1] = 0;
        cnt[1] = slot + 1;
        nr = 2;
    }
    int32_t ch = 32L * 128 * 128;
    for (uint8_t d = 0; d < CLM_HEAD_DIM; d++) {
        ch -= 128 * (int32_t)q[d];
    }
    PROF(m, CLM_PH_SCORES);
    /* Scores last key first, as for one run of mk keys; run 0 goes first (the scratch entry
     * below its last score is run 1's). Run r's weights go to wr[r] = its part of ew + (s0 & 7):
     * its dense P.V window starts at slot s0 & ~7 and is zero-padded to whole groups of 8 keys,
     * so it reads cache rows inside [0, cap) only. */
    clm_i24 mx = 0;
    clm_i24 *sc = m->sc + mk;
    uint8_t *w = m->ew;
    for (uint8_t r = 0; r < nr; r++) {
        unsigned d0 = r ? slot : mk - 1;                   /* distance of the run's first key */
        sc -= cnt[r];
        clm_i24 s = att_scores(q, ch - ((int32_t)d0 << es), (int32_t)1 << es,
                               K + s0[r] * CLM_KROW, (uint16_t)cnt[r], sc);
        mx = (r == 0 || s > mx) ? s : mx;
        uint8_t f = (uint8_t)s0[r] & 7;
        wr[r] = w + f;
        w += (f + cnt[r] + 7) & ~7u;
    }
    PROF(m, CLM_PH_SOFTMAX);
    uint32_t z = 0;
    sc = m->sc + mk;
    for (uint8_t r = 0; r < nr; r++) {
        sc -= cnt[r];
        z += att_softmax(L, sc, (uint16_t)cnt[r], mx, wr[r]);
    }
    PROF(m, CLM_PH_PV);
#if defined(__ez80__)
    unsigned *au = (unsigned *)(void *)m->y1;   /* [2][HEAD_DIM]; y1 is the FFN's, free here */
    unsigned nk = CV_CAP + 1;
    if (mk >= CV_MIN_KEYS) {                     /* compact both runs into one list */
        uint8_t *cv = (uint8_t *)m->y, *ce = (uint8_t *)m->sc;   /* both free here */
        nk = 0;
        for (uint8_t r = 0; r < nr && nk <= CV_CAP; r++) {
            nk += clm_a_compact(V + s0[r] * CLM_HEAD_DIM, wr[r], cnt[r], cv + nk * CLM_HEAD_DIM,
                                ce + nk, CV_CAP - nk);
        }
        if (nk <= CV_CAP) {
            for (unsigned j = nk; j & 7; j++) {
                ce[j] = 0;                      /* clm_a_pv works in groups of 8 keys */
            }
            clm_a_pv(cv, ce, nk, au);
        }
    }
    if (nk > CV_CAP) {
        for (uint8_t r = 0; r < nr; r++) {
            uint8_t f = (uint8_t)s0[r] & 7;
            unsigned span = f + cnt[r];
            for (uint8_t j = 0; j < f; j++) {
                wr[r][-1 - j] = 0;
            }
            for (unsigned j = span; j & 7; j++) {
                wr[r][j - f] = 0;
            }
            clm_a_pv(V + (s0[r] - f) * CLM_HEAD_DIM, wr[r] - f, span, au + r * CLM_HEAD_DIM);
        }
        for (uint8_t d = 0; nr == 2 && d < CLM_HEAD_DIM; d++) {
            au[d] += au[CLM_HEAD_DIM + d];      /* < 255^3 < 2^24 */
        }
    }
    uint8_t zb = clm_bitlen(z);
    uint32_t rz = (((uint32_t)1 << (zb + 14)) + (z >> 1)) / z;
    clm_a_havg(au, z, (unsigned)(zb - 8), (unsigned)rz, m->v32 + h * CLM_HEAD_DIM);
#else
    uint32_t au[2][CLM_HEAD_DIM];
    for (uint8_t r = 0; r < nr; r++) {
        att_pv(V + s0[r] * CLM_HEAD_DIM, wr[r], (uint16_t)cnt[r], au[r]);
    }
    for (uint8_t d = 0; nr == 2 && d < CLM_HEAD_DIM; d++) {
        au[0][d] += au[1][d];
    }
    att_havg(au[0], z, m->v32 + h * CLM_HEAD_DIM);
#endif
}

static void attention(clm_t *m, uint8_t l, uint16_t pos) {
    clm_layer *L = &m->L[l];
    uint16_t cap = m->mem.ctx_cap;
#if defined(CLM_KV4)
    uint8_t *K = m->kv4_scratch;              /* the decoded window: attention reads it */
    uint8_t *V = K + (uint32_t)cap * CLM_KROW;
#else
    uint8_t *K = m->mem.kc + (uint32_t)l * cap * CLM_KROW;
    uint8_t *V = m->mem.vc + (uint32_t)l * cap * CLM_HEAD_DIM;
#endif
    /* ring (format 2, full cache): this token's slot and the n keys held, in slot order */
    uint16_t slot = m->slide ? (uint16_t)(pos & (CLM_CTX - 1)) : pos;
    uint16_t n = pos < cap ? (uint16_t)(pos + 1) : cap;
    PROF(m, CLM_PH_NORM);
    clm_pf sx = clm_norm_quant(m->r, CLM_DIM, L->attn_g, L->attn_eg, m->sqrtd, m->v32, m->x8,
                               m->sc);
    PROF(m, CLM_PH_BUILD);
    tmv_d_build(m->x8, m->mem.tbl);
    rows_q8(m, &L->wq, CLM_QDIM, sx, m->q8);
    rows_q8(m, &L->wk, CLM_HEAD_DIM, sx, (int8_t *)(K + slot * CLM_KROW));
    rows_q8(m, &L->wv, CLM_HEAD_DIM, sx, (int8_t *)(V + slot * CLM_HEAD_DIM));
#if defined(CLM_KV4)
    PROF(m, CLM_PH_KV4);
    kv4_store(m, l, slot, K + slot * CLM_KROW, V + slot * CLM_HEAD_DIM);
    kv4_unpack(m, l, n, 0);
    PROF(m, CLM_PH_SCORES);
#else
    PROF(m, CLM_PH_SCORES);
    bias_row(K + slot * CLM_KROW, 1);
    bias_row(V + slot * CLM_HEAD_DIM, 0);
#endif
    uint8_t *qu = (uint8_t *)m->q8;
    for (clm_qidx i = 0; i < CLM_QDIM; i++) {
        qu[i] ^= 0x80;                          /* q + 128 */
    }
    for (uint8_t h = 0; h < CLM_HEADS; h++) {
        uint16_t mk = n;                        /* keys head h attends to */
        if (m->horizons && m->alibi) {
            uint8_t e = (uint8_t)(CLM_HEADS - 1 - h);
            uint16_t hz = (uint16_t)((m->lut_max + (1u << e) - 1) >> e);
            mk = hz < n ? hz : n;
        }
        if (mk < n) {
            attention_head_hz(m, L, K, V, qu + h * CLM_HEAD_DIM, mk, slot, h);
        } else {
            attention_head(m, L, K, V, qu + h * CLM_HEAD_DIM, n, slot, h);
        }
    }
    PROF(m, CLM_PH_QUANT);
    clm_pf unit = {L->sv.m, (int16_t)(L->sv.e - 8)};
    clm_pf so = clm_quant_vec(m->v32, CLM_QDIM, unit, m->x8);
    if (so.m) {
        PROF(m, CLM_PH_BUILD);
        tmv_d_build(m->x8, m->mem.tbl);
        PROF(m, CLM_PH_ROWS);
        tmv_d_rows(L->wo.w, m->y, CLM_DIM, m->mem.tbl);
        residual_add(m, &L->wo, so);
    }
}

#if !defined(__ez80__)
static int16_t silu_q8(const clm_t *m, int16_t y1) {
    if (y1 < -SILU_EDGE) {
        return 0;
    }
    if (y1 >= SILU_EDGE) {
        return y1;
    }
    int16_t idx = (int16_t)(((y1 + 2) >> 2) + SILU_HALF);   /* rsr(y1, 2) + 512 in [0, 1024] */
    return RD16S(m->silu + 2 * idx);
}
#endif

static void ffn(clm_t *m, uint8_t l) {
    clm_layer *L = &m->L[l];
    PROF(m, CLM_PH_NORM);
    clm_pf sx = clm_norm_quant(m->r, CLM_DIM, L->ffn_g, L->ffn_eg, m->sqrtd, m->v32, m->x8, m->sc);
    if (!sx.m) {
        return;                                   /* zero input: the block adds nothing */
    }
    PROF(m, CLM_PH_BUILD);
    tmv_d_build(m->x8, m->mem.tbl);
    PROF(m, CLM_PH_ROWS);
    tmv_d_rows(L->w1.w, m->y, CLM_HIDDEN, m->mem.tbl);
    PROF(m, CLM_PH_SCALE);
    clm_scale_rows(&L->w1, m->y, CLM_HIDDEN, sx, 8, 1, m->y1, (uint8_t *)m->v32);
    PROF(m, CLM_PH_ROWS);
    tmv_d_rows(L->w3.w, m->y, CLM_HIDDEN, m->mem.tbl);
    PROF(m, CLM_PH_FFN);
    uint32_t peak = clm_t3(m->y, &L->w3, L->e3, CLM_HIDDEN, m->y);   /* t3 in place, int24 */
    int8_t c3 = (int8_t)(clm_bitlen(peak) - 15);
#if defined(__ez80__)
    clm_a_silu(m->y1, CLM_HIDDEN, m->silu);                          /* in place */
#else
    for (uint16_t i = 0; i < CLM_HIDDEN; i++) {
        m->y1[i] = silu_q8(m, m->y1[i]);                             /* in place */
    }
#endif
    /* h = silu(y1) * normalize16(t3), |h| < 2^30 */
    clm_mulnorm(m->y, CLM_HIDDEN, c3, (const uint8_t *)m->y1, m->v32, 0, 0);
    PROF(m, CLM_PH_QUANT);
    clm_pf unit = {(uint16_t)(sx.m << 7), (int16_t)(sx.e - 7 + L->e3 + c3 - 8)};
    clm_pf sh = clm_quant_vec(m->v32, CLM_HIDDEN, unit, m->h8);
    if (sh.m) {
        PROF(m, CLM_PH_BUILD);
        tmv_h_build(m->h8, m->mem.tbl);
        PROF(m, CLM_PH_ROWS);
        tmv_h_rows(L->w2.w, m->y, CLM_DIM, m->mem.tbl);
        residual_add(m, &L->w2, sh);
    }
}

static void tap(clm_t *m, uint8_t i) {
    if (m->tap) {
        for (uint16_t j = 0; j < CLM_DIM; j++) {
            m->tap[i * CLM_DIM + j] = m->r[j];
        }
    }
}

void clm_forward(clm_t *m, uint16_t tok, uint16_t pos) {
    if (!m->tokens || tok >= m->vocab || (!m->slide && pos >= m->mem.ctx_cap)) {
        return;                                   /* not loaded, or out of range */
    }
#if defined(CLM_KV4)
    if (!m->kv4_ok) {
        return;                                   /* no CLMKV4 accepted since the last load */
    }
#endif
    PROF(m, CLM_PH_EMBED);
    embed(m, tok, pos);
    for (uint8_t l = 0; l < CLM_LAYERS; l++) {
        attention(m, l, pos);
        tap(m, (uint8_t)(2 * l));
        ffn(m, l);
        tap(m, (uint8_t)(2 * l + 1));
    }
    PROF(m, CLM_PH_NORM);
    clm_pf sx = clm_norm_quant(m->r, CLM_DIM, m->norm_g, m->norm_eg, m->sqrtd, m->v32, m->x8, m->sc);
    if (sx.m) {
        PROF(m, CLM_PH_BUILD);
        tmv_d_build(m->x8, m->mem.tbl);
        PROF(m, CLM_PH_ROWS);
        tmv_d_rows(m->out.w, m->y, m->vocab, m->mem.tbl);
    }
    PROF(m, CLM_PH_SCALE);
    if (sx.m) {
        clm_scale_rows(&m->out, m->y, m->vocab, sx, 8, 2, m->logits, (uint8_t *)m->v32);
    } else {
        for (uint16_t i = 0; i < m->vocab; i++) {
            m->logits[i] = 0;
        }
    }
    PROF(m, CLM_PH_END);
}
