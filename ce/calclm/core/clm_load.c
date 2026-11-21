/* Load a CalcLM export from its AppVar payloads: check every AppVar header, the blob header
 * (magic, format, dims, scalar ranges), walk the tensors in blob order (INTREF.md section 6,
 * model/export.py build_tensors) and resolve each one to a pointer inside one AppVar.
 * Format 1 has a position table; format 2 (ALiBi) has none and one more header field. */
#include "clm_int.h"

#if CLM_HEADS * CLM_HEAD_DIM != CLM_DIM
#error "wo takes n_heads * head_dim inputs; the core reuses the d-input kernel for it"
#endif

#define HDR_FIELDS 16
#define LAYER_FIELDS 6
#define TAIL_FIELDS 1                 /* format 2: pos_scheme */
#define HDR_N (HDR_FIELDS + LAYER_FIELDS * CLM_LAYERS)
#define MAGIC 0x304D4C43UL            /* "CLM0" little-endian */
#define ALIBI_BIAS_MAX (1L << 22)     /* lut_max << sh: ALiBi scores stay inside int24 */

static int32_t rd32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
                     (uint32_t)p[3] << 24);
}

static int in_range(int32_t v, int32_t lo, int32_t hi) {
    return v >= lo && v <= hi;
}

/* tensor walk state: the running blob offset, and pointer resolution (or only sizing) */
typedef struct {
    clm_t *m;
    uint32_t off;
    int resolve;
    int err, bad;
} walk_t;

static int seg_of(const clm_t *m, uint32_t off) {
    for (uint8_t i = 0; i < m->n_seg; i++) {
        if (off >= m->seg[i].start && off < m->seg[i].end) {
            return i;
        }
    }
    return -1;
}

/* next tensor of n bytes: its pointer if it lies inside one AppVar */
static const uint8_t *take(walk_t *w, uint32_t n) {
    uint32_t off = w->off;
    w->off += n;
    if (!w->resolve || w->err) {
        return 0;
    }
    int i = seg_of(w->m, off);
    if (i < 0 || off + n > w->m->seg[i].end) {
        w->err = CLM_E_SPLIT;
        w->bad = i;
        return 0;
    }
    return w->m->seg[i].data + (off - w->m->seg[i].start);
}

/* a tensor of `rows` rows that may be split between rows (embeddings): returns its offset */
static uint32_t take_rows(walk_t *w, uint32_t rows, uint32_t row_bytes) {
    uint32_t off = w->off, end = off + rows * row_bytes;
    w->off = end;
    if (w->resolve && !w->err) {
        for (uint8_t i = 0; i < w->m->n_seg; i++) {
            uint32_t b = w->m->seg[i].end;     /* every AppVar boundary inside must be a row edge */
            if (b > off && b < end && (b - off) % row_bytes) {
                w->err = CLM_E_SPLIT;
                w->bad = i;
            }
        }
    }
    return off;
}

static void take_lin(walk_t *w, clm_lin *l, uint16_t rows, uint16_t groups) {
    l->w = take(w, (uint32_t)rows * groups);
    l->m = take(w, rows);
    l->e = (const int8_t *)take(w, rows);
}

static void walk(walk_t *w, const int32_t *hdr, uint32_t hdr_bytes) {
    clm_t *m = w->m;
    uint16_t V = m->vocab;
    w->off = hdr_bytes;
    for (uint8_t l = 0; l < CLM_LAYERS; l++) {
        clm_layer *L = &m->L[l];
        L->attn_g = take(w, 2 * CLM_DIM);
        L->lut = take(w, (uint32_t)hdr[HDR_FIELDS + LAYER_FIELDS * l + 3]);
        take_lin(w, &L->wq, CLM_QDIM, CLM_G_DIM);
        take_lin(w, &L->wk, CLM_HEAD_DIM, CLM_G_DIM);
        take_lin(w, &L->wv, CLM_HEAD_DIM, CLM_G_DIM);
        take_lin(w, &L->wo, CLM_DIM, CLM_G_DIM);
        L->ffn_g = take(w, 2 * CLM_DIM);
        take_lin(w, &L->w1, CLM_HIDDEN, CLM_G_DIM);
        take_lin(w, &L->w3, CLM_HIDDEN, CLM_G_DIM);
        take_lin(w, &L->w2, CLM_DIM, CLM_G_HID);
    }
    m->norm_g = take(w, 2 * CLM_DIM);
    take_lin(w, &m->out, V, CLM_G_DIM);
    m->tok_off = take_rows(w, V, CLM_DIM);
    m->tok_m = take(w, V);
    m->tok_e = (const int8_t *)take(w, V);
    if (!m->alibi) {
        m->pos_off = take_rows(w, CLM_CTX, CLM_DIM);
        m->pos_m = take(w, CLM_CTX);
        m->pos_e = (const int8_t *)take(w, CLM_CTX);
    }
    m->silu = take(w, 2 * (uint32_t)hdr[10]);
    m->expw = take(w, 2 * (uint32_t)hdr[11]);
}

static int check_appvars(clm_t *m, const uint8_t *const *data, const uint32_t *size,
                         uint8_t count, int *bad) {
    if (count == 0 || count > CLM_MAX_APPVARS) {
        return CLM_E_COUNT;
    }
    for (uint8_t i = 0; i < count; i++) {
        const uint8_t *d = data[i];
        *bad = i;
        if (!d || size[i] < CLM_APPVAR_HEADER || d[0] != 'C' || d[1] != 'L' || d[2] != i) {
            return CLM_E_APPVAR;
        }
        if (d[3] != count) {
            /* CLM00 says how many AppVars the export has */
            return i == 0 ? CLM_E_COUNT : CLM_E_APPVAR;
        }
        uint32_t start = (uint32_t)d[4] | (uint32_t)d[5] << 8 | (uint32_t)d[6] << 16;
        if (start != (i ? m->seg[i - 1].end : 0)) {
            return CLM_E_GAP;
        }
        m->seg[i].start = start;
        m->seg[i].end = start + size[i] - CLM_APPVAR_HEADER;
        m->seg[i].data = d + CLM_APPVAR_HEADER;
    }
    m->n_seg = count;
    *bad = -1;
    return CLM_OK;
}

/* the format from the first two fields: 1 or 2, or an error */
static int check_format(const int32_t *h) {
    if ((uint32_t)h[0] != MAGIC) {
        return CLM_E_MAGIC;
    }
    if (h[1] != CLM_FORMAT_LEARNED && h[1] != CLM_FORMAT_RELATIVE) {
        return CLM_E_VERSION;
    }
    return (int)h[1];
}

static int check_header(clm_t *m, const int32_t *h) {
    m->alibi = h[1] == CLM_FORMAT_RELATIVE;
    if (m->alibi && h[HDR_N] != CLM_POS_ALIBI) {
        return CLM_E_VERSION;                    /* NoPE or an unknown scheme */
    }
    if (h[2] != CLM_DIM || h[3] != CLM_LAYERS || h[4] != CLM_HEADS || h[5] != CLM_HEAD_DIM ||
        h[6] != CLM_HIDDEN || h[8] != CLM_CTX || !in_range(h[7], 4 + CLM_MAX_TOPK, CLM_VOCAB_MAX)) {
        return CLM_E_SHAPE;
    }
    /* F, silu_len (the SiLU index assumes 1025 entries), expw_len, norm_eg, sqrtd */
    if (!in_range(h[9], 0, 22) || h[10] != 1025 || !in_range(h[11], 1, 4096) ||
        !in_range(h[12], -64, 64) || !in_range(h[13], 1, 4096) || !in_range(h[14], 16384, 32767) ||
        !in_range(h[15], -64, 64)) {
        return CLM_E_RANGE;
    }
    for (uint8_t l = 0; l < CLM_LAYERS; l++) {
        const int32_t *f = h + HDR_FIELDS + LAYER_FIELDS * l;  /* eg, eg, sh, lut_len, sv_m, sv_e */
        if (!in_range(f[0], -64, 64) || !in_range(f[1], -64, 64) || !in_range(f[2], 0, 24) ||
            !in_range(f[3], 1, 4096) || !in_range(f[4], 16384, 32767) || !in_range(f[5], -64, 64)) {
            return CLM_E_RANGE;
        }
        if (m->alibi && (f[2] > 22 || h[13] > (ALIBI_BIAS_MAX >> f[2]))) {
            return CLM_E_RANGE;                  /* the largest ALiBi penalty, in score units */
        }
        clm_layer *L = &m->L[l];
        L->attn_eg = (int8_t)f[0];
        L->ffn_eg = (int8_t)f[1];
        L->sh = (int8_t)f[2];
        L->lut_len = (int16_t)f[3];
        L->sv.m = (uint16_t)f[4];
        L->sv.e = (int16_t)f[5];
    }
    m->vocab = (uint16_t)h[7];
    m->lut_max = (uint16_t)h[13];
    m->F = (int8_t)h[9];
    m->expw_len = (int16_t)h[11];
    m->norm_eg = (int8_t)h[12];
    m->sqrtd.m = (uint16_t)h[14];
    m->sqrtd.e = (int16_t)h[15];
    return CLM_OK;
}

/* PF7 row mantissas must be 64..127 (scale_out's products stay below 2^31) */
static int check_rows(const clm_lin *w, uint16_t rows) {
    for (uint16_t i = 0; i < rows; i++) {
        if (w->m[i] < 64 || w->m[i] > 127) {
            return CLM_E_RANGE;
        }
    }
    return CLM_OK;
}

/* Tables the arithmetic relies on: every PF7 row mantissa in 64..127, each softmax LUT
 * starting at 255 (so Z >= 255 and the head average stays in range), EXPW[0] > 0 (so the
 * sampling total is positive). */
static int check_tables(const clm_t *m) {
    int rc = CLM_OK;
    for (uint8_t l = 0; l < CLM_LAYERS && rc == CLM_OK; l++) {
        const clm_layer *L = &m->L[l];
        if (L->lut[0] != 255) {
            return CLM_E_RANGE;
        }
        rc |= check_rows(&L->wq, CLM_QDIM) | check_rows(&L->wk, CLM_HEAD_DIM) |
              check_rows(&L->wv, CLM_HEAD_DIM) | check_rows(&L->wo, CLM_DIM) |
              check_rows(&L->w1, CLM_HIDDEN) | check_rows(&L->w3, CLM_HIDDEN) |
              check_rows(&L->w2, CLM_DIM);
    }
    if (rc != CLM_OK || check_rows(&m->out, m->vocab) != CLM_OK || RD16U(m->expw) == 0) {
        return CLM_E_RANGE;
    }
    return CLM_OK;
}

/* embeddings are added as rsr(e8 * em, -(ee + F)); a left shift above 8 could leave int32 */
static int check_embeddings(const clm_t *m) {
    for (uint16_t t = 0; t < m->vocab; t++) {
        if (m->tok_e[t] + m->F > 8) {
            return CLM_E_RANGE;
        }
    }
    for (uint16_t p = 0; !m->alibi && p < CLM_CTX; p++) {
        if (m->pos_e[p] + m->F > 8) {
            return CLM_E_RANGE;
        }
    }
    return CLM_OK;
}

static void lin_range(clm_lin *w, uint16_t rows) {
    int8_t lo = 127, hi = -128;
    for (uint16_t i = 0; i < rows; i++) {
        lo = w->e[i] < lo ? w->e[i] : lo;
        hi = w->e[i] > hi ? w->e[i] : hi;
    }
    w->emin = lo;
    w->emax = hi;
}

static void place_activations(clm_t *m, uint8_t *p) {
    /* 4-byte items first (host alignment), then 2-byte, then bytes */
    m->r = (clm_i24 *)(void *)p;
    p += 4 * CLM_DIM;
    m->v32 = (int32_t *)(void *)p;
    p += 4 * CLM_HIDDEN;
    m->y = (clm_i24 *)(void *)p;
    p += sizeof(clm_i24) * (CLM_VOCAB_MAX > CLM_HIDDEN ? CLM_VOCAB_MAX : CLM_HIDDEN);
    p += sizeof(clm_i24);         /* clm_a_scores pushes one scratch entry just below sc */
    m->sc = (clm_i24 *)(void *)p;
    p += sizeof(clm_i24) * CLM_CTX;
    m->logits = (clm_i24 *)(void *)p;
    p += sizeof(clm_i24) * CLM_VOCAB_MAX;
    m->y1 = (int16_t *)(void *)p;
    p += 2 * CLM_HIDDEN;
    m->cand = (uint16_t *)(void *)p;
    p += 2 * CLM_MAX_TOPK;
    m->wts = (uint16_t *)(void *)p;
    p += 2 * CLM_MAX_TOPK;
    m->x8 = (int8_t *)p;
    p += 5 * CLM_G_DIM;
    m->h8 = (int8_t *)p;
    p += 5 * CLM_G_HID;
    m->q8 = (int8_t *)p;
    p += CLM_QDIM;
    m->ew = p;
    for (uint16_t i = CLM_DIM; i < 5 * CLM_G_DIM; i++) {
        m->x8[i] = 0;             /* kernel input padding stays zero */
    }
    for (uint16_t i = CLM_HIDDEN; i < 5 * CLM_G_HID; i++) {
        m->h8[i] = 0;
    }
}

int clm_load(clm_t *m, const uint8_t *const *data, const uint32_t *size, uint8_t count,
             const clm_mem *mem, int *bad) {
    int dummy, rc;
    int32_t hdr[HDR_N + TAIL_FIELDS];
    bad = bad ? bad : &dummy;
    *bad = -1;
    m->done = 1;
    m->n_tok = 0;
    m->tokens = 0;                            /* not loaded until everything checks out */
    m->tap = 0;
    m->prof = 0;
    m->n_seg = 0;
    m->masks = 0;
    m->horizons = 0;
    m->ld2 = 0;
    m->ld = 0;
#if defined(CLM_KV4)
    m->kv4_ok = 0;                            /* tables are accepted per load (clm_kv4_load) */
#endif
    m->cap = 0;
    if (!mem->kc || !mem->vc || !mem->act || !mem->tbl || ((uintptr_t)mem->tbl & 0xFFFF) ||
        mem->ctx_cap == 0 || mem->ctx_cap > CLM_CTX || (mem->ctx_cap & 7) || !mem->tok ||
        mem->tok_cap < 2) {
        return CLM_E_MEM;
    }
    if ((rc = check_appvars(m, data, size, count, bad)) != CLM_OK) {
        return rc;
    }
    if (m->seg[0].end < 8) {
        *bad = 0;
        return CLM_E_SPLIT;
    }
    hdr[0] = rd32(m->seg[0].data);
    hdr[1] = rd32(m->seg[0].data + 4);
    if ((rc = check_format(hdr)) < 0) {
        return rc;
    }
    uint8_t n_hdr = (uint8_t)(rc == CLM_FORMAT_RELATIVE ? HDR_N + TAIL_FIELDS : HDR_N);
    if (m->seg[0].end < 4UL * n_hdr) {
        *bad = 0;
        return CLM_E_SPLIT;
    }
    for (uint8_t i = 2; i < n_hdr; i++) {
        hdr[i] = rd32(m->seg[0].data + 4 * i);
    }
    if ((rc = check_header(m, hdr)) != CLM_OK) {
        return rc;
    }
    walk_t w = {m, 0, 0, CLM_OK, -1};
    uint32_t hdr_bytes = 4UL * n_hdr;
    walk(&w, hdr, hdr_bytes);                 /* sizes only */
    if (w.off != m->seg[count - 1].end) {
        return CLM_E_SIZE;
    }
    w.resolve = 1;
    walk(&w, hdr, hdr_bytes);
    if (w.err) {
        *bad = w.bad;
        return w.err;
    }
    if ((rc = check_embeddings(m)) != CLM_OK || (rc = check_tables(m)) != CLM_OK) {
        return rc;
    }
    for (uint8_t l = 0; l < CLM_LAYERS; l++) {
        clm_layer *L = &m->L[l];
        lin_range(&L->wq, CLM_QDIM);
        lin_range(&L->wk, CLM_HEAD_DIM);
        lin_range(&L->wv, CLM_HEAD_DIM);
        lin_range(&L->wo, CLM_DIM);
        lin_range(&L->w1, CLM_HIDDEN);
        lin_range(&L->w3, CLM_HIDDEN);
        lin_range(&L->w2, CLM_DIM);
    }
    lin_range(&m->out, m->vocab);
    for (uint8_t l = 0; l < CLM_LAYERS; l++) {
        clm_layer *L = &m->L[l];
        int8_t e3 = -128;
        for (uint16_t i = 0; i < CLM_HIDDEN; i++) {
            e3 = L->w3.e[i] > e3 ? L->w3.e[i] : e3;
        }
        L->e3 = e3;
    }
    m->mem = *mem;
    m->slide = m->alibi && mem->ctx_cap == CLM_CTX;
    place_activations(m, mem->act);
    m->tokens = mem->tok;
    return CLM_OK;
}
