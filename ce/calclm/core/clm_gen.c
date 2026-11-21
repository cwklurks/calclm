/* Sampling and the generation loop: INTREF.md sections 5 and 13 (model/intref.py choose,
 * generate; the masks themselves are in clm_mask.c). */
#include "clm_int.h"

#define SEED0 2463534242UL

static uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/* First k allowed ids ranked by logit (descending), ties by id (ascending), into m->cand.
 * An id enters only when its logit beats the current k-th, so equal logits keep the smaller
 * id first; with masks, banned ids never enter (fewer than k are left if fewer are allowed).
 * The candidates' logits sit in m->sc (free while sampling). */
__attribute__((always_inline))   /* one copy per call site: the unmasked one has no ban test */
static inline uint8_t top_k_scan(clm_t *m, uint8_t k, const uint8_t *ban) {
    uint16_t *cand = m->cand;
    clm_i24 *val = m->sc;
    const clm_i24 *lg = m->logits;
    unsigned n = 0, vocab = m->vocab;
    clm_i24 thr = 0;
    const clm_i24 *lp = lg + CLM_EOS;               /* walk a pointer: no index multiply */
    for (unsigned v = CLM_EOS; v < vocab; v++, lp++) {
        clm_i24 x = *lp;
        if (v == CLM_UNK || (n == k && x <= thr) || (ban && (ban[v >> 3] >> (v & 7)) & 1)) {
            continue;
        }
        unsigned i = n < k ? n++ : (unsigned)k - 1;
        while (i > 0 && val[i - 1] < x) {
            cand[i] = cand[i - 1];
            val[i] = val[i - 1];
            i--;
        }
        cand[i] = (uint16_t)v;
        val[i] = x;
        thr = val[n - 1];
    }
    return (uint8_t)n;
}

static uint8_t top_k(clm_t *m, uint8_t k) {
    return m->masks ? top_k_scan(m, k, m->ban) : top_k_scan(m, k, 0);
}

static uint16_t choose(clm_t *m) {
    if (m->greedy) {
        top_k(m, 1);
        return m->cand[0];
    }
    uint8_t n = top_k(m, m->top_k);
    int32_t top = m->logits[m->cand[0]];
    uint32_t total = 0;
    for (uint8_t j = 0; j < n; j++) {
        int32_t d = top - m->logits[m->cand[j]];
        uint32_t idx = (((uint32_t)(d > 32767 ? 32767 : d) * m->inv_t) + 32768u) >> 16;
        if (idx > (uint32_t)m->expw_len - 1) {
            idx = (uint32_t)m->expw_len - 1;
        }
        m->wts[j] = RD16U(m->expw + 2 * idx);
        total += m->wts[j];
    }
    m->state = xorshift32(m->state);
    uint32_t target = m->state % total;           /* total >= EXPW[0] = 65535 */
    uint32_t acc = 0;
    for (uint8_t j = 0; j < n; j++) {
        acc += m->wts[j];
        if (acc > target) {
            return m->cand[j];
        }
    }
    return m->cand[n - 1];                        /* unreachable */
}

int clm_reset(clm_t *m, uint32_t seed, uint16_t temperature_q8, uint8_t top_k_, uint8_t greedy,
              uint16_t cap, uint8_t masks) {
    if (!m->tokens || cap == 1 || masks > CLM_MASKS_SAFETY ||
        (masks && m->vocab != clm_mask_vocab) ||
        (m->ld2 && (!m->ld || !masks || greedy || cap > CLM_DEFAULT_CAP))) {
        return CLM_E_ARG;
    }
    if (!greedy && (temperature_q8 < 33 || top_k_ < 1 || top_k_ > CLM_MAX_TOPK)) {
        return CLM_E_ARG;
    }
    uint16_t limit = m->mem.tok_cap;              /* what the memory holds */
    if (!m->slide && m->mem.ctx_cap < limit) {
        limit = m->mem.ctx_cap;                   /* no sliding: the K/V cache is the limit */
    }
    cap = cap ? cap : CLM_DEFAULT_CAP;
    m->cap = cap < limit ? cap : limit;
    m->greedy = greedy ? 1 : 0;
    m->top_k = top_k_;
    m->inv_t = greedy ? 0 : (uint16_t)((32UL << 16) / temperature_q8);
    m->state = seed ? seed : SEED0;
    m->masks = masks;
    m->tokens[0] = CLM_BOS;
    m->n_tok = 1;
    m->n_prompt = 1;
    m->done = 0;
    if (m->masks) {
        clm_track(m, CLM_BOS, 1);
    }
    return CLM_OK;
}

int clm_feed(clm_t *m, uint16_t tok) {
    if (!m->tokens || m->done || m->n_tok != m->n_prompt || tok < 4 || tok >= m->vocab ||
        m->n_tok + 1 >= m->cap) {
        return CLM_E_ARG;
    }
    clm_forward(m, m->tokens[m->n_tok - 1], (uint16_t)(m->n_tok - 1));
    m->tokens[m->n_tok++] = tok;
    m->n_prompt = m->n_tok;
    if (m->masks) {
        if (m->prof) {
            m->prof(CLM_PH_MASK);
        }
        clm_track(m, tok, 0);
    }
    if (m->prof) {
        m->prof(CLM_PH_END);
    }
    return CLM_OK;
}

uint32_t clm_mix_seed(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6BUL;
    h ^= h >> 13;
    h *= 0xC2B2AE35UL;
    h ^= h >> 16;
    return h;
}

int clm_step(clm_t *m) {
    if (m->done || m->n_tok >= m->cap) {
        m->done = 1;
        return CLM_DONE;
    }
    clm_forward(m, m->tokens[m->n_tok - 1], (uint16_t)(m->n_tok - 1));
    if (m->masks) {
        if (m->prof) {
            m->prof(CLM_PH_MASK);
        }
        clm_mask(m);
    }
    if (m->prof) {
        m->prof(CLM_PH_SAMPLE);
    }
    int forced = m->ld2 ? clm_ld2_forced(m) : -1;
    uint16_t tok;
    if (forced >= 0) {
        tok = (uint16_t)forced;
    } else if (m->ld2 && clm_ld2_stop(m)) {
        tok = CLM_EOS;
    } else {
        tok = choose(m);
        if (m->ld2 && tok == CLM_EOS && m->trk.n_unres && m->cap - m->n_tok > 4) {
            if (m->feat.cat[0] == TRK_CTX_START && !m->trk.pend_kind && !m->trk.if_state) {
                tok = 55;                        /* Lbl */
            } else {
                tok = 4;                         /* newline, then queued Lbl */
                m->ld->queued = 1;
            }
        }
    }
    m->tokens[m->n_tok++] = tok;
    if (tok == CLM_EOS) {
        m->done = 1;
    } else if (m->masks) {
        if (m->prof) {
            m->prof(CLM_PH_MASK);
        }
        clm_track(m, tok, 0);                     /* the tracker reads every token fed */
    }
    if (m->prof) {
        m->prof(CLM_PH_END);
    }
    return tok;
}
