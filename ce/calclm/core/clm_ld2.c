/* model/linker.py LD2: XT exit ledger, Row, rules 1-7. All name sets use the
 * tracker's exact 37 + 37*37 label numbering. No allocation or OS/library calls. */
#include "clm_int.h"
#include "clm_mask_ids.h"
#include "trk_tables.h"

#define LABELS 1406
#define LBL 55
#define STOP 138
#define RET 150
#define COLON 15
#define NK_LBL 1
#define NK_GOTO 2
#define NK_MENU 3

static uint8_t get(const uint8_t *a, uint16_t i) {
    return (a[i >> 3] >> (i & 7)) & 1;
}

static void put(uint8_t *a, uint16_t i, uint8_t v) {
    uint8_t bit = (uint8_t)(1u << (i & 7));
    a[i >> 3] = (uint8_t)((a[i >> 3] & (uint8_t)~bit) | (v ? bit : 0));
}

static uint8_t role(uint16_t t) { return t < TRK_VOCAB ? tracker_classes()[t] : CL_OTHER; }

static uint16_t name_id(const uint8_t *name, uint8_t n) {
    return n == 1 ? name[0] : (uint16_t)(37 + (uint16_t)name[0] * 37 + name[1]);
}

static int row_name(const uint16_t *t, uint16_t n) {
    if (!n || role(t[0]) >= 37) {
        return -1;
    }
    return n == 1 || role(t[1]) >= 37 ? role(t[0])
        : 37 + (int)role(t[0]) * 37 + role(t[1]);
}

static void row_ref(clm_ld2 *l, int id) {
    if (id >= 0 && !get(l->row_defined, (uint16_t)id)) {
        put(l->row_pending, (uint16_t)id, 1);
    }
}

static void row_end(clm_ld2 *l) {
    uint16_t n = l->row_n, *t = l->row_toks;
    l->row_n = 0;
    if (!n) {
        return;
    }
    uint8_t r = role(t[0]), body = 0;
    if (l->row_if) {
        l->row_if = 0;
        if (r == CL_THEN) {
            l->row_depth++;
            l->row_transfer = 0;
            return;
        }
        body = 1;
    }
    uint16_t depth = l->row_depth;
    if (r == CL_END && !depth && l->row_transfer && !body) {
        return;
    }
    if (r == CL_IF) {
        l->row_if = 1;
    } else if (r == CL_FOR || r == CL_WHILE || r == CL_REPEAT) {
        l->row_depth++;
    } else if (r == CL_END) {
        if (l->row_depth) {
            l->row_depth--;
        }
    } else if (r == CL_LBL) {
        int id = row_name(t + 1, n - 1);
        if (id >= 0) {
            put(l->row_defined, (uint16_t)id, 1);
            put(l->row_pending, (uint16_t)id, 0);
        }
    } else if (r == CL_GOTO) {
        row_ref(l, row_name(t + 1, n - 1));
    } else if (r == CL_MENU) {
        uint16_t start = 1, d = 0, arg = 0;
        uint8_t ins = 0;
        for (uint16_t i = 1; i <= n; i++) {
            uint8_t c = i < n ? role(t[i]) : CL_OTHER;
            if (i == n || (!ins && ((c == CL_COMMA && !d) || (c == CL_RPAREN && !d)))) {
                if (arg >= 2 && !(arg & 1)) {
                    row_ref(l, row_name(t + start, i - start));
                }
                if (i == n || c == CL_RPAREN) {
                    break;
                }
                arg++;
                start = i + 1;
            } else if (ins) {
                if (c == CL_QUOTE) {
                    ins = 0;
                }
            } else if (c == CL_QUOTE) {
                ins = 1;
            } else if (c >= CL_OPEN_FIRST && c <= CL_OPEN_LAST) {
                d++;
            } else if (c == CL_RPAREN) {
                d--;
            }
        }
    }
    l->row_transfer = (r == CL_GOTO || r == CL_MENU || t[0] == STOP || t[0] == RET)
                      && !body && !depth;
}

static void row_step(clm_ld2 *l, uint16_t t) {
    uint8_t c = role(t);
    if (l->row_string) {
        if (c == CL_QUOTE || c == CL_STORE) {
            l->row_string = 0;
        } else if (c == CL_NEWLINE) {
            l->row_string = 0;
            row_end(l);
            return;
        }
    } else {
        if (c == CL_NEWLINE || c == CL_COLON) {
            row_end(l);
            return;
        }
        if (c == CL_QUOTE) {
            l->row_string = 1;
        }
    }
    l->row_toks[l->row_n++] = t;       /* clm_reset bounds the entire program to 1024 */
}

void clm_ld2_before_track(clm_t *m, uint16_t tok, uint8_t reset) {
    clm_ld2 *l = m->ld;
    const trk_state *s = &m->trk;
    if (reset) {
        uint8_t *p = l->exit_since;
        for (unsigned i = 0; i < sizeof(*l) - sizeof(l->E); i++) {
            p[i] = 0;
        }
        return;                         /* Row deliberately never sees BOS */
    }
    /* XT sees these tokens even inside strings. A completed Goto also exits all labels. */
    uint8_t c = role(tok);
    uint8_t finish = !(s->flags & 1) && s->name_len &&
        (c == CL_NEWLINE || c == CL_COLON || c == CL_EOS ||
         (s->name_kind == NK_MENU && (c == CL_COMMA || c == CL_RPAREN)));
    if (tok == 25 || tok == 137 || tok == 140 || tok == 88 || tok == 456 || tok == 438 ||
        tok == STOP || tok == RET || tok == 112 || tok == 122 || tok == 92 || tok == 157 ||
        (finish && s->name_kind == NK_GOTO)) {
        for (unsigned i = 0; i < TRK_LABEL_BYTES; i++) {
            l->exit_since[i] = 255;
        }
    }
    if (finish && s->name_kind == NK_LBL) {
        uint16_t id = name_id(s->name, s->name_len);
        if (!get(s->lbl_def, id)) {
            put(l->exit_since, id, 0);
        }
    }
    row_step(l, tok);
}

void clm_ld2_after_track(clm_t *m, uint16_t tok) {
    if (role(tok) == CL_GOTO && m->trk.name_kind == NK_GOTO && !m->trk.name_len) {
        m->ld->goto_seen = 1;
        m->ld->goto_if = m->trk.if_state;
        m->ld->goto_depth = m->trk.depth;
    }
}

/* Candidate names extending the tracker's prefix. Bitsets avoid sorting; only the union of
 * next characters and terminators matters, so reference string order cannot affect draws. */
static uint8_t prefix(clm_t *m, uint8_t defined) {
    const trk_state *s = &m->trk;
    uint8_t next[37] = {0}, end = 0, any = 0;
    for (uint16_t id = 0; id < LABELS; id++) {
        if (!get(s->lbl_unres, id) && !(defined && get(s->lbl_def, id))) {
            continue;
        }
        uint8_t n = id < 37 ? 1 : 2;
        uint8_t a = id < 37 ? (uint8_t)id : (uint8_t)((id - 37) / 37);
        uint8_t b = id < 37 ? 0 : (uint8_t)((id - 37) % 37);
        if (s->name_len > n || (s->name_len && s->name[0] != a) ||
            (s->name_len == 2 && s->name[1] != b)) {
            continue;
        }
        any = 1;
        if (s->name_len == n) {
            end = 1;
        } else {
            next[s->name_len ? b : a] = 1;
        }
    }
    if (any) {
        for (uint16_t v = 0; v < m->vocab; v++) {
            uint8_t c = role(v);
            if (!((c < 37 && next[c]) || (end && (c == CL_NEWLINE || c == CL_COLON)))) {
                put(m->ban, v, 1);
            }
        }
    }
    return any;
}

static void no_dup(clm_t *m, uint8_t exits, uint8_t *out) {
    const trk_state *s = &m->trk;
    if (!s->name_len) {
        return;
    }
    uint16_t id = name_id(s->name, s->name_len);
    if (get(s->lbl_def, id) && (!exits || !get(m->ld->exit_since, id))) {
        put(out, MASK_NEWLINE, 1);
        put(out, COLON, 1);
        put(out, CLM_EOS, 1);
    }
    if (s->name_len == 1) {
        for (uint8_t k = 0; k < 37; k++) {
            id = (uint16_t)(37 + (uint16_t)s->name[0] * 37 + k);
            if (get(s->lbl_def, id) && (!exits || !get(m->ld->exit_since, id))) {
                put(out, clm_mask_names()[k], 1);
            }
        }
    }
}

int clm_ld2_forced(clm_t *m) {
    clm_ld2 *l = m->ld;
    trk_state *s = &m->trk;
    uint16_t R = m->cap - m->n_tok;
    uint32_t D = 1;
    for (uint16_t id = 0; id < LABELS; id++) {
        if (get(s->lbl_unres, id)) {
            D += 3 + (id < 37 ? 1 : 2) + 24;
        }
    }
    if (s->name_kind == NK_LBL) {
        if (s->n_unres) {
            prefix(m, 0);
        } else {
            no_dup(m, 0, m->ban);
        }
    }
    if ((s->name_kind == NK_GOTO || s->name_kind == NK_MENU) && R <= D + 3 + 24) {
        prefix(m, 1);
    }
    if (s->name_kind == NK_GOTO && l->goto_seen && !l->goto_if && !l->goto_depth) {
        uint8_t xb[(CLM_VOCAB_MAX + 7) / 8] = {0};
        no_dup(m, 1, xb);
        uint8_t any = 0;
        for (uint16_t v = 4; v < m->vocab; v++) {
            if (!get(m->ban, v) && !get(xb, v)) {
                any = 1;
                break;
            }
        }
        if (any) {
            for (unsigned i = 0; i < (m->vocab + 7u) / 8; i++) {
                m->ban[i] |= xb[i];
            }
        }
    }
    uint8_t at_start = (s->flags & 2) && !(s->flags & 1) && !s->name_kind &&
                       !s->pend_kind && !s->if_state;
    uint8_t closure = !l->row_string && !l->row_n && l->row_transfer && !l->row_if;
    uint8_t pending = 0;
    for (unsigned i = 0; i < TRK_LABEL_BYTES; i++) {
        pending |= l->row_pending[i];
    }
    if (l->queued) {
        l->queued = 0;
        return LBL;
    }
    if (closure && !s->n_unres && !pending) {
        return CLM_EOS;                                 /* R1 */
    }
    if (s->n_unres && at_start && R <= D) {
        return LBL;                                     /* deadline */
    }
    if (closure && (s->n_unres || pending) && at_start) {
        return LBL;                                     /* R2 */
    }
    return -1;
}

uint8_t clm_ld2_stop(clm_t *m) {
    if (!m->trk.n_unres && !get(m->ban, CLM_EOS) && m->n_tok > 1) {
        int32_t top = CLM_I24_MIN;
        for (uint16_t v = CLM_EOS; v < m->vocab; v++) {
            if (v != CLM_UNK && !get(m->ban, v) && m->logits[v] > top) {
                top = m->logits[v];
            }
        }
        uint32_t W = 0, we = 0;
        for (uint16_t v = CLM_EOS; v < m->vocab; v++) {
            if (v == CLM_UNK || get(m->ban, v)) {
                continue;
            }
            int32_t d = top - m->logits[v];
            /* T=1.0: inv_t = 8192. Round exactly as intref's rsr(...,16). */
            uint32_t idx = ((uint32_t)(d > 32767 ? 32767 : d) + 4) >> 3;
            if (idx >= (uint32_t)m->expw_len) {
                idx = (uint32_t)m->expw_len - 1;
            }
            uint16_t w = RD16U(m->expw + 2 * idx);
            W += w;
            if (v == CLM_EOS) {
                we = w;
            }
        }
        unsigned state = m->trk.depth ? 3 : (m->prev == STOP || m->prev == RET ? 0 : 1);
        unsigned pos = m->n_tok - 1;
        unsigned cell = 4 * state + (pos >= 256) + (pos >= 512) + (pos >= 1024);
        uint64_t num = (uint64_t)we * m->ld->E[cell];
        uint64_t den = num + (uint64_t)(W - we) * 65536UL;
        uint32_t h = den ? (uint32_t)((num * 65536UL) / den) : 0;
        uint32_t x = m->state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        m->state = x;                                    /* extra draw, even if h=0 */
        if ((x >> 16) < h) {
            return 1;
        }
    }
    put(m->ban, CLM_EOS, 1);                              /* EOS only by rule 7 or R1 */
    for (uint16_t v = 4; v < m->vocab; v++) {
        if (!get(m->ban, v)) {
            return 0;
        }
    }
    put(m->ban, CLM_EOS, 0);                              /* all-banned fallback */
    return 0;
}
