/* The tracker and its logit masks: INTREF.md section 13, model/intmask.py IntMask, rule for
 * rule. Each rule is a one-token look-ahead on the C tracker (tracker/tracker.c, the file
 * training and lab/masks.py use): copy the state, step the candidate token(s) on the copy, and
 * compare one of the tracker's counters before and after. Level 2 (CLM_MASKS_V24,
 * model/safety.py) adds fixed lists, the window-variable stores and the string automaton;
 * level 3 (CLM_MASKS_SAFETY) is those additions and v2.3's data-destroying bans, without v2.3's
 * grammar rules. */
#include "clm_int.h"
#include "clm_mask_ids.h"     /* generated from the vocab of tracker/trk_tables.h (tested) */

#define N_LETTERS 26          /* A-Z: the letters a list-name probe may use */
#define HINT_END 37           /* hint bit: the name read so far is a candidate */

enum { K_UNDEF, K_BAD, K_EXTRA_END };

const uint16_t clm_mask_vocab = MASK_VOCAB;
const uint32_t clm_mask_vocab_hash = MASK_VOCAB_HASH;

const uint16_t *clm_mask_names(void) { return MASK_NAMES; }

static uint16_t counter(const trk_state *s, uint8_t key) {
    return key == K_UNDEF ? s->n_undef : key == K_BAD ? s->n_bad : s->n_extra_end;
}

static void ban(clm_t *m, uint16_t v) {
    m->ban[v >> 3] |= (uint8_t)(1u << (v & 7));
}

static uint8_t is_banned(const clm_t *m, uint16_t v) {
    return (uint8_t)((m->ban[v >> 3] >> (v & 7)) & 1);
}

/* does counter `key` grow when the n tokens t0 [t1 [t2]] are stepped on a copy of the state? */
static uint8_t grows(clm_t *m, uint8_t key, uint8_t n, uint16_t t0, uint16_t t1, uint16_t t2) {
    m->trial = m->trk;
    tracker_step(&m->trial, t0, 0);
    if (n > 1) {
        tracker_step(&m->trial, t1, 0);
    }
    if (n > 2) {
        tracker_step(&m->trial, t2, 0);
    }
    return counter(&m->trial, key) > counter(&m->trk, key);
}

static uint8_t hint(const clm_t *m, uint8_t k) {
    return (uint8_t)((m->feat.hint[k >> 3] >> (k & 7)) & 1);
}

/* bag bits 58-63 (var bits 20-25): a custom list in some bucket is assigned */
static uint8_t any_list(const clm_t *m) {
    return (uint8_t)(((m->feat.var[2] >> 4) | (m->feat.var[3] << 4)) & 0x3F);
}

/* Is the list name being started (after `ʟ`, prefixed) or read (inside it) a read? Probe:
 * end it as a one-letter name no assigned list starts with (the first letter whose hint bit
 * is clear; every hint is clear after a fresh `ʟ`) and see whether the tracker counts an
 * undefined read. No such letter counts as a read. */
static uint8_t reads_list(clm_t *m, uint8_t prefixed) {
    uint8_t k = 0;
    while (!prefixed && k < N_LETTERS && hint(m, k)) {
        k++;
    }
    if (k == N_LETTERS) {
        return 1;
    }
    return prefixed ? grows(m, K_UNDEF, 3, MASK_LISTL, MASK_NAMES[k], MASK_NEWLINE)
                    : grows(m, K_UNDEF, 2, MASK_NAMES[k], MASK_NEWLINE, 0);
}

static void list_rules(clm_t *m) {
    if (m->feat.cat[0] != TRK_CTX_LIST) {
        if (!any_list(m) && reads_list(m, 1)) {
            ban(m, MASK_LISTL);                   /* a read, and no list is assigned */
        }
        return;
    }
    if (m->list_read == CLM_LIST_UNKNOWN) {       /* first mask inside this name */
        m->list_read = reads_list(m, 0);
    }
    if (!m->list_read) {
        return;
    }
    for (uint8_t k = 0; k < MASK_N_NAMES; k++) {
        if (!hint(m, k)) {
            ban(m, MASK_NAMES[k]);                /* does not extend an assigned list's name */
        }
    }
    if (!hint(m, HINT_END)) {                     /* the name so far is no assigned list: */
        for (uint16_t v = 4; v < m->vocab; v++) { /* only name characters may follow */
            ban(m, v);
        }
        for (uint8_t k = 0; k < MASK_N_NAMES; k++) {
            if (hint(m, k)) {
                m->ban[MASK_NAMES[k] >> 3] &= (uint8_t)~(1u << (MASK_NAMES[k] & 7));
            }
        }
        ban(m, CLM_EOS);
    }
}

/* the token's class inside a string: MASK_STR_IDS is sorted; ids not in it are class 0 (W) */
static uint8_t str_class(uint16_t tok) {
    for (uint8_t i = 0; i < MASK_N_STR && MASK_STR_IDS[i] <= tok; i++) {
        if (MASK_STR_IDS[i] == tok) {
            return MASK_STR_CLS[i];
        }
    }
    return 0;
}

void clm_track(clm_t *m, uint16_t tok, uint8_t reset) {
    uint8_t was_str = m->feat.cat[0] == TRK_CTX_STRING;
    if (m->ld2) {
        clm_ld2_before_track(m, tok, reset);
    }
    if (reset) {
        tracker_reset(&m->trk);
    }
    tracker_step(&m->trk, tok, &m->feat);
    if (reset) {
        m->str_carry = 0;
    }
    if (m->feat.cat[0] == TRK_CTX_STRING) {       /* the string automaton (IntMask._str_step) */
        if (was_str) {
            m->str_state = MASK_STR_NEXT[m->str_state][str_class(tok)];
        } else if (m->str_carry != 2) {
            m->str_state = MASK_STR_START;        /* a string just opened */
        }
        m->str_carry = 0;
    } else if (was_str && tok == MASK_QUOTE) {
        m->str_carry = 1;                         /* closed by a quote: + may continue it */
    } else if (m->str_carry == 1 && tok == MASK_PLUS) {
        m->str_carry = 2;
    } else {
        m->str_state = MASK_STR_START;
        m->str_carry = 0;
    }
    if (m->ld2) {
        clm_ld2_after_track(m, tok);
    }
    if (m->feat.cat[0] != TRK_CTX_LIST) {
        m->list_read = CLM_LIST_UNKNOWN;
    }
    m->prev = tok;
}

/* level 2: calls and setting commands; window variables where they would be assigned (after
 * ->, in an Input or Prompt statement, as a For( IS>( DS<( variable); in a string, every id of
 * the classes the string automaton bans (only listed ids: no state bans class W, which the
 * generator checks). The fixed sets are bitmaps OR-ed in byte by byte. */
static void safety_rules(clm_t *m) {
    uint8_t ctx = m->feat.cat[0];
    for (uint8_t i = 0; i < MASK_BAN_BYTES; i++) {
        m->ban[i] |= MASK_ALWAYS_BITS[i];
    }
    const uint8_t *cl = tracker_classes();
    uint8_t io = (m->trk.stmt == cl[MASK_IO[0]] || m->trk.stmt == cl[MASK_IO[1]]) &&
                 ctx != TRK_CTX_START;
    if (ctx == TRK_CTX_STORE || io || m->prev == MASK_LOOPVAR[0] || m->prev == MASK_LOOPVAR[1] ||
        m->prev == MASK_LOOPVAR[2]) {
        for (uint8_t i = 0; i < MASK_BAN_BYTES; i++) {
            m->ban[i] |= MASK_WINDOW_BITS[i];
        }
    }
    uint16_t bits = MASK_STR_BAN[m->str_state];
    if (ctx == TRK_CTX_STRING && bits) {
        for (uint8_t j = 0; j < MASK_N_STR; j++) {
            if ((bits >> MASK_STR_CLS[j]) & 1) {
                ban(m, MASK_STR_IDS[j]);
            }
        }
    }
}

/* v2.3's grammar rules (INTREF.md 13): every level but CLM_MASKS_SAFETY */
static void v23_rules(clm_t *m) {
    if (grows(m, K_EXTRA_END, 1, MASK_END, 0, 0)) {
        ban(m, MASK_END);                         /* End with no open block */
    }
    if (grows(m, K_BAD, 1, MASK_ELSE, 0, 0)) {
        ban(m, MASK_ELSE);                        /* Else outside an If-Then */
    }
    uint8_t prev_var = 0;
    for (uint8_t i = 0; i < MASK_N_VARS; i++) {
        if (grows(m, K_UNDEF, 1, MASK_VARS[i], 0, 0)) {
            ban(m, MASK_VARS[i]);                 /* a Str or matrix read before assignment */
        }
        prev_var |= m->prev == MASK_VARS[i];
    }
    if (prev_var) {                               /* lab/masks.py asks only here */
        for (uint8_t i = 0; i < 2; i++) {
            if (grows(m, K_UNDEF, 1, MASK_PENDING[i], 0, 0)) {
                ban(m, MASK_PENDING[i]);          /* `,` / `(` resolving a pending read */
            }
        }
    }
    list_rules(m);
}

void clm_mask(clm_t *m) {
    for (uint16_t i = 0; i < (m->vocab + 7) / 8; i++) {
        m->ban[i] = 0;
    }
    for (uint8_t i = 0; i < MASK_N_SAFETY; i++) {
        ban(m, MASK_SAFETY[i]);                   /* data-destroying tokens: every level */
    }
    if (m->masks != CLM_MASKS_SAFETY) {
        v23_rules(m);
    }
    if (m->masks == CLM_MASKS_V24 || m->masks == CLM_MASKS_SAFETY) {
        safety_rules(m);
    }
    for (uint16_t v = CLM_EOS; v < m->vocab; v++) {
        if (v != CLM_UNK && !is_banned(m, v)) {
            return;
        }
    }
    m->ban[CLM_EOS >> 3] &= (uint8_t)~(1u << (CLM_EOS & 7));   /* nothing left: <eos> */
}
