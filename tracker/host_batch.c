/* Host-only helpers around tracker.c for training (tracker/bind.py): whole-stream feature
 * extraction and per-program summaries. Not part of the calculator build. */
#include <stddef.h>
#include <stdint.h>

#include "tracker.h"

size_t trk_sizeof_state(void) { return sizeof(trk_state); }
size_t trk_sizeof_feat(void) { return sizeof(trk_feat); }
/* where trk_state.stmt sits (model/intmask.py reads it for the v2.4 window-variable rule,
 * as ce/calclm/core/clm_mask.c reads m->trk.stmt) */
size_t trk_offsetof_stmt(void) { return offsetof(trk_state, stmt); }

/* Features for a uint16 token stream (programs separated by <bos>, which resets the state).
 * Row i is the state after token i: cats[i * TRK_NCAT + g], bags[i] = hint bits 0..37 |
 * var bits << 38. The stream starts from the reset state. */
void trk_batch(const uint16_t *toks, size_t n, uint8_t *cats, uint64_t *bags)
{
    trk_state st;
    trk_feat f;
    size_t i;
    int g;
    tracker_reset(&st);
    for (i = 0; i < n; i++) {
        uint64_t bag = 0;
        tracker_step(&st, toks[i], &f);
        for (g = 0; g < TRK_NCAT; g++)
            cats[i * TRK_NCAT + (size_t)g] = f.cat[g];
        for (g = 0; g < 5; g++)
            bag |= (uint64_t)f.hint[g] << (8 * g);
        bag &= ((uint64_t)1 << 38) - 1;
        for (g = 0; g < 4; g++)
            bag |= (uint64_t)f.var[g] << (38 + 8 * g);
        bags[i] = bag;
    }
}

/* One summary (8 uint16, trk_summary order) per program: a program runs from a <bos> (or the
 * stream start) to just before the next <bos>. Returns the number of programs found; writes
 * at most max_progs. */
size_t trk_summaries(const uint16_t *toks, size_t n, uint16_t bos, uint16_t *out,
                     size_t max_progs)
{
    trk_state st;
    trk_summary sm;
    size_t i, k = 0;
    tracker_reset(&st);
    for (i = 0; i < n; i++) {
        tracker_step(&st, toks[i], NULL);
        if (i + 1 == n || toks[i + 1] == bos) {
            if (k < max_progs) {
                tracker_summary(&st, &sm);
                out[8 * k + 0] = sm.n_unres;
                out[8 * k + 1] = sm.depth;
                out[8 * k + 2] = sm.n_undef;
                out[8 * k + 3] = sm.first_undef_pos;
                out[8 * k + 4] = sm.n_bad;
                out[8 * k + 5] = sm.first_bad_pos;
                out[8 * k + 6] = sm.n_extra_end;
                out[8 * k + 7] = sm.n_prgm;
            }
            k++;
        }
    }
    return k;
}
