/* CEmu test driver for the tracker on the eZ80 (tracker/ce_bench.py). Linked with tracker.c
 * into a flat binary; the harness calls these entry points like a debugger. */
#include <stdint.h>

#include "tracker.h"

void trk_drv_reset(trk_state *s) { tracker_reset(s); }

unsigned trk_drv_sizes(void) { return (unsigned)(sizeof(trk_state) | (sizeof(trk_feat) << 12)); }

/* step one token (cycle measurement per call) */
void trk_drv_step(trk_state *s, unsigned tok, trk_feat *out) { tracker_step(s, tok, out); }

/* step n tokens, writing one trk_feat per token (bit-exactness against the host) */
void trk_drv_run(trk_state *s, const uint16_t *toks, unsigned n, trk_feat *out)
{
    unsigned i;
    for (i = 0; i < n; i++)
        tracker_step(s, toks[i], &out[i]);
}

/* step n tokens without writing features: the per-token cost the calculator pays */
void trk_drv_run_last(trk_state *s, const uint16_t *toks, unsigned n, trk_feat *out)
{
    unsigned i;
    for (i = 0; i < n; i++)
        tracker_step(s, toks[i], out);
}
