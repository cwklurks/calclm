/* Cheap optional-decoder binding. CLMVOC supplies its existing six-byte export SHA;
 * clm_load has already checked the CLM slice layout. FNV-1a protects all offset bytes.
 * Full model SHA-256 verification belongs to the host packager, not the calculator. */
#include "clm.h"

static uint32_t rd(const uint8_t *p, uint8_t n) {
    uint32_t v = 0;
    while (n) {
        v = (v << 8) | p[--n];
    }
    return v;
}

int clm_ld2_load(clm_t *m, clm_ld2 *workspace, const uint8_t *payload, uint32_t size,
                  const uint8_t *export_sha) {
    m->ld2 = 0;
    m->ld = 0;
    if (!workspace || !payload || !export_sha || !m->tokens || !m->alibi ||
        m->vocab != clm_mask_vocab || size != 83 || payload[0] != 'L' ||
        payload[1] != 'D' || payload[2] != '2' || payload[3] != 2 ||
        rd(payload + 4, 2) != m->vocab || rd(payload + 6, 3) != m->seg[m->n_seg - 1].end) {
        return CLM_E_ARG;
    }
    for (uint8_t i = 0; i < 6; i++) {
        if (payload[9 + i] != export_sha[i]) {
            return CLM_E_ARG;
        }
    }
    uint32_t h = 0x811C9DC5UL;
    for (uint8_t i = 0; i < 79; i++) {
        h = (h ^ payload[i]) * 0x01000193UL;
    }
    if (h != rd(payload + 79, 4)) {
        return CLM_E_ARG;
    }
    for (uint8_t i = 0; i < 16; i++) {
        workspace->E[i] = rd(payload + 15 + 4 * i, 4);
    }
    m->ld = workspace;
    return CLM_OK;
}

/* Shared by the OS load path and the driver: missing/rejected offsets preserve v2.1. */
void clm_program_decoder(clm_t *m, clm_ld2 *workspace, const uint8_t *payload, uint32_t size,
                          const uint8_t *export_sha) {
    m->horizons = m->alibi;
    m->ld2 = clm_ld2_load(m, workspace, payload, size, export_sha) == CLM_OK;
}
