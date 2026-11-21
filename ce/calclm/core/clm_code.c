/* Loader for the flash-resident routines (see clm_code.h and tools/build_flash.py). */
#include "clm_code.h"

#if defined(__ez80__)

#ifndef CLM_CODE_ID
#error "CLM_CODE_ID (the assembly build id) must be defined by tools/build_core.sh"
#endif

/* RAM entry points, in the payload's entry order (tools/build_flash.py ENTRIES) */
void clm_a_quant(void);
void clm_a_scale(void);
void clm_a_mulnorm(void);
void clm_a_t3(void);
void clm_a_maxabs32(void);
void clm_a_scores(void);
void clm_a_softmax(void);
void clm_a_pv(void);
void clm_a_havg(void);
void clm_a_silu(void);
void clm_a_maxabs24(void);
void clm_a_compact(void);
void clm_a_emb(void);
void tmv_d_build(void);
void tmv_d_rows(void);
void tmv_h_build(void);
void tmv_h_rows(void);

static void (*const entries[CLM_CODE_ENTRIES])(void) = {
    clm_a_quant, clm_a_scale, clm_a_mulnorm, clm_a_t3, clm_a_maxabs32, clm_a_scores,
    clm_a_softmax, clm_a_pv, clm_a_havg, clm_a_silu, clm_a_maxabs24, clm_a_compact, clm_a_emb,
    tmv_d_build, tmv_d_rows, tmv_h_build, tmv_h_rows,
};

static uint32_t rd24(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16;
}

static void jp_to(uint8_t *at, uint32_t target) {
    at[0] = 0xC3;                                   /* jp nn */
    at[1] = (uint8_t)target;
    at[2] = (uint8_t)(target >> 8);
    at[3] = (uint8_t)(target >> 16);
}

int clm_code_attach(const uint8_t *p, uint32_t size) {
    if (size < 13 || p[0] != 'C' || p[1] != 'C' || p[2] != 1 || p[3] != CLM_CODE_ENTRIES) {
        return CLM_CODE_E_FORMAT;
    }
    uint32_t id = rd24(p + 9) | (uint32_t)p[12] << 24;
    if (id != (uint32_t)CLM_CODE_ID) {
        return CLM_CODE_E_BUILD;
    }
    uint16_t nt = (uint16_t)(p[4] | p[5] << 8);
    uint32_t code_len = rd24(p + 6);
    uint32_t head = 13 + 3UL * CLM_CODE_ENTRIES + 6UL * nt;
    if (head + code_len > size) {
        return CLM_CODE_E_FORMAT;
    }
    const uint8_t *code = p + head;
    const uint8_t *tr = p + 13 + 3 * CLM_CODE_ENTRIES;
    for (uint16_t i = 0; i < nt; i++) {            /* check everything before writing */
        uint32_t ram = rd24(tr + 6 * i), off = rd24(tr + 6 * i + 3);
        if (ram < CLM_FRAM || ram + 4 > CLM_FRAM + CLM_FRAM_SIZE || off >= code_len) {
            return CLM_CODE_E_TABLE;
        }
    }
    for (uint8_t e = 0; e < CLM_CODE_ENTRIES; e++) {
        if (rd24(p + 13 + 3 * e) >= code_len || *(const uint8_t *)entries[e] != 0xC3) {
            return CLM_CODE_E_TABLE;
        }
    }
    for (uint16_t i = 0; i < nt; i++) {
        jp_to((uint8_t *)rd24(tr + 6 * i), (uint32_t)code + rd24(tr + 6 * i + 3));
    }
    for (uint8_t e = 0; e < CLM_CODE_ENTRIES; e++) {
        jp_to((uint8_t *)entries[e], (uint32_t)code + rd24(p + 13 + 3 * e));
    }
    return CLM_CODE_OK;
}

void clm_code_detach(void) {
    for (uint8_t e = 0; e < CLM_CODE_ENTRIES; e++) {
        uint8_t *at = (uint8_t *)entries[e];
        jp_to(at, (uint32_t)at + 4);               /* the body follows the entry jump */
    }
}

#else

int clm_code_attach(const uint8_t *payload, uint32_t size) {
    (void)payload;
    (void)size;
    return CLM_CODE_E_HOST;
}

void clm_code_detach(void) {}

#endif
