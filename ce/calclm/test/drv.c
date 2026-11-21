/* Test/benchmark driver for the CalcLM core: one entry point, drv_cmd(block), shared by the
 * host shared library (ctypes) and the eZ80 flat binary run in CEmu (ce/tools/clm.py).
 *
 * Block layout (little-endian; P = sizeof(void *): 8 on the host, 3 on the eZ80):
 *   [0] u8 command; the reply code is the return value; [1] i8 `bad` index from clm_load
 *   LOAD     [1] u8 count  [2] u16 ctx_cap  [4] u8 flags (1: tap, 2: profiling hook)
 *            [6] u16 tok_cap  [8] ptr kc, vc, tbl, act, tap, tok
 *            then count x (ptr data, u32 size)
 *   RESET    [4] u32 seed  [8] u16 temperature_q8  [10] u8 top_k  [11] u8 greedy
 *            [12] u16 cap (0: default)  [14] u8 masks
 *   STEP     -> token or CLM_DONE
 *   GENERATE [4] u16 n  [8] ptr out (u16 tokens)  -> number of tokens written (stops after
 *            <eos> or at the cap)
 *   FORWARD  [4] u16 tok  [6] u16 pos  [8] ptr logits_out (int32 [vocab]; NULL: no copy)
 *   SEEK     [4] u16 pos  [6] u16 filler: as if `pos` tokens (<bos>, filler...) were generated,
 *            so the next STEP runs the forward pass at position pos (benchmarks)
 *   PROF     [4] u8 on: profiling hook on/off
 *   ATTACH   [4] u24 size  [8] ptr CLMCODE payload: run the flash-resident routines
 *   DETACH   back to the RAM routines
 *   TRACK    [4] u16 n  [6] u8 reset  [8] ptr tokens (u16): the tracker reads <bos> (reset)
 *            then the n tokens
 *   MASK     [8] ptr out (ban bitmap, (vocab + 7) / 8 bytes): the masks for the tracker
 *            state  -> number of banned ids (out NULL: nothing copied or counted, returns 0)
 *   TRACKER  [8] ptr out: trk_state, trk_feat, u16 prev  -> sizeof(trk_state)
 *   HORIZONS [4] u8 on: ALiBi horizons on/off (clm_t.horizons; clm_load turns them off)
 *   LD2      [4] u8 on: LD2 on/off (also cleared by clm_load)
 *   E        [4] 16 x u32: trusted raw hazard offsets, copied into driver-owned workspace
 *   LD2_LOAD [4] u32 size [8] ptr: verify CLMLD2; [16] 6-byte CLMVOC export SHA
 *   PROGRAM_DECODER same layout: apply the OS program post-load defaults
 *   FEED     [4] u16 n  [8] ptr tokens (u16): clm_feed each token until one is refused
 *            -> CLM_OK or the first error (v2.5 topics; command 30)
 *   MIX      [4] u32 seed -> [8] u32 clm_mix_seed(seed) (command 31)
 * kv4 build only (CLM_KV4, E20 spike):
 *   KV4_LOAD [4] u32 size [8] ptr: verify CLMKV4 (clm_kv4_load); [16] 6-byte CLMVOC export
 *            SHA; [22] u8 ram: copy the tables to RAM (placement B)
 *   KV4_UNPACK [4] u8 layer [5] u8 c_only [6] u16 n: test-only, unpack slots 0..n-1 of the
 *            layer into the scratch window
 *   KV4_ENCODE [4] u8 layer [5] u8 kind [6] u8 channel [8] ptr out (512 bytes): test-only, the
 *            code and biased decoded value of x = -128..127
 */
#include <stdint.h>

#include "clm.h"
#include "clm_code.h"

enum { DRV_LOAD = 1, DRV_RESET, DRV_STEP, DRV_GENERATE, DRV_FORWARD, DRV_SEEK, DRV_PROF,
       DRV_ATTACH, DRV_DETACH, DRV_TRACK, DRV_MASK, DRV_TRACKER, DRV_HORIZONS,
       DRV_LD2, DRV_E, DRV_LD2_LOAD, DRV_PROGRAM_DECODER,
#if defined(CLM_KV4)
       DRV_KV4_LOAD, DRV_KV4_UNPACK, DRV_KV4_ENCODE,
#endif
       DRV_FEED = 30, DRV_MIX = 31,     /* fixed numbers: the kv4 build's commands stay put */
};

#define P ((unsigned)sizeof(void *))

static clm_t M;
static clm_ld2 LD;
volatile uint8_t drv_phase;


static uint32_t rd(const uint8_t *p, unsigned n) {
    uint32_t v = 0;
    for (unsigned i = n; i--;) {
        v = v << 8 | p[i];
    }
    return v;
}

static void *rdp(const uint8_t *p) {
    uintptr_t v = 0;
    for (unsigned i = P; i--;) {
        v = v << 8 | p[i];
    }
    return (void *)v;
}

#if defined(__ez80__)
/* Profiling hook: record the phase and stop the CPU (HALT with interrupts off); the harness
 * reads the cycle counter and resumes after the HALT, like a debugger breakpoint. */
static void drv_prof(uint8_t phase) {
    drv_phase = phase;
    __asm__ volatile("halt");
}
#else
static void drv_prof(uint8_t phase) { drv_phase = phase; }
#endif

static int load(uint8_t *b) {
    const uint8_t *data[CLM_MAX_APPVARS];
    uint32_t size[CLM_MAX_APPVARS];
    uint8_t count = b[1];
    clm_mem mem;
    int bad = -1;
    if (count > CLM_MAX_APPVARS) {
        return CLM_E_COUNT;
    }
    mem.ctx_cap = (uint16_t)rd(b + 2, 2);
    mem.tok_cap = (uint16_t)rd(b + 6, 2);
    mem.kc = rdp(b + 8);
    mem.vc = rdp(b + 8 + P);
    mem.tbl = rdp(b + 8 + 2 * P);
    mem.act = rdp(b + 8 + 3 * P);
    mem.tok = rdp(b + 8 + 5 * P);
    for (unsigned i = 0; i < count; i++) {
        const uint8_t *e = b + 8 + 6 * P + i * (P + 4);
        data[i] = rdp(e);
        size[i] = rd(e + P, 4);
    }
    int rc = clm_load(&M, data, size, count, &mem, &bad);
    b[1] = (uint8_t)bad;
    M.tap = (b[4] & 1) ? rdp(b + 8 + 4 * P) : 0;
    M.prof = (b[4] & 2) ? drv_prof : 0;
    return rc;
}

int drv_cmd(uint8_t *b) {
    switch (b[0]) {
    case DRV_LOAD:
        return load(b);
    case DRV_RESET:
        return clm_reset(&M, rd(b + 4, 4), (uint16_t)rd(b + 8, 2), b[10], b[11],
                         (uint16_t)rd(b + 12, 2), b[14]);
    case DRV_STEP:
        return clm_step(&M);
    case DRV_GENERATE: {
        unsigned n = rd(b + 4, 2), k = 0;
        uint16_t *out = rdp(b + 8);
        while (k < n) {
            int t = clm_step(&M);
            if (t < 0) {
                break;
            }
            out[k++] = (uint16_t)t;
            if (t == CLM_EOS) {
                break;
            }
        }
        return (int)k;
    }
    case DRV_FORWARD: {
        int32_t *out = rdp(b + 8);
        clm_forward(&M, (uint16_t)rd(b + 4, 2), (uint16_t)rd(b + 6, 2));
        for (unsigned i = 0; out && i < M.vocab; i++) {
            out[i] = M.logits[i];
        }
        return 0;
    }
    case DRV_SEEK: {
        unsigned pos = rd(b + 4, 2);
        if (!M.tokens || pos >= M.cap) {
            return -1;
        }
        M.tokens[0] = CLM_BOS;
        for (unsigned i = 1; i <= pos; i++) {
            M.tokens[i] = (uint16_t)rd(b + 6, 2);
        }
        M.n_tok = (uint16_t)(pos + 1);
        M.done = 0;
        return 0;
    }
    case DRV_TRACK: {
        const uint8_t *t = rdp(b + 8);
        if (b[6]) {
            clm_track(&M, CLM_BOS, 1);
        }
        for (unsigned i = 0, n = rd(b + 4, 2); i < n; i++) {
            clm_track(&M, (uint16_t)rd(t + 2 * i, 2), 0);
        }
        return 0;
    }
    case DRV_MASK: {
        uint8_t *out = rdp(b + 8);
        int n = 0;
        clm_mask(&M);
        for (unsigned i = 0; out && i < (M.vocab + 7u) / 8u; i++) {
            for (uint8_t k = 0; k < 8; k++) {
                n += (M.ban[i] >> k) & 1;
            }
            out[i] = M.ban[i];
        }
        return n;
    }
    case DRV_TRACKER: {
        uint8_t *out = rdp(b + 8);
        const uint8_t *st = (const uint8_t *)&M.trk, *ft = (const uint8_t *)&M.feat;
        unsigned i;
        for (i = 0; i < sizeof M.trk; i++) {
            out[i] = st[i];
        }
        for (unsigned k = 0; k < sizeof M.feat; k++) {
            out[i + k] = ft[k];
        }
        out[i + sizeof M.feat] = (uint8_t)M.prev;
        out[i + sizeof M.feat + 1] = (uint8_t)(M.prev >> 8);
        return (int)sizeof M.trk;
    }
    case DRV_PROF:
        M.prof = b[4] ? drv_prof : 0;
        return 0;
    case DRV_HORIZONS:
        M.horizons = b[4] ? 1 : 0;
        return 0;
    case DRV_LD2:
        M.ld2 = b[4] ? 1 : 0;
        return 0;
    case DRV_E:
        for (unsigned i = 0; i < 16; i++) {
            LD.E[i] = rd(b + 4 + 4 * i, 4);
        }
        M.ld = &LD;
        return 0;
    case DRV_LD2_LOAD:
        return clm_ld2_load(&M, &LD, rdp(b + 8), rd(b + 4, 4), b + 16);
    case DRV_PROGRAM_DECODER:
        clm_program_decoder(&M, &LD, rdp(b + 8), rd(b + 4, 4), b + 16);
        return 0;
    case DRV_FEED: {
        const uint8_t *t = rdp(b + 8);
        for (unsigned i = 0, n = rd(b + 4, 2); i < n; i++) {
            int rc = clm_feed(&M, (uint16_t)rd(t + 2 * i, 2));
            if (rc != CLM_OK) {
                return rc;
            }
        }
        return CLM_OK;
    }
    case DRV_MIX: {
        uint32_t h = clm_mix_seed(rd(b + 4, 4));
        for (unsigned i = 0; i < 4; i++) {
            b[8 + i] = (uint8_t)(h >> (8 * i));
        }
        return 0;
    }
    case DRV_ATTACH:
        return clm_code_attach(rdp(b + 8), rd(b + 4, 3));
    case DRV_DETACH:
        clm_code_detach();
        return 0;
#if defined(CLM_KV4)
    case DRV_KV4_LOAD:
        return clm_kv4_load(&M, rdp(b + 8), rd(b + 4, 4), b + 16, b[22]);
    case DRV_KV4_UNPACK:
        return clm_kv4_unpack_test(&M, b[4], (uint16_t)rd(b + 6, 2), b[5]);
    case DRV_KV4_ENCODE:
        return clm_kv4_encode_test(&M, b[4], b[5], b[6], rdp(b + 8));
#endif
    default:
        return -100;
    }
}
