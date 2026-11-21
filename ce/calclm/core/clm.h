/* CalcLM inference core: integer transformer inference, bit-exact with model/intref.py
 * (contract: model/INTREF.md). No OS calls; all memory comes from the caller.
 *
 * Builds for the eZ80 (ez80-clang, CE toolchain; matvecs via the M2 assembly kernels) and for
 * the host (gcc; a C stand-in for the kernels) so the same source can be tested both ways.
 *
 *   clm_load()   validate the CLMnn AppVar payloads and the blob header, resolve tensors
 *   clm_reset()  start a generation: <bos> at position 0, sampler settings, PRNG seed, cap,
 *                logit masks
 *   clm_feed()   optional, before the first clm_step: a prompt token (v2.5 topics)
 *   clm_step()   one forward pass + one sampled token (CLM_DONE at <eos> or the cap)
 *
 * Export formats (INTREF.md 6): 1 = learned positions (v1; generation stops when the
 * CLM_CTX-token context is full); 2 = ALiBi relative positions (v2; a CLM_CTX-token sliding
 * window over a ring K/V cache, generation up to a cap). The tracker (tracker/tracker.c) and
 * its logit masks (INTREF.md 13) are compiled in and work with either format.
 */
#ifndef CLM_H
#define CLM_H

#include <stdint.h>

#include "clm_shape.h"
#include "tracker.h"

#define CLM_FORMAT_LEARNED 1      /* blob header version 1 */
#define CLM_FORMAT_RELATIVE 2     /* blob header version 2, pos_scheme after the layer fields */
#define CLM_POS_ALIBI 1           /* the only relative scheme this core runs */
#define CLM_DEFAULT_CAP 1024      /* tokens (<bos> included) when clm_reset gets cap 0 */

#define CLM_PAD 0
#define CLM_BOS 1
#define CLM_EOS 2
#define CLM_UNK 3
#define CLM_DONE (-1)

/* clm_reset's masks: the logit mask level */
#define CLM_MASKS_V23 1      /* v2.3: DelVar ... ClrList, End/Else, reads, lists (INTREF.md 13) */
#define CLM_MASKS_V24 2      /* v2.4: also calls, settings, window stores, credit lines in strings
                                (model/safety.py, docs/agents/phase1-safety-masks.md) */
#define CLM_MASKS_SAFETY 3   /* v2.4's additions and v2.3's destructive bans, without v2.3's
                                grammar rules: the calculator's Masks row off (the destructive
                                and safety bans are always on) */

/* error codes (clm_load / clm_reset) */
#define CLM_OK 0
#define CLM_E_COUNT (-1)     /* number of AppVars differs from the count in their headers */
#define CLM_E_APPVAR (-2)    /* an AppVar header is wrong (magic, index, count, size) */
#define CLM_E_GAP (-3)       /* AppVar slices are not contiguous (mixed exports?) */
#define CLM_E_MAGIC (-4)     /* blob does not start with "CLM0" */
#define CLM_E_VERSION (-5)   /* blob format (version, position scheme) unknown to this program */
#define CLM_E_SHAPE (-6)     /* model dims differ from the ones this program was built for
                                (vocab: 68..CLM_VOCAB_MAX) */
#define CLM_E_SIZE (-7)      /* tensor walk does not end where the last AppVar ends */
#define CLM_E_SPLIT (-8)     /* a tensor (or embedding row) straddles two AppVars */
#define CLM_E_RANGE (-9)     /* a header scalar or table is outside what the core supports */
#define CLM_E_MEM (-10)      /* bad buffers (table bank alignment, context capacity) */
#define CLM_E_ARG (-11)      /* sampler settings out of range (or masks for another vocab) */

#define CLM_APPVAR_HEADER 7  /* "CL", index, count, blob_start (u24 LE) */

#define CLM_G_DIM ((CLM_DIM + 4) / 5)          /* kernel groups for d-input matrices */
#define CLM_G_HID ((CLM_HIDDEN + 4) / 5)       /* kernel groups for hidden-input matrices */
#define CLM_QDIM (CLM_HEADS * CLM_HEAD_DIM)
/* K cache row: head_dim keys stored as k + 128 (uint8), then -128 * sum(k + 128) as int24 LE.
 * V cache row: head_dim values stored as v + 128. (Unsigned operands for the eZ80 MLT.) */
#define CLM_KROW (CLM_HEAD_DIM + 3)
#define CLM_MAX_TOPK 64
#if defined(CLM_KV4)
/* E20 spike, 4-bit K/V (PLAN-E20.md; core/clm_kv4.h). kc holds CLM_LAYERS * ctx_cap packed K
 * rows of CLM_KV4_KROW bytes (16 bytes of 4-bit codes, then -128 * sum of the decoded biased
 * keys as int24 LE); vc holds CLM_LAYERS * ctx_cap packed V rows of CLM_KV4_VROW bytes, then
 * the scratch (one layer's decoded window in the layout above: ctx_cap K rows of CLM_KROW
 * bytes, then ctx_cap V rows of CLM_HEAD_DIM bytes), then, for tables in RAM, the
 * CLM_KV4_TBL_BYTES biased tables 512-aligned (each (layer, kind) block inside one 64 KB bank,
 * as the eZ80 unpack needs) within CLM_KV4_SLACK bytes of slack (which also keeps the end of the
 * region 8-aligned for what follows): CLM_KV4_VC_BYTES(ctx_cap, ram). */
#define CLM_KV4_KROW (CLM_HEAD_DIM / 2 + 3)
#define CLM_KV4_VROW (CLM_HEAD_DIM / 2)
#define CLM_KV4_SROW (CLM_KROW + CLM_HEAD_DIM)
#define CLM_KV4_TBL_BYTES (CLM_LAYERS * 2 * CLM_HEAD_DIM * 16)
#define CLM_KV4_HDR 18
#define CLM_KV4_ALIGN 512
#define CLM_KV4_SLACK 512
#define CLM_KV4_SIZE (CLM_KV4_HDR + CLM_LAYERS * 2 * CLM_HEAD_DIM * 2 + CLM_KV4_TBL_BYTES)
#define CLM_KV4_VC_BYTES(cap, ram) ((uint32_t)CLM_LAYERS * (cap) * CLM_KV4_VROW + \
    (uint32_t)(cap) * CLM_KV4_SROW + ((ram) ? CLM_KV4_TBL_BYTES + CLM_KV4_SLACK : 0))
#endif

#if defined(__ez80__)
typedef int clm_i24;              /* native 24-bit int: matvec outputs, scores, logits */
#else
typedef int32_t clm_i24;
#endif

typedef struct {
    uint16_t m;                   /* PF15: 2^14 <= m < 2^15 (PF8: 128 <= m < 256, or 0) */
    int16_t e;
} clm_pf;

typedef struct {
    const uint8_t *w;             /* packed ternary rows, ceil(n_in/5) bytes each */
    const uint8_t *m;             /* PF7 mantissa per row */
    const int8_t *e;              /* exponent per row */
    int8_t emin, emax;            /* range of e (set at load; picks the fast scale_out path) */
} clm_lin;

typedef struct {
    const uint8_t *attn_g, *ffn_g;    /* int16 LE gains */
    const uint8_t *lut;               /* uint8 exp table */
    clm_lin wq, wk, wv, wo, w1, w3, w2;
    int16_t lut_len;
    int8_t attn_eg, ffn_eg, sh, e3;   /* e3 = max exponent of w3 rows */
    clm_pf sv;                        /* static V grid step (PF15) */
} clm_layer;

typedef struct {
    uint32_t start, end;              /* blob byte range [start, end) */
    const uint8_t *data;              /* blob byte `start` */
} clm_seg;

/* Optional LD2 workspace, owned by the caller. Kept out of clm_t so the shipped program
 * pays only for a pointer and flag until it opts in. Row stores at most one capped program. */
typedef struct {
    uint32_t E[16];
    uint8_t exit_since[TRK_LABEL_BYTES];
    uint8_t row_defined[TRK_LABEL_BYTES], row_pending[TRK_LABEL_BYTES];
    uint16_t row_toks[CLM_DEFAULT_CAP], row_n, row_depth;
    uint16_t goto_depth;
    uint8_t goto_if, goto_seen, row_string, row_if, row_transfer, queued;
} clm_ld2;

/* Caller-provided memory. tbl: a 64 KB-aligned bank; the kernels use pages 0..242 x columns
 * 0..2*CLM_G_HID of it (bank offsets up to 0xF2CE for v2.2, 0xF2E8 for W).
 * act: CLM_ACT_BYTES of scratch.
 * kc: CLM_LAYERS * ctx_cap * CLM_KROW bytes; vc: CLM_LAYERS * ctx_cap * CLM_HEAD_DIM bytes;
 * ctx_cap (a multiple of 8, at most CLM_CTX: the eZ80 P.V reads keys in groups of 8) is the
 * number of K/V entries. Format 2 slides over them only when ctx_cap == CLM_CTX (the model's
 * window); otherwise, and for format 1, generation stops when ctx_cap tokens (<bos>
 * included) are in the cache. tok: the generated tokens, tok_cap entries (<bos> first); the
 * cap of a generation is at most tok_cap. */
typedef struct {
    uint8_t *kc, *vc;
    uint8_t *tbl;
    uint8_t *act;
    uint16_t *tok;
    uint16_t ctx_cap, tok_cap;
} clm_mem;

/* profiling hook phases (the hook, if set, is called at the start of each phase) */
enum {
    CLM_PH_EMBED, CLM_PH_NORM, CLM_PH_BUILD, CLM_PH_ROWS, CLM_PH_SCALE, CLM_PH_SCORES,
    CLM_PH_SOFTMAX, CLM_PH_PV, CLM_PH_QUANT, CLM_PH_FFN, CLM_PH_SAMPLE, CLM_PH_END,
    CLM_PH_MASK,                      /* tracker step and logit masks */
#if defined(CLM_KV4)
    CLM_PH_KV4,                       /* 4-bit K/V: encode at the write, unpack to scratch */
#endif
    CLM_N_PHASES
};

#define CLM_LIST_UNKNOWN 0xFF        /* clm_t.list_read before the first mask in a list name */

typedef struct clm {
    /* model (pointers into the AppVar payloads) */
    clm_layer L[CLM_LAYERS];
    clm_lin out;
    const uint8_t *norm_g;
    const uint8_t *tok_m, *pos_m;
    const int8_t *tok_e, *pos_e;
    const uint8_t *silu;              /* int16 LE [silu_len] */
    const uint8_t *expw;              /* uint16 LE [expw_len] */
    uint32_t tok_off, pos_off;        /* blob offsets of tok_emb / pos_emb (rows resolved per use) */
    clm_seg seg[CLM_MAX_APPVARS];
    clm_pf sqrtd;
    uint16_t vocab;                   /* from the blob header */
    uint16_t lut_max;                 /* header lut_max: the ALiBi penalty cap in table steps */
    uint8_t n_seg;
    uint8_t alibi;                    /* format 2: ALiBi, no position table */
    uint8_t slide;                    /* alibi with a full CLM_CTX-entry cache: ring K/V */
    uint8_t horizons;                 /* ALiBi horizons (PLAN-v2.1.md E1; intref horizons=True):
                                         head h attends only to distances d < ceil(lut_max /
                                         2^(HEADS-1-h)). 0 after clm_load; the caller sets it;
                                         format 2 only (ignored for format 1) */
    int16_t expw_len;
    int8_t F, norm_eg;

    /* memory */
    clm_mem mem;
    clm_i24 *r;                       /* residual (int24) [DIM] */
    int8_t *x8;                       /* kernel input, padded [5 * G_DIM] */
    int8_t *h8;                       /* kernel input, padded [5 * G_HID] */
    clm_i24 *y;                       /* kernel output [max(VOCAB_MAX, HIDDEN)] */
    int8_t *q8;                       /* [QDIM] */
    int32_t *v32;                     /* z / o / h [HIDDEN] */
    int16_t *y1;                      /* [HIDDEN] */
    clm_i24 *sc;                      /* attention scores [CTX] */
    uint8_t *ew;                      /* attention weights [CTX + 8] */
    clm_i24 *logits;                  /* [vocab] Q8 */
    uint16_t *tokens;                 /* mem.tok [mem.tok_cap] */
    uint16_t *cand;                   /* [MAX_TOPK] */
    uint16_t *wts;                    /* [MAX_TOPK] */

    /* generation state */
    uint32_t state;
    uint16_t inv_t, n_tok, cap;
    uint16_t n_prompt;                /* n_tok after clm_reset and each clm_feed (clm_feed's
                                         guard: no clm_step yet) */
    uint8_t top_k, greedy, done, masks;
    uint8_t ld2;                     /* opt in after load; requires masks and ld workspace */
    clm_ld2 *ld;
#if defined(CLM_KV4)
    const uint8_t *kv4_m;             /* reciprocals M[l][kind][c], uint16 LE (CLMKV4 payload) */
    const uint8_t *kv4_t;             /* biased tables [l][kind][c][16]: in place or the RAM copy */
    uint8_t *kv4_scratch;             /* one layer's decoded window (in vc, after the packed V) */
    uint8_t kv4_ok;                   /* CLMKV4 accepted since the last clm_load */
#endif

    /* tracker and logit masks (clm_mask.c, INTREF.md 13) */
    trk_state trk, trial;             /* the program so far; a scratch copy for look-aheads */
    trk_feat feat;                    /* features after the last token read */
    uint16_t prev;                    /* last token read */
    uint8_t list_read;                /* in a list name: 1 read, 0 definition, or LIST_UNKNOWN */
    uint8_t str_state;                /* the string automaton's state (level 2 credit lines) */
    uint8_t str_carry;                /* 1 after a closing quote, 2 after its + (the state runs on) */
    uint8_t ban[(CLM_VOCAB_MAX + 7) / 8];   /* ids the masks exclude from the ranking */

    /* debugging / profiling (NULL in the shipped program) */
    int32_t *tap;                     /* residual after each sublayer: [2 * LAYERS][DIM] */
    void (*prof)(uint8_t phase);
} clm_t;

/* activation scratch layout (bytes) */
#define CLM_ACT_BYTES (4 * CLM_DIM + 5 * CLM_G_DIM + 5 * CLM_G_HID + \
    (int)sizeof(clm_i24) * ((CLM_VOCAB_MAX > CLM_HIDDEN ? CLM_VOCAB_MAX : CLM_HIDDEN) + CLM_CTX + \
    CLM_VOCAB_MAX) + \
    CLM_QDIM + 4 * CLM_HIDDEN + 2 * CLM_HIDDEN + (int)sizeof(clm_i24) + CLM_CTX + 8 + \
    4 * CLM_MAX_TOPK)

/* Check AppVar payloads data[i] (size[i] bytes each, as stored in the calculator; data[i] is
 * CLMnn, nn = i) and load the model. The AppVar count and the vocab size come from the export
 * itself. On CLM_E_APPVAR / CLM_E_GAP / CLM_E_SPLIT, *bad (if not NULL) is the AppVar index. */
int clm_load(clm_t *m, const uint8_t *const *data, const uint32_t *size, uint8_t count,
             const clm_mem *mem, int *bad);

/* temperature_q8 = round(256 T) in [33, 65535]; top_k in [1, 64]; greedy ignores both.
 * cap: tokens in all, <bos> included (0: CLM_DEFAULT_CAP); it is lowered to what the memory
 * holds (mem.tok_cap, and ctx_cap unless the cache slides). masks: 0 off, CLM_MASKS_V23,
 * CLM_MASKS_V24 or CLM_MASKS_SAFETY, the tracker's logit masks at that level (needs the vocab
 * the tracker was built for, TRK_VOCAB ids); any other value is CLM_E_ARG. */
int clm_reset(clm_t *m, uint32_t seed, uint16_t temperature_q8, uint8_t top_k, uint8_t greedy,
              uint16_t cap, uint8_t masks);

/* Feed one prompt token (lab/e14.py feed_prompt; PHASE3-topic-menu.md 2a): only after
 * clm_reset and before the first clm_step. The forward pass runs on the previous token (the
 * core is one token behind, so the first clm_step forwards the last prompt token), `tok` is
 * appended and, with masks on, read by the tracker, the string automaton and LD2's state. No
 * bans are computed and no PRNG step is taken. CLM_E_ARG for an id below 4 or not in the
 * vocab, a prompt that would leave no room for one decided token under the cap, or a call
 * after clm_step. */
int clm_feed(clm_t *m, uint16_t tok);

/* MurmurHash3's fmix32: the seed transform of a prompted (topic) generation (model/topics.py
 * mix_seed). A bijection with clm_mix_seed(0) = 0. */
uint32_t clm_mix_seed(uint32_t seed);

/* Feed the last token, sample the next one. Returns the token id (CLM_EOS ends the program),
 * or CLM_DONE once <eos> was returned or cap tokens are written. */
int clm_step(clm_t *m);

/* One forward pass: logits for token `tok` at position `pos` into m->logits (also writes the
 * KV cache at pos, or at pos mod CLM_CTX when the cache slides). Exposed for tests and
 * benchmarks; does nothing unless the model is loaded, tok < vocab and (unless the cache
 * slides) pos < ctx_cap. */
void clm_forward(clm_t *m, uint16_t tok, uint16_t pos);

/* The tracker (clm_mask.c), exposed for tests and benchmarks: clm_track reads one token
 * (tracker_reset first when `reset`); clm_mask fills m->ban for the next draw. */
void clm_track(clm_t *m, uint16_t tok, uint8_t reset);
void clm_mask(clm_t *m);
const uint16_t *clm_mask_names(void);
/* LD2 state updates bracket tracker_step; selection applies rules 1-7 to m->ban/logits. */
void clm_ld2_before_track(clm_t *m, uint16_t tok, uint8_t reset);
void clm_ld2_after_track(clm_t *m, uint16_t tok);
int clm_ld2_forced(clm_t *m);
uint8_t clm_ld2_stop(clm_t *m);
/* Verify the 83-byte version-2 CLMLD2 using the already-accepted CLMVOC's six-byte
 * export_sha, loaded vocab/blob size and payload checksum. Host packaging verifies the
 * full model SHA-256. Copy E and attach caller-owned workspace; failure disables/detaches.
 * Does not enable LD2. */
int clm_ld2_load(clm_t *m, clm_ld2 *workspace, const uint8_t *payload, uint32_t size,
                  const uint8_t *export_sha);
/* Program defaults after each successful load: horizons for format 2, LD2 if accepted.
 * Callers pass NULL for missing/unavailable offsets or an incompatible tracker vocab. */
void clm_program_decoder(clm_t *m, clm_ld2 *workspace, const uint8_t *payload, uint32_t size,
                          const uint8_t *export_sha);
#if defined(CLM_KV4)
/* Verify a CLMKV4 payload (core/clm_kv4.h: header, the accepted CLMVOC's six-byte export sha,
 * the loaded blob size, layers, head_dim, size, FNV-1a) after a successful clm_load, and
 * accept its tables; ram: copy the biased tables into vc after the scratch (placement B).
 * Until it returns CLM_OK after the last clm_load, clm_forward does nothing. */
int clm_kv4_load(clm_t *m, const uint8_t *payload, uint32_t size, const uint8_t *export_sha,
                 uint8_t ram);
/* Test-only (the driver's KV4_UNPACK, KV4_ENCODE): unpack slots 0..n-1 of layer l into the
 * scratch (c_only: the C unpack even where the eZ80 build has an assembly one); the code and
 * the biased decoded value of every int8 input for one (layer, kind, channel) into out[512]. */
int clm_kv4_unpack_test(clm_t *m, uint8_t l, uint16_t n, uint8_t c_only);
int clm_kv4_encode_test(clm_t *m, uint8_t l, uint8_t kind, uint8_t c, uint8_t *out);
#endif
extern const uint16_t clm_mask_vocab;   /* the vocab size the tracker and mask ids are for */
/* FNV-1a over that vocab's token byte counts and bytes, id by id (tools/gen_mask_ids.py): the
 * OS layer compares it with the CLMVOC of the export before it allows the masks */
extern const uint32_t clm_mask_vocab_hash;

#endif
