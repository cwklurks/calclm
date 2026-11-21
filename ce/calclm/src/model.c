/* CalcLM OS layer: locate the archived model AppVars (fileioc), check them, set up memory,
 * and hand everything to the inference core (ce/calclm/core, the code the emulator tests ran). */
#include <fileioc.h>
#include <string.h>
#include <ti/vars.h>

#include "clm_code.h"
#include "memmap.h"
#include "model.h"

#define TMP_NAME "CLMTMP"         /* temporary RAM AppVar: V cache, optional LD2 workspace */
#define VOC_NAME "CLMVOC"
#define CODE_NAME "CLMCODE"
#define LD2_NAME "CLMLD2"
#define RAM_MARGIN 4096           /* left free for prgmGEN and the OS */
#define MIN_CTX 32
#if defined(CLM_KV4)
/* E20 spike (PLAN-E20.md step P): 4-bit K/V. The packed K cache is in BSS; CLMTMP holds the
 * packed V cache, the decoded window (scratch) and the tables copied from CLMKV4 (placement
 * B), then the optional LD2 workspace: CLM_KV4_VC_BYTES(ctx, KV4_RAM) + LD2 bytes. */
#define KV4_NAME "CLMKV4"
#define KV4_RAM 1
static const uint8_t *kv4_data;   /* CLMKV4 payload (archived) */
static uint32_t kv4_size;
#endif

clm_t M;
#if defined(CLM_KV4)
static uint8_t kcache[CLM_LAYERS * CLM_CTX * CLM_KV4_KROW];   /* BSS: packed 4-bit K cache */
#else
static uint8_t kcache[CLM_LAYERS * CLM_CTX * CLM_KROW];   /* BSS (OS scratch RAM) */
#endif
/* the generated tokens: BSS too, at a fixed address (saving prgmGEN moves RAM variables) */
static uint16_t tokens[CLM_DEFAULT_CAP];
static const uint8_t *av_data[CLM_MAX_APPVARS];
static uint32_t av_size[CLM_MAX_APPVARS];
static const uint8_t *voc;        /* CLMVOC payload */
static const uint8_t *voc_offs;   /* u16 offsets */
static const uint8_t *voc_rec;    /* record area */
static uint16_t ctx;
static uint8_t av_count;
static uint8_t masks_ok;          /* the export's vocab is the one the tracker was built for */
static clm_mem mem;

typedef char static_check_act[(CLM_ACT_BYTES <= MM_ACT_LIMIT) ? 1 : -1];

static void name_of(char *name, uint8_t i) {
    memcpy(name, "CLM", 3);
    name[3] = (char)('0' + i / 10);
    name[4] = (char)('0' + i % 10);
    name[5] = 0;
}

static char *put(char *p, const char *s) {
    while (*s) {
        *p++ = *s++;
    }
    *p = 0;
    return p;
}

/* an archived AppVar's data pointer and size; 0 if missing, 1 if in RAM, 2 if fine */
static int find_appvar(const char *name, const uint8_t **data, uint32_t *size) {
    uint8_t h = ti_Open(name, "r");
    if (!h) {
        return 0;
    }
    int archived = ti_IsArchived(h);
    *data = ti_GetDataPtr(h);
    *size = ti_GetSize(h);
    ti_Close(h);
    return archived ? 2 : 1;
}

static int find_model(char *msg) {
    char name[8];
    uint8_t count = 1;
    for (uint8_t i = 0; i < count; i++) {
        name_of(name, i);
        int st = find_appvar(name, &av_data[i], &av_size[i]);
        if (st != 2) {
            char *p = put(msg, name);
            put(p, st ? " is in RAM.\nArchive it (2nd MEM,\nMem Mgmt/Del)." :
                        " is missing.\nSend CLM00..CLMnn\n(archived) from the\nsame export.");
            return 1;
        }
        if (i == 0) {
            const uint8_t *d = av_data[0];
            if (av_size[0] < CLM_APPVAR_HEADER || d[0] != 'C' || d[1] != 'L' || d[3] == 0 ||
                d[3] > CLM_MAX_APPVARS) {
                put(msg, "CLM00 is not a CalcLM\nmodel AppVar.");
                return 1;
            }
            count = d[3];
        }
    }
    return (int)count | 0x100;   /* low byte: AppVar count */
}

static int find_vocab(char *msg) {
    uint32_t size;
    int st = find_appvar(VOC_NAME, &voc, &size);
    if (st != 2) {
        put(msg, st ? "CLMVOC is in RAM.\nArchive it (2nd MEM,\nMem Mgmt/Del)." :
                      "CLMVOC is missing.\nSend CLMVOC.8xv made\nfrom the same export.");
        return 1;
    }
    if (size < 16 || voc[0] != 'C' || voc[1] != 'V' || voc[2] != 1) {
        put(msg, "CLMVOC is not a CalcLM\ntoken table (format 1).");
        return 1;
    }
    uint16_t v = (uint16_t)(voc[4] | voc[5] << 8);
    uint32_t blob = (uint32_t)voc[6] | (uint32_t)voc[7] << 8 | (uint32_t)voc[8] << 16;
    if (v != M.vocab || blob != M.seg[M.n_seg - 1].end) {
        put(msg, "CLMVOC belongs to a\ndifferent export than\nCLM00..CLMnn.");
        return 1;
    }
    voc_offs = voc + 16 + 8 * voc[3];
    voc_rec = voc_offs + 2 * ((uint32_t)v + 1);
    return 0;
}

/* The tracker's token classes and the mask ids are compiled for one vocab: compare its hash
 * (tools/gen_mask_ids.py vocab_hash) with this export's token bytes in CLMVOC. */
static uint8_t tracker_vocab(void) {
    if (M.vocab != clm_mask_vocab) {
        return 0;
    }
    uint32_t h = 0x811C9DC5UL;
    for (uint16_t id = 0; id < M.vocab; id++) {
        uint8_t nb, nd;
        const uint8_t *disp;
        const uint8_t *b = vocab_token(id, &nb, &disp, &nd);
        h = (h ^ nb) * 0x01000193UL;
        for (uint8_t i = 0; i < nb; i++) {
            h = (h ^ b[i]) * 0x01000193UL;
        }
    }
    return h == clm_mask_vocab_hash;
}

/* V cache: a RAM AppVar sized for the largest context that leaves RAM_MARGIN free */
static uint8_t *alloc_vcache(char *msg, uint16_t extra) {
    ti_Delete(TMP_NAME);                         /* leftover from an interrupted run */
    size_t avail = os_MemChk(NULL);
#if defined(CLM_KV4)
    uint32_t per = (uint32_t)CLM_LAYERS * CLM_KV4_VROW + CLM_KV4_SROW;
    uint32_t reserve = RAM_MARGIN + 64 + extra + (KV4_RAM ? CLM_KV4_TBL_BYTES + CLM_KV4_SLACK : 0);
#else
    uint32_t per = (uint32_t)CLM_LAYERS * CLM_HEAD_DIM;
    uint32_t reserve = RAM_MARGIN + 64 + extra;
#endif
    uint32_t c = avail > reserve ? (avail - reserve) / per : 0;
    ctx = (uint16_t)(c > CLM_CTX ? CLM_CTX : c) & ~7u;      /* the core wants a multiple of 8 */
    if (ctx < MIN_CTX) {
#if defined(CLM_KV4)
        put(msg, "Not enough free RAM.\nNeed about 66 KB free:\ndelete or archive\nvariables.");
#else
        put(msg, "Not enough free RAM.\nNeed about 55 KB free:\ndelete or archive\nvariables.");
#endif
        return 0;
    }
    uint8_t h = ti_Open(TMP_NAME, "w");
#if defined(CLM_KV4)
    uint16_t bytes = (uint16_t)(CLM_KV4_VC_BYTES(ctx, KV4_RAM) + extra);
#else
    uint16_t bytes = per * ctx + extra;
#endif
    if (!h || ti_Resize(bytes, h) != bytes) {
        if (h) {
            ti_Close(h);
        }
        ti_Delete(TMP_NAME);
        put(msg, "Could not create the\nCLMTMP work AppVar.");
        return 0;
    }
    uint8_t *p = ti_GetDataPtr(h);
    ti_Close(h);
    return p;
}

/* Only archived, matching offsets with the tracker vocab may enable LD2. Before
 * allocating CLMTMP, use activation scratch to validate E; no generation runs there. */
static void decoder(clm_ld2 *workspace) {
    const uint8_t *data = 0;
    uint32_t size = 0;
    if (!masks_ok || find_appvar(LD2_NAME, &data, &size) != 2) {
        data = 0;
    }
    clm_program_decoder(&M, workspace, data, size, vocab_sha());
}

static const char *load_error(int rc) {
    switch (rc) {
    case CLM_E_COUNT: return "AppVar count differs\nfrom CLM00's header.";
    case CLM_E_APPVAR: return "A CLMnn AppVar header\nis wrong (resend all).";
    case CLM_E_GAP: return "CLMnn AppVars come\nfrom different exports.";
    case CLM_E_MAGIC: return "CLM00 is not a CalcLM\nmodel.";
    case CLM_E_VERSION: return "Model format version\ndiffers from CALCLM.";
    case CLM_E_SHAPE: return "Model shape differs\nfrom this CALCLM build.";
    case CLM_E_SIZE: return "Model AppVars are\ntruncated or extra.";
    case CLM_E_SPLIT: return "Model AppVar split is\nnot supported.";
    case CLM_E_RANGE: return "Model scalars out of\nthe supported range.";
    default: return "Model load failed.";
    }
}

int model_open(char *msg) {
    int st = find_model(msg);
    if (!(st & 0x100)) {
        return 1;
    }
    uint8_t count = (uint8_t)st;
    av_count = count;
    /* Validation does not read/write the V cache. Allocate it after the optional
     * decoder checks, so absent or rejected LD2 retains v2.1's RAM/context budget. */
    mem.vc = MM_ACT;
    mem.kc = kcache;
    mem.tbl = MM_TBL;
    mem.act = MM_ACT;
    mem.tok = tokens;
    mem.ctx_cap = CLM_CTX;
    mem.tok_cap = CLM_DEFAULT_CAP;
    int bad = -1;
    int rc = clm_load(&M, av_data, av_size, count, &mem, &bad);
    if (rc != CLM_OK) {
        model_close();
        put(msg, load_error(rc));
        return 1;
    }
    if (find_vocab(msg)) {
        model_close();
        return 1;
    }
    masks_ok = tracker_vocab();
    decoder((clm_ld2 *)MM_ACT);
    mem.vc = alloc_vcache(msg, M.ld2 ? sizeof(clm_ld2) : 0);
    if (!mem.vc) {
        model_close();
        return 1;
    }
    mem.ctx_cap = ctx;
    /* Loading only resolved tensors and activation pointers. Adopt the real cache;
     * all other buffers are unchanged. No extra model scan is needed before graphx. */
    M.mem = mem;
    M.slide = M.alibi && ctx == CLM_CTX;
#if defined(CLM_KV4)
    if (find_appvar(KV4_NAME, &kv4_data, &kv4_size) != 2 ||
        clm_kv4_load(&M, kv4_data, kv4_size, vocab_sha(), KV4_RAM) != CLM_OK) {
        model_close();
        put(msg, "CLMKV4 is missing, in\nRAM, or from another\nexport than CLM00.");
        return 1;
    }
    decoder((clm_ld2 *)(mem.vc + CLM_KV4_VC_BYTES(ctx, KV4_RAM)));
#else
    decoder((clm_ld2 *)(mem.vc + (uint32_t)CLM_LAYERS * ctx * CLM_HEAD_DIM));
#endif
    return 0;
}

uint8_t model_masks_ok(void) {
    return masks_ok;
}

void model_close(void) {
    clm_code_detach();
    ti_Delete(TMP_NAME);
}

int model_prepare(uint8_t *fast) {
    int bad;
    if (clm_load(&M, av_data, av_size, av_count, &mem, &bad) != CLM_OK) {
        return 1;
    }
#if defined(CLM_KV4)
    if (clm_kv4_load(&M, kv4_data, kv4_size, vocab_sha(), KV4_RAM) != CLM_OK) {
        return 1;                                /* clm_load cleared the tables */
    }
    decoder((clm_ld2 *)(mem.vc + CLM_KV4_VC_BYTES(ctx, KV4_RAM)));
#else
    decoder((clm_ld2 *)(mem.vc + (uint32_t)CLM_LAYERS * ctx * CLM_HEAD_DIM));
#endif
    const uint8_t *code;
    uint32_t size;
    *fast = find_appvar(CODE_NAME, &code, &size) == 2 && clm_code_attach(code, size) == CLM_CODE_OK;
    return 0;
}

uint16_t model_ctx(void) {
    return ctx;
}

const uint8_t *vocab_glyphs(uint8_t *count) {
    *count = voc[3];
    return voc + 16;
}

const uint8_t *vocab_sha(void) {
    return voc + 9;
}

const uint8_t *vocab_token(uint16_t id, uint8_t *nbytes, const uint8_t **disp, uint8_t *ndisp) {
    uint16_t a = (uint16_t)(voc_offs[2 * id] | voc_offs[2 * id + 1] << 8);
    uint16_t b = (uint16_t)(voc_offs[2 * id + 2] | voc_offs[2 * id + 3] << 8);
    const uint8_t *r = voc_rec + a;
    *nbytes = r[0];
    *disp = r + 1 + r[0];
    *ndisp = (uint8_t)(b - a - 1 - r[0]);
    return r + 1;
}
