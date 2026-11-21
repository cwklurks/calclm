/* CalcLM browser demo: the calculator's inference core (ce/calclm/core, tracker/tracker.c,
 * the host matvec stand-in) and the pure part of its start screen (src/menu.c, src/topics.h),
 * compiled to WebAssembly (demo/tools/build_wasm.sh). docs/agents/phase6-browser-demo.md.
 *
 * The model is loaded as src/model.c loads it (clm_load on the AppVar payloads, masks only for
 * the tracker's vocab, clm_program_decoder with CLMLD2) and a run is started as src/main.c's
 * generate() starts it (menu_run, M.ld2 = r.ld2 && M.ld, clm_reset, clm_feed per opening token,
 * then clm_step). The module imports nothing; the page copies the payloads in with demo_alloc.
 */
#include <stdint.h>

#include "clm.h"
#include "menu.h"
#include "topics.h"

/* the calculator's Random defaults (main.c: temps[4], topks[3], caps[2], sampled) */
#define RANDOM_TEMP_Q8 205
#define RANDOM_TOP_K 20
#define RANDOM_CAP CLM_DEFAULT_CAP

static clm_t M;
static clm_ld2 LD;
static uint8_t kcache[CLM_LAYERS * CLM_CTX * CLM_KROW];
static uint8_t vcache[CLM_LAYERS * CLM_CTX * CLM_HEAD_DIM];
static uint8_t tbl[0x10000] __attribute__((aligned(0x10000)));
static uint8_t act[CLM_ACT_BYTES] __attribute__((aligned(8)));
static uint16_t tokens[CLM_DEFAULT_CAP];

static uint8_t loaded, masks_ok, topics = 0xFF;
static menu_run_t run;
static uint16_t fed;

/* ---- memory for the payloads (bump allocation; one model per instance) ------------------- */

extern unsigned char __heap_base;
static uintptr_t brk_;

uint8_t *demo_alloc(uint32_t n) {
    if (!brk_) {
        brk_ = (uintptr_t)&__heap_base;
    }
    uintptr_t p = (brk_ + 7u) & ~(uintptr_t)7u;
    uintptr_t end = p + n;
    uintptr_t have = (uintptr_t)__builtin_wasm_memory_size(0) * 65536u;
    if (end < p) {
        return 0;
    }
    if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535u) / 65536u) == (uintptr_t)-1) {
        return 0;
    }
    brk_ = end;
    return (uint8_t *)p;
}

/* ---- load ------------------------------------------------------------------------------ */

/* model.c tracker_vocab(): FNV-1a over every id's token byte count and bytes (CLMVOC's records,
 * here `voc`: per id one count byte, then the bytes) equals the masks' vocab hash */
static uint8_t tracker_vocab(const uint8_t *voc, uint32_t size) {
    if (M.vocab != clm_mask_vocab) {
        return 0;
    }
    uint32_t h = 0x811C9DC5UL, at = 0;
    for (uint16_t id = 0; id < M.vocab; id++) {
        if (at >= size || at + 1u + voc[at] > size) {
            return 0;
        }
        uint8_t nb = voc[at++];
        h = (h ^ nb) * 0x01000193UL;
        for (uint8_t i = 0; i < nb; i++) {
            h = (h ^ voc[at++]) * 0x01000193UL;
        }
    }
    return at == size && h == clm_mask_vocab_hash;
}

/* data/size: `count` AppVar payloads (CLM00..); voc: the token table; ld2: the CLMLD2 payload
 * (size 0: none); sha6: CLMVOC's 6-byte export sha. Returns clm_load's code. */
int demo_load(uint8_t count, const uint8_t *const *data, const uint32_t *size,
              const uint8_t *voc, uint32_t voc_size, const uint8_t *ld2, uint32_t ld2_size,
              const uint8_t *sha6) {
    clm_mem mem = {kcache, vcache, tbl, act, tokens, CLM_CTX, CLM_DEFAULT_CAP};
    int bad = -1;
    loaded = masks_ok = 0;
    topics = 0xFF;
    int rc = clm_load(&M, data, size, count, &mem, &bad);
    if (rc != CLM_OK) {
        return rc;
    }
    masks_ok = tracker_vocab(voc, voc_size);
    /* model.c decoder(): no CLMLD2 without the masks' vocab */
    clm_program_decoder(&M, &LD, masks_ok && ld2_size ? ld2 : 0, masks_ok ? ld2_size : 0, sha6);
    topics = menu_topics_reason(masks_ok, sha6, M.ld ? M.ld->E : 0, mem.ctx_cap);
    loaded = 1;
    return CLM_OK;
}

int demo_masks_ok(void) { return masks_ok; }
int demo_ld2_ok(void) { return M.ld != 0; }
int demo_topics_reason(void) { return topics; }
const char *demo_reason_text(void) { return menu_reason_text(topics); }

/* ---- the topic table (topics.h, through menu.c where it has an accessor) ------------------- */

int demo_topic_n(void) { return TOPIC_N; }
const char *demo_topic_label(int i) { return i >= 0 && i < TOPIC_N ? menu_label((uint8_t)i) : ""; }
const char *demo_topic_check(int i) { return i >= 0 && i < TOPIC_N ? menu_check((uint8_t)i) : ""; }
const char *demo_topic_rate(int i) { return i >= 0 && i < TOPIC_N ? topic_rate[i] : RANDOM_RATE; }
const char *demo_topic_time(int i) { return i >= 0 && i < TOPIC_N ? topic_time[i] : RANDOM_TIME; }
const uint16_t *demo_topic_ids(int i) { return i >= 0 && i < TOPIC_N ? topic_ids + topic_start[i] : 0; }
int demo_topic_len(int i) {
    return i >= 0 && i < TOPIC_N ? topic_start[i + 1] - topic_start[i] : 0;
}
uint32_t demo_mix_seed(uint32_t seed) { return clm_mix_seed(seed); }

/* ---- a run ------------------------------------------------------------------------------- */

/* topic: 0..TOPIC_N-1, or -1 for Random. level: 2 is the calculator (Masks row on); 3 is the
 * Masks row off (Random) ; 1 is v2.3's masks with LD2 (rc2's hardware-check sheet). For a
 * topic, levels 1 and 3 replace only menu_run's masks and LD2 (not something the calculator
 * does; the page says so). Returns clm_reset's code, or CLM_E_ARG. */
int demo_start(int topic, uint32_t seed, int level) {
    if (!loaded || level < 1 || level > 3 || topic < -1 || topic >= TOPIC_N ||
        (topic >= 0 && topics != TOPICS_OK)) {
        return CLM_E_ARG;
    }
    uint8_t t = topic < 0 ? MENU_RANDOM : (uint8_t)topic;
    menu_run(t, seed, RANDOM_TEMP_Q8, RANDOM_TOP_K, 0, RANDOM_CAP, level != 3, masks_ok,
             M.ld != 0, &run);
    if (level == 1 && masks_ok) {
        run.masks = CLM_MASKS_V23;
    } else if (level == 3 && t != MENU_RANDOM) {
        run.masks = CLM_MASKS_SAFETY;
        run.ld2 = 0;
    }
    M.ld2 = run.ld2 && M.ld;
    fed = 0;
    return clm_reset(&M, run.seed, run.temperature_q8, run.top_k, run.greedy, run.cap, run.masks);
}

/* the next token: the opening's tokens through clm_feed, then clm_step. Returns the token id,
 * CLM_DONE (-1) after <eos> or at the cap, or a clm_feed error (< -1). */
int demo_step(void) {
    if (fed < run.n_prompt) {
        int rc = clm_feed(&M, run.prompt[fed]);
        return rc != CLM_OK ? rc : run.prompt[fed++];
    }
    return clm_step(&M);
}

/* the run's settings, as the core received them */
uint32_t demo_run_seed(void) { return run.seed; }
int demo_run_temp(void) { return run.temperature_q8; }
int demo_run_top_k(void) { return run.top_k; }
int demo_run_cap(void) { return M.cap; }
int demo_run_masks(void) { return M.masks; }
int demo_run_ld2(void) { return M.ld2; }
int demo_run_prompt(void) { return run.n_prompt; }
const uint16_t *demo_tokens(void) { return M.tokens; }
int demo_n_tok(void) { return M.n_tok; }

/* menu.c's menu_draw needs these; the page draws its own screen */
void ui_text(uint8_t col, uint8_t y, const char *s) { (void)col, (void)y, (void)s; }
void ui_line_clear(uint8_t y) { (void)y; }
