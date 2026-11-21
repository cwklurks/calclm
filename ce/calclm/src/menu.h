/* CalcLM v2.5's start screen: "What should CalcLM write?" (PHASE3-topic-menu.md 1a). The
 * entries are the measured topics of topics.h, in menu order, then Random program. Everything
 * but menu_draw is pure (no OS calls), so it links into a host test (test_calclm_menu.py). */
#ifndef CALCLM_MENU_H
#define CALCLM_MENU_H

#include <stdint.h>

#define MENU_RANDOM 0xFF          /* menu_topic() of the Random entry */

/* menu_key() actions */
enum { MENU_NONE, MENU_PICK, MENU_SETTINGS, MENU_QUIT };

/* why topics cannot run (menu_topics_reason); 0: they can */
enum { TOPICS_OK, TOPICS_NO_MASKS, TOPICS_OTHER_MODEL, TOPICS_NO_LD2, TOPICS_SHORT_CTX,
       TOPICS_OTHER_FIT };

typedef struct {
    uint8_t sel;                  /* selected entry */
    uint8_t n;                    /* entries: TOPIC_N + 1 with topics, else 1 (Random) */
} menu_t;

/* what one generation uses */
typedef struct {
    uint32_t seed;                /* the decoder's seed (mixed for a topic) */
    uint16_t temperature_q8, cap;
    uint8_t top_k, greedy, masks, ld2;
    const uint16_t *prompt;       /* the opening (a topic), else 0 */
    uint16_t n_prompt;
} menu_run_t;

/* Topics run only with the masks' vocab, on the measured model (CLMVOC's export sha), with the
 * accepted CLMLD2's offsets equal to the measured fit (ld2_E: 16 offsets, 0 without LD2), with
 * the full 256-token context, and not in the E20 4-bit K/V build: the measurement covers
 * nothing else. */
uint8_t menu_topics_reason(uint8_t masks_ok, const uint8_t *export_sha6, const uint32_t *ld2_E,
                           uint16_t ctx);
const char *menu_reason_text(uint8_t reason);

void menu_init(menu_t *s, uint8_t topics_ok);
/* [up] [down] move, 1-9 jump to a topic, 0 to Random, [enter] pick, [mode] settings,
 * [clear] quit */
uint8_t menu_key(menu_t *s, uint8_t key);
uint8_t menu_topic(const menu_t *s);   /* topic index of the selection, or MENU_RANDOM */

/* a topic: its fixed settings (T 0.8, top-k 20, sampled, cap 256, masks level 2, LD2) and
 * the seed mixed by clm_mix_seed; Random: the settings rows, the raw seed, masks level 2 with
 * the Masks row on and the destructive and safety bans alone (level 3) with it off */
void menu_run(uint8_t topic, uint32_t seed, uint16_t temperature_q8, uint8_t top_k,
              uint8_t greedy, uint16_t cap, uint8_t masks_row, uint8_t masks_ok,
              uint8_t ld2_ok, menu_run_t *out);
uint32_t menu_retry_seed(uint32_t seed);   /* [2nd] another try: the next seed */

const char *menu_label(uint8_t topic);
const char *menu_check(uint8_t topic);     /* the check line (at most 40 columns) */
/* one menu row, 40 columns at most: marker, digit, label, rate, time */
char *menu_line(char *buf, const menu_t *s, uint8_t entry);

void menu_draw(const menu_t *s, uint8_t reason);   /* the screen (ui.h) */

#endif
