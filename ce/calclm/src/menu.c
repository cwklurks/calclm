/* CalcLM v2.5's start screen (menu.h, PHASE3-topic-menu.md 1a). */
#include <string.h>

#include "clm.h"
#include "menu.h"
#include "topics.h"
#include "ui.h"

#define ROW_Y(i) ((uint8_t)(UI_TEXT_Y + 6 + 13 * (i)))

uint8_t menu_topics_reason(uint8_t masks_ok, const uint8_t *export_sha6, const uint32_t *ld2_E,
                           uint16_t ctx) {
    if (!masks_ok) {
        return TOPICS_NO_MASKS;
    }
#if defined(CLM_KV4)
    (void)export_sha6;
    (void)ld2_E;
    (void)ctx;
    return TOPICS_OTHER_MODEL;            /* 4-bit K/V writes other programs than measured */
#else
    if (memcmp(export_sha6, TOPIC_EXPORT_SHA, 6) != 0) {
        return TOPICS_OTHER_MODEL;
    }
    if (!ld2_E) {
        return TOPICS_NO_LD2;
    }
    if (memcmp(ld2_E, topic_ld2_E, sizeof topic_ld2_E) != 0) {
        return TOPICS_OTHER_FIT;
    }
    return ctx == CLM_CTX ? TOPICS_OK : TOPICS_SHORT_CTX;
#endif
}

const char *menu_reason_text(uint8_t reason) {
    switch (reason) {
    case TOPICS_NO_MASKS: return "Topics off: this vocab has no masks";
    case TOPICS_OTHER_MODEL: return "Topics off: not the measured model";
    case TOPICS_NO_LD2: return "Topics off: CLMLD2 missing or refused";
    case TOPICS_SHORT_CTX: return "Topics off: RAM short of 256 tokens";
    case TOPICS_OTHER_FIT: return "Topics off: CLMLD2 not the measured fit";
    default: return "";
    }
}

void menu_init(menu_t *s, uint8_t topics_ok) {
    s->n = topics_ok ? TOPIC_N + 1 : 1;
    s->sel = 0;
}

uint8_t menu_key(menu_t *s, uint8_t key) {
    if (key == K_ENTER) {
        return MENU_PICK;
    }
    if (key == K_MODE) {
        return MENU_SETTINGS;
    }
    if (key == K_CLEAR) {
        return MENU_QUIT;
    }
    if (key == K_UP && s->sel) {
        s->sel--;
    } else if (key == K_DOWN && s->sel + 1 < s->n) {
        s->sel++;
    } else if (key == K_0) {
        s->sel = (uint8_t)(s->n - 1);
    } else if (key >= K_1 && key <= K_9 && (uint8_t)(key - K_1) < s->n - 1) {
        s->sel = (uint8_t)(key - K_1);
    }
    return MENU_NONE;
}

uint8_t menu_topic(const menu_t *s) {
    return s->sel + 1 < s->n ? s->sel : MENU_RANDOM;
}

void menu_run(uint8_t topic, uint32_t seed, uint16_t temperature_q8, uint8_t top_k,
              uint8_t greedy, uint16_t cap, uint8_t masks_row, uint8_t masks_ok,
              uint8_t ld2_ok, menu_run_t *out) {
    if (topic != MENU_RANDOM) {
        out->seed = clm_mix_seed(seed);
        out->temperature_q8 = TOPIC_TEMP_Q8;
        out->top_k = TOPIC_TOP_K;
        out->greedy = 0;
        out->cap = TOPIC_CAP;
        out->masks = TOPIC_MASKS;
        out->ld2 = 1;
        out->prompt = topic_ids + topic_start[topic];
        out->n_prompt = (uint16_t)(topic_start[topic + 1] - topic_start[topic]);
        return;
    }
    out->seed = seed;
    out->temperature_q8 = temperature_q8;
    out->top_k = top_k;
    out->greedy = greedy;
    out->cap = cap;
    out->masks = !masks_ok ? 0 : masks_row ? CLM_MASKS_V24 : CLM_MASKS_SAFETY;
    out->ld2 = ld2_ok && masks_ok && masks_row && !greedy;
    out->prompt = 0;
    out->n_prompt = 0;
}

uint32_t menu_retry_seed(uint32_t seed) {
    return seed + 1;
}

const char *menu_label(uint8_t topic) {
    return topic == MENU_RANDOM ? RANDOM_LABEL : topic_label[topic];
}

const char *menu_check(uint8_t topic) {
    return topic == MENU_RANDOM ? "" : topic_check[topic];
}

/* s right-aligned so that it ends before column `end` */
static void right(char *buf, uint8_t end, const char *s) {
    size_t n = strlen(s);
    memcpy(buf + end - n, s, n);
}

char *menu_line(char *buf, const menu_t *s, uint8_t entry) {
    uint8_t random = entry + 1 == s->n;
    uint8_t topic = random ? MENU_RANDOM : entry;
    const char *label = menu_label(topic);
    memset(buf, ' ', 38);
    buf[38] = 0;
    buf[0] = (char)(s->sel == entry ? '>' : ' ');
    buf[1] = (char)(random ? '0' : entry < 9 ? '1' + entry : ' ');
    memcpy(buf + 3, label, strlen(label));
    right(buf, 31, random ? RANDOM_RATE : topic_rate[topic]);
    right(buf, 38, random ? RANDOM_TIME : topic_time[topic]);
    return buf;
}

void menu_draw(const menu_t *s, uint8_t reason) {
    char line[41];
    ui_line_clear(UI_TEXT_Y - 8);
    ui_text(0, UI_TEXT_Y - 8, "What should CalcLM write? right   time");
    for (uint8_t i = 0; i < s->n; i++) {
        ui_line_clear(ROW_Y(i));
        ui_text(0, ROW_Y(i), menu_line(line, s, i));
    }
    if (reason != TOPICS_OK) {
        ui_line_clear(ROW_Y(2));
        ui_text(1, ROW_Y(2), menu_reason_text(reason));
    }
}
