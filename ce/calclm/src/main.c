/* CALCLM: a ternary transformer writes TI-BASIC on the calculator.
 *
 * Thin OS layer around the inference core (ce/calclm/core): finds the archived model
 * (CLM00..CLMnn) and token table (CLMVOC), opens on the topic menu (menu.c, v2.5: a measured
 * formula or a random program), lets the user pick seed / temperature / top-k / greedy / logit
 * masks / length for Random ([mode]), streams the program as TI-BASIC text with tokens/s
 * (a topic's opening first, as it is fed), and saves it as prgmGEN. Keys: arrows, digits,
 * [del], [enter], [clear], [mode], [2nd]. A format 1 export (v1) stops at its 256-token
 * context whatever the length setting.
 */
#include <fileioc.h>
#include <stdint.h>
#include <string.h>
#include <sys/power.h>
#include <sys/rtc.h>
#include <ti/getcsc.h>
#include <ti/screen.h>
#include <time.h>

#include "menu.h"
#include "model.h"
#include "ui.h"

#define GEN_NAME "GEN"
#define NOTE_Y (UI_TEXT_Y + UI_LINE * UI_ROWS)   /* 204: the status line above 216, 228 */
static const uint16_t temps[] = {64, 128, 154, 179, 205, 230, 256, 282, 307, 384};  /* T * 256 */
static const char *const temp_txt[] = {"0.25", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1",
                                       "1.2", "1.5"};
static const uint8_t topks[] = {1, 5, 10, 20, 40, 64};
static const uint16_t caps[] = {256, 512, CLM_DEFAULT_CAP};   /* tokens, <bos> included */

enum { DONE_QUIT, DONE_MENU, DONE_RETRY };

static uint32_t seed;
static uint8_t t_i = 4, k_i = 3, c_i = 2, greedy, sel, masks;
static uint16_t n_gen;
static uint8_t fast;
/* initialized, so they live in the program image, not in the (nearly full) BSS */
static uint8_t topics = 0xFF;     /* menu_topics_reason(): TOPICS_OK offers the topics */
static uint32_t rate100;          /* tokens/s * 100 (BSS, as in v2.4) */
static menu_t menu = {0, 1};

static void show_error(const char *msg) {
    os_ClrHome();
    os_PutStrFull("CalcLM");
    os_NewLine();
    const char *p = msg;
    char line[27];
    while (*p) {
        uint8_t n = 0;
        while (*p && *p != '\n' && n < 26) {
            line[n++] = *p++;
        }
        line[n] = 0;
        if (*p == '\n') {
            p++;
        }
        os_NewLine();
        os_PutStrFull(line);
    }
    os_NewLine();
    os_NewLine();
    os_PutStrFull("Press any key.");
    while (!os_GetCSC()) {
    }
}

static void status(const char *a, const char *b) {
    ui_line_clear(216);
    ui_line_clear(228);
    ui_text(0, 216, a);
    ui_text(0, 228, b);
}

static void note(const char *s) {
    ui_line_clear(NOTE_Y);
    ui_text(0, NOTE_Y, s);
}

/* the title bar; ld2: the coming run uses LD2 */
static void header(uint8_t ld2) {
    char s[41], *p = s;
    ui_line_clear(2);
    p = memcpy(p, "CalcLM  ctx ", 12), p += 12;
    p = ui_utoa(p, model_ctx());
    p = memcpy(p, "  vocab ", 8), p += 8;
    p = ui_utoa(p, M.vocab);
    p = memcpy(p, fast ? "  fast" : "  slow", 6), p += 6;
    memcpy(p, ld2 ? " ld2" : "", ld2 ? 5 : 1);
    ui_invert(1);
    ui_text(0, 2, s);
    ui_invert(0);
}

/* tokens a generation can write: the cache stops a model that does not slide (format 1, or
 * a cache shortened by low RAM) */
static uint16_t length(void) {
    uint16_t c = caps[c_i];
    return M.slide || c < model_ctx() ? c : model_ctx();
}

static void settings_draw(void) {
    static const char *const label[] = {"Seed  ", "Temp  ", "Top-k ", "Mode  ", "Masks ",
                                        "Length "};
    char s[41];
    header(M.ld && masks && !greedy);
    for (uint8_t i = 0; i < 6; i++) {
        uint8_t y = (uint8_t)(UI_TEXT_Y + 6 + 14 * i);
        char *p = s;
        *p++ = (char)(sel == i ? '>' : ' ');
        p = memcpy(p, label[i], 6), p += 6;
        if (i == 0) {
            p = ui_utoa(p, seed);
        } else if (i == 1) {
            p = memcpy(p, temp_txt[t_i], strlen(temp_txt[t_i])), p += strlen(temp_txt[t_i]);
        } else if (i == 2) {
            p = ui_utoa(p, topks[k_i]);
        } else if (i == 3) {
            p = memcpy(p, greedy ? "greedy" : "sampled", greedy ? 6 : 7), p += greedy ? 6 : 7;
        } else if (i == 4) {
            /* off: v2.3's grammar masks and LD2 off; destructive and safety bans stay on */
            const char *t = !model_masks_ok() ? "n/a" : masks ? "on" : "off (safety rules on)";
            p = memcpy(p, t, strlen(t)), p += strlen(t);
        } else {
            p = ui_utoa(p, length());
            p = memcpy(p, " tokens", 7), p += 7;
        }
        *p = 0;
        ui_line_clear(y);
        ui_text(1, y, s);
    }
    ui_text(1, UI_TEXT_Y + 98, "digits: seed  [del]: clear seed");
    ui_text(1, UI_TEXT_Y + 110, "[<] [>]: change   [up] [down]: move");
    if (greedy) {
        ui_text(1, UI_TEXT_Y + 122, "greedy ignores seed, temp, top-k  ");
    } else {
        ui_line_clear(UI_TEXT_Y + 122);
    }
    ui_text(1, UI_TEXT_Y + 140, "Topics use their own settings:");
    ui_text(1, UI_TEXT_Y + 152, "only the seed applies to them.");
    status("[mode] back to the menu", "[clear] quit");
}

/* returns 1 back to the menu, 0 to quit */
static int settings(void) {
    ui_clear();
    settings_draw();
    for (;;) {
        uint8_t k = ui_wait_key();
        if (k == K_CLEAR) {
            return 0;
        }
        if (k == K_MODE || k == K_ENTER) {
            return 1;
        }
        if (k == K_UP && sel) {
            sel--;
        } else if (k == K_DOWN && sel < 5) {
            sel++;
        } else if (k >= K_0 && k <= K_9) {
            uint32_t d = (uint32_t)(k - K_0);
            seed = seed < 100000000UL ? seed * 10 + d : d;
            sel = 0;
        } else if (k == K_DEL) {
            seed = 0;
        } else if (k == K_LEFT || k == K_RIGHT) {
            int8_t step = k == K_LEFT ? -1 : 1;
            if (sel == 1 && t_i + step >= 0 && t_i + step < (int8_t)sizeof temps / 2) {
                t_i = (uint8_t)(t_i + step);
            } else if (sel == 2 && k_i + step >= 0 && k_i + step < (int8_t)sizeof topks) {
                k_i = (uint8_t)(k_i + step);
            } else if (sel == 3) {
                greedy ^= 1;
            } else if (sel == 4) {
                masks ^= 1;
            } else if (sel == 5 && c_i + step >= 0 && c_i + step < (int8_t)(sizeof caps / 2)) {
                c_i = (uint8_t)(c_i + step);
            }
        }
        settings_draw();
    }
}

/* the start screen; returns MENU_PICK, MENU_SETTINGS or MENU_QUIT */
static uint8_t menu_screen(void) {
    ui_clear();
    header(M.ld && masks && !greedy);
    menu_draw(&menu, topics);
    status("[enter] write  [mode] settings", "[clear] quit");
    for (;;) {
        uint8_t a = menu_key(&menu, ui_wait_key());
        if (a != MENU_NONE) {
            return a;
        }
        menu_draw(&menu, topics);
    }
}

static void progress(uint32_t ticks, const char *tail) {
    char a[41], *p = a;
    p = memcpy(p, "tokens ", 7), p += 7;
    p = ui_utoa(p, n_gen);
    p = memcpy(p, "   ", 3), p += 3;
    rate100 = ticks ? (uint32_t)n_gen * (CLOCKS_PER_SEC * 100UL) / ticks : 0;
    p = ui_fixed2(p, rate100);
    memcpy(p, " tok/s", 7);
    status(a, tail);
}

static void stream_token(uint16_t t) {
    uint8_t nb, nd;
    const uint8_t *disp;
    vocab_token(t, &nb, &disp, &nd);
    ui_stream_bytes(disp, nd);
}

/* returns 1 when the program ended on its own (<eos>, the cap or a full window) */
static int generate(uint8_t topic) {
    menu_run_t r;
    menu_run(topic, seed, temps[t_i], topks[k_i], greedy, caps[c_i], masks, model_masks_ok(),
             M.ld != 0, &r);
    ui_clear();
    header(r.ld2 && M.ld);
    char s[41], *p = s;
    if (topic != MENU_RANDOM) {
        const char *l = menu_label(topic);
        p = memcpy(p, l, strlen(l)), p += strlen(l);
        p = memcpy(p, "  seed ", 7), p += 7;
        ui_utoa(p, seed);
    } else {
        p = memcpy(p, "seed ", 5), p += 5;
        p = ui_utoa(p, seed);
        if (greedy) {
            memcpy(p, "  greedy", 9);
        } else {
            p = memcpy(p, "  T ", 4), p += 4;
            p = memcpy(p, temp_txt[t_i], strlen(temp_txt[t_i])), p += strlen(temp_txt[t_i]);
            p = memcpy(p, "  top-k ", 8), p += 8;
            ui_utoa(p, topks[k_i]);
        }
    }
    ui_text(0, 12, s);
    ui_stream_reset();
    M.ld2 = r.ld2 && M.ld;
    n_gen = 0;
    rate100 = 0;
    if (clm_reset(&M, r.seed, r.temperature_q8, r.top_k, r.greedy, r.cap, r.masks) != CLM_OK) {
        return 0;                                /* "stopped: 0 tok" */
    }
    clock_t t0 = clock();
    for (uint16_t i = 0; i < r.n_prompt; i++) {   /* the topic's opening, as it is fed */
        if (clm_feed(&M, r.prompt[i]) != CLM_OK) {
            return 0;
        }
        n_gen++;
        stream_token(r.prompt[i]);
        p = memcpy(s, "reading the topic ", 18), p += 18;
        p = ui_utoa(p, i + 1u);
        *p++ = '/';
        ui_utoa(p, r.n_prompt);
        status(s, "[clear] stop");
        if (ui_clear_held()) {
            return 0;
        }
    }
    progress(0, "[clear] stop");
    for (;;) {
        int t = clm_step(&M);
        if (t < 0) {
            return 1;
        }
        n_gen++;
        uint32_t ticks = (uint32_t)(clock() - t0);
        if (t == CLM_EOS) {
            progress(ticks, "[clear] stop");
            return 1;
        }
        stream_token((uint16_t)t);
        progress(ticks, "[clear] stop");
        if (ui_clear_held()) {
            return 0;
        }
    }
}

static uint32_t program_bytes(void) {
    uint32_t n = 0;
    for (uint16_t i = 1; i < M.n_tok; i++) {
        uint8_t nb, nd;
        const uint8_t *disp;
        if (M.tokens[i] >= 4) {
            vocab_token(M.tokens[i], &nb, &disp, &nd);
            n += nb;
        }
    }
    return n;
}

/* save the generated tokens (a topic's opening included) as prgmGEN; returns a status line */
static const char *save(void) {
    uint8_t h = ti_OpenVar(GEN_NAME, "r", OS_TYPE_PRGM);
    if (h) {
        ti_Close(h);
        status("prgmGEN exists. Overwrite?", "[enter] yes   [clear] no");
        uint8_t k;
        while ((k = ui_wait_key()) != K_ENTER && k != K_CLEAR) {
        }
        if (k == K_CLEAR) {
            return "not saved";
        }
    }
    h = ti_OpenVar(GEN_NAME, "w", OS_TYPE_PRGM);
    if (!h) {
        return "save failed (RAM full?)";
    }
    for (uint16_t i = 1; i < M.n_tok; i++) {
        uint8_t nb, nd;
        const uint8_t *disp;
        if (M.tokens[i] >= 4) {
            const uint8_t *b = vocab_token(M.tokens[i], &nb, &disp, &nd);
            if (ti_Write(b, 1, nb, h) != nb) {
                ti_Close(h);
                return "save failed (RAM full?)";
            }
        }
    }
    ti_Close(h);
    return "saved as prgmGEN";
}

/* after a program: DONE_RETRY (another try, seed + 1), DONE_MENU or DONE_QUIT. The done
 * count is the tokens after <bos> (a topic's opening included), as in hw_check's sheet. */
static uint8_t done(int finished, uint8_t topic) {
    char a[41], *p = a;
    p = memcpy(p, finished ? "done: " : "stopped: ", finished ? 6 : 9);
    p += finished ? 6 : 9;
    p = ui_utoa(p, n_gen);
    p = memcpy(p, " tok, ", 6), p += 6;
    p = ui_fixed2(p, rate100);
    memcpy(p, " tok/s", 7);
    const char *saved = 0;
    for (;;) {
        note(saved ? saved : a);
        status(topic != MENU_RANDOM ? menu_check(topic) : "[clear] quit",
               "[enter] save  [2nd] retry  [mode] menu");
        uint8_t k = ui_wait_key();
        if (k == K_CLEAR) {
            return DONE_QUIT;
        }
        if (k == K_MODE) {
            return DONE_MENU;
        }
        if (k == K_2ND) {
            return DONE_RETRY;
        }
        if (k == K_ENTER && !saved) {
            saved = program_bytes() ? save() : "nothing to save";
        }
    }
}

/* after gfx_Begin or a save (which may move RAM variables, the V cache AppVar among them):
 * load again and decide whether topics can run; returns 0 on success */
static int prepare(void) {
    if (model_prepare(&fast)) {
        return 1;
    }
    uint8_t was = topics;
    topics = menu_topics_reason(model_masks_ok(), vocab_sha(), M.ld ? M.ld->E : 0,
                                model_ctx());
    if (topics != was) {
        menu_init(&menu, topics == TOPICS_OK);
    }
    return 0;
}

int main(void) {
    static char msg[160];
    if (model_open(msg)) {
        show_error(msg);
        return 0;
    }
    seed = rtc_Time() % 1000000UL;
    masks = M.alibi;                  /* v2's deployment setting; v1 as shipped (no masks) */
    os_DisableAPD();                  /* a 1024-token program takes about 20 minutes */
    uint8_t ng;
    const uint8_t *glyphs = vocab_glyphs(&ng);
    if (ui_begin(glyphs, ng)) {
        model_close();
        os_EnableAPD();
        show_error("graphx did not take the\nscreen at 0xD52C00.");
        return 0;
    }
    if (prepare()) {                            /* VRAM was cleared by gfx_Begin */
        model_close();
        ui_end();
        os_EnableAPD();
        show_error("Model reload failed.");
        return 0;
    }
    for (;;) {
        uint8_t a = menu_screen();
        if (a == MENU_QUIT) {
            break;
        }
        if (a == MENU_SETTINGS) {
            if (!settings()) {
                break;
            }
            continue;
        }
        uint8_t topic = menu_topic(&menu), next;
        do {
            next = done(generate(topic), topic);
            if (next == DONE_RETRY) {
                seed = menu_retry_seed(seed);
            }
            /* saving may have moved RAM variables (the V cache AppVar): set up again */
            model_close();
            if (model_open(msg) || prepare()) {
                model_close();
                ui_end();
                os_EnableAPD();
                show_error(msg);
                return 0;
            }
            glyphs = vocab_glyphs(&ng);
            ui_font(glyphs, ng);
        } while (next == DONE_RETRY && (topic == MENU_RANDOM || topics == TOPICS_OK));
        if (next == DONE_QUIT) {
            break;
        }
    }
    model_close();
    ui_end();
    os_EnableAPD();
    return 0;
}
