/* Screen and keys for CalcLM. graphx runs in 8bpp; the visible frame and graphx's drawing
 * buffer are both moved to the second half of VRAM (MM_SCREEN), because the first half holds
 * the matvec kernels' table bank and the activations. Nothing here may draw to the buffer. */
#include <graphx.h>
#include <keypadc.h>
#include <string.h>
#include <sys/lcd.h>

#include "memmap.h"
#include "ui.h"

#define C_BG 255
#define C_FG 0

static uint8_t font[256 * 8];
static uint8_t col, row;
static uint8_t prev[8];

void ui_font(const uint8_t *glyphs, uint8_t nglyphs) {
    const uint8_t *def = gfx_SetFontData(font);     /* returns the default font */
    if (def != font) {
        memcpy(font, def, 128 * 8);
    }
    memset(font + 128 * 8, 0, 128 * 8);
    memcpy(font + 128 * 8, glyphs, (size_t)nglyphs * 8);
    gfx_SetMonospaceFont(8);
}

int ui_begin(const uint8_t *glyphs, uint8_t nglyphs) {
    gfx_Begin();
    lcd_UpBase = MM_SCREEN;                         /* show the second half of VRAM */
    lcd_LpBase = MM_SCREEN;                         /* graphx's current drawing buffer */
    gfx_SetDrawScreen();
    if (lcd_LpBase != MM_SCREEN || lcd_UpBase != MM_SCREEN) {
        gfx_End();                                  /* never draw over the table bank */
        return 1;
    }
    ui_font(glyphs, nglyphs);
    gfx_SetTextFGColor(C_FG);
    gfx_SetTextBGColor(C_BG);
    gfx_SetTextTransparentColor(254);
    memset(prev, 0xff, sizeof prev);                /* ignore keys held at start */
    ui_clear();
    return 0;
}

void ui_end(void) {
    gfx_End();
}

void ui_clear(void) {
    gfx_FillScreen(C_BG);
}

void ui_invert(uint8_t on) {
    gfx_SetTextFGColor(on ? C_BG : C_FG);
    gfx_SetTextBGColor(on ? C_FG : C_BG);
}

void ui_text(uint8_t c, uint8_t y, const char *s) {
    gfx_SetTextXY(8 * c, y);
    while (*s && c < UI_COLS) {
        gfx_PrintChar(*s++);
        c++;
    }
}

void ui_line_clear(uint8_t y) {
    gfx_SetColor(C_BG);
    gfx_FillRectangle_NoClip(0, y, 320, UI_LINE);
}

void ui_stream_reset(void) {
    col = 0;
    row = 0;
    for (uint8_t r = 0; r < UI_ROWS; r++) {
        ui_line_clear((uint8_t)(UI_TEXT_Y + UI_LINE * r));
    }
    ui_text(0, UI_TEXT_Y, ":");
    col = 1;
}

static void newline(uint8_t colon) {
    if (row + 1 < UI_ROWS) {
        row++;
    } else {
        gfx_SetClipRegion(0, UI_TEXT_Y, 320, UI_TEXT_Y + UI_LINE * UI_ROWS);
        gfx_ShiftUp(UI_LINE);
        gfx_SetClipRegion(0, 0, 320, 240);
        ui_line_clear((uint8_t)(UI_TEXT_Y + UI_LINE * row));
    }
    col = 0;
    if (colon) {
        ui_text(0, (uint8_t)(UI_TEXT_Y + UI_LINE * row), ":");
    }
    col = 1;
}

void ui_stream_bytes(const uint8_t *s, uint8_t n) {
    for (uint8_t i = 0; i < n; i++) {
        if (s[i] == 0x0A) {
            newline(1);
            continue;
        }
        if (col >= UI_COLS) {
            newline(0);
        }
        gfx_SetTextXY(8 * col, UI_TEXT_Y + UI_LINE * row);
        gfx_PrintChar((char)s[i]);
        col++;
    }
}

static const struct {
    uint8_t group, mask, key;
} keymap[] = {
    {7, kb_Up, K_UP}, {7, kb_Down, K_DOWN}, {7, kb_Left, K_LEFT}, {7, kb_Right, K_RIGHT},
    {6, kb_Enter, K_ENTER}, {6, kb_Clear, K_CLEAR}, {1, kb_Del, K_DEL}, {1, kb_Mode, K_MODE},
    {1, kb_2nd, K_2ND}, {3, kb_0, K_0}, {3, kb_1, K_1}, {4, kb_2, K_2}, {5, kb_3, K_3},
    {3, kb_4, K_4}, {4, kb_5, K_5}, {5, kb_6, K_6}, {3, kb_7, K_7}, {4, kb_8, K_8},
    {5, kb_9, K_9},
};

uint8_t ui_key(void) {
    uint8_t now[8], hit = K_NONE;
    kb_Scan();
    for (uint8_t g = 1; g < 8; g++) {
        now[g] = kb_Data[g];
    }
    for (uint8_t i = 0; i < sizeof keymap / sizeof keymap[0]; i++) {
        uint8_t g = keymap[i].group, m = keymap[i].mask;
        if ((now[g] & m) && !(prev[g] & m) && hit == K_NONE) {
            hit = keymap[i].key;
        }
    }
    for (uint8_t g = 1; g < 8; g++) {
        prev[g] = now[g];
    }
    return hit;
}

uint8_t ui_wait_key(void) {
    uint8_t k;
    while ((k = ui_key()) == K_NONE) {
    }
    return k;
}

uint8_t ui_clear_held(void) {
    kb_Scan();
    return (kb_Data[6] & kb_Clear) != 0;
}

char *ui_utoa(char *p, uint32_t v) {
    char tmp[11];
    uint8_t n = 0;
    do {
        tmp[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n) {
        *p++ = tmp[--n];
    }
    *p = 0;
    return p;
}

char *ui_fixed2(char *p, uint32_t v100) {
    p = ui_utoa(p, v100 / 100);
    *p++ = '.';
    *p++ = (char)('0' + v100 / 10 % 10);
    *p++ = (char)('0' + v100 % 10);
    *p = 0;
    return p;
}
