/* Screen and keys for CalcLM (graphx 8bpp on the second half of VRAM, keypadc). */
#ifndef CALCLM_UI_H
#define CALCLM_UI_H

#include <stdint.h>

enum {
    K_NONE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ENTER, K_CLEAR, K_DEL, K_MODE, K_2ND,
    K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9
};

#define UI_COLS 40
#define UI_TEXT_Y 24              /* first text line */
#define UI_LINE 10                /* pixels per text line */
#define UI_ROWS 18                /* text lines in the scrolling area (y 24..203; 204 is the
                                     done screen's third status line) */

int ui_begin(const uint8_t *glyphs, uint8_t nglyphs);   /* 0 on success */
void ui_end(void);
void ui_font(const uint8_t *glyphs, uint8_t nglyphs);  /* (re)install the font */
void ui_clear(void);
void ui_text(uint8_t col, uint8_t y, const char *s);   /* ASCII at a pixel row */
void ui_line_clear(uint8_t y);
void ui_invert(uint8_t on);

/* streaming text area: TI-BASIC listing with ':' at each line start */
void ui_stream_reset(void);
void ui_stream_bytes(const uint8_t *s, uint8_t n);

uint8_t ui_key(void);             /* newly pressed key, K_NONE if none (non-blocking) */
uint8_t ui_wait_key(void);
uint8_t ui_clear_held(void);      /* [clear] down right now */

char *ui_utoa(char *p, uint32_t v);                    /* returns the end */
char *ui_fixed2(char *p, uint32_t v100);               /* v / 100 with 2 decimals */

#endif
