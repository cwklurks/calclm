/* Finding and validating model/token/optional decoder AppVars and their runtime RAM. */
#ifndef CALCLM_MODEL_H
#define CALCLM_MODEL_H

#include <stdint.h>

#include "clm.h"

extern clm_t M;

/* Find CLM00..CLMnn and CLMVOC, allocate the V cache, load the core. Returns 0, or writes a
 * short message (lines separated by '\n', 26 columns) into msg and returns nonzero. */
int model_open(char *msg);
void model_close(void);           /* delete the temporary V cache and LD2 workspace AppVar */

/* After gfx_Begin (which clears VRAM, where the activations and the flash routines' RAM
 * window live): load the core again, restore decoder defaults and attach CLMCODE if present.
 * Returns 0 on success;
 * *fast tells whether the flash-resident routines are in use. */
int model_prepare(uint8_t *fast);

uint16_t model_ctx(void);         /* context window actually available (RAM permitting) */
uint8_t model_masks_ok(void);     /* the logit masks can run on this export's vocab */
const uint8_t *vocab_glyphs(uint8_t *count);
const uint8_t *vocab_sha(void);   /* 6 bytes */
/* token id -> token bytes (n <= 2) and display string (ASCII, 0x0A, 0x80+ glyph codes) */
const uint8_t *vocab_token(uint16_t id, uint8_t *nbytes, const uint8_t **disp, uint8_t *ndisp);

#endif
