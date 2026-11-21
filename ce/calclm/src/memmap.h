/* CalcLM memory map on the TI-84 Plus CE (see ce/calclm/README.md).
 *
 *   0xD052C6..       BSS (toolchain region, OS scratch RAM): K cache (52.5 KB), the generated
 *                    tokens (2 KB), core state (with the tracker), font
 *   user RAM         this program (copied there by the OS), its libraries, and the temporary
 *                    AppVar CLMTMP holding the V cache (48 KB at the full 256-token context)
 *                    and, only with accepted LD2 offsets, its 2,652-byte workspace
 *   0xD40000-D4F2FF  VRAM bank 0: the matvec kernels' table bank (pages 0..242)
 *   0xD4F300-D529FF  VRAM: activations (CLM_ACT_BYTES: 12,177 for v2.2, 13,011 for W)
 *   0xD52A00-D52BFF  VRAM: RAM window of the flash-resident routines (core/clm_code.h)
 *   0xD52C00-D657FF  VRAM: the visible 8bpp screen (graphx draws only here)
 *   archive          CLM00..CLMnn (weights, read in place; 11 AppVars for v2.2, 16 for W),
 *                    CLMVOC (token table), CLMCODE (optional: the assembly routines, executed
 *                    in place)
 */
#ifndef CALCLM_MEMMAP_H
#define CALCLM_MEMMAP_H

#include <stdint.h>

#define MM_TBL ((uint8_t *)0xD40000)
#define MM_ACT ((uint8_t *)0xD4F300)
#define MM_SCREEN 0xD52C00
#define MM_FRAM 0xD52A00
#define MM_ACT_LIMIT (MM_FRAM - 0xD4F300)

#endif
