/* Optional flash-resident copy of the core's assembly routines (tools/build_flash.py).
 *
 * The RAM build of every routine starts with a patchable `jp` to its own body. clm_code_attach
 * checks a CLMCODE payload (format, build id, tables), writes the flash addresses into the
 * RAM trampolines of the flash copy (all inside CLM_FRAM .. CLM_FRAM + CLM_FRAM_SIZE) and
 * points the entry jumps at the flash copies; clm_code_detach points them back. Results are
 * identical either way (same source); only the speed differs. On the host both are no-ops. */
#ifndef CLM_CODE_H
#define CLM_CODE_H

#include <stdint.h>

#define CLM_FRAM 0xD52A00UL
#define CLM_FRAM_SIZE 512
#define CLM_CODE_ENTRIES 17

#define CLM_CODE_OK 0
#define CLM_CODE_E_FORMAT (-1)    /* not a CLMCODE payload, or truncated */
#define CLM_CODE_E_BUILD (-2)     /* built from different assembly sources than this core */
#define CLM_CODE_E_TABLE (-3)     /* a table entry points outside the code or the RAM window */
#define CLM_CODE_E_HOST (-4)      /* not on the eZ80 */

int clm_code_attach(const uint8_t *payload, uint32_t size);
void clm_code_detach(void);

#endif
