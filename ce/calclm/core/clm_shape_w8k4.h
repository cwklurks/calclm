/* E20 spike (PLAN-E20.md Design 1): W8 with 4-bit K/V. The dims are tools/gen_shape.py's for
 * runs/e16/export/W8-s10/model.json (ce/tests/test_calclm_kv4_build.py checks them); the last
 * two lines are the spike's: CLM_MAX_APPVARS 24 (D5; W8 has 20 model AppVars) and CLM_KV4,
 * which turns on the 4-bit K/V cache (core/clm_kv4.h) in the core, the driver and the program. */
#ifndef CLM_SHAPE_H
#define CLM_SHAPE_H

#define CLM_DIM 256
#define CLM_LAYERS 8
#define CLM_HEADS 8
#define CLM_HEAD_DIM 32
#define CLM_HIDDEN 576
#define CLM_CTX 256

/* runtime-sized: limits only */
#define CLM_VOCAB_MAX 1024
#define CLM_MAX_APPVARS 24

#define CLM_KV4 1

#endif
