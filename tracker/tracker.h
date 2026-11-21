/* TI-BASIC program-state tracker for CalcLM v2.
 *
 * One C99 implementation shared by training (host gcc, through tracker/bind.py) and the
 * calculator (CE toolchain, ez80-clang). Integer only, no allocation, no globals (only const
 * tables), fixed-size state. Semantics: tracker/README.md; Python spec: tracker/spec.py.
 *
 *     trk_state st; trk_feat f;
 *     tracker_reset(&st);                 // or step the <bos> token
 *     tracker_step(&st, token_id, &f);    // f = state AFTER the token
 *
 * The features of position i describe the program after token i has been read: they are the
 * model's extra input at position i (aligned with the input token, not the target).
 */
#ifndef TRACKER_H
#define TRACKER_H

#include <stdint.h>

#define TRK_NCAT 9          /* categorical groups, see trk_feat.cat */
#define TRK_MAX_BLOCKS 32   /* block types kept; deeper blocks report TRK_BLK_DEEP */
#define TRK_MAX_LISTS 16    /* custom list names kept */
#define TRK_LABEL_BYTES 176 /* 1406 label names (1-2 of A-Z 0-9 theta), one bit each */

/* cat[0] ctx: lexical context for the next token */
#define TRK_CTX_START 0     /* next token begins a statement */
#define TRK_CTX_EXPR 1      /* inside a statement */
#define TRK_CTX_STRING 2    /* inside a string literal */
#define TRK_CTX_STORE 3     /* right after -> (a store target comes next) */
#define TRK_CTX_LBL 4       /* reading a Lbl name */
#define TRK_CTX_REF 5       /* reading a Goto or Menu( label name */
#define TRK_CTX_LIST 6      /* reading a custom list name */
#define TRK_CTX_PRGM 7      /* reading a prgm name */
#define TRK_CTX_DELVAR 8    /* right after DelVar */
/* cat[1] name_len: 0, 1, 2+ name characters read (name contexts only) */
/* cat[2] paren: open ( and NAME( depth 0, 1, 2, 3+ */
/* cat[3] lit */
#define TRK_LIT_NONE 0
#define TRK_LIT_BRACE 1
#define TRK_LIT_BRACKET 2
/* cat[4] menu: Menu( argument kind */
#define TRK_MENU_NONE 0
#define TRK_MENU_TITLE 1
#define TRK_MENU_TEXT 2
#define TRK_MENU_LABEL 3
/* cat[5] blk_top: innermost open block */
#define TRK_BLK_NONE 0
#define TRK_BLK_IF 1        /* If-Then */
#define TRK_BLK_ELSE 2      /* If-Then after Else */
#define TRK_BLK_FOR 3
#define TRK_BLK_WHILE 4
#define TRK_BLK_REPEAT 5
#define TRK_BLK_DEEP 6      /* deeper than TRK_MAX_BLOCKS */
/* cat[6] blk_depth: 0, 1, 2, 3, 4+ */
/* cat[7] if_state */
#define TRK_IF_NONE 0
#define TRK_IF_COND 1       /* inside If cond (or IS>( DS<() */
#define TRK_IF_AFTER 2      /* If statement ended: Then or a one-statement body follows */
#define TRK_IF_BODY 3       /* inside the one-statement body of a single-line If */
/* cat[8] unres: referenced-but-undefined labels 0, 1, 2+ */

typedef struct {
    uint8_t cat[TRK_NCAT];
    uint8_t hint[5];        /* bits 0..36 next name character (name index), bit 37 END */
    uint8_t var[4];         /* bits 0..9 Str (AA00+k), 10..19 [A]+k, 20..25 list buckets */
} trk_feat;

typedef struct {            /* end-of-program diagnostics */
    uint16_t n_unres, depth, n_undef, first_undef_pos, n_bad, first_bad_pos, n_extra_end,
        n_prgm;
} trk_summary;

typedef struct {
    uint8_t lbl_def[TRK_LABEL_BYTES];    /* labels defined (Lbl) */
    uint8_t lbl_unres[TRK_LABEL_BYTES];  /* referenced (Goto, Menu() and not defined */
    uint8_t def_first[5], unres_first[5]; /* first characters present in each label set */
    uint8_t list_name[TRK_MAX_LISTS][5]; /* name indices, 0xFF padded */
    uint8_t blk[TRK_MAX_BLOCKS / 2];     /* block types, 4 bits each */
    uint8_t var[4];                      /* same layout as trk_feat.var */
    uint8_t list_set[2];                 /* list_name entries currently assigned (bits) */
    uint16_t pic_set, gdb_set;           /* Pic n / GDB n stored */
    uint16_t depth, n_unres;
    uint16_t pos, n_undef, first_undef_pos, n_bad, first_bad_pos, n_extra_end, n_prgm;
    uint16_t pend_pos;
    uint8_t list_n;
    uint8_t flags, mode, stmt, if_state;  /* mode: low 4 bits mode, high bits Pic/GDB op */
    uint8_t paren, brace, bracket;
    uint8_t menu_depth, menu_arg;
    uint8_t fn_kind, fn_depth;
    uint8_t pend_kind, pend_idx;
    uint8_t name_kind, name_role, name_len;
    uint8_t name[5];
} trk_state;

void tracker_reset(trk_state *s);
/* Read one token id (ids >= the vocab size count as ordinary tokens). out may be NULL. */
void tracker_step(trk_state *s, unsigned tok, trk_feat *out);
void tracker_features(const trk_state *s, trk_feat *out);
void tracker_summary(const trk_state *s, trk_summary *out);
/* Read-only generated token classes, for decoder state that uses the same roles. */
const uint8_t *tracker_classes(void);

#endif
