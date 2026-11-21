/* TI-BASIC program-state tracker (see tracker.h, README.md). C99, integer only, no
 * allocation, no writable globals. Mirrors tracker/spec.py statement for statement. */
#include "tracker.h"
#include "trk_tables.h"

const uint8_t *tracker_classes(void) { return TRK_CLASS; }

#define U16_MAX 0xFFFFu
#define NAME_N 37u                  /* name characters A-Z 0-9 theta */
#define NONE8 0xFFu

/* flags */
#define F_IN_STR 1u
#define F_AT_START 2u
#define F_FIRST_LBRACE 4u
#define F_ARG_START 8u
#define F_MENU_ON 16u
#define F_LIST_OVER 32u
#define F_STATS 64u                  /* a statistics command has appeared */
#define F_KEEP (F_LIST_OVER | F_STATS) /* program-wide flags */
/* one-token modes */
#define M_NONE 0u
#define M_STORE 1u
#define M_STORE_DIM 2u
#define M_DELVAR 3u
#define M_PICOP 4u
/* Pic/GDB operation, stored in the high nibble of mode */
#define PO_GDB 1u
#define PO_DEFINE 2u
#define M_MASK 15u
/* names in progress */
#define NK_NONE 0u
#define NK_LBL 1u
#define NK_GOTO 2u
#define NK_MENU 3u
#define NK_LIST 4u
#define NK_PRGM 5u
#define NK_STORE_RUN 6u
#define NK_SETUP_RUN 7u
/* list name roles */
#define LR_READ 0u
#define LR_DEF 1u
#define LR_UNDEF 2u
/* pending decisions */
#define PK_NONE 0u
#define PK_MAT_STORE 1u
#define PK_STR_INPUT 2u
/* var bit offsets */
#define V_STR 0u
#define V_MAT 10u
#define V_LIST 20u
#define N_BUCKETS 6u

static const uint8_t BIT[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* ---- small helpers ------------------------------------------------------------------- */

static uint8_t bit_get(const uint8_t *a, uint16_t i) { return a[i >> 3] & BIT[i & 7u]; }
/* 8-bit index versions for the short sets (hint, var, list_set) */
static uint8_t get8(const uint8_t *a, uint8_t i) { return a[i >> 3] & BIT[i & 7u]; }
static void set8(uint8_t *a, uint8_t i) { a[i >> 3] |= BIT[i & 7u]; }
static void clr8(uint8_t *a, uint8_t i) { a[i >> 3] &= (uint8_t)~BIT[i & 7u]; }
static void bit_set(uint8_t *a, uint16_t i) { a[i >> 3] |= BIT[i & 7u]; }
static void bit_clr(uint8_t *a, uint16_t i) { a[i >> 3] &= (uint8_t)~BIT[i & 7u]; }
static uint16_t inc16(uint16_t v) { return v == U16_MAX ? v : (uint16_t)(v + 1u); }
static uint8_t inc8(uint8_t v) { return v == NONE8 ? v : (uint8_t)(v + 1u); }
static int is_char(uint8_t c) { return c < NAME_N; }
static int is_letter(uint8_t c) { return c < 26u || c == 36u; }
static int is_digit(uint8_t c) { return c >= 26u && c <= 35u; }
static int is_sep(uint8_t c) { return c == CL_NEWLINE || c == CL_COLON || c == CL_EOS; }

static void bad(trk_state *s)
{
    s->n_bad = inc16(s->n_bad);
    if (!s->first_bad_pos)
        s->first_bad_pos = s->pos;
}

static void undef(trk_state *s, uint16_t pos)
{
    s->n_undef = inc16(s->n_undef);
    if (!s->first_undef_pos)
        s->first_undef_pos = pos;
}

/* ---- blocks ---------------------------------------------------------------------------- */

static uint8_t blk_get(const trk_state *s, uint16_t i)
{
    uint8_t b = s->blk[i >> 1];
    return (i & 1u) ? (uint8_t)(b >> 4) : (uint8_t)(b & 15u);
}

static void blk_put(trk_state *s, uint16_t i, uint8_t t)
{
    uint8_t b = s->blk[i >> 1];
    s->blk[i >> 1] = (i & 1u) ? (uint8_t)((b & 15u) | (t << 4)) : (uint8_t)((b & 0xF0u) | t);
}

static void push(trk_state *s, uint8_t t)
{
    if (s->depth == U16_MAX)
        return;
    if (s->depth < TRK_MAX_BLOCKS)
        blk_put(s, s->depth, t);
    s->depth++;
}

/* ---- variables ------------------------------------------------------------------------- */

/* entry holding the finished name in s->name (padded with NONE8 by pad_name), or NONE8 */
static uint8_t list_find(const trk_state *s)
{
    uint8_t i;
    const uint8_t (*row)[5] = s->list_name;  /* one row per entry: stays inside the table */
    const uint8_t *e, *w = s->name;
    for (i = 0; i < s->list_n; i++, row++) {
        e = *row;
        if (e[0] == w[0] && e[1] == w[1] && e[2] == w[2] && e[3] == w[3] && e[4] == w[4])
            return i;
    }
    return NONE8;
}

static void pad_name(trk_state *s)
{
    uint8_t j;
    for (j = s->name_len; j < 5u; j++)
        s->name[j] = NONE8;
}

/* var bit of a list name's bucket: V_LIST + (first character + 1) mod 6 (feats.list_bucket) */
static const uint8_t BUCKET_BIT[NAME_N] = {
    21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 20, 21,
    22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 20, 21};

static void list_buckets(trk_state *s)
{
    uint8_t i;
    uint8_t (*row)[5] = s->list_name;
    s->var[2] &= 0x0Fu;                  /* var bits 20..25 */
    s->var[3] &= 0xFCu;
    for (i = 0; i < s->list_n; i++, row++)
        if (get8(s->list_set, i))
            set8(s->var, BUCKET_BIT[(*row)[0]]);
}

static void list_define(trk_state *s)
{
    uint8_t i = list_find(s), j;
    if (i == NONE8) {
        if (s->list_n >= TRK_MAX_LISTS) {
            s->flags |= F_LIST_OVER;
            return;
        }
        i = s->list_n++;
        for (j = 0; j < 5u; j++)
            s->list_name[i][j] = s->name[j];
    }
    set8(s->list_set, i);
    set8(s->var, BUCKET_BIT[s->name[0]]);
}

static void list_undefine(trk_state *s)
{
    uint8_t i = list_find(s);
    if (i != NONE8) {
        clr8(s->list_set, i);
        list_buckets(s);
    }
}

static void list_read(trk_state *s)
{
    uint8_t i = list_find(s);
    if (i != NONE8) {
        if (!get8(s->list_set, i))
            undef(s, s->pos);
    } else if (!(s->flags & F_LIST_OVER)) {
        undef(s, s->pos);
    }
}

static void var_read(trk_state *s, uint8_t bit, uint16_t pos)
{
    if (!get8(s->var, bit))
        undef(s, pos);
}

static void pic_use(trk_state *s, uint8_t gdb, uint8_t define, uint8_t n)
{
    uint16_t *set = gdb ? &s->gdb_set : &s->pic_set;
    uint16_t m = (uint16_t)(1u << n);
    if (define)
        *set = (uint16_t)(*set | m);
    else if (!(*set & m))
        undef(s, s->pos);
}

/* ---- statements ------------------------------------------------------------------------ */

static void resolve_pending(trk_state *s, uint8_t c)
{
    uint8_t kind = s->pend_kind;
    s->pend_kind = PK_NONE;
    if (kind == PK_MAT_STORE) {          /* ->[A]( is an element store: needs [A] */
        if (c == CL_LPAREN)
            var_read(s, (uint8_t)(V_MAT + s->pend_idx), s->pend_pos);
        else
            set8(s->var, (uint8_t)(V_MAT + s->pend_idx));
    } else {                             /* Input Str1,... uses Str1 as the prompt */
        if (c == CL_COMMA)
            var_read(s, (uint8_t)(V_STR + s->pend_idx), s->pend_pos);
        else
            set8(s->var, (uint8_t)(V_STR + s->pend_idx));
    }
}

static void end_statement(trk_state *s)
{
    if (s->pend_kind)
        resolve_pending(s, CL_EOS);
    if (s->flags & F_AT_START) {         /* empty statement: it is a single-line If body */
        if (s->if_state == TRK_IF_AFTER)
            s->if_state = TRK_IF_NONE;
    } else if (s->if_state == TRK_IF_COND) {
        s->if_state = TRK_IF_AFTER;
    } else if (s->if_state == TRK_IF_BODY) {
        s->if_state = TRK_IF_NONE;
    }
    s->flags = (uint8_t)((s->flags & F_KEEP) | F_AT_START);
    s->stmt = NONE8;
    s->paren = s->brace = s->bracket = 0;
    s->fn_kind = 0;
    s->mode = M_NONE;
}

static void do_store(trk_state *s)
{
    s->paren = s->brace = s->bracket = 0;
    s->flags &= (uint8_t)~F_MENU_ON;
    s->fn_kind = 0;
    s->mode = M_STORE;
}

static void open_paren(trk_state *s) { s->paren = inc8(s->paren); }

static void start_name(trk_state *s, uint8_t kind, uint8_t role)
{
    s->name_kind = kind;
    s->name_role = role;
    s->name_len = 0;
}

/* ---- names ----------------------------------------------------------------------------- */

/* first id of the 2-character labels starting with name character c */
static uint16_t second_base(uint8_t c) { return (uint16_t)(NAME_N + (uint16_t)c * NAME_N); }

/* any bit set in a[lo .. lo+n), n >= 1 */
static uint8_t any_bits(const uint8_t *a, uint16_t lo, uint8_t n)
{
    uint16_t last = (uint16_t)(lo + n - 1u);
    uint8_t b = (uint8_t)(lo >> 3), b1 = (uint8_t)(last >> 3);
    uint8_t m0 = (uint8_t)(0xFFu << (lo & 7u)), m1 = (uint8_t)(0xFFu >> (7u - (last & 7u)));
    if (b == b1)
        return a[b] & m0 & m1;
    if (a[b] & m0)
        return 1;
    for (b++; b < b1; b++)
        if (a[b])
            return 1;
    return a[b1] & m1;
}

static void register_label(trk_state *s)
{
    uint8_t c = s->name[0];
    uint16_t id = s->name_len == 1u ? c : (uint16_t)(second_base(c) + s->name[1]);
    if (s->name_kind == NK_LBL) {
        bit_set(s->lbl_def, id);
        set8(s->def_first, c);
        if (bit_get(s->lbl_unres, id)) {
            bit_clr(s->lbl_unres, id);
            s->n_unres--;
            if (!bit_get(s->lbl_unres, c) && !any_bits(s->lbl_unres, second_base(c), NAME_N))
                clr8(s->unres_first, c);
        }
    } else if (!bit_get(s->lbl_def, id) && !bit_get(s->lbl_unres, id)) {
        bit_set(s->lbl_unres, id);
        set8(s->unres_first, c);
        s->n_unres++;
    }
}

static void finish_list(trk_state *s)
{
    s->name_kind = NK_NONE;
    if (!s->name_len) {
        bad(s);
        return;
    }
    pad_name(s);
    if (s->name_role == LR_DEF) {
        list_define(s);
    } else if (s->name_role == LR_UNDEF) {
        list_undefine(s);
    } else {
        list_read(s);
    }
}

static void finish_run(trk_state *s)
{
    uint8_t kind = s->name_kind;
    s->name_kind = NK_NONE;
    pad_name(s);
    if (kind == NK_SETUP_RUN || s->name_len >= 2u || (s->flags & F_FIRST_LBRACE))
        list_define(s);
}

/* returns 1 if the token was consumed by the name in progress */
static int name_step(trk_state *s, uint8_t c)
{
    uint8_t kind = s->name_kind;
    if (kind == NK_LBL || kind == NK_GOTO || kind == NK_MENU) {
        int term;
        if (is_char(c)) {
            if (s->name_len < 2u) {
                s->name[s->name_len++] = c;
            } else {
                bad(s);
                s->name_kind = NK_NONE;
            }
            return 1;
        }
        term = is_sep(c) || (kind == NK_MENU && (c == CL_COMMA || c == CL_RPAREN));
        if (s->name_len && term)
            register_label(s);
        else
            bad(s);
        s->name_kind = NK_NONE;
        return 0;
    }
    if (kind == NK_PRGM) {
        if (is_char(c) && s->name_len < 8u) {
            s->name_len++;
            return 1;
        }
        s->name_kind = NK_NONE;
        if (s->name_role) {              /* a call statement: 1-8 characters, then a separator */
            if (is_char(c) || !s->name_len || !is_sep(c))
                bad(s);
            return is_char(c);           /* a 9th character goes with the bad name */
        }
        return 0;
    }
    if (kind == NK_LIST) {
        if (is_char(c) && (s->name_len || is_letter(c))) {
            if (s->name_len < 5u) {
                s->name[s->name_len++] = c;
                return 1;
            }
            finish_list(s);
            bad(s);
            return 1;
        }
        finish_list(s);
        if (s->name_role == LR_UNDEF && !is_sep(c))
            end_statement(s);            /* DelVar lNAME chains: this token starts a statement */
        return 0;
    }
    /* NK_STORE_RUN, NK_SETUP_RUN: list names written without the list token */
    if (is_char(c)) {
        if (s->name_len < 5u) {
            s->name[s->name_len++] = c;
            return 1;
        }
        finish_run(s);
        bad(s);
        return 1;
    }
    finish_run(s);
    return 0;
}

/* ---- tokens ---------------------------------------------------------------------------- */

static void statement(trk_state *s, uint8_t c)
{
    switch (c) {
    case CL_IF:
        s->if_state = TRK_IF_COND;
        break;
    case CL_ISDS:
        s->if_state = TRK_IF_COND;
        open_paren(s);
        break;
    case CL_THEN:                        /* not right after an If: ERR:SYNTAX */
        bad(s);
        push(s, TRK_BLK_IF);
        break;
    case CL_ELSE:
        if (s->depth > TRK_MAX_BLOCKS)
            break;
        if (s->depth && blk_get(s, (uint16_t)(s->depth - 1u)) == TRK_BLK_IF)
            blk_put(s, (uint16_t)(s->depth - 1u), TRK_BLK_ELSE);
        else
            bad(s);
        break;
    case CL_END:
        if (s->depth)
            s->depth--;
        else
            s->n_extra_end = inc16(s->n_extra_end);
        break;
    case CL_FOR:
        push(s, TRK_BLK_FOR);
        open_paren(s);
        break;
    case CL_WHILE:
        push(s, TRK_BLK_WHILE);
        break;
    case CL_REPEAT:
        push(s, TRK_BLK_REPEAT);
        break;
    case CL_LBL:
        start_name(s, NK_LBL, 0);
        break;
    case CL_GOTO:
        start_name(s, NK_GOTO, 0);
        break;
    case CL_MENU:
        open_paren(s);
        s->flags |= F_MENU_ON;
        s->menu_depth = s->paren;
        s->menu_arg = 0;
        break;
    case CL_DELVAR:
        s->mode = M_DELVAR;
        break;
    case CL_SETUPED:
        s->flags |= F_ARG_START;
        break;
    case CL_STOREPIC:
    case CL_RECALLPIC:
    case CL_STOREGDB:
    case CL_RECALLGDB:
        s->mode = (uint8_t)(M_PICOP
                            | (((c == CL_STOREGDB || c == CL_RECALLGDB) ? PO_GDB : 0u) << 4)
                            | (((c == CL_STOREPIC || c == CL_STOREGDB) ? PO_DEFINE : 0u) << 4));
        break;
    default:                             /* Input, Prompt: only the statement kind matters */
        break;
    }
}

static int is_statement(uint8_t c)
{
    return (c >= CL_IF && c <= CL_GOTO) || c == CL_FOR || c == CL_MENU || c == CL_ISDS
           || (c >= CL_DELVAR && c <= CL_RECALLGDB);
}

static int store_target(trk_state *s, uint8_t c)
{
    if (c >= CL_STR0 && c < CL_STR0 + 10u) {
        set8(s->var, (uint8_t)(V_STR + (c - CL_STR0)));
    } else if (c >= CL_MAT0 && c < CL_MAT0 + 10u) {
        s->pend_kind = PK_MAT_STORE;
        s->pend_idx = (uint8_t)(c - CL_MAT0);
        s->pend_pos = s->pos;
    } else if (c == CL_LISTL) {
        start_name(s, NK_LIST, LR_DEF);
    } else if (c == CL_DIM) {
        open_paren(s);
        s->mode = M_STORE_DIM;
    } else if (is_letter(c)) {
        start_name(s, NK_STORE_RUN, 0);
        s->name[0] = c;
        s->name_len = 1;
    } else {
        return 0;
    }
    return 1;
}

static void delvar_operand(trk_state *s, uint8_t c)
{
    if (c == CL_LISTL) {
        start_name(s, NK_LIST, LR_UNDEF);
        return;
    }
    if (c >= CL_STR0 && c < CL_STR0 + 10u)
        clr8(s->var, (uint8_t)(V_STR + (c - CL_STR0)));
    else if (c >= CL_MAT0 && c < CL_MAT0 + 10u)
        clr8(s->var, (uint8_t)(V_MAT + (c - CL_MAT0)));
    else if (c >= CL_PIC0 && c < CL_PIC0 + 10u)
        s->pic_set = (uint16_t)(s->pic_set & ~(1u << (c - CL_PIC0)));
    else if (c >= CL_GDB0 && c < CL_GDB0 + 10u)
        s->gdb_set = (uint16_t)(s->gdb_set & ~(1u << (c - CL_GDB0)));
    end_statement(s);                    /* DelVar X chains: next token starts a statement */
}

static int pic_operand(trk_state *s, uint8_t c, uint8_t picop)
{
    uint8_t gdb = picop & PO_GDB, n;
    uint8_t first = gdb ? CL_GDB0 : CL_PIC0;
    if (is_digit(c))
        n = (uint8_t)(c - 26u);
    else if (c >= first && c < first + 10u)
        n = (uint8_t)(c - first);
    else
        return 0;
    pic_use(s, gdb, picop & PO_DEFINE, n);
    return 1;
}

static void general(trk_state *s, uint8_t c)
{
    if (c == CL_QUOTE) {
        s->flags |= F_IN_STR;
    } else if (c == CL_STORE) {
        do_store(s);
    } else if (c >= CL_OPEN_FIRST && c <= CL_OPEN_LAST) {
        open_paren(s);
        if ((c == CL_EQUSTR || c == CL_LISTMATR || c == CL_MATRLIST) && !s->fn_kind) {
            s->fn_kind = c;
            s->fn_depth = s->paren;
        }
        if (c == CL_STATOPEN)
            s->flags |= F_STATS;
    } else if (c == CL_RPAREN) {
        if (s->paren)
            s->paren--;
        if ((s->flags & F_MENU_ON) && s->paren < s->menu_depth)
            s->flags &= (uint8_t)~F_MENU_ON;
        if (s->fn_kind && s->paren < s->fn_depth)
            s->fn_kind = 0;
    } else if (c == CL_LBRACE) {
        s->brace = inc8(s->brace);
    } else if (c == CL_RBRACE) {
        if (s->brace)
            s->brace--;
    } else if (c == CL_LBRACKET) {
        s->bracket = inc8(s->bracket);
    } else if (c == CL_RBRACKET) {
        if (s->bracket)
            s->bracket--;
    } else if (c == CL_COMMA) {
        if ((s->flags & F_MENU_ON) && s->paren == s->menu_depth) {
            s->menu_arg = inc8(s->menu_arg);
            if (s->menu_arg >= 2u && !(s->menu_arg & 1u))
                start_name(s, NK_MENU, 0);
        }
        if (s->stmt == CL_SETUPED && !s->paren)
            s->flags |= F_ARG_START;
    } else if (c == CL_LISTL) {
        int def = s->stmt == CL_INPUT || s->stmt == CL_PROMPT || s->stmt == CL_SETUPED
                  || s->fn_kind == CL_MATRLIST;
        start_name(s, NK_LIST, def ? LR_DEF : LR_READ);
    } else if (c >= CL_STR0 && c < CL_STR0 + 10u) {
        uint8_t k = (uint8_t)(c - CL_STR0);
        if (s->stmt == CL_INPUT) {
            s->pend_kind = PK_STR_INPUT;
            s->pend_idx = k;
            s->pend_pos = s->pos;
        } else if (s->stmt == CL_PROMPT || s->fn_kind == CL_EQUSTR) {
            set8(s->var, (uint8_t)(V_STR + k));
        } else {
            var_read(s, (uint8_t)(V_STR + k), s->pos);
        }
    } else if (c >= CL_MAT0 && c < CL_MAT0 + 10u) {
        uint8_t bit = (uint8_t)(V_MAT + (c - CL_MAT0));
        if (s->stmt == CL_INPUT || s->stmt == CL_PROMPT || s->fn_kind == CL_LISTMATR)
            set8(s->var, bit);
        else
            var_read(s, bit, s->pos);
    } else if (c == CL_PRGM) {          /* inside an expression (Asm(prgmX), ...) */
        s->n_prgm = inc16(s->n_prgm);
        start_name(s, NK_PRGM, 0);
    } else if (c == CL_LBL || c == CL_GOTO) {
        bad(s);
    } else if (c >= CL_PIC0 && c < CL_GDB0 + 10u) { /* Archive Pic1, UnArchive GDB2, ... */
        if (c < CL_GDB0)
            pic_use(s, 0, 0, (uint8_t)(c - CL_PIC0));
        else
            pic_use(s, 1, 0, (uint8_t)(c - CL_GDB0));
    } else if (c == CL_STATVAR) {
        if (!(s->flags & F_STATS))
            undef(s, s->pos);
    } else if (c == CL_STATCMD) {
        s->flags |= F_STATS;
    }
}

static void token(trk_state *s, uint8_t c)
{
    uint8_t mode = s->mode & M_MASK, picop = (uint8_t)(s->mode >> 4);
    uint8_t arg_start = s->flags & F_ARG_START;
    s->mode = M_NONE;
    s->flags &= (uint8_t)~F_ARG_START;
    if (s->flags & F_AT_START) {
        s->flags &= (uint8_t)~F_AT_START;
        if (s->if_state == TRK_IF_AFTER) {
            if (c == CL_THEN) {
                s->stmt = CL_THEN;
                push(s, TRK_BLK_IF);
                s->if_state = TRK_IF_NONE;
                return;
            }
            s->if_state = TRK_IF_BODY;
        }
        s->stmt = c;
        if (c == CL_LBRACE)
            s->flags |= F_FIRST_LBRACE;
        if (is_statement(c)) {
            statement(s, c);
            return;
        }
        if (c == CL_PRGM) {
            s->n_prgm = inc16(s->n_prgm);
            start_name(s, NK_PRGM, 1);
            return;
        }
    }
    if (mode == M_STORE && store_target(s, c))
        return;
    if (mode == M_STORE_DIM) {
        if (c == CL_LISTL) {
            start_name(s, NK_LIST, LR_DEF);
            return;
        }
        if (c >= CL_MAT0 && c < CL_MAT0 + 10u) {
            set8(s->var, (uint8_t)(V_MAT + (c - CL_MAT0)));
            return;
        }
    }
    if (mode == M_DELVAR) {
        delvar_operand(s, c);
        return;
    }
    if (mode == M_PICOP && pic_operand(s, c, picop))
        return;
    if (is_letter(c) && arg_start && s->stmt == CL_SETUPED && !s->paren) {
        start_name(s, NK_SETUP_RUN, 0);
        s->name[0] = c;
        s->name_len = 1;
        return;
    }
    general(s, c);
}

/* ---- public API ------------------------------------------------------------------------ */

void tracker_reset(trk_state *s)
{
    uint8_t *p = (uint8_t *)s;
    uint8_t n = (uint8_t)(sizeof *s / 8u);
    while (n--) {                        /* unrolled: this runs once per program */
        p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0;
        p[4] = 0; p[5] = 0; p[6] = 0; p[7] = 0;
        p += 8;
    }
    for (n = 0; n < (uint8_t)(sizeof *s % 8u); n++)
        p[n] = 0;
    s->flags = F_AT_START;
    s->stmt = NONE8;
}

void tracker_step(trk_state *s, unsigned tok, trk_feat *out)
{
    uint8_t c = tok < TRK_VOCAB ? TRK_CLASS[tok] : (uint8_t)CL_OTHER;
    if (c == CL_BOS) {
        tracker_reset(s);
    } else if (c != CL_PAD) {
        s->pos = inc16(s->pos);
        if (c == CL_EOS) {
            if (s->name_kind)
                name_step(s, c);
            if (s->if_state == TRK_IF_COND || s->if_state == TRK_IF_AFTER)
                bad(s);                  /* single-line If with no statement after it */
            end_statement(s);
        } else if (s->flags & F_IN_STR) {
            if (c == CL_QUOTE) {
                s->flags &= (uint8_t)~F_IN_STR;
            } else if (c == CL_STORE) {
                s->flags &= (uint8_t)~F_IN_STR;
                do_store(s);
            } else if (c == CL_NEWLINE) {
                end_statement(s);
            }
        } else {
            if (s->pend_kind)
                resolve_pending(s, c);
            if (!(s->name_kind && name_step(s, c))) {
                if (c == CL_NEWLINE || c == CL_COLON)
                    end_statement(s);
                else
                    token(s, c);
            }
        }
    }
    if (out)
        tracker_features(s, out);
}

/* hint bits for label candidates in `set` (first-character set `first`) extending the
 * current name prefix */
static void label_hint(const trk_state *s, const uint8_t *set, const uint8_t *first,
                       uint8_t *hint)
{
    uint8_t i;
    if (s->name_len == 0u) {
        for (i = 0; i < 5u; i++)
            hint[i] = first[i];
    } else if (s->name_len == 1u) {
        uint16_t base = second_base(s->name[0]);
        uint8_t at = (uint8_t)(base >> 3), sh = (uint8_t)(base & 7u);
        const uint8_t *p = set + at;
        for (i = 0; i < 5u; i++) {  /* bits base .. base+39, never past the last byte */
            uint8_t hi = (at + i + 1u < TRK_LABEL_BYTES) ? p[i + 1u] : 0u;
            hint[i] = sh ? (uint8_t)((p[i] >> sh) | (hi << (8u - sh))) : p[i];
        }
        hint[4] &= 0x1Fu;           /* keep the 37 second characters */
        if (bit_get(set, s->name[0]))
            hint[4] |= 0x20u;       /* END */
    } else if (bit_get(set, (uint16_t)(second_base(s->name[0]) + s->name[1]))) {
        hint[4] = 0x20u;
    }
}

/* byte and bit of hint bit k (k = 0..37), so the loop below needs no shifts */
static const uint8_t HINT_BYTE[NAME_N + 1] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                                              4, 4, 4, 4, 4, 4};

static void list_hint(const trk_state *s, uint8_t *hint)
{
    uint8_t i, j, k, n = s->name_len, bits = s->list_set[0], m = 1;
    const uint8_t (*row)[5] = s->list_name;
    const uint8_t *e;
    for (i = 0; i < s->list_n; i++, row++, m = (uint8_t)(m + m)) {
        e = *row;
        if (i == 8u) {
            bits = s->list_set[1];
            m = 1;
        }
        if (!(bits & m))
            continue;
        for (j = 0; j < n; j++)
            if (e[j] != s->name[j])
                break;
        if (j < n)
            continue;
        k = (n < 5u && e[n] != NONE8) ? e[n] : (n ? (uint8_t)NAME_N : NONE8);
        if (k != NONE8)
            hint[HINT_BYTE[k]] |= BIT[k & 7u];
    }
}

void tracker_features(const trk_state *s, trk_feat *out)
{
    uint8_t ctx, i;
    uint8_t k = s->name_kind;
    if (s->flags & F_IN_STR)
        ctx = TRK_CTX_STRING;
    else if (k == NK_LBL)
        ctx = TRK_CTX_LBL;
    else if (k == NK_GOTO || k == NK_MENU)
        ctx = TRK_CTX_REF;
    else if (k == NK_PRGM)
        ctx = TRK_CTX_PRGM;
    else if (k == NK_STORE_RUN && s->name_len < 2u)
        ctx = TRK_CTX_EXPR;
    else if (k != NK_NONE)
        ctx = TRK_CTX_LIST;
    else if (s->mode == M_STORE)
        ctx = TRK_CTX_STORE;
    else if (s->mode == M_DELVAR)
        ctx = TRK_CTX_DELVAR;
    else
        ctx = (s->flags & F_AT_START) ? TRK_CTX_START : TRK_CTX_EXPR;
    out->cat[0] = ctx;
    out->cat[1] = (ctx >= TRK_CTX_LBL && ctx <= TRK_CTX_PRGM)
                      ? (s->name_len < 2u ? s->name_len : 2u) : 0u;
    out->cat[2] = s->paren < 3u ? s->paren : 3u;
    out->cat[3] = s->bracket ? TRK_LIT_BRACKET : (s->brace ? TRK_LIT_BRACE : TRK_LIT_NONE);
    if (!(s->flags & F_MENU_ON))
        out->cat[4] = TRK_MENU_NONE;
    else if (!s->menu_arg)
        out->cat[4] = TRK_MENU_TITLE;
    else
        out->cat[4] = (s->menu_arg & 1u) ? TRK_MENU_TEXT : TRK_MENU_LABEL;
    if (!s->depth)
        out->cat[5] = TRK_BLK_NONE;
    else if (s->depth > TRK_MAX_BLOCKS)
        out->cat[5] = TRK_BLK_DEEP;
    else
        out->cat[5] = blk_get(s, (uint16_t)(s->depth - 1u));
    out->cat[6] = s->depth < 4u ? (uint8_t)s->depth : 4u;
    out->cat[7] = s->if_state;
    out->cat[8] = s->n_unres < 2u ? (uint8_t)s->n_unres : 2u;
    for (i = 0; i < 5u; i++)
        out->hint[i] = 0;
    if (ctx == TRK_CTX_LBL)
        label_hint(s, s->lbl_unres, s->unres_first, out->hint);
    else if (ctx == TRK_CTX_REF)
        label_hint(s, s->lbl_def, s->def_first, out->hint);
    else if (ctx == TRK_CTX_LIST)
        list_hint(s, out->hint);
    for (i = 0; i < 4u; i++)
        out->var[i] = s->var[i];
}

void tracker_summary(const trk_state *s, trk_summary *out)
{
    out->n_unres = s->n_unres;
    out->depth = s->depth;
    out->n_undef = s->n_undef;
    out->first_undef_pos = s->first_undef_pos;
    out->n_bad = s->n_bad;
    out->first_bad_pos = s->first_bad_pos;
    out->n_extra_end = s->n_extra_end;
    out->n_prgm = s->n_prgm;
}
