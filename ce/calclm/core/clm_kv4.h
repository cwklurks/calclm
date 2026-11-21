/* E20 spike: 4-bit K/V for the inference core (PLAN-E20.md Design 2-6), bit-exact with
 * model/kv4.py's Kv4Model. Included by clm_fwd.c in a CLM_KV4 build only (the shape headers
 * core/clm_shape_w8k4.h and clm_shape_wk4.h define CLM_KV4); every function here is static
 * except the entry points declared in clm.h.
 *
 * Storage: code j in -7..7 is the nibble j + 7, channel 2i in the low nibble (clm.h's
 * CLM_KV4_* layout). At the cache write the int8 K and V rows from scale_out are encoded as
 * model/kv4.py encodes them; each layer's slots held are then decoded through the biased tables
 * (T[j] + 128) into the scratch window, in the int8 cache's layout (K rows with the int24 sum
 * term), and the shipped attention routines run on the scratch unchanged (decode-to-scratch).
 *
 * CLMKV4 payload (ce/calclm/tools/make_kv4.py), little-endian:
 *   0 "KV", format 1, layers, head_dim; 5 the export's 6-byte sha (as CLMVOC); 11 blob size (u24);
 *   14 FNV-1a (u32) of every byte after the 18-byte header; 18 the reciprocals M[l][kind][c]
 *   (uint16); then the biased tables [l][kind][c][16] (entry 15 a pad).
 */
#ifndef CLM_KV4_H
#define CLM_KV4_H

#define KV4_NREC ((uint16_t)CLM_LAYERS * 2 * CLM_HEAD_DIM)   /* (layer, kind, channel) */

static uint32_t kv4_rd(const uint8_t *p, uint8_t n) {
    uint32_t v = 0;
    while (n) {
        v = (v << 8) | p[--n];
    }
    return v;
}

int clm_kv4_load(clm_t *m, const uint8_t *payload, uint32_t size, const uint8_t *export_sha,
                 uint8_t ram) {
    m->kv4_ok = 0;
    if (!m->tokens || !payload || !export_sha || size != CLM_KV4_SIZE || payload[0] != 'K' ||
        payload[1] != 'V' || payload[2] != 1 || payload[3] != CLM_LAYERS ||
        payload[4] != CLM_HEAD_DIM || kv4_rd(payload + 11, 3) != m->seg[m->n_seg - 1].end) {
        return CLM_E_ARG;
    }
    for (uint8_t i = 0; i < 6; i++) {
        if (payload[5 + i] != export_sha[i]) {
            return CLM_E_ARG;
        }
    }
    uint32_t h = 0x811C9DC5UL;
    for (const uint8_t *p = payload + CLM_KV4_HDR; p < payload + CLM_KV4_SIZE; p++) {
        h = (h ^ *p) * 0x01000193UL;
    }
    if (h != kv4_rd(payload + 14, 4)) {
        return CLM_E_ARG;
    }
    uint16_t cap = m->mem.ctx_cap;
    const uint8_t *t = payload + CLM_KV4_HDR + 2 * KV4_NREC;
    m->kv4_m = payload + CLM_KV4_HDR;
    m->kv4_scratch = m->mem.vc + (uint32_t)CLM_LAYERS * cap * CLM_KV4_VROW;
    if (ram) {                                /* placement B: aligned copy after the scratch */
        uintptr_t a = (uintptr_t)(m->kv4_scratch + (uint32_t)cap * CLM_KV4_SROW);
        uint8_t *dst = (uint8_t *)((a + CLM_KV4_ALIGN - 1) & ~(uintptr_t)(CLM_KV4_ALIGN - 1));
        for (uint16_t i = 0; i < CLM_KV4_TBL_BYTES; i++) {
            dst[i] = t[i];
        }
        t = dst;
    }
    m->kv4_t = t;
    m->kv4_ok = 1;
    return CLM_OK;
}

/* model/kv4.py encode: p = |x| M (<= 128 * 65535 < 2^23: native 24-bit), q = p >> 16,
 * r = p & 0xFFFF, j = q + (r > 32768 or (r == 32768 and q odd)), at most 7, x's sign; returned
 * as the nibble j + 7 */
static uint8_t kv4_code(int8_t x, unsigned mr) {
    unsigned p = (unsigned)(x < 0 ? -x : x) * mr;
    unsigned q = p >> 16, r = p & 0xFFFFu;
    unsigned j = q + (r > 0x8000u || (r == 0x8000u && (q & 1u)));
    if (j > 7) {
        j = 7;
    }
    return (uint8_t)(x < 0 ? 7 - j : 7 + j);
}

/* one int8 row (K or V of layer l) into a packed row; K also gets -128 * sum(T[j] + 128) */
static void kv4_encode_row(const clm_t *m, uint8_t l, uint8_t kind, const int8_t *x, uint8_t *dst) {
    uint16_t rec = (uint16_t)(2 * l + kind) * CLM_HEAD_DIM;
    const uint8_t *mr = m->kv4_m + 2 * rec, *t = m->kv4_t + 16 * rec;
    int sum = 0;                              /* <= 32 * 255: native int */
    for (uint8_t i = 0; i < CLM_HEAD_DIM; i += 2, mr += 4, t += 32) {
        uint8_t lo = kv4_code(x[i], RD16U(mr)), hi = kv4_code(x[i + 1], RD16U(mr + 2));
        dst[i >> 1] = (uint8_t)(lo | hi << 4);
        sum += t[lo] + t[16 + hi];
    }
    if (!kind) {
        uint32_t nk = (uint32_t)(-128 * sum);   /* int24 LE */
        dst[CLM_HEAD_DIM / 2] = (uint8_t)nk;
        dst[CLM_HEAD_DIM / 2 + 1] = (uint8_t)(nk >> 8);
        dst[CLM_HEAD_DIM / 2 + 2] = (uint8_t)(nk >> 16);
    }
}

/* n packed rows (srow bytes apart) through one (layer, kind)'s tables t into decoded rows (drow
 * bytes apart); sum: copy the K row's three sum bytes after the head_dim values */
static void kv4_unpack_rows_c(const uint8_t *src, uint8_t *dst, uint16_t n, const uint8_t *t,
                              uint8_t srow, uint8_t drow, uint8_t sum) {
    for (uint16_t j = 0; j < n; j++, src += srow, dst += drow) {
        const uint8_t *tc = t;
        for (uint8_t i = 0; i < CLM_HEAD_DIM / 2; i++, tc += 32) {
            uint8_t b = src[i];
            dst[2 * i] = tc[b & 15];
            dst[2 * i + 1] = tc[16 + (b >> 4)];
        }
        if (sum) {
            dst[CLM_HEAD_DIM] = src[CLM_HEAD_DIM / 2];
            dst[CLM_HEAD_DIM + 1] = src[CLM_HEAD_DIM / 2 + 1];
            dst[CLM_HEAD_DIM + 2] = src[CLM_HEAD_DIM / 2 + 2];
        }
    }
}

#if defined(__ez80__)
/* The eZ80 unpack (PLAN-E20 Design 5, the time-boxed assembly loop), bit-identical to
 * kv4_unpack_rows_c (T6). int clm_kv4_unpack_a(src, dst, n, t, kind): n packed rows of one
 * (layer, kind) from src through its 32 tables t (16 bytes each, any alignment, both
 * placements) into decoded rows at dst; kind 0: K (19-byte rows into 35-byte rows, sum term
 * copied), 1: V (16 into 32). 1 <= n <= 256. Returns 0, or 1 (nothing done) when t's 512 bytes
 * cross a 64 KB boundary (the caller then runs the C unpack).
 *
 * The rows are fully unrolled (16 packed bytes, 32 lookups). A lookup is
 *     and 15; add a, lo(table); ld l, a; adc a, hi(table); sub l; ld h, a; ld a, (hl)
 * so a table may cross a 256-byte page (HLU is t's 64 KB bank, set once). The 32 tables' low and
 * high address bytes are patched into the `add`/`adc` immediates at each call: the routine runs
 * from RAM (clm_fwd.o's text, in the CE program and in the test binary), never from CLMCODE.
 * Registers: DE source, IY destination row, C the packed byte, B the row count. */
#if CLM_HEAD_DIM != 32
#error "the assembly unpack is written for head_dim 32 (16 packed bytes per row)"
#endif
int clm_kv4_unpack_a(const uint8_t *src, uint8_t *dst, unsigned n, const uint8_t *t,
                     unsigned kind);
__asm__(
    "\t.section\t.bss._clm_kv4_unpack_a,\"aw\",@nobits\n"
    "kv4_a_end:\t.skip\t3\n"
    "\t.section\t.text._clm_kv4_unpack_a,\"ax\",@progbits\n"
    "\t.macro\tKV4_PAIR p, i\n"
    "\tld\ta, (de)\n\tinc\tde\n\tld\tc, a\n"
    "\tand\ta, 15\n.L\\p\\()slo\\i:\n\tadd\ta, 0\n\tld\tl, a\n.L\\p\\()hlo\\i:\n"
    "\tadc\ta, 0\n\tsub\ta, l\n\tld\th, a\n\tld\ta, (hl)\n\tld\t(iy + 2 * \\i), a\n"
    "\tld\ta, c\n\trrca\n\trrca\n\trrca\n\trrca\n"
    "\tand\ta, 15\n.L\\p\\()shi\\i:\n\tadd\ta, 0\n\tld\tl, a\n.L\\p\\()hhi\\i:\n"
    "\tadc\ta, 0\n\tsub\ta, l\n\tld\th, a\n\tld\ta, (hl)\n\tld\t(iy + 2 * \\i + 1), a\n"
    "\t.endm\n"
    "\t.global\t_clm_kv4_unpack_a\n"
    "_clm_kv4_unpack_a:\n"
    "\tpush\tix\n\tld\tix, 0\n\tadd\tix, sp\n"
    "\tld\thl, (ix + 15)\n\tld\tde, 511\n\tadd\thl, de\n\tld\t(kv4_a_end), hl\n"
    "\tld\ta, (kv4_a_end + 2)\n\tcp\ta, (ix + 17)\n\tjr\tz, .Lkv4_ok\n"
    "\tld\thl, 1\n\tpop\tix\n\tret\n"
    ".Lkv4_ok:\n"
    "\tld\ta, i\n\tpush\taf\n\tdi\n"
    "\tld\tiy, .Lkv4_sites_k\n"
    "\tld\ta, (ix + 18)\n\tor\ta, a\n\tjr\tz, 1f\n\tld\tiy, .Lkv4_sites_v\n1:\n"
    "\tld\thl, (ix + 15)\n\tld\tb, 32\n"           /* patch the 32 tables' address bytes */
    ".Lkv4_patch:\n"
    "\tld\tde, (iy)\n\tinc\tde\n\tld\ta, l\n\tld\t(de), a\n"
    "\tld\tde, (iy + 3)\n\tinc\tde\n\tld\ta, h\n\tld\t(de), a\n"
    "\tld\tde, 16\n\tadd\thl, de\n\tlea\tiy, iy + 6\n\tdjnz\t.Lkv4_patch\n"
    "\tld\thl, (ix + 15)\n"                        /* HLU: the tables' bank */
    "\tld\tde, (ix + 6)\n\tld\tiy, (ix + 9)\n\tld\tb, (ix + 12)\n"   /* n = 256: b = 0 */
    "\tld\ta, (ix + 18)\n\tor\ta, a\n\tjp\tnz, .Lkv4_vrow\n"
    ".Lkv4_krow:\n"
    "\t.irp\ti, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n\tKV4_PAIR k, \\i\n\t.endr\n"
    "\tld\ta, (de)\n\tinc\tde\n\tld\t(iy + 32), a\n"
    "\tld\ta, (de)\n\tinc\tde\n\tld\t(iy + 33), a\n"
    "\tld\ta, (de)\n\tinc\tde\n\tld\t(iy + 34), a\n"
    "\tlea\tiy, iy + 35\n\tdec\tb\n\tjp\tnz, .Lkv4_krow\n\tjp\t.Lkv4_done\n"
    ".Lkv4_vrow:\n"
    "\t.irp\ti, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n\tKV4_PAIR v, \\i\n\t.endr\n"
    "\tlea\tiy, iy + 32\n\tdec\tb\n\tjp\tnz, .Lkv4_vrow\n"
    ".Lkv4_done:\n"
    "\tor\ta, a\n\tsbc\thl, hl\n"
    "\tpop\taf\n\tpop\tix\n\tret\tpo\n\tei\n\tret\n"
    ".Lkv4_sites_k:\n"
    "\t.irp\ti, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
    "\t.d24\t.Lkslo\\i, .Lkhlo\\i, .Lkshi\\i, .Lkhhi\\i\n\t.endr\n"
    ".Lkv4_sites_v:\n"
    "\t.irp\ti, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
    "\t.d24\t.Lvslo\\i, .Lvhlo\\i, .Lvshi\\i, .Lvhhi\\i\n\t.endr\n");

static void kv4_unpack_rows(const uint8_t *src, uint8_t *dst, uint16_t n, const uint8_t *t,
                            uint8_t srow, uint8_t drow, uint8_t sum) {
    if (!n || clm_kv4_unpack_a(src, dst, n, t, sum ? 0 : 1)) {
        kv4_unpack_rows_c(src, dst, n, t, srow, drow, sum);
    }
}
#else
#define kv4_unpack_rows kv4_unpack_rows_c
#endif

/* slots 0..n-1 of layer l into the scratch: K rows, then V rows ctx_cap rows later */
static void kv4_unpack(clm_t *m, uint8_t l, uint16_t n, uint8_t c_only) {
    uint16_t cap = m->mem.ctx_cap;
    const uint8_t *kp = m->mem.kc + (uint32_t)l * cap * CLM_KV4_KROW;
    const uint8_t *vp = m->mem.vc + (uint32_t)l * cap * CLM_KV4_VROW;
    const uint8_t *t = m->kv4_t + (uint16_t)(2 * l) * CLM_HEAD_DIM * 16;
    uint8_t *ks = m->kv4_scratch, *vs = ks + (uint32_t)cap * CLM_KROW;
    if (c_only) {
        kv4_unpack_rows_c(kp, ks, n, t, CLM_KV4_KROW, CLM_KROW, 1);
        kv4_unpack_rows_c(vp, vs, n, t + CLM_HEAD_DIM * 16, CLM_KV4_VROW, CLM_HEAD_DIM, 0);
    } else {
        kv4_unpack_rows(kp, ks, n, t, CLM_KV4_KROW, CLM_KROW, 1);
        kv4_unpack_rows(vp, vs, n, t + CLM_HEAD_DIM * 16, CLM_KV4_VROW, CLM_HEAD_DIM, 0);
    }
}

/* The cache write of layer l at slot: the int8 rows scale_out wrote into the scratch slot (k8,
 * v8) are encoded into the packed slot; the unpack then overwrites the scratch slot with the
 * decoded values. */
static void kv4_store(clm_t *m, uint8_t l, uint16_t slot, const uint8_t *k8, const uint8_t *v8) {
    uint16_t cap = m->mem.ctx_cap;
    kv4_encode_row(m, l, 0, (const int8_t *)k8,
                   m->mem.kc + ((uint32_t)l * cap + slot) * CLM_KV4_KROW);
    kv4_encode_row(m, l, 1, (const int8_t *)v8,
                   m->mem.vc + ((uint32_t)l * cap + slot) * CLM_KV4_VROW);
}

int clm_kv4_unpack_test(clm_t *m, uint8_t l, uint16_t n, uint8_t c_only) {
    if (!m->kv4_ok || l >= CLM_LAYERS || n > m->mem.ctx_cap) {
        return CLM_E_ARG;
    }
    kv4_unpack(m, l, n, c_only);
    return CLM_OK;
}

int clm_kv4_encode_test(clm_t *m, uint8_t l, uint8_t kind, uint8_t c, uint8_t *out) {
    if (!m->kv4_ok || l >= CLM_LAYERS || kind > 1 || c >= CLM_HEAD_DIM || !out) {
        return CLM_E_ARG;
    }
    uint16_t rec = (uint16_t)(2 * l + kind) * CLM_HEAD_DIM + c;
    unsigned mr = RD16U(m->kv4_m + 2 * rec);
    const uint8_t *t = m->kv4_t + 16 * rec;
    for (unsigned i = 0; i < 256; i++) {
        uint8_t j = kv4_code((int8_t)((int)i - 128), mr);
        out[i] = j;
        out[256 + i] = t[j];
    }
    return CLM_OK;
}

#endif
