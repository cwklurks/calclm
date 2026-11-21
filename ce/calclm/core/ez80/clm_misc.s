; CalcLM eZ80 small loops.
;
; void clm_a_silu(int16_t *y, unsigned n, const int16_t *silu)
;   in place: y < -2048 -> 0; y >= 2048 -> y; else silu[rsr(y, 2) + 512] (1025-entry table)
;   (rsr(y, 2) + 512 = (y + 2050) >> 2 for y in [-2048, 2047]; the byte offset is
;    ((y + 2050) >> 1) & ~1 and 2048 = 8 * 256, so with A = high byte + 8:
;    y + 2050 = A * 256 + low + 2)
; unsigned clm_a_maxabs24(const clm_i24 *v, unsigned n): max |v| (|v| <= 2^23)
	.assume	adl=1

	.section .text.clm_a_silu, "ax", @progbits
	.global	_clm_a_silu
_clm_a_silu:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	iy, (ix + 12)		; silu table
	ld	bc, (ix + 9)		; n
	ld	ix, (ix + 6)		; y
.Lsi:
	ld	l, (ix + 0)
	ld	a, (ix + 1)
	add	a, 8			; high byte + 8: in [0, 15] iff y in [-2048, 2047]
	cp	a, 16
	jr	c, 2f
	sub	a, 8			; back to the high byte
	jp	p, 3f			; y >= 2048: unchanged
	xor	a, a			; y < -2048: 0
	ld	(ix + 0), a
	ld	(ix + 1), a
	jr	3f
2:	ld	de, 0
	ld	d, a
	ld	e, l
	inc	de
	inc	de			; y + 2050 (16 bits, DEU = 0)
	srl	d
	rr	e
	res	0, e			; byte offset into the table
	lea	hl, iy + 0
	add	hl, de
	ld	a, (hl)
	ld	(ix + 0), a
	inc	hl
	ld	a, (hl)
	ld	(ix + 1), a
3:	lea	ix, ix + 2
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .Lsi
	pop	ix
	ret

	.section .text.clm_a_maxabs24, "ax", @progbits
	.global	_clm_a_maxabs24
_clm_a_maxabs24:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	iy, (ix + 6)		; v
	ld	bc, (ix + 9)		; n
	ld	de, 0			; peak
.Lmx24:
	ld	hl, (iy + 0)
	bit	7, (iy + 2)
	jr	z, 1f
	push	de
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de			; |v| (2^23 for -2^23, as unsigned)
	pop	de
1:	or	a, a
	sbc	hl, de
	jr	c, 2f
	add	hl, de
	ex	de, hl			; new peak
2:	lea	iy, iy + 3
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .Lmx24
	ex	de, hl
	pop	ix
	ret

; unsigned clm_a_compact(const uint8_t *V, const uint8_t *ew, unsigned n, uint8_t *cv,
;                        uint8_t *ce, unsigned cap)
;   copies the 32-byte V rows j < n with ew[j] != 0 (in order) to cv and their weights to ce;
;   returns their count, or cap + 1 as soon as there are more than cap of them (cv, ce then
;   hold cap rows: the caller falls back to the dense P.V).
	.section .bss.clm_a_compact, "aw", @nobits
cp_left:	.skip	3
cp_cnt:		.skip	3
cp_cap:		.skip	3

	.section .text.clm_a_compact, "ax", @progbits
	.global	_clm_a_compact
_clm_a_compact:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	hl, (ix + 12)
	ld	(cp_left), hl
	ld	hl, (ix + 21)
	ld	(cp_cap), hl
	or	a, a
	sbc	hl, hl
	ld	(cp_cnt), hl
	ld	hl, (ix + 6)		; V
	ld	de, (ix + 15)		; cv
	ld	iy, (ix + 18)		; ce
	ld	ix, (ix + 9)		; ew
.Lcp:
	ld	a, (ix + 0)
	inc	ix
	or	a, a
	jr	z, 2f
	ld	(iy + 0), a
	inc	iy
	push	hl
	ld	hl, (cp_cnt)
	inc	hl
	ld	(cp_cnt), hl
	ld	bc, (cp_cap)
	or	a, a
	sbc	hl, bc
	pop	hl
	jr	z, 1f			; count == cap: this row still fits
	jr	nc, .Lcp_full
1:	ld	bc, 32
	ldir
	jr	3f
2:	ld	bc, 32
	add	hl, bc
3:	ld	bc, (cp_left)		; HL = next V row: keep it
	dec	bc
	ld	(cp_left), bc
	ld	a, b
	or	a, c
	jr	nz, .Lcp
	ld	hl, (cp_cnt)
	pop	ix
	ret
.Lcp_full:
	ld	hl, (cp_cnt)
	pop	ix
	ret

	.section .note.GNU-stack,"",%progbits

	.section .note.GNU-stack,"",%progbits

; void clm_a_emb(const int8_t *x, unsigned n, unsigned mm, int s, clm_i24 *r, unsigned add)
;   t = rsr(x[i] * mm, s) (mm < 256, s >= -8); add = 0: r[i] = t;
;   add = 1: r[i] = clamp(r[i] + t, int24).
;   With X = |x| * mm < 2^15: t = +-((X - neg + 2^(s-1)) >> s) for 1 <= s <= 15 (16 bits
;   suffice), t = 0 for s >= 16, t = +-(X << -s) for s <= 0 (|t| < 2^23).
	.section .bss.clm_a_emb, "aw", @nobits
em_s:	.skip	1
em_add:	.skip	1
em_rnd:	.skip	3

	.section .text.clm_a_emb, "ax", @progbits
	.global	_clm_a_emb
_clm_a_emb:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, (ix + 15)
	ld	(em_s), a
	ld	a, (ix + 21)
	ld	(em_add), a
	ld	hl, 0			; rnd = 2^(s-1) for s in [1, 15]
	ld	a, (em_s)
	bit	7, a
	jr	nz, 2f
	or	a, a
	jr	z, 2f
	cp	a, 16
	jr	nc, 2f
	inc	hl
	dec	a
	jr	z, 2f
1:	add	hl, hl
	dec	a
	jr	nz, 1b
2:	ld	(em_rnd), hl
	ld	iy, (ix + 6)		; x
	ld	c, (ix + 12)		; mm (kept in C)
	ld	de, (ix + 9)		; n
	push	de
	ld	ix, (ix + 18)		; r
.Lem:
	ld	a, (iy + 0)
	inc	iy
	ld	b, a			; sign in bit 7 of B
	bit	7, a
	jr	z, 3f
	neg
3:	ld	h, a
	ld	l, c
	mlt	hl			; X, HLU = 0
	ld	a, (em_s)
	bit	7, a
	jr	nz, .Lem_left
	or	a, a
	jr	z, .Lem_sign		; s = 0
	cp	a, 16
	jr	c, 4f
	ld	hl, 0			; s >= 16: 0
	jr	.Lem_sign
4:	bit	7, b
	jr	z, 5f
	dec	hl			; X - 1 for negative x
5:	ld	de, (em_rnd)
	add	hl, de			; < 2^16
6:	srl	h
	rr	l
	dec	a
	jr	nz, 6b
	jr	.Lem_sign
.Lem_left:
	neg				; -s in [1, 8]
7:	add	hl, hl
	dec	a
	jr	nz, 7b
.Lem_sign:
	bit	7, b
	jr	z, 8f
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de
8:	ld	a, (em_add)
	or	a, a
	jr	z, 9f
	ld	de, (ix + 0)		; r + t, saturated to int24
	or	a, a
	adc	hl, de
	jp	po, 9f
	ld	hl, 0x7fffff
	jp	m, 9f
	ld	hl, 0x800000
9:	ld	(ix + 0), hl
	lea	ix, ix + 3
	pop	de
	dec	de
	push	de
	ld	a, d
	or	a, e
	jr	nz, .Lem
	pop	de
	pop	ix
	ret
