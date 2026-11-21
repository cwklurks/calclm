; CalcLM eZ80 hot loops for RMSNorm and the SwiGLU product (INTREF.md sections 3, 4).
;
; unsigned clm_a_mulnorm(const clm_i24 *x, unsigned n, int c, const int16_t *w, int32_t *out,
;                        unsigned want_ss, clm_i24 *tmp)
;   x'  = normalize16 element of x: c >= 1: clamp(rsr(x, c), +-32767); c <= 0: x << -c
;   out = x' * w (exact int32); returns sum of rsr(x', 8)^2 when want_ss (else 0)
;   Preconditions: 1 <= n, c in [-15, 8], |x| < 2^(c+15) (c comes from the peak of x).
;   Used for norm_quant (x = residual, w = gain, ss) and the FFN (x = t3, w = silu(y1)).
;   x is streamed through SP and the loop pushes below SP, which overwrites consumed elements:
;   pass tmp (3n bytes) to work on a copy and keep x intact, or tmp = 0 to consume x.
;
; int clm_a_t3(const clm_i24 *acc, const uint8_t *bm, unsigned n, int E3, clm_i24 *t3,
;              const int8_t *be)
;   t3 = rsr(acc * bm[i], E3 - be[i]); returns max |t3|
;   Preconditions: 1 <= n, |acc| < 2^16, 0 <= E3 - be[i] <= 8.
;   Method: rsr(acc * b, d) = rsr(acc * (b << (8 - d)), 8), and b << (8 - d) < 2^15.
;
; void clm_a_maxabs32(const int32_t *v, unsigned n, uint32_t *peak): max |v| (|v| < 2^31)
	.assume	adl=1
	.include "clm_mac.inc"

	.section .bss.clm_a_vec, "aw", @nobits
mn_sp:	.skip	3
mn_ss:	.skip	3
mn_k1:	.skip	1		; c >= 1: left shift 8 - c; c <= 0: left shift -c
mn_ssf:	.skip	1		; want_ss
mn_E3:	.skip	1
mn_n:	.skip	3
mn_pk:	.skip	3

; NORM1 pos: HL = x (int24) -> DE = x' (int16). pos = 1 for c >= 1, 0 for c <= 0.
; k1 comes from mn_k1. Clobbers AF, B, C, HL.
.macro NORM1 pos
	ld	a, (mn_k1)
	or	a, a
	jr	z, 2f
	ld	b, a
1:	add	hl, hl
	djnz	1b
2:
.if \pos
	ld	de, 128
	add	hl, de			; Y = (x << (8 - c)) + 128
	ld	c, l
	HLU_A
	ld	d, a
	ld	e, h			; x' = bits 8..23 of Y
	cp	a, 0x80
	jr	nz, 3f
	ld	a, e
	or	a, a
	jr	nz, 3f
	ld	de, 0x7fff		; pattern 0x8000: +32768 if Y's low byte < 128, else -32768
	bit	7, c
	jr	z, 3f
	ld	de, 0x8001
3:
.else
	ex	de, hl			; |x << -c| < 2^15
.endif
.endm

; MULNORM_BODY pos: one element; SP = x, IX = out, shadow HL' = w, BC' = count
.macro MULNORM_BODY pos, loop
	pop	hl
	NORM1	\pos
	ld	a, (mn_ssf)
	or	a, a
	jr	z, 4f
	ld	a, e			; r8 = rsr(x', 8) = D + (E >= 128), |r8| <= 128
	add	a, 128
	ld	a, d
	adc	a, 0
	bit	7, a
	jr	z, 5f
	neg				; |r8| (0x80 stays 128)
5:	ld	b, a
	ld	c, a
	mlt	bc
	ld	hl, (mn_ss)
	add	hl, bc
	ld	(mn_ss), hl
4:	exx
	ld	iy, (hl)		; w (IYU: don't care)
	inc	hl
	inc	hl
	exx
	ld	a, d
	xor	a, iyh
	and	a, 0x80			; sign of the product
	push	af
	bit	7, d
	jr	z, 6f
	xor	a, a			; |x'|
	sub	a, e
	ld	e, a
	ld	a, 0
	sbc	a, d
	ld	d, a
6:	ld	a, iyh
	bit	7, a
	jr	z, 7f
	xor	a, a			; |w|
	sub	a, iyl
	ld	iyl, a
	ld	a, 0
	sbc	a, iyh
	ld	iyh, a
7:	MUL16R				; HL = P >> 16, D = (P >> 8) & 255, E = P & 255
	pop	af
	or	a, a
	jr	z, 8f
	xor	a, a			; negate the 32-bit product
	sub	a, e
	ld	e, a
	ld	a, 0
	sbc	a, d
	ld	d, a
	ex	de, hl			; HL = low 16 bits, DE = P >> 16
	push	hl			; (writes below SP: consumed x entries)
	ld	hl, 0
	sbc	hl, de			; high part, with the borrow
	pop	de			; DE = low 16 bits
8:	ld	(ix + 0), e
	ld	(ix + 1), d
	ld	(ix + 2), l
	ld	(ix + 3), h
	lea	ix, ix + 4
	exx
	dec	bc
	ld	a, b
	or	a, c
	exx
	jp	nz, \loop
.endm

	.section .text.clm_a_mulnorm, "ax", @progbits
	.global	_clm_a_mulnorm
_clm_a_mulnorm:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	or	a, a
	sbc	hl, hl
	ld	(mn_ss), hl
	ld	a, (ix + 21)
	ld	(mn_ssf), a
	ld	a, (ix + 12)		; c
	ld	hl, (ix + 15)		; w
	ld	bc, (ix + 9)		; n
	exx
	ld	(mn_sp), sp
	ld	hl, (ix + 6)
	ld	de, (ix + 24)		; tmp
	push	af
	ld	a, (ix + 26)
	or	a, d
	or	a, e
	jr	z, 1f
	push	de			; copy x -> tmp (3n bytes), stream the copy
	push	hl
	ld	hl, (ix + 9)
	push	hl
	pop	bc
	add	hl, hl
	add	hl, bc
	push	hl
	pop	bc
	pop	hl
	ldir
	pop	hl
1:	pop	af
	ld	sp, hl			; SP = x (or its copy)
	bit	7, a
	jp	nz, .Lmn_neg
	or	a, a
	jp	z, .Lmn_neg
	neg
	add	a, 8			; k1 = 8 - c
	ld	(mn_k1), a
	ld	ix, (ix + 18)		; out
.Lmn_pos:
	MULNORM_BODY 1, .Lmn_pos
	jp	.Lmn_done
.Lmn_neg:
	neg				; k1 = -c
	ld	(mn_k1), a
	ld	ix, (ix + 18)
.Lmn_nloop:
	MULNORM_BODY 0, .Lmn_nloop
.Lmn_done:
	ld	sp, (mn_sp)
	ld	hl, (mn_ss)
	pop	af
	pop	ix
	ret	po
	ei
	ret

	.section .text.clm_a_t3, "ax", @progbits
	.global	_clm_a_t3
_clm_a_t3:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	ld	a, (ix + 15)
	ld	(mn_E3), a
	or	a, a
	sbc	hl, hl
	ld	(mn_pk), hl
	ld	hl, (ix + 21)		; be
	ld	de, (ix + 9)		; bm
	or	a, a
	sbc	hl, de
	ex	de, hl			; DE = be - bm (row i's exponent is at bm + i + DE)
	ld	hl, (ix + 9)
	ld	bc, (ix + 12)
	exx				; shadow: HL' = bm, DE' = be - bm, BC' = count
	ld	(mn_sp), sp
	ld	hl, (ix + 6)
	ld	sp, hl			; SP = acc
	ld	ix, (ix + 18)		; IX = t3
.Lt3:
	pop	de			; acc first: the pushes below write over its slot, never t3
	exx
	ld	a, (hl)			; b
	ld	iyl, a
	push	hl
	add	hl, de
	ld	a, (mn_E3)
	sub	a, (hl)			; d = E3 - be in [0, 8]
	pop	hl
	inc	hl
	exx
	; m = b << (8 - d): d = 0 -> (b, 0); else b * (128 >> (d - 1))
	jr	nz, 1f
	ld	a, iyl
	ld	iyh, a
	ld	iyl, 0
	jr	3f
1:	ld	b, 0x80
	dec	a
	jr	z, 2f
5:	srl	b
	dec	a
	jr	nz, 5b
2:	ld	c, iyl
	mlt	bc
	ld	iyl, c
	ld	iyh, b
3:	ld	hl, 0x800000
	add	hl, de			; carry: acc < 0
	sbc	a, a
	push	af			; A = 0xff if negative
	or	a, a
	jr	z, 4f
	ld	hl, 0
	sbc	hl, de			; (carry clear after `or a`)
	ex	de, hl			; |acc|
4:	MUL16R				; X = |acc| * m: HL = X >> 16, D, E low bytes
	pop	af
	ld	c, a			; C = sign mask
	or	a, a
	jr	z, 6f
	ld	a, e			; X - 1 (negative: round half up of -X)
	sub	a, 1
	ld	e, a
	ld	a, d
	sbc	a, 0
	ld	d, a
	jr	nc, 6f
	dec	hl
6:	add	hl, hl			; R = (X >> 8) + bit 7 of X (X < 2^31: R16 < 2^15)
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	ld	l, d
	bit	7, e
	jr	z, 7f
	inc	hl
7:	; peak = max(peak, R)
	ex	de, hl
	ld	hl, (mn_pk)
	or	a, a
	sbc	hl, de
	jr	nc, 8f
	ld	(mn_pk), de
8:	ex	de, hl			; HL = R
	ld	a, c
	or	a, a
	jr	z, 9f
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de
9:	ld	(ix + 0), hl
	lea	ix, ix + 3
	exx
	dec	bc
	ld	a, b
	or	a, c
	exx
	jp	nz, .Lt3
	ld	sp, (mn_sp)
	ld	hl, (mn_pk)
	pop	af
	pop	ix
	ret	po
	ei
	ret

	.section .text.clm_a_maxabs32, "ax", @progbits
	.global	_clm_a_maxabs32
_clm_a_maxabs32:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	iy, (ix + 6)		; v
	ld	bc, (ix + 9)		; n
	ld	hl, (ix + 12)
	ld	(mn_pk), hl		; result pointer
	ld	de, 0			; peak: DE = bits 8..31, IXL = bits 0..7
	ld	ixl, 0
.Lmx:
	ld	hl, (iy + 1)		; bits 8..31
	ld	a, (iy + 0)		; bits 0..7
	bit	7, (iy + 3)
	jr	z, 1f
	neg				; |v| = -v: low byte, borrow in carry
	push	de
	ex	de, hl
	ld	hl, 0
	sbc	hl, de
	pop	de
1:	push	hl
	scf
	ccf
	sbc	hl, de			; compare bits 8..31 (A preserved)
	pop	hl
	jr	c, 3f			; smaller
	jr	nz, 2f			; larger
	cp	a, ixl			; equal high part: compare the low byte
	jr	c, 3f
	jr	z, 3f
2:	ex	de, hl
	ld	ixl, a
3:	lea	iy, iy + 4
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .Lmx
	ld	hl, (mn_pk)
	ld	a, ixl
	ld	(hl), a
	inc	hl
	ld	(hl), de		; bits 8..31 (top byte of DE is bits 24..31)
	pop	ix
	ret

	.section .note.GNU-stack,"",%progbits
