; CalcLM eZ80 hot loops for attention (INTREF.md section 4), one head at a time.
; Cache format (clm.h): K row = 32 keys as k + 128, then -128 * sum(k + 128) as int24;
; V row = 32 values as v + 128. All products below are unsigned 8 x 8 MLTs.
;
; int clm_a_scores(const uint8_t *qu, int ch, const uint8_t *K, unsigned n, clm_i24 *sc, int step)
;   s_j = sum_d qu[d] * K[j][d] + K[j].nk + ch + j * step   (= q . k exactly when step = 0;
;   ch = 2^19 - 128 sum qu, minus an ALiBi penalty that falls by `step` per key otherwise)
;   for j < n into sc[n - 1 - j] (pushed); returns max s_j. Needs 1 <= n and
;   -0x500000 < s_j < 2^20 (|q . k| <= 127^2 * 32 < 2^19, ALiBi penalties <= 2^22 by
;   clm_load.c), so max - s_j stays inside 24 bits.
; unsigned clm_a_softmax(const clm_i24 *sc, unsigned n, int mx, int sh, const uint8_t *lut,
;                        int lut_len, uint8_t *ew)
;   ew[j] = lut[min(rsr(mx - s_j, sh), lut_len - 1)]; returns sum ew. Requires 1 <= sh <= 7,
;   lut_len <= 512 (so every unsaturated mx - s + 2^(sh-1) is below 2^16).
; void clm_a_pv(const uint8_t *V, const uint8_t *ew, unsigned n, unsigned *a)
;   a[d] = sum_j ew[j] * V[j][d] (< 2^24) for d < 32; reads ew and V rows up to n rounded up
;   to a multiple of 8 (ew must hold zeros there, V must be readable there).
; void clm_a_havg(const unsigned *a, unsigned z, unsigned k, unsigned rz, int32_t *o)
;   o[d] = rsr(rsr(a[d] - 128 z, k) * rz, 14) for d < 32; k = bitlen(z) - 8 in [0, 8],
;   rz <= 2^15, |a[d] - 128 z| < 2^(k+15).
	.assume	adl=1
	.include "clm_mac.inc"

	.section .bss.clm_a_attn, "aw", @nobits
at_sp:	.skip	3
at_ch:	.skip	3
at_step:	.skip	3
at_mx:	.skip	3
at_T:	.skip	3		; softmax: saturation threshold on mx' - s
at_lut:	.skip	3
at_sat:	.skip	1		; lut[L - 1]
at_c:	.skip	1		; 2^(8 - sh)

; ---------------------------------------------------------------------------------------
	.section .text.clm_a_scores, "ax", @progbits
; one key: IX = start value + q . k_j (32 unsigned MACs + the row term), pushed as s_j; the
; running max HL' is updated (BC' counts keys); Z is set after the last key
.macro SCORE_KEY
	.irp	d, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	ld	c, (hl)
	inc	hl
	ld	b, (iy + \d)
	mlt	bc
	add	ix, bc
	.endr
	ld	de, (hl)		; nk = -128 sum(k + 128)
	add	ix, de
	inc	hl
	inc	hl
	inc	hl
	push	ix			; s_j
	exx
	pop	de
	push	de			; DE' = s_j
	push	hl			; (scratch below SP: the next score's slot)
	or	a, a
	sbc	hl, de			; max - s, |.| < 2^23
	pop	hl
	jp	p, 1f
	ex	de, hl			; new max
1:	dec	bc
	ld	a, b
	or	a, c
	exx
.endm

	.global	_clm_a_scores
_clm_a_scores:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	ld	hl, (ix + 9)
	ld	(at_ch), hl		; ch
	ld	de, (ix + 21)
	ld	(at_step), de		; step
	ld	hl, 0
	or	a, a
	sbc	hl, de			; Z: step = 0 (the plain loop); HL = -step
	jr	z, 1f
	ld	de, (at_ch)
	add	hl, de			; (24-bit add: Z is kept)
	ld	(at_ch), hl		; ch - step: the stepped loop adds step before each key
1:	push	af			; keep Z (the loop choice) across the setup
	ld	iy, (ix + 6)		; qu
	ld	hl, (ix + 15)		; n
	push	hl
	pop	bc
	add	hl, hl
	add	hl, bc
	ld	de, (ix + 18)
	add	hl, de			; sc + 3n
	push	hl
	ld	hl, -0x500000		; running max (below any score)
	exx				; shadow: BC' = n, HL' = max
	pop	de
	pop	af
	ld	hl, (ix + 12)		; HL = K
	ld	(at_sp), sp
	ex	de, hl
	ld	sp, hl			; SP = sc + 3n: s_j is pushed to sc + 3 (n - 1 - j)
	ex	de, hl
	jp	nz, .Lstep
.Lkey:
	ld	ix, (at_ch)
	SCORE_KEY
	jp	nz, .Lkey
	jp	.Lend
.Lstep:
	ld	ix, (at_ch)
	ld	de, (at_step)
	add	ix, de
	ld	(at_ch), ix		; this key's start value
	SCORE_KEY
	jp	nz, .Lstep
.Lend:
	exx
	ld	sp, (at_sp)
	pop	af
	pop	ix
	ret	po
	ei
	ret

; ---------------------------------------------------------------------------------------
	.section .text.clm_a_softmax, "ax", @progbits
	.global	_clm_a_softmax
_clm_a_softmax:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	ld	b, (ix + 15)		; sh in [1, 7]
	ld	de, (ix + 18)		; lut
	ld	(at_lut), de
	ld	hl, (ix + 21)		; L
	dec	hl
	push	hl
	add	hl, de
	ld	a, (hl)
	ld	(at_sat), a		; lut[L - 1]
	pop	hl
	ld	a, b
1:	add	hl, hl
	dec	a
	jr	nz, 1b
	ld	(at_T), hl		; T = (L - 1) << sh: saturate when mx' - s >= T
	ld	hl, 1
	ld	a, b
	dec	a
	jr	z, 3f
2:	add	hl, hl
	dec	a
	jr	nz, 2b
3:	ld	de, (ix + 12)		; mx
	add	hl, de
	ld	(at_mx), hl		; mx' = mx + 2^(sh - 1)
	ld	a, 8
	sub	a, b
	ld	b, a
	ld	a, 1
4:	add	a, a
	djnz	4b
	ld	(at_c), a		; c = 2^(8 - sh)
	ld	hl, (ix + 24)		; ew
	ld	de, (ix + 9)		; n
	add	hl, de
	dec	hl
	push	hl
	ld	bc, (ix + 9)
	ld	hl, (ix + 6)		; sc
	exx				; BC' = n
	pop	ix			; IX = ew + n - 1 (scores come out last key first)
	exx
	ld	(at_sp), sp
	ld	sp, hl			; SP = sc
	exx
	ld	iy, 0			; Z
.Lsm:
	pop	de			; s
	ld	hl, (at_mx)
	or	a, a
	sbc	hl, de			; d' = mx' - s >= 0
	ld	de, (at_T)
	or	a, a
	sbc	hl, de
	jr	nc, .Lsat
	add	hl, de			; d' < T < 2^16 (HLU = 0)
	ld	a, (at_c)
	ld	d, l
	ld	e, a
	mlt	de			; dL * c
	ld	l, a
	mlt	hl			; dH * c
	ld	e, d
	ld	d, 0			; DEU = 0 after mlt
	add	hl, de			; idx = (d' * c) >> 8 < L - 1
	ld	de, (at_lut)
	add	hl, de
	ld	a, (hl)
	jr	.Lput
.Lsat:
	ld	a, (at_sat)
.Lput:
	ld	(ix + 0), a
	dec	ix
	ld	bc, 0
	ld	c, a
	add	iy, bc
	exx
	dec	bc
	ld	a, b
	or	a, c
	exx
	jp	nz, .Lsm
	ld	sp, (at_sp)
	lea	hl, iy + 0
	pop	af
	pop	ix
	ret	po
	ei
	ret

; ---------------------------------------------------------------------------------------
	.section .text.clm_a_pv, "ax", @progbits
	.global	_clm_a_pv
_clm_a_pv:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	ld	hl, (ix + 12)		; n -> groups of 8 keys: (n + 7) / 8 (n < 2^16)
	ld	de, 7
	add	hl, de
	srl	h
	rr	l
	srl	h
	rr	l
	srl	h
	rr	l
	ld	(at_T), hl
	ld	hl, (ix + 15)
	ld	(at_mx), hl		; output pointer
	ld	hl, (ix + 6)
	ld	(at_lut), hl		; V + d
	ld	hl, (ix + 9)
	ld	(at_ch), hl		; ew
	ld	a, 32
	ld	(at_c), a		; dims left
.Ldim:
	ld	iy, (at_ch)
	ld	hl, (at_lut)
	ld	de, 32
	ld	ix, 0
	exx
	ld	bc, (at_T)
	exx
.Lgrp:
	.irp	k, 0,1,2,3,4,5,6,7
	ld	c, (hl)
	add	hl, de
	ld	b, (iy + \k)
	mlt	bc
	add	ix, bc
	.endr
	lea	iy, iy + 8
	exx
	dec	bc
	ld	a, b
	or	a, c
	exx
	jr	nz, .Lgrp
	ld	hl, (at_mx)
	ld	(hl), ix
	inc	hl
	inc	hl
	inc	hl
	ld	(at_mx), hl
	ld	hl, (at_lut)
	inc	hl
	ld	(at_lut), hl
	ld	a, (at_c)
	dec	a
	ld	(at_c), a
	jp	nz, .Ldim
	pop	af
	pop	ix
	ret	po
	ei
	ret

; ---------------------------------------------------------------------------------------
	.section .text.clm_a_havg, "ax", @progbits
	.global	_clm_a_havg
_clm_a_havg:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di
	ld	hl, (ix + 9)		; z
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	ld	(at_ch), hl		; 128 z
	ld	a, 8
	sub	a, (ix + 12)
	ld	(at_c), a		; left shift 8 - k in [0, 8]
	ld	iy, (ix + 15)		; rz <= 2^15 (m for MUL16R)
	ld	hl, (ix + 6)		; a
	ld	ix, (ix + 18)		; IX = o
	exx
	ld	b, 32
	exx
	ld	(at_sp), sp
	ld	sp, hl			; SP = a (pushes below only overwrite consumed entries)
.Lhav:
	pop	hl
	ld	de, (at_ch)
	or	a, a
	sbc	hl, de			; a - 128 z, |.| < 2^(k+15)
	ld	a, (at_c)
	or	a, a
	jr	z, 2f
	ld	b, a
1:	add	hl, hl
	djnz	1b
2:	ld	de, 128
	add	hl, de			; as = bits 8..23 (rsr(a, k); k = 0: just a)
	HLU_A
	ld	d, a
	ld	e, h			; as, |as| <= 32512
	and	a, 0x80
	push	af
	jr	z, 4f
	xor	a, a			; |as|
	sub	a, e
	ld	e, a
	ld	a, 0
	sbc	a, d
	ld	d, a
4:	MUL16R				; X = |as| * rz
	pop	af
	ld	c, a			; C = sign
	or	a, a
	jr	z, 5f
	ld	a, e			; X - 1 for negative values
	sub	a, 1
	ld	e, a
	ld	a, d
	sbc	a, 0
	ld	d, a
	jr	nc, 5f
	dec	hl
5:	sla	e			; R = ((X - neg) << 2) >> 16 + bit 15
	rl	d
	adc	hl, hl
	sla	e
	rl	d
	adc	hl, hl
	bit	7, d
	jr	z, 6f
	inc	hl
6:	ld	a, c
	or	a, a
	jr	z, 7f
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de
7:	ld	(ix + 0), hl		; |o| < 2^15: byte 3 is the sign of H
	ld	a, h
	rla
	sbc	a, a
	ld	(ix + 3), a
	lea	ix, ix + 4
	exx
	dec	b
	exx
	jp	nz, .Lhav
	ld	sp, (at_sp)
	pop	af
	pop	ix
	ret	po
	ei
	ret

	.section .note.GNU-stack,"",%progbits
