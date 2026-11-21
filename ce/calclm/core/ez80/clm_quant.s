; CalcLM eZ80 hot loop: quant_vec's per-element pass (INTREF.md section 3).
;
; void clm_a_quant(const int32_t *v, unsigned n, int c, unsigned gm, int S, int8_t *x8)
;   v'  = normalize16 element: c >= 1: clamp(rsr(v, c), +-32767); c <= 0: v << -c
;   x8  = clamp(rsr(v' * gm, S), +-127)
; Preconditions (checked by the C caller, which otherwise uses the C path):
;   1 <= n <= 65535, c in [-15, 16] with |v| < 2^(c+15) for every element (c comes from the
;   vector's peak), gm < 2^15, S in [17, 24], |rsr(v' * gm, S)| <= 128 for every element.
;
; Method (all exact):
;   c >= 1: X = the 24-bit signed value at v + o (o = 1 if c >= 9 else 0) is floor(v / 2^(8o))
;           and rsr(v, c) = rsr(X, k) with k = c - 8o in [1, 8]. With Y = (X << (8 - k)) + 128
;           (|X << (8-k)| < 2^23), v' = bits 8..23 of Y; the one ambiguous pattern 0x8000 is
;           +32768 (clamped to 32767) if Y's low byte < 128, else -32768 (clamped to -32767).
;   P = v' * gm: unsigned product of v' mod 2^16 and gm, minus gm * 2^16 when v' < 0, so
;           P >> 16 = R16 - gm; x = rsr(P, S) = bits 8..15 of (R16 << (24 - S)) + 128 (taken
;           mod 256; the only out-of-range pattern 0x80 means +-128 and is clamped by sign).
	.assume	adl=1
	.include "clm_mac.inc"

; QTAIL loop: v' in DE -> x8 store, count, and branch back to `loop` while elements remain
.macro QTAIL loop
	push	de
	MUL16R				; HL = (v' mod 2^16) * gm >> 16
	pop	bc			; B = v' high byte (sign)
	bit	7, b
	jr	z, 3f
	lea	de, iy + 0
	or	a, a
	sbc	hl, de			; signed: R16 - gm
3:	exx
	ld	a, e			; k2
	exx
	or	a, a
	jr	z, 5f
4:	add	hl, hl
	dec	a
	jr	nz, 4b
5:	ld	de, 128
	add	hl, de
	ld	a, h			; x mod 256
	cp	a, 0x80
	jr	nz, 6f
	ld	a, 0x7f			; +128 -> 127
	bit	7, b
	jr	z, 6f
	ld	a, 0x81			; -128 -> -127
6:	exx
	ld	(hl), a
	inc	hl
	dec	bc
	ld	a, b
	or	a, c
	exx
	jp	nz, \loop
.endm

	.section .text.clm_a_quant, "ax", @progbits
	.global	_clm_a_quant
_clm_a_quant:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i			; P/V = IEF1 (restored on return)
	push	af
	di				; shadow registers in use below
	ld	iy, (ix + 15)		; gm (IYU = 0)
	ld	a, 24
	sub	a, (ix + 18)		; k2 = 24 - S in [0, 7]
	ld	e, a
	ld	a, (ix + 12)		; c
	ld	hl, (ix + 6)		; v
	ld	bc, (ix + 9)		; n
	bit	7, a
	jp	nz, .Lneg_c
	or	a, a
	jp	z, .Lneg_c		; c = 0: a left shift by 0
	cp	a, 9
	jr	c, 1f
	inc	hl			; c >= 9: read X at v + 1, k = c - 8
	sub	a, 8
1:	ld	d, a			; k
	ld	a, 8
	sub	a, d
	ld	d, a			; k1 = 8 - k in [0, 7]
	push	hl
	ld	hl, (ix + 21)		; x8
	exx
	pop	ix			; IX = v (+o); shadow: HL' = x8, BC' = n, D' = k1, E' = k2
.Lloop:
	ld	hl, (ix + 0)		; X
	lea	ix, ix + 4
	exx
	ld	a, d			; k1
	exx
	or	a, a
	jr	z, 2f
	ld	b, a
1:	add	hl, hl
	djnz	1b
2:	ld	de, 128
	add	hl, de			; Y
	ld	c, l			; Y low byte (resolves the 0x8000 pattern)
	HLU_A
	ld	d, a
	ld	e, h			; DE = v' (16 bits)
	cp	a, 0x80
	jr	nz, .Lmul
	ld	a, e
	or	a, a
	jr	nz, .Lmul
	ld	de, 0x7fff		; +32768 -> 32767
	bit	7, c
	jr	z, .Lmul
	ld	de, 0x8001		; -32768 -> -32767
.Lmul:
	QTAIL	.Lloop
	jr	.Ldone

	; c <= 0: v' = v << -c with |v| < 2^(15+c): the low 24 bits of v hold it exactly
.Lneg_c:
	neg				; k1 = -c in [0, 15]
	ld	d, a
	push	hl
	ld	hl, (ix + 21)
	exx
	pop	ix
.Lnloop:
	ld	hl, (ix + 0)
	lea	ix, ix + 4
	exx
	ld	a, d
	exx
	or	a, a
	jr	z, 2f
	ld	b, a
1:	add	hl, hl
	djnz	1b
2:	ld	d, h
	ld	e, l			; DE = v' (|v'| <= 32767)
	QTAIL	.Lnloop

.Ldone:
	pop	af
	pop	ix
	ret	po
	ei
	ret

	.section .note.GNU-stack,"",%progbits
