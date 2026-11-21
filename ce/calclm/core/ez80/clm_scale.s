; CalcLM eZ80 hot loop: scale_out (INTREF.md section 3), one matrix output at a time.
;
; void clm_a_scale(const clm_i24 *acc, const uint8_t *bm, unsigned n, unsigned sxm, int sb,
;                  void *out, unsigned mode, uint8_t *rec, const int8_t *be)
;   row i: m = bm[i] * sxm, S = sb - be[i];
;          R = rsr(acc[i] * m, S), then by mode:
;            0: out int8[i]  = clamp(R, -127, 127)
;            1: out int16[i] = clamp(R, -32768, 32767)
;            2: out int24[i] = clamp(R, -2^23, 2^23 - 1)
;            3: out int24[i] = clamp(out[i] + clamp(R, int24), int24)   (residual add)
; Preconditions (the C caller checks them per matrix, else it uses the C path): 1 <= n,
;   |acc| < 2^16, 1 <= S <= 24 for every row; rec: 1536 bytes of scratch (256 records).
;
; Method: rows are processed in chunks of up to 256. A pre-pass pushes one 6-byte record per
; row onto a private stack in `rec` ({m, S, cpow}); the main pass pops them (so it walks the
; chunk backwards). Work is on magnitudes: X = |acc| * m (MUL16R gives X >> 16 = R16 and the
; low bytes), and rsr(+-X, S) = +-((X - neg + 2^(S-1)) >> S) for S >= 1, neg = acc < 0.
;   S <= 16: shift the 32-bit X - neg left by 16 - S; R = (X >> 16) + bit 15.
;   S >= 17: with c = 2^(24-S) <= 128 and d = neg and (X mod 2^16 == 0):
;            R = (R16 * c + 128 - d * c) >> 8 = R16H * c + ((R16L * c + 128 - d * c) >> 8).
	.assume	adl=1
	.include "clm_mac.inc"

	.section .bss.clm_a_scale, "aw", @nobits
sc_sp:		.skip	3		; caller's SP
sc_acc:		.skip	3		; next acc row (forward)
sc_out:		.skip	3		; next out row (forward)
sc_left:	.skip	3		; rows left
sc_bm:		.skip	3		; next bm
sc_n:		.skip	3
sc_rec:		.skip	3		; end of the record area
sc_be:		.skip	3		; next be
sc_mode:	.skip	1
sc_sxm:		.skip	1
sc_sb:		.skip	1
sc_cnt:		.skip	1		; rows in this chunk (0 = 256)

	.section .text.clm_a_scale, "ax", @progbits
	.global	_clm_a_scale
_clm_a_scale:
	push	ix
	ld	ix, 0
	add	ix, sp
	ld	a, i
	push	af
	di				; SP and shadow registers are data registers below
	ld	hl, (ix + 6)
	ld	(sc_acc), hl
	ld	hl, (ix + 9)
	ld	(sc_bm), hl
	ld	hl, (ix + 12)
	ld	(sc_n), hl
	ld	(sc_left), hl
	ld	a, (ix + 15)
	ld	(sc_sxm), a
	ld	a, (ix + 18)
	ld	(sc_sb), a
	ld	hl, (ix + 21)
	ld	(sc_out), hl
	ld	a, (ix + 24)
	ld	(sc_mode), a
	ld	hl, (ix + 30)
	ld	(sc_be), hl
	ld	hl, (ix + 27)
	ld	de, 1536
	add	hl, de
	ld	(sc_rec), hl
	ld	(sc_sp), sp

.Lchunk:
	; cnt = min(left, 256)
	ld	hl, (sc_left)
	ld	de, 256
	or	a, a
	sbc	hl, de
	jr	nc, 1f
	add	hl, de			; left < 256
	ld	a, l
	ld	hl, 0
	jr	2f
1:	xor	a, a			; 256 rows (count 0)
2:	ld	(sc_left), hl
	ld	(sc_cnt), a

	; ---- pre-pass: records for the chunk's rows, pushed in row order --------------------
	ld	sp, (sc_rec)		; private stack: pushes fill `rec` downwards
	ld	hl, (sc_bm)		; HL = bm ptr
	ld	de, (sc_be)		; DE = be ptr
	ld	a, (sc_sxm)
	ld	iyl, a
	ld	a, (sc_sb)
	ld	iyh, a
	ld	a, (sc_cnt)
	ld	ixl, a			; row counter (0 = 256)
.Lpre:
	ld	a, iyh			; sb
	ex	de, hl
	sub	a, (hl)			; S = sb - be
	inc	hl
	ex	de, hl
	ld	c, a			; C = S
	ld	b, 0			; cpow = 0 for S <= 16
	sub	a, 17
	jr	c, 4f
	ld	b, 0x80			; S >= 17: cpow = 128 >> (S - 17)
	jr	z, 4f
3:	srl	b
	dec	a
	jr	nz, 3b
4:	push	bc			; record bytes 3..5: S, cpow, (unused)
	ld	b, (hl)			; bm
	inc	hl
	ld	c, iyl			; sxm
	mlt	bc			; m = bm * sxm < 2^15
	push	bc			; record bytes 0..2: m (3 bytes, top byte 0)
	dec	ixl
	jr	nz, .Lpre
	ld	(sc_bm), hl
	ld	(sc_be), de

	; ---- main pass: last row of the chunk first ------------------------------------------
	ld	a, (sc_cnt)
	ld	b, a			; B' = row count (0 = 256)
	dec	a			; cnt - 1 (0..255)
	ld	e, a
	ld	d, 3
	mlt	de			; 3 (cnt - 1)
	ld	ix, (sc_acc)
	add	ix, de			; IX = last acc row
	ld	e, a
	ld	a, (sc_mode)
	ld	d, 1			; element size 1, 2, 3, 3
	or	a, a
	jr	z, 5f
	inc	d
	dec	a
	jr	z, 5f
	inc	d
5:	mlt	de			; size * (cnt - 1)
	ld	hl, (sc_out)
	add	hl, de			; HL' = last out element
	exx

.Lrow:
	pop	iy			; m (IYU = 0)
	pop	bc			; C = S, B = cpow
	ld	de, (ix + 0)		; acc (|acc| < 2^16)
	lea	ix, ix - 3
	ld	hl, 0x800000
	add	hl, de			; carry: acc < 0
	jr	nc, 1f
	ld	hl, 0
	or	a, a
	sbc	hl, de
	ex	de, hl			; DE = |acc|
	set	7, c			; C bit 7 = neg
1:	push	bc
	MUL16R				; X = |acc| * m: HL = X >> 16, D = (X >> 8) & 255, E = X & 255
	pop	bc
	ld	a, c
	and	a, 0x7f			; S
	cp	a, 17
	jr	nc, .Lhi

	; ---- S <= 16: R = ((X - neg) << (16 - S)) >> 16 + bit 15 ----------------------------
	bit	7, c
	jr	z, 2f
	ld	a, e			; X - 1 (X >= 1 when acc < 0)
	sub	a, 1
	ld	e, a
	ld	a, d
	sbc	a, 0
	ld	d, a
	jr	nc, 2f
	dec	hl
2:	ld	a, c
	and	a, 0x7f
	neg
	add	a, 16			; n = 16 - S in [0, 15]
	jr	z, 7f
	cp	a, 8
	jr	c, 5f
	add	hl, hl			; n >= 8: shift by a byte (HL < 2^15: no overflow)
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	add	hl, hl
	ld	l, d
	ld	d, e
	ld	e, 0
	sub	a, 8
	jr	z, 7f
5:	sla	e
	rl	d
	adc	hl, hl
	jr	c, .Lsat		; R >= 2^24
	jp	m, .Lsat		; R >= 2^23 (sign of the adc result; later shifts only grow it)
	dec	a
	jr	nz, 5b
7:	bit	7, d
	jr	z, .Lstore
	inc	hl			; HL < 2^23 here, no wrap
	jr	.Lstore

.Lsat:
	ld	hl, 0x800000		; any magnitude >= 2^23 saturates in every mode
	jr	.Lstore

	; ---- S >= 17: R = R16H * c + ((R16L * c + 128 - d * c) >> 8) ------------------------
.Lhi:
	ld	a, d
	or	a, e			; Z: low 16 bits of X are zero
	ld	a, 128
	jr	nz, 6f			; d = 0
	bit	7, c
	jr	z, 6f			; acc >= 0: d = 0
	sub	a, b			; d = 1: 128 - c
6:	ld	e, l
	ld	d, b
	mlt	de			; R16L * c
	add	a, e
	ld	e, a
	jr	nc, 8f
	inc	d
8:	ld	l, h
	ld	h, b
	mlt	hl			; R16H * c (HLU = 0)
	ld	e, d
	ld	d, 0			; DE = high byte of the low term (DEU = 0 after mlt)
	add	hl, de			; R < 2^16

	; ---- clamp, sign, store by mode; HL = |R|, C bit 7 = neg --------------------------
.Lstore:
	ld	a, (sc_mode)
	or	a, a
	jp	z, .Lm8
	dec	a
	jp	z, .Lm16
	; modes 2, 3: int24, limit 2^23 - 1 + neg
	ld	de, 0x800000
	bit	7, c
	jr	z, 1f
	inc	de
1:	or	a, a
	sbc	hl, de
	jr	c, 2f
	ld	hl, 0
	dec	de			; HL = limit (after the add below)
2:	add	hl, de
	bit	7, c
	jr	z, 3f
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de			; -R
3:	ld	a, (sc_mode)
	push	hl
	exx				; (B' is the row counter: keep BC' intact)
	pop	de			; DE' = R (int24)
	cp	a, 3
	jr	nz, 4f
	push	hl			; residual add, saturated
	ld	hl, (hl)		; r[i]
	or	a, a
	adc	hl, de
	jp	po, 5f			; no signed overflow
	ld	hl, 0x7fffff		; overflow: a negative-looking sum means it was too positive
	jp	m, 5f
	ld	hl, 0x800000
5:	ex	de, hl
	pop	hl
4:	ld	(hl), de
	dec	hl
	dec	hl
	dec	hl
	jr	.Lnext

.Lm16:	; int16: limit 32767 + neg
	ld	de, 32768
	bit	7, c
	jr	z, 1f
	inc	de
1:	or	a, a
	sbc	hl, de
	jr	c, 2f
	ld	hl, 0
	dec	de
2:	add	hl, de
	bit	7, c
	jr	z, 3f
	ex	de, hl
	ld	hl, 0
	or	a, a
	sbc	hl, de
3:	ld	a, l
	ld	b, h
	exx
	ld	(hl), a
	inc	hl
	exx
	ld	a, b
	exx
	ld	(hl), a
	dec	hl
	dec	hl
	dec	hl
	jr	.Lnext

.Lm8:	; int8: limit 127
	ld	de, 128
	or	a, a
	sbc	hl, de
	ld	a, 127
	jr	nc, 1f
	add	hl, de
	ld	a, l
1:	bit	7, c
	jr	z, 2f
	neg
2:	exx
	ld	(hl), a
	dec	hl

.Lnext:					; shadow set active
	djnz	1f
	exx
	jr	.Lchunk_end
1:	exx
	jp	.Lrow

.Lchunk_end:
	; advance acc and out past the chunk
	ld	a, (sc_cnt)
	ld	de, 0
	ld	e, a
	or	a, a
	jr	nz, 1f
	inc	d			; 256
1:	ld	hl, (sc_acc)
	add	hl, de
	add	hl, de
	add	hl, de
	ld	(sc_acc), hl
	ld	hl, (sc_out)
	ld	a, (sc_mode)
	add	hl, de
	or	a, a
	jr	z, 2f
	add	hl, de
	dec	a
	jr	z, 2f
	add	hl, de
2:	ld	(sc_out), hl
	ld	hl, (sc_left)
	add	hl, de
	or	a, a
	sbc	hl, de			; flags for left == 0
	jp	nz, .Lchunk

	ld	sp, (sc_sp)
	pop	af
	pop	ix
	ret	po
	ei
	ret

	.section .note.GNU-stack,"",%progbits
