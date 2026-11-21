; Shared macros for the CalcLM eZ80 hot loops (ADL mode, GNU as syntax).
;
; Cycle notes (CEmu timing, code in RAM): every opcode byte costs 4 cycles, a data read 4, a
; write 2, MLT 4 extra. The upper byte of a 24-bit register (e.g. HLU) is not directly
; addressable; HLU_A gets it through the stack.

; MUL16R: unsigned 16 x 16 -> 32-bit product P = (D:E) * (IYH:IYL).
;   out: HL = P >> 16 (HLU = 0), D = (P >> 8) & 255, E = P & 255
;   clobbers AF, AF', BC
.macro MUL16R
	ld	h, d
	ld	a, iyl
	ld	l, a
	mlt	hl			; t1 = aH*mL (HLU = 0)
	ld	b, e
	ld	c, a
	mlt	bc			; t0 = aL*mL
	ld	a, c
	ex	af, af'			; A' = p0
	ld	c, b
	ld	b, 0			; BC = t0 >> 8 (BCU = 0 after mlt)
	add	hl, bc			; s = t1 + t0H < 2^16
	ld	b, e
	ld	c, iyh
	mlt	bc			; t2 = aL*mH
	ld	e, iyh
	mlt	de			; t3 = aH*mH (DEU = 0)
	ld	a, l
	add	a, c			; mid = s + t2: low byte, carry
	ld	c, a			; C = midL
	ld	a, h
	adc	a, b			; mid high byte, carry = bit 16 of mid
	ld	l, a
	ld	a, 0
	adc	a, a
	ld	h, a			; HL = mid >> 8
	add	hl, de			; HL = t3 + (mid >> 8) = P >> 16
	ld	d, c
	ex	af, af'
	ld	e, a
.endm

; HLU_A: A = HLU. Interrupt-safe (an interrupt between push and pop only overwrites L).
.macro HLU_A
	push	hl
	inc	sp
	pop	af
	dec	sp
.endm
