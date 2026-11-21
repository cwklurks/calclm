/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Thin C API over the CEmu core for the Python ctypes wrapper.
 * Adapted from the author's earlier CEmu shim and extended for CalcLM M2 with
 * debugger-style instrumentation (ce/tools/bench.py): direct writes into the
 * emulated flash array and RAM, setting PC/SP/ADL with interrupts disabled, running until the
 * CPU executes HALT with a cycle limit, and reading CEmu's exact cycle counters.
 * Implements the gui_* callbacks the core needs, like tests/autotester does.
 * Time unit for shim_run is milliseconds of emulated time (run rate 1000), unthrottled.
 *
 * Licence: GPL-3.0-or-later, unlike the rest of CalcLM (MIT, see LICENSE). This file follows
 * CEmu-style code and is built and linked with the CEmu core (GPL-3.0,
 * https://github.com/CE-Programming/CEmu) into ce/build/libcebench.so (ce/tools/build.sh), a
 * combined work under the GPL. CEmu itself is not part of this repository. */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emu.h"
#include "asic.h"
#include "cpu.h"
#include "mem.h"
#include "keypad.h"
#include "lcd.h"
#include "link.h"
#include "vat.h"
#include "extras.h"
#include "schedule.h"
#include "flash.h"
#include "control.h"
#include "registers.h"

static int verbose;
static volatile int transfer_done = 1;

void gui_console_clear(void) {}

void gui_console_printf(const char *format, ...) {
    if (verbose) {
        va_list ap;
        va_start(ap, format);
        vfprintf(stderr, format, ap);
        va_end(ap);
    }
}

void gui_console_err_printf(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}

asic_rev_t gui_handle_reset(const boot_ver_t *boot_ver, asic_rev_t loaded_rev,
                            asic_rev_t default_rev, bool *python) {
    (void)boot_ver; (void)loaded_rev; (void)default_rev; (void)python;
    return ASIC_REV_AUTO;
}

void shim_set_verbose(int on) { verbose = on; }

/* 0 = valid, 1 = invalid, 2 = not a CE */
int shim_load_rom(const char *path) {
    int st = emu_load(EMU_DATA_ROM, path);
    emu_set_run_rate(1000);
    return st;
}

int shim_load_state(const char *path) {
    int st = emu_load(EMU_DATA_IMAGE, path);
    emu_set_run_rate(1000);
    return st;
}

int shim_save_state(const char *path) { return emu_save(EMU_DATA_IMAGE, path) ? 0 : 1; }

void shim_run(uint32_t ms) { emu_run(ms); }

void shim_key(unsigned row, unsigned col, int press) { emu_keypad_event(row, col, press != 0); }

/* Inject an OS keycode (as GetKey would return it); 1 = accepted, 0 = OS busy. */
int shim_os_key(uint16_t key) { return sendKey(key) ? 1 : 0; }

static bool progress(void *ctx, int value, int total) {
    (void)ctx;
    if (value == 1 && total == 1) {
        transfer_done = 1;
    }
    return false;
}

/* location: 0 RAM, 1 Archive. Starts the transfer; poll shim_transfer_done while running. */
int shim_send(const char *const *files, int num, int location) {
    transfer_done = 0;
    int r = emu_send_variables(files, num, location, progress, NULL);
    if (r != LINK_GOOD) {
        transfer_done = 1;
    }
    return r;
}

int shim_transfer_done(void) { return transfer_done; }

/* Receive one variable (type + raw VAT name bytes) to a TI file. 0 = ok. */
int shim_receive(const char *file, int type, const uint8_t *name, int namelen) {
    calc_var_t want;
    memset(&want, 0, sizeof want);
    want.type = (calc_var_type_t)type;
    want.type1 = (uint8_t)type;
    want.namelen = (uint8_t)namelen;
    memcpy(want.name, name, namelen > 8 ? 8 : namelen);
    return emu_receive_variable(file, &want, 1);
}

void shim_read(uint32_t addr, uint8_t *out, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        out[i] = mem_peek_byte(addr + i);
    }
}

void shim_write(uint32_t addr, const uint8_t *in, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        mem_poke_byte(addr + i, in[i]);
    }
}

/* Raw flash copy (archive inspection). */
void shim_read_flash(uint32_t addr, uint8_t *out, uint32_t len) {
    if (addr + len > SIZE_FLASH) {
        len = addr < SIZE_FLASH ? SIZE_FLASH - addr : 0;
    }
    memcpy(out, mem.flash.block + addr, len);
}

/* 320x240 RGB888 into out (230400 bytes). */
void shim_lcd_rgb(uint8_t *out) {
    static uint32_t frame[LCD_SIZE];
    memset(frame, 0, sizeof frame);
    emu_lcd_drawframe(frame);
    for (int i = 0; i < LCD_SIZE; i++) {
        uint32_t p = frame[i];
        out[3 * i + 0] = (uint8_t)(p >> 0);
        out[3 * i + 1] = (uint8_t)(p >> 8);
        out[3 * i + 2] = (uint8_t)(p >> 16);
    }
}

/* VAT listing: records of 16 bytes {type, namelen, archived, pad, size u16le, pad2[2], name[8]}.
 * Returns the number of records written (at most cap). */
int shim_vat(uint8_t *out, int cap) {
    calc_var_t var;
    int n = 0;
    vat_search_init(&var);
    while (n < cap && vat_search_next(&var)) {
        uint8_t *r = out + 16 * n;
        memset(r, 0, 16);
        r[0] = (uint8_t)var.type;
        r[1] = var.namelen;
        r[2] = var.archived;
        r[4] = (uint8_t)(var.size & 0xFF);
        r[5] = (uint8_t)(var.size >> 8);
        memcpy(r + 8, var.name, 8);
        n++;
    }
    return n;
}

/* Copy a variable's data (as the VAT walk sees it). Returns its size or -1 if missing. */
int shim_var_data(int type, const uint8_t *name, int namelen, uint8_t *out, int cap) {
    calc_var_t var;
    vat_search_init(&var);
    while (vat_search_next(&var)) {
        if ((int)var.type == type && var.namelen == namelen && !memcmp(var.name, name, namelen)) {
            int n = var.size < cap ? var.size : cap;
            memcpy(out, var.data, n);
            return var.size;
        }
    }
    return -1;
}

/* Cumulative CPU cycles and cycles spent halted (OS waiting for a key). */
void shim_cpu_stats(uint64_t *total, uint64_t *halted, int *is_halted) {
    *total = sched_total_cycles();
    *halted = cpu.haltCycles;
    *is_halted = cpu.halted;
}

/* ------------------------------------------------------------------------------------------
 * CalcLM M2 benchmark instrumentation (debugger-style; the emulated CPU is pointed at our code).
 * ------------------------------------------------------------------------------------------ */

/* Copy bytes straight into the emulator's private flash array (no flash command emulation).
 * Returns 0 on success, 1 if the range is outside the 4 MB chip. */
int shim_write_flash(uint32_t addr, const uint8_t *in, uint32_t len) {
    if (addr >= SIZE_FLASH || len > SIZE_FLASH - addr) {
        return 1;
    }
    memcpy(mem.flash.block + addr, in, len);
    return 0;
}

/* Copy bytes straight into emulated RAM (CPU addresses 0xD00000..0xD657FF). */
int shim_write_ram(uint32_t addr, const uint8_t *in, uint32_t len) {
    if (addr < 0xD00000 || addr - 0xD00000 >= SIZE_RAM || len > SIZE_RAM - (addr - 0xD00000)) {
        return 1;
    }
    memcpy(mem.ram.block + (addr - 0xD00000), in, len);
    return 0;
}

int shim_read_ram(uint32_t addr, uint8_t *out, uint32_t len) {
    if (addr < 0xD00000 || addr - 0xD00000 >= SIZE_RAM || len > SIZE_RAM - (addr - 0xD00000)) {
        return 1;
    }
    memcpy(out, mem.ram.block + (addr - 0xD00000), len);
    return 0;
}

/* Hardware facts relevant to timing. out[] (uint32):
 * 0 asic revision (1=A, 2=I, 3=M)   1 serial flash (1) / parallel (0)   2 python edition
 * 3 CPU clock Hz                   4 flash.waitStates                    5 flash.mask
 * 6 flash.mappedBytes              7 control.privileged                  8 protectedStart
 * 9 protectedEnd                   10 stackLimit                         11 cpuSpeed (0..3)
 * 12 flash port 0x05 (parallel wait-state register)                     13 MBASE */
void shim_hw_info(uint32_t *out) {
    out[0] = (uint32_t)asic.revision;
    out[1] = asic.serFlash ? 1 : 0;
    out[2] = asic.python ? 1 : 0;
    out[3] = sched_get_clock_rate(CLOCK_CPU);
    out[4] = flash.waitStates;
    out[5] = flash.mask;
    out[6] = flash.mappedBytes;
    out[7] = control.privileged;
    out[8] = control.protectedStart;
    out[9] = control.protectedEnd;
    out[10] = control.stackLimit;
    out[11] = control.cpuSpeed;
    out[12] = flash.ports[0x05];
    out[13] = cpu.registers.MBASE;
}

/* CEmu's optional accurate LCD DMA emulation: when on, the LCD controller's VRAM reads are
 * scheduled as DMA and stall CPU RAM accesses (cpu.dmaCycles), as on hardware. Off by default
 * in the core (the GUI exposes it as an option). Not part of the saved state's semantics, so
 * set it after every state load. */
void shim_set_lcd_dma(int enable) { emu_set_lcd_dma(enable ? 1 : 0); }

/* Registers: out[] (uint32) = AF BC DE HL IX IY SPL PC, MBASE, ADL, IEF1, halted. */
void shim_get_regs(uint32_t *out) {
    eZ80registers_t *r = &cpu.registers;
    out[0] = r->AF;
    out[1] = r->BC;
    out[2] = r->DE;
    out[3] = r->HL;
    out[4] = r->IX;
    out[5] = r->IY;
    out[6] = r->SPL;
    out[7] = r->PC;
    out[8] = r->MBASE;
    out[9] = cpu.ADL;
    out[10] = cpu.IEF1;
    out[11] = cpu.halted;
}

/* Point the CPU at `pc` in ADL mode with interrupts disabled (like a debugger "set PC"),
 * optionally flush the serial-flash cache (cold cache), then run until the CPU executes HALT
 * or `max_cycles` active cycles elapse. stats[] (uint64):
 * 0 active cycles (total minus halted)   1 total cycles   2 halted cycles
 * 3 DMA stall cycles (LCD/USB DMA stealing RAM bandwidth)  4 flash accesses (cache touches)
 * 5 flash cache misses  6 reached HALT (1/0)  7 final PC
 * (4 and 5 are differences of CEmu's 32-bit counters; CEmu rescales flashTotalAccesses when it
 * wraps after 2^32 accesses, so 4 is only meaningful for calls shorter than that.)
 * Returns 0 if HALT was reached, 1 on timeout. */
static int run_until_halt(uint64_t max_cycles, uint64_t *stats);

int shim_call(uint32_t pc, uint32_t sp, int flush_cache, uint64_t max_cycles, uint64_t *stats) {
    eZ80registers_t *r = &cpu.registers;

    if (flush_cache) {
        flash_flush_cache();
    }
    cpu.IEF1 = cpu.IEF2 = false;
    cpu.IEF_wait = false;
    cpu.NMI = false;
    cpu.halted = false;
    cpu.inBlock = false;
    cpu.PREFIX = 0;
    cpu.SUFFIX = false;
    cpu.ADL = cpu.L = cpu.IL = true;
    cpu.MADL = false;
    r->SPL = sp & 0xFFFFFF;
    r->PC = pc & 0xFFFFFF;
    cpu.prefetch = mem_peek_byte(r->PC);
    return run_until_halt(max_cycles, stats);
}

/* Continue after a HALT (like a debugger "continue" after a breakpoint): clear the halted
 * state and run on from the current PC with all registers as they are, until the next HALT
 * with interrupts disabled or `max_cycles` active cycles. Same stats and return as shim_call.
 * CalcLM M4 uses it for phase profiling (the code under test HALTs at each phase start). */
int shim_resume(uint64_t max_cycles, uint64_t *stats) {
    cpu.halted = false;
    return run_until_halt(max_cycles, stats);
}

static int run_until_halt(uint64_t max_cycles, uint64_t *stats) {
    eZ80registers_t *r = &cpu.registers;
    uint64_t t0 = sched_total_cycles();
    uint64_t h0 = cpu.haltCycles;
    uint64_t d0 = cpu.dmaCycles;
    uint32_t fa0 = cpu.flashTotalAccesses;
    uint32_t fm0 = cpu.flashCacheMisses;

    int reached = 0;
    for (;;) {
        emu_run(1); /* 1 ms of emulated time per slice; counters below are exact */
        if (cpu.halted && !cpu.IEF1) {
            reached = 1;
            break;
        }
        if (sched_total_cycles() - t0 - (cpu.haltCycles - h0) > max_cycles) {
            break;
        }
    }
    uint64_t total = sched_total_cycles() - t0;
    uint64_t halted = cpu.haltCycles - h0;
    stats[0] = total - halted;
    stats[1] = total;
    stats[2] = halted;
    stats[3] = cpu.dmaCycles - d0;
    stats[4] = (uint32_t)(cpu.flashTotalAccesses - fa0);
    stats[5] = (uint32_t)(cpu.flashCacheMisses - fm0);
    stats[6] = (uint64_t)reached;
    stats[7] = r->PC;
    return reached ? 0 : 1;
}
