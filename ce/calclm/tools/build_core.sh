#!/usr/bin/env bash
# Build the CalcLM inference core for the eZ80 with the CE toolchain (~/CEdev):
#
#   ce/build/calclm/core.o      relocatable core (C + M2 kernels + asm), linked by the CE
#                               program (ce/calclm/makefile, EXTRA_LIBS) and by the test binary
#   ce/build/calclm/test.bin    core.o + test/drv.c, flat binary at <org> for CEmu tests
#   ce/build/calclm/test.sym    its symbols
#
#   build_core.sh [org] [-f]     (org defaults to 0xD1A87F, the CE program load address)
#
# Model shape: CLM_SHAPE=<header> (default core/clm_shape.h, v2.2's dims; core/clm_shape_w.h is
# the wide model W) is force-included ahead of core/clm_shape.h, whose include guard it shares,
# and generates the matvec kernels. Build each shape into its own CALCLM_OUT; CALCLM_BIN is
# where CLMCODE.8xv goes (default ce/calclm/bin).
#
# C flags are the CE makefile's own (meta/makefile.mk EZCFLAGS); the core is compiled once, so
# the program and the emulator tests run identical machine code. The TI-BASIC tracker
# (tracker/tracker.c, the file training uses) is compiled into core.o for the logit masks.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CALCLM="$(cd "$HERE/.." && pwd)"
CE="$(cd "$CALCLM/.." && pwd)"
TRACKER="$(cd "$CE/../tracker" && pwd)"
CEDEV="${CEDEV:-$HOME/CEdev}"
BIN="$CEDEV/bin"
BU="$CEDEV/binutils/bin"
OUT="${CALCLM_OUT:-$CE/build/calclm}"
SHAPE_IN="${CLM_SHAPE:-$CALCLM/core/clm_shape.h}"
[ -f "$SHAPE_IN" ] || { echo "CLM_SHAPE: no file $SHAPE_IN" >&2; exit 1; }
SHAPE="$(cd "$(dirname "$SHAPE_IN")" && pwd)/$(basename "$SHAPE_IN")"
APPVAR_DIR="${CALCLM_BIN:-$CALCLM/bin}"
PYTHON="${PYTHON:-python3}"
ORG="${1:-0xD1A87F}"
FORCE="${2:-}"
mkdir -p "$OUT/obj"

CFLAGS=(-nostdinc -isystem "$CEDEV/include" -include "$SHAPE" -I"$CALCLM/core" -I"$TRACKER"
        -fno-autolink -fno-addrsig
        -fno-threadsafe-statics -mllvm -profile-guided-section-prefix=false -mllvm -z80-gas-style
        -ffunction-sections -fdata-sections -fno-math-errno -D__TICE__=1 -DNDEBUG=1
        -fno-builtin -Wall -Wextra -Werror ${CLM_OPT:--Oz})
CORE_C=(clm_int.c clm_load.c clm_fwd.c clm_gen.c clm_ld2.c clm_ld2_load.c clm_mask.c clm_code.c)
CORE_S=()
[ -d "$CALCLM/core/ez80" ] && while IFS= read -r f; do CORE_S+=("$f"); done < <(find "$CALCLM/core/ez80" -name '*.s' | sort)

stale() {  # stale <target> <deps...>: target missing or older than any dep (or forced)
    local t="$1"; shift
    [ "$FORCE" = "-f" ] && return 0
    [ -f "$t" ] || return 0
    for d in "$@"; do [ "$d" -nt "$t" ] && return 0; done
    return 1
}

HDRS=("$CALCLM"/core/*.h "$TRACKER"/tracker.h "$TRACKER"/trk_tables.h)
# build id shared by core.o and CLMCODE: hash of everything that shapes the assembly
CODE_ID="0x$(cat "$CALCLM"/core/ez80/*.s "$CALCLM"/core/ez80/*.inc "$SHAPE" \
    "$CE/kernel/gen_tmv.py" "$HERE/gen_kernels.py" "$HERE/asm_place.py" "$HERE/build_flash.py" \
    | sha256sum | cut -c1-8)"
CFLAGS+=(-DCLM_CODE_ID="$CODE_ID")
objs=()
"$PYTHON" "$HERE/gen_kernels.py" "$SHAPE" -o "$OUT/obj/kernels_src.s"
"$PYTHON" "$HERE/asm_place.py" ram "$OUT/obj/kernels_src.s" -o "$OUT/obj/kernels.s"
if stale "$OUT/obj/kernels.o" "$OUT/obj/kernels.s"; then
    "$BU/z80-none-elf-as" -march=ez80+full "$OUT/obj/kernels.s" -o "$OUT/obj/kernels.o"
fi
objs+=("$OUT/obj/kernels.o")
cmp -s "$SHAPE" "$OUT/clm_shape.h" || cp "$SHAPE" "$OUT/clm_shape.h"   # what was built (bundle.py)
echo "$CODE_ID" > "$OUT/obj/code_id.new"
cmp -s "$OUT/obj/code_id.new" "$OUT/obj/code_id" || mv "$OUT/obj/code_id.new" "$OUT/obj/code_id"
for c in "${CORE_C[@]}"; do
    o="$OUT/obj/${c%.c}.o"
    if stale "$o" "$CALCLM/core/$c" "${HDRS[@]}" "$0" "$OUT/obj/code_id"; then
        "$BIN/ez80-clang" -S "${CFLAGS[@]}" "$CALCLM/core/$c" -o "$OUT/obj/${c%.c}.s"
        "$BU/z80-none-elf-as" -march=ez80+full "$OUT/obj/${c%.c}.s" -o "$o"
    fi
    objs+=("$o")
done
o="$OUT/obj/tracker.o"
if stale "$o" "$TRACKER/tracker.c" "${HDRS[@]}" "$0"; then
    "$BIN/ez80-clang" -S "${CFLAGS[@]}" "$TRACKER/tracker.c" -o "$OUT/obj/tracker.s"
    "$BU/z80-none-elf-as" -march=ez80+full "$OUT/obj/tracker.s" -o "$o"
fi
objs+=("$o")
for s in "${CORE_S[@]+"${CORE_S[@]}"}"; do
    o="$OUT/obj/$(basename "${s%.s}").o"
    if stale "$o" "$s" "$CALCLM"/core/ez80/*.inc "$HERE/asm_place.py"; then
        "$PYTHON" "$HERE/asm_place.py" ram "$s" -o "$OUT/obj/$(basename "${s%.s}")_ram.s"
        "$BU/z80-none-elf-as" -march=ez80+full -I "$CALCLM/core/ez80" \
            "$OUT/obj/$(basename "${s%.s}")_ram.s" -o "$o"
    fi
    objs+=("$o")
done
if stale "$OUT/core.o" "${objs[@]}"; then   # relinked only when an object changed
    "$BU/z80-none-elf-ld" -r "${objs[@]}" -o "$OUT/core.o"
fi

# test binary: core.o + driver, flat at ORG (same link script as the M2 kernels)
"$BIN/ez80-clang" -S "${CFLAGS[@]}" "$CALCLM/test/drv.c" -o "$OUT/obj/drv.s"
"$BU/z80-none-elf-as" -march=ez80+full "$OUT/obj/drv.s" -o "$OUT/obj/drv.o"
# keep every assembly entry point (ce/tests/test_calclm_asm.py calls them directly)
keep=($("$BU/z80-none-elf-nm" "$OUT/core.o" | awk '$2 == "T" && $3 ~ /^_clm_a_/ {print "-u", $3}'))
"$BU/z80-none-elf-ld" -static --no-warn-rwx-segments --no-warn-execstack --gc-sections -e _drv_cmd \
    "${keep[@]}" -T "$CE/kernel/link.ld" \
    --defsym "ORG=$ORG" "$OUT/obj/drv.o" "$OUT/core.o" \
    --start-group "$CEDEV/lib/crt/libcrt.a" --end-group -o "$OUT/test.elf"
"$BU/z80-none-elf-objcopy" -O binary "$OUT/test.elf" "$OUT/test.bin"
"$BU/z80-none-elf-nm" "$OUT/test.elf" > "$OUT/test.sym"
# the core must not call libc or the OS: only libcrt helpers (__*) may be undefined
undef="$("$BU/z80-none-elf-nm" -u "$OUT/core.o" | awk '{print $2}' | grep -v '^__' || true)"
if [ -n "$undef" ]; then
    echo "core.o references non-runtime symbols: $undef" >&2
    exit 1
fi
# the flash-resident copy of the same assembly (CLMCODE AppVar payload)
"$PYTHON" "$HERE/build_flash.py" "$OUT" "$CODE_ID" --shape "$SHAPE" \
    --appvar "$APPVAR_DIR/CLMCODE.8xv" >/dev/null
echo "core_o=$OUT/core.o"
echo "clmcode=$OUT/clmcode.bin"
echo "test_bin=$OUT/test.bin"
echo "test_sym=$OUT/test.sym"
# load address of the flat binary (the linker may align ORG up to the section alignment)
echo "test_org=0x$("$BU/z80-none-elf-objdump" -h "$OUT/test.elf" | awk '$2 == ".text" {print $4}')"
