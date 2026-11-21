#!/usr/bin/env bash
# Build the CalcLM ternary LUT matvec kernels as raw, position-specific binaries with the CE
# C toolchain (~/CEdev v15.0). Outputs go to ce/build/kernel/<name>.{s,o,elf,bin,sym}.
#
#   build.sh asm <G> <org> [vars] [variant] [c0] [pic]   generated assembly kernel for G groups
#   build.sh c <org>                               C reference (tmv_ref.c, ez80-clang -O2)
#
# org   load address (RAM 0xD1xxxx, or flash for the flash-resident variant)
# vars  RAM address for the kernel's variables (required when org is in flash)
# pic   1 = position-independent (long jumps via RAM trampolines the loader patches); the .bin
#       is then byte-identical for any org and can run from an archived AppVar wherever it lands
# The steps are the CEdev makefile's own: ez80-clang -S -> z80-none-elf-as -march=ez80+full ->
# z80-none-elf-ld (flat linker script link.ld, --defsym ORG=...) -> objcopy -O binary.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CEDEV="${CEDEV:-$HOME/CEdev}"
BIN="$CEDEV/bin"
BU="$CEDEV/binutils/bin"
OUT="${KERNEL_OUT:-$HERE/../build/kernel}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"

link() {  # link <name> <org> [libs...]
    local name="$1" org="$2"; shift 2
    "$BU/z80-none-elf-ld" -static --no-warn-rwx-segments -T "$HERE/link.ld" \
        --defsym "ORG=$org" "$OUT/$name.o" "$@" -o "$OUT/$name.elf"
    "$BU/z80-none-elf-objcopy" -O binary "$OUT/$name.elf" "$OUT/$name.bin"
    "$BU/z80-none-elf-nm" "$OUT/$name.elf" > "$OUT/$name.sym"
}

case "${1:-}" in
asm)
    G="$2"; ORG="$3"; VARS="${4:-}"; VARIANT="${5:-pop}"; C0="${6:-0}"; PIC="${7:-0}"
    name="tmv_${VARIANT}_g${G}_c${C0}_${ORG}${VARS:+_v$VARS}$([ "$PIC" = 1 ] && echo _pic || true)"
    "$PYTHON" "$HERE/gen_tmv.py" -G "$G" --c0 "$C0" --variant "$VARIANT" \
        ${VARS:+--vars "$VARS"} $([ "$PIC" = 1 ] && echo --pic || true) -o "$OUT/$name.s"
    "$BU/z80-none-elf-as" -march=ez80+full "$OUT/$name.s" -o "$OUT/$name.o"
    link "$name" "$ORG"
    echo "$OUT/$name.bin"
    ;;
c)
    ORG="$2"
    name="tmv_ref_${ORG}"
    "$BIN/ez80-clang" -S -O2 -nostdinc -isystem "$CEDEV/include" -fno-autolink -fno-addrsig \
        -ffunction-sections -fdata-sections -mllvm -z80-gas-style -D__TICE__=1 -Wall -Wextra \
        "$HERE/tmv_ref.c" -o "$OUT/$name.s"
    "$BU/z80-none-elf-as" -march=ez80+full "$OUT/$name.s" -o "$OUT/$name.o"
    link "$name" "$ORG" --start-group "$CEDEV/lib/crt/libcrt.a" --end-group
    echo "$OUT/$name.bin"
    ;;
*)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
