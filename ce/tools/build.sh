#!/usr/bin/env bash
# Build ce/build/libcebench.so: the CEmu v2.0 core from the CEmu harness checkout's
# vendor/CEmu/core (CALCLM_CEMU_ROOT; CEMU_CORE names the core folder itself; compiled READ-ONLY
# from that checkout; objects go to ce/build/cemu-obj) plus ce/tools/shim.c,
# linked as one shared library for ce/tools/bench.py. Nothing is written outside ce/build.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CE="$(cd "$HERE/.." && pwd)"
CORE="${CEMU_CORE:-${CALCLM_CEMU_ROOT:?set CALCLM_CEMU_ROOT to the CEmu harness checkout, or CEMU_CORE}/vendor/CEmu/core}"
OBJ="$CE/build/cemu-obj"
OUT="$CE/build/libcebench.so"
CC="${CC:-gcc}"
CFLAGS="-O2 -fPIC -std=gnu11 -w"

[ -f "$CORE/emu.h" ] || { echo "missing CEmu core at $CORE" >&2; exit 1; }
mkdir -p "$OBJ"

objs=()
pids=()
while IFS= read -r src; do
    rel="${src#"$CORE"/}"
    obj="$OBJ/${rel//\//_}.o"
    objs+=("$obj")
    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
        "$CC" $CFLAGS -c "$src" -o "$obj" &
        pids+=($!)
    fi
done < <(find "$CORE" -name '*.c' -not -path '*/arm/*' | sort)
for p in "${pids[@]+"${pids[@]}"}"; do wait "$p"; done

"$CC" $CFLAGS -I"$CORE" -c "$HERE/shim.c" -o "$OBJ/shim.o"
"$CC" -shared -o "$OUT" "$OBJ/shim.o" "${objs[@]}" -lm -lpthread
echo "built $OUT"
