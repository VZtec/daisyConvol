#!/usr/bin/env bash
# Every test: the host suites (CMake + CTest, any C++14 compiler), then the
# firmware examples when the Arm toolchain is on PATH.
#
#   test/all.sh
#   LIBDAISY_DIR=/path/to/libDaisy test/all.sh
#
# LIBDAISY_DIR (default: a libDaisy next to this repository) adds the suites
# on the target's CMSIS-DSP FFT built for the host, and, if that libDaisy is
# built (`make` in it), the firmware builds. Without it only the suites on
# the reference FFT run. CMAKE_ARGS passes extra options to CMake, e.g.
# CMAKE_ARGS=-DDAISYCONVOL_SANITIZE=ON.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BUILD=${BUILD:-$ROOT/build/test}

# Absolute paths, in the form native tools understand (Git Bash: C:/...).
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }

LIBDAISY_DIR=${LIBDAISY_DIR:-$ROOT/../libDaisy}
if [ -d "$LIBDAISY_DIR" ]; then
    LIBDAISY_DIR=$(abspath "$LIBDAISY_DIR")
    echo "all.sh: libDaisy at $LIBDAISY_DIR"
else
    echo "all.sh: no libDaisy at $LIBDAISY_DIR: reference FFT suites only"
    LIBDAISY_DIR=
fi

# --- Host suites --------------------------------------------------------------
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DLIBDAISY_DIR="$LIBDAISY_DIR" ${CMAKE_ARGS:-}
cmake --build "$BUILD" --config Release -j 4
ctest --test-dir "$BUILD" -C Release --output-on-failure

# --- Firmware -----------------------------------------------------------------
if ! command -v arm-none-eabi-gcc > /dev/null 2>&1; then
    echo "all.sh: firmware builds skipped (no arm-none-eabi-gcc on PATH)"
    exit 0
fi
if [ -z "$LIBDAISY_DIR" ] || [ ! -f "$LIBDAISY_DIR/build/libdaisy.a" ]; then
    echo "all.sh: firmware builds skipped (libDaisy not found or not built)"
    exit 0
fi
for ex in seed cabsim hall_reverb church_reverb irswitch convolver; do
    echo "== examples/$ex"
    make -C "$ROOT/examples/$ex" LIBDAISY_DIR="$LIBDAISY_DIR" -j4 -s
done
arm-none-eabi-size "$ROOT/examples/seed/build/daisyconvol_seed.elf"
echo "all.sh: OK"
