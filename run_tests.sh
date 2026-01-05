#!/bin/sh
set -eu

. ./scripts/build_common.sh

mkdir -p build

for test_src in tests/*.cpp; do
    [ -e "$test_src" ] || continue

echo "==============================================="
echo "Building and running test: $test_src"
echo "==============================================="

OUT_BIN="build/$(basename "${test_src%.cpp}")"
mkdir -p "$(dirname "$OUT_BIN")"

if [ "$ON_APPLE" = "true" ]; then
    # Build command for Apple (macOS).
    $CXX $CXXFLAGS $INCLUDE_FLAGS $LIB_CPP_SOURCES "$test_src" \
    $OBJCXXFLAGS $LIB_MM_SOURCES $FRAMEWORKS -o "$OUT_BIN"
    # Copy metallib next to the binary output dir.
    if [ -n "${METALLIB_PATH:-}" ] && [ -f "$METALLIB_PATH" ]; then
        cp "$METALLIB_PATH" "$(dirname "$OUT_BIN")/"
    fi
else
    # Build command for other platforms (Linux).
    $CXX $CXXFLAGS $INCLUDE_FLAGS $LIB_CPP_SOURCES "$test_src" -o "$OUT_BIN"
fi

echo "Running: $OUT_BIN"
"$OUT_BIN"
done
