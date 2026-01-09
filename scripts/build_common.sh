#!/bin/sh

set -eu

# Config
: "${CXX:=clang++}"
: "${DEBUG:=false}"
: "${CPPGRAD_DEBUG:=false}"

: "${CXXFLAGS:=-std=c++17}"
: "${OBJCXXFLAGS:=-fobjc-arc}"

: "${INCLUDE_FLAGS:=-I src -I .}"
: "${FRAMEWORKS:=-framework Metal -framework Foundation -framework MetalPerformanceShaders}"

: "${SANITIZE_ADDRESS:=false}"
: "${SANITIZE_THREAD:=false}"
: "${FFP_CONTRACT_OFF:=false}"
: "${FAST_MATH:=true}"

COMMON_FLAGS=""
COMMON_DEFS=""

# Plaform Detection
ON_APPLE=false
case "$(uname -s)" in
  Darwin) ON_APPLE=true ;;
esac

# Defines
if [ "$DEBUG" = "true" ] || [ "$CPPGRAD_DEBUG" = "true" ]; then
    COMMON_DEFS="$COMMON_DEFS -DCPPGRAD_DEBUG=1"
fi

# Flags
if [ "$DEBUG" = "true" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -g -O0"
else
    COMMON_FLAGS="$COMMON_FLAGS -O3"
fi

if [ "$SANITIZE_ADDRESS" = "true" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -fsanitize=address -fno-omit-frame-pointer"
fi
if [ "$SANITIZE_THREAD" = "true" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -fsanitize=thread"
fi
if [ "$FFP_CONTRACT_OFF" = "true" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -ffp-contract=off"
fi
if [ "$FAST_MATH" = "false" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -fno-fast-math"
fi

# Final composed flags
CXXFLAGS="$CXXFLAGS $COMMON_FLAGS $COMMON_DEFS"
OBJCXXFLAGS="$OBJCXXFLAGS $COMMON_FLAGS $COMMON_DEFS"

# C++ Sources
LIB_CPP_SOURCES=$(find src -name '*.cpp' ! -path 'src/cppgrad/backend/metal/*')

LIB_MM_SOURCES=""
LIB_METAL_SOURCES=""
METALLIB_PATH=""

# Metal Compilation on Apple
if [ "$ON_APPLE" = "true" ]; then
    echo "Apple platform detected. Preparing Metal backend..."
    CXXFLAGS="$CXXFLAGS -DCPPGRAD_WITH_METAL=1 -DCPPGRAD_ON_APPLE=1"
    if command -v xcrun >/dev/null 2>&1; then
        LIB_MM_SOURCES=$(find src/cppgrad/backend/metal -name '*.mm' 2>/dev/null || true)
        LIB_METAL_SOURCES=$(find src/cppgrad/backend/metal -name '*.metal' 2>/dev/null || true)

    METAL_BUILD_DIR="build/metal"
    METALLIB_PATH="$METAL_BUILD_DIR/default.metallib"
    mkdir -p "$METAL_BUILD_DIR"

    if [ -n "$LIB_METAL_SOURCES" ]; then
        METAL_AIRS=""

      for msrc in $LIB_METAL_SOURCES; do
          base=$(basename "$msrc" .metal)
          air="$METAL_BUILD_DIR/$base.air"

        echo "Compiling Metal: $msrc -> $air"
        xcrun -sdk macosx metal -std=macos-metal2.3 -O3 -c "$msrc" -o "$air"

        METAL_AIRS="$METAL_AIRS $air"
      done

      echo "Linking Metal AIRs into metallib: $METALLIB_PATH"
      xcrun -sdk macosx metallib $METAL_AIRS -o "$METALLIB_PATH"
    fi
    else
        echo "Warning: xcrun not found; Metal backend will be disabled."
        CXXFLAGS="$CXXFLAGS -DCPPGRAD_WITH_METAL=0"
    fi
else
    echo "Non-Apple platform detected. Metal backend will be disabled."
    CXXFLAGS="$CXXFLAGS -DCPPGRAD_WITH_METAL=0"
fi
