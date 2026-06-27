// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cstdint>
#include <cstring>

#include "cppgrad/common/dtype.h"

namespace cppgrad::common { 

// Host-side bfloat16: the top 16 bits of an IEEE-754 fp32 (1 sign, 8 exponent, 7 mantissa). Same
// bit layout as Metal's native `bfloat`, so device kernels and host code agree on stored bytes.
// Conversions go through fp32; arithmetic operators upconvert to float, so bfloat16 drops into the
// generic templated kernels (it is NOT a fast compute type -- the win is half the storage/bandwidth).
struct bfloat16 {
    uint16_t bits;

    bfloat16() = default;
    bfloat16(float f) : bits(from_float(f)) {}
    operator float() const { return to_float(bits); }

    // fp32 -> bf16, round-to-nearest-even (matches hardware bfloat conversion).
    static uint16_t from_float(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((x >> 16) | 0x0040u);  // NaN -> quiet NaN
        const uint32_t rounding_bias = 0x7fffu + ((x >> 16) & 1u);
        x += rounding_bias;
        return (uint16_t)(x >> 16);
    }
    // bf16 -> fp32 (exact: just zero-pad the low mantissa bits).
    static float to_float(uint16_t h) {
        uint32_t x = (uint32_t)h << 16;
        float f;
        std::memcpy(&f, &x, sizeof(f));
        return f;
    }
};

static_assert(sizeof(bfloat16) == 2, "bfloat16 must be 2 bytes");

// Hook bfloat16 into the runtime dtype machinery (dtype_of / dtype_v were UNKNOWN for it).
template<> constexpr DType dtype_of<bfloat16>() { return DType::BFLOAT16; }

} // namespace cppgrad::common
