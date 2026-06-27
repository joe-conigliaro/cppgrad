// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cppgrad::common { 

enum class DType {
    INT8,
    INT32,
    INT64,
    UINT8,
    UINT32,
    FLOAT16,
    BFLOAT16,
    FLOAT32,
    FLOAT64,
    BOOL8,
    UNKNOWN
};

inline const char* to_string(DType dt) {
    switch (dt) {
        case DType::INT8:     return "int8";
        case DType::INT32:    return "int32";
        case DType::INT64:    return "int64";
        case DType::UINT8:    return "uint8";
        case DType::UINT32:   return "uint32";
        case DType::FLOAT16:  return "float16";
        case DType::BFLOAT16: return "bfloat16";
        case DType::FLOAT32:  return "float32";
        case DType::FLOAT64:  return "float64";
        case DType::BOOL8:    return "bool8";
        default:              return "unknown";
    }
}

// Size in bytes.
inline constexpr size_t size(DType dtype) {
    switch (dtype) {
        case DType::INT8:     return sizeof(int8_t);
        case DType::INT32:    return sizeof(int32_t);
        case DType::INT64:    return sizeof(int64_t);
        case DType::UINT8:    return sizeof(uint8_t);
        case DType::UINT32:   return sizeof(uint32_t);
        case DType::FLOAT16:  return 2;
        case DType::BFLOAT16: return 2;
        case DType::FLOAT32:  return sizeof(float);
        case DType::FLOAT64:  return sizeof(double);
        case DType::BOOL8:    return sizeof(uint8_t);
        default:              return 0;
    }
}

// Type-promotion rank: float kinds outrank ints; wider/higher-precision wins. Used to pick the
// result dtype of a mixed-dtype op (e.g. fp32 activations x bf16 weights -> fp32).
inline constexpr int promote_rank(DType d) {
    switch (d) {
        case DType::FLOAT64:  return 6;
        case DType::FLOAT32:  return 5;
        case DType::FLOAT16:  return 4;
        case DType::BFLOAT16: return 4;  // same width as fp16; fp32 still wins over both
        case DType::INT64:    return 3;
        case DType::UINT32:
        case DType::INT32:    return 2;
        case DType::UINT8:
        case DType::INT8:     return 1;
        default:              return 0;
    }
}

inline DType promote(DType dt1, DType dt2) {
    if (dt1 == dt2) return dt1;
    return promote_rank(dt1) >= promote_rank(dt2) ? dt1 : dt2;
}

// Compile-time mapping.
template<typename T>
constexpr DType dtype_of() {
    // using U = std::remove_cvref_t<T>; // C++20
    using U = std::remove_cv_t<std::remove_reference_t<T>>; // C++17
    if      constexpr (std::is_same<U, float>::value)    return DType::FLOAT32;
    else if constexpr (std::is_same<U, double>::value)   return DType::FLOAT64;
    else if constexpr (std::is_same<U, int8_t>::value)   return DType::INT8;
    else if constexpr (std::is_same<U, int32_t>::value)  return DType::INT32;
    else if constexpr (std::is_same<U, int64_t>::value)  return DType::INT64;
    else if constexpr (std::is_same<U, uint8_t>::value)  return DType::UINT8;
    else if constexpr (std::is_same<U, uint32_t>::value) return DType::UINT32;
    else if constexpr (std::is_same<U, bool>::value)     return DType::BOOL8;
    else return DType::UNKNOWN;
}

// Convenience alias.
template<typename T>
constexpr DType dtype_v = dtype_of<T>();

} // namespace cppgrad::common
