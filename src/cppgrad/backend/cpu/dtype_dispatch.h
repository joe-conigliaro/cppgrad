// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <stdexcept>

#include "cppgrad/common/dtype.h"

namespace cppgrad::backend::cpu {

// common::DType dispatcher
// Calls f.template operator()<T>() based on runtime dtype.
// Example:
//   dispatch_dtype(out.dtype(), [&] <typename T> () {
//     unary_view_kernel<T>(...);
//   });
template <typename F> inline void dispatch_dtype(common::DType dt, const F &f) {
    switch (dt) {
    // TODO: add f16 type
    // case common::DType::FLOAT16: f.template operator()<f16>();      break;
    case common::DType::FLOAT32:
        f.template operator()<float>();
        break;
    case common::DType::FLOAT64:
        f.template operator()<double>();
        break;
    case common::DType::INT8:
        f.template operator()<int8_t>();
        break;
    case common::DType::INT32:
        f.template operator()<int32_t>();
        break;
    case common::DType::INT64:
        f.template operator()<int64_t>();
        break;
    case common::DType::UINT8:
        f.template operator()<uint8_t>();
        break;
    case common::DType::UINT32:
        f.template operator()<uint32_t>();
        break;
    case common::DType::BOOL8:
        f.template operator()<uint8_t>();
        break;
    default:
        throw std::runtime_error("dispatch_dtype: unsupported dtype");
    }
}

// Type tag
template <typename T> struct type_tag {
    using type = T;
};
// Wrapper that lets dispatch_dtype call f.template operator()<T>()
// and you can provide either:
//  - a templated body (C++20), or
//  - a tag body (C++17).
template <typename Body> struct TemplatedBody {
    Body body;
    // Prefer a templated body if present (C++20 path)
    template <typename T, typename B = Body, typename = decltype(std::declval<const B &>().template operator()<T>())>
    void operator()() const {
        body.template operator()<T>();
    }
    // Fallback to tag body (C++17 path)
    template <typename T, typename B = Body, typename = void,
              typename = decltype(std::declval<const B &>()(type_tag<T>{}))>
    void operator()() const {
        body(type_tag<T>{});
    }
};
template <typename Body> TemplatedBody<std::decay_t<Body>> make_templated(Body &&body) {
    return {std::forward<Body>(body)};
}

} // namespace cppgrad::backend::cpu
