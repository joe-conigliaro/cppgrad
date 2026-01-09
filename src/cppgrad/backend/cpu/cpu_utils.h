// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <stdexcept>
#include "cppgrad/backend/buffer.h"

namespace cppgrad {
namespace backend {

inline void* as_host_ptr(const Buffer& b) {
    if (b.device_type() != DeviceType::CPU) {
        throw std::runtime_error("as_host_ptr: not a CPU buffer");
    }
    return b.data();
}

} // namespace backend
} // namespace cppgrad
