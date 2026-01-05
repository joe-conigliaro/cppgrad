// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/device_manager.h"

namespace cppgrad {
namespace backend {

inline void copy(Buffer& dst, const Buffer& src) {
    if (src.size_bytes() == 0 || src.data() == dst.data()) return;
    if (src.dtype() != dst.dtype() || src.size_bytes() != dst.size_bytes())
        throw std::runtime_error("backend::copy: dtype/size mismatch");

    auto* src_dev = DeviceManager::device(src.device());
    auto* dst_dev = DeviceManager::device(dst.device());
    if (!src_dev || !dst_dev) throw std::runtime_error("backend::copy: device not found");

    // Same-device copy.
    if (src.device() == dst.device()) {
        src_dev->allocator()->copy_device_to_device(dst, src);
        return;
    }

    // Cross-device via host staging using allocators.
    std::vector<uint8_t> host(dst.size_bytes());
    src_dev->allocator()->copy_device_to_host(host.data(), src);
    dst_dev->allocator()->copy_host_to_device(dst, host.data());
}

} // namespace backend
} // namespace cppgrad
