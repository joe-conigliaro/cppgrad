// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/backend/device.h"
#include "cppgrad/backend/dtype.h"
#include <cstddef>
#include <memory>

namespace cppgrad {
namespace backend {

// Forward declaration.
class Allocator;

class Buffer {
public:
    Buffer(void* ptr, size_t size_bytes, DType dtype, DeviceType device_type, Allocator* allocator);
    ~Buffer();

    // Make Buffer move-only.
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void* data() const { return _ptr; }
    size_t size_bytes() const { return _size_bytes; }
    size_t numel() const { return _size_bytes == 0 ? 0 : _size_bytes / size(_dtype); }
    DType dtype() const { return _dtype; }
    DeviceType device_type() const { return _device_type; }

private:
    void*      _ptr;
    size_t     _size_bytes;
    DType      _dtype;
    DeviceType _device_type;
    Allocator* _allocator;
};

} // namespace backend
} // namespace cppgrad
