// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/allocator.h"

namespace cppgrad {
namespace backend {

Buffer::Buffer(void* data, size_t size_bytes, DType dtype, DeviceType device_type, Allocator* allocator)
    : _ptr(data), _size_bytes(size_bytes), _dtype(dtype), _device_type(device_type), _allocator(allocator) {}

Buffer::~Buffer() {
    if (_ptr && _allocator) {
        _allocator->deallocate(_ptr);
    }
}

} // namespace backend
} // namespace cppgrad
