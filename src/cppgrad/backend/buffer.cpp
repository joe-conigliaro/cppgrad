// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/allocator.h"

namespace cppgrad {
namespace backend {

Buffer::Buffer(void* data, size_t size_bytes, DType dtype, DeviceType device, Allocator* allocator)
    : _ptr(data), _size_bytes(size_bytes), _dtype(dtype), _device(device), _allocator(allocator) {}

Buffer::~Buffer() {
    if (_ptr && _allocator) {
        _allocator->deallocate(_ptr);
    }
}

} // namespace backend
} // namespace cppgrad
