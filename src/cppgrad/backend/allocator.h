// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include "cppgrad/backend/dtype.h"

namespace cppgrad {
namespace backend {

class Buffer;

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual std::shared_ptr<Buffer> allocate(size_t num_elements, DType dtype) = 0;
    virtual std::shared_ptr<Buffer> allocate(const void* src, size_t num_elements, DType dtype) = 0;
    virtual void deallocate(void* ptr) = 0;

    virtual void copy_device_to_host(void* host_dst, const Buffer& device_src) const = 0;
    virtual void copy_host_to_device(Buffer& device_dst, const void* host_src) const = 0;
    virtual void copy_device_to_device(Buffer& device_dst, const Buffer& device_src) const = 0;
};

} // namespace backend
} // namespace cppgrad
