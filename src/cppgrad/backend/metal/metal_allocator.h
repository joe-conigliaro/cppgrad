// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include "cppgrad/backend/allocator.h"

namespace cppgrad::backend::metal {

// Forward-declare metal execution context
class MetalExecutionContext;
// Forward-declare the private implementation struct
struct MetalAllocatorImpl;

class MetalAllocator : public Allocator {
public:
    explicit MetalAllocator(void* native_device, MetalExecutionContext* exec_ctx = nullptr);
    ~MetalAllocator() override;

    std::shared_ptr<Buffer> allocate(size_t num_elements, common::DType dtype) override;
    std::shared_ptr<Buffer> allocate(const void* src, size_t num_elements, common::DType dtype) override;
    void deallocate(void* ptr) override;

    void copy_device_to_host(void* host_dst, const Buffer& device_src) const override;
    void copy_host_to_device(Buffer& device_dst, const void* host_src) const override;
    void copy_device_to_device(Buffer& device_dst, const Buffer& device_src) const override;

private:
    std::unique_ptr<MetalAllocatorImpl> _impl;
};

} // namespace cppgrad::backend::metal
