// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cstring>
#include "cppgrad/backend/cpu/cpu_allocator.h"
#include "cppgrad/backend/buffer.h"

namespace cppgrad {
namespace backend {
namespace cpu {

std::shared_ptr<Buffer> CPUAllocator::allocate(size_t num_elements, DType dtype) {
    size_t bytes = num_elements * size(dtype);
    if (bytes == 0) {
        return std::make_shared<Buffer>(nullptr, 0, dtype, DeviceType::CPU, this);
    }
    void* ptr = new char[bytes];
    return std::make_shared<Buffer>(ptr, bytes, dtype, DeviceType::CPU, this);
}

std::shared_ptr<Buffer> CPUAllocator::allocate(const void* src, size_t num_elements, DType dtype) {
    auto buffer = allocate(num_elements, dtype);
    if (src && buffer->data()) {
        std::memcpy(buffer->data(), src, buffer->size_bytes());
    }
    return buffer;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) {
        delete[] static_cast<char*>(ptr);
    }
}

// Device -> Host
void CPUAllocator::copy_device_to_host(void* host_dst, const Buffer& device_src) const {
    // device_src must be a CPU buffer
    if (device_src.size_bytes() == 0) return;
    if (device_src.device() != DeviceType::CPU) {
        throw std::runtime_error("CPUAllocator::copy_device_to_host: src is not a CPU buffer");
    }
    if (!host_dst || !device_src.data()) return;
    std::memcpy(host_dst, device_src.data(), device_src.size_bytes());
}

// Device -> Host
void CPUAllocator::copy_host_to_device(Buffer& device_dst, const void* host_src) const {
    // device_dst must be a CPU buffer
    if (device_dst.size_bytes() == 0) return;
    if (device_dst.device() != DeviceType::CPU) {
        throw std::runtime_error("CPUAllocator::copy_host_to_device: dst is not a CPU buffer");
    }
    if (!device_dst.data() || !host_src) return;
    std::memcpy(device_dst.data(), host_src, device_dst.size_bytes());
}

// Device -> Device (same CPU device)
void CPUAllocator::copy_device_to_device(Buffer& device_dst, const Buffer& device_src) const {
    if (device_src.size_bytes() == 0) return;
    if (device_dst.device() != DeviceType::CPU || device_src.device() != DeviceType::CPU) {
        throw std::runtime_error("CPUAllocator::copy_device_to_device: src and dst must both be CPU buffers");
    }
    if (!device_dst.data() || !device_src.data()) return;
    std::memcpy(device_dst.data(), device_src.data(), device_src.size_bytes());
}

} // namespace cpu
} // namespace backend
} // namespace cppgrad
