// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <Metal/Metal.h>
#include <string>
#include <unordered_map>

namespace cppgrad {
namespace backend {
namespace metal {

class MetalKernelCache {
public:
    // Constructor takes the device to create the library and pipelines.
    MetalKernelCache(id<MTLDevice> device);
    id<MTLComputePipelineState> get(const std::string& name);

private:
    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> _cache;
};

} // namespace metal
} // namespace backend
} // namespace cppgrad
