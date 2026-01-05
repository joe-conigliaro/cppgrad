// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Foundation/Foundation.h>
#include <string>
#include <stdexcept>
#include "cppgrad/backend/metal/metal_kernel_cache.h"

namespace cppgrad {
namespace backend {
namespace metal {

MetalKernelCache::MetalKernelCache(id<MTLDevice> device) : _device(device) {
    NSError* error = nil;
    _library = [_device newDefaultLibrary];
    if (!_library) {
        _library = [_device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&error];
    }
    if (!_library) {
        std::string err_str = "Failed to create default Metal library. Ensure your .metal file is compiled and linked.";
        if (error) {
            err_str += " Details: " + std::string([[error localizedDescription] UTF8String]);
        }
        throw std::runtime_error(err_str);
    }
}

id<MTLComputePipelineState> MetalKernelCache::get(const std::string& name) {
    auto it = _cache.find(name);
    if (it != _cache.end()) {
        return it->second;
    }

    NSError* error = nil;
    id<MTLFunction> func = [_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!func) {
        throw std::runtime_error("Failed to find Metal function: " + name);
    }

    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        std::string err = "Failed to create PSO for " + name + ": ";
        if (error) { err += [[error localizedDescription] UTF8String]; }
        throw std::runtime_error(err);
    }

    _cache[name] = pso;
    return pso;
}

} // namespace metal
} // namespace backend
} // namespace cppgrad
