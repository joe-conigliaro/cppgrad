// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/metal/metal_allocator.h"
#include "cppgrad/backend/metal/metal_backend.h"
#include "cppgrad/backend/metal/metal_execution_context.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <exception>
#include <iostream>
#include <utility>

namespace cppgrad::backend::metal {
namespace {

void register_device() {
    @try {
        id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
        if (mtlDevice) {
            id<MTLCommandQueue> mtlQueue = [mtlDevice newCommandQueue];

            // Bridge the objects to void* for the constructors. These are non-owning pointers.
            void *device_ptr = (__bridge void *)mtlDevice;
            void *queue_ptr = (__bridge void *)mtlQueue;

            // Shared execution context: batches compute work and commits it at
            // GraphScope boundaries / on readback. Lives for the application
            // lifetime (intentionally leaked; the backend + allocator hold non-owning
            // pointers to it).
            auto exec_ctx = std::make_unique<MetalExecutionContext>(device_ptr, queue_ptr);

            DeviceManager::instance().register_device(std::make_unique<Device>(
                DeviceType::METAL, std::make_unique<MetalBackend>(device_ptr, queue_ptr, exec_ctx.get()),
                std::make_unique<MetalAllocator>(device_ptr, exec_ctx.get())));

            exec_ctx.release();
        }
    } @catch (NSException *exception) {
        std::cerr << "Warning: Exception during Metal device registration: " << [[exception reason] UTF8String]
                  << std::endl;
    }
}

// Self-register the Metal device at static-initialization time. This runs before
// main(), so the body must never throw out (it would be uncatchable there): any
// C++ exception (e.g. metallib load failure) is swallowed so the program simply
// falls back to CPU instead of terminating.
struct AutoRegister {
    AutoRegister() {
        try {
            register_device();
        } catch (const std::exception &e) {
            std::cerr << "Warning: Metal device registration failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Warning: Metal device registration failed (unknown error)." << std::endl;
        }
    }
};
const AutoRegister _auto_register;

} // namespace
} // namespace cppgrad::backend::metal
