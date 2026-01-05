// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <utility>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/metal/metal_backend.h"
#include "cppgrad/backend/metal/metal_allocator.h"

namespace cppgrad::backend::metal {

void register_device() {
    @try {
        id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
        if (mtlDevice) {
            id<MTLCommandQueue> mtlQueue = [mtlDevice newCommandQueue];

            // Bridge the objects to void* for the constructors. These are non-owning pointers.
            void* device_ptr = (__bridge void*)mtlDevice;
            void* queue_ptr = (__bridge void*)mtlQueue;

            DeviceManager::instance().register_device(std::make_unique<Device>(
                DeviceType::METAL,
                std::make_unique<MetalBackend>(device_ptr, queue_ptr),
                std::make_unique<MetalAllocator>(device_ptr)
            ));
        }
    } @catch (NSException *exception) {
        std::cerr << "Warning: Exception during Metal device registration: " << [[exception reason] UTF8String] << std::endl;
    }
}

} // namespace cppgrad::backend::metal
