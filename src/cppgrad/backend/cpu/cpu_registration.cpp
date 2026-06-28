// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iostream>
#include <memory>

#include "cppgrad/backend/cpu/cpu_allocator.h"
#include "cppgrad/backend/cpu/cpu_backend.h"
#include "cppgrad/backend/device_manager.h"

namespace cppgrad::backend::cpu {
namespace {

// Self-register the CPU device at static-initialization time.
struct AutoRegister {
    AutoRegister() {
        DeviceManager::instance().register_device(std::make_unique<Device>(
            DeviceType::CPU, std::make_unique<CPUBackend>(), std::make_unique<CPUAllocator>()));
    }
};
const AutoRegister _auto_register;

} // namespace
} // namespace cppgrad::backend::cpu
