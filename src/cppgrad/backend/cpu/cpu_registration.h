// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/cpu/cpu_backend.h"
#include "cppgrad/backend/cpu/cpu_allocator.h"

namespace cppgrad::backend::cpu {

void register_device() {
    DeviceManager::instance().register_device(std::make_unique<Device>(
        DeviceType::CPU,
        std::make_unique<CPUBackend>(),
        std::make_unique<CPUAllocator>()
    ));
}

} // namespace cppgrad::backend::cpu
