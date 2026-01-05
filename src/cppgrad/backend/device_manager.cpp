// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/cpu/cpu_backend.h"
#include "cppgrad/backend/cpu/cpu_allocator.h"
#include "cppgrad/backend/cpu/cpu_registration.h"
#include "cppgrad/executor/interpreter/interpreter_executor.h"

// Forward declare mm function.
#ifdef CPPGRAD_ON_APPLE
namespace cppgrad::backend::metal {
void register_device();
}
#endif

namespace cppgrad {
namespace backend {

std::mutex DeviceManager::_mutex;

DeviceManager& DeviceManager::instance() {
    static DeviceManager inst;
    return inst;
}

Device* DeviceManager::device(DeviceType type) {
    auto& inst = instance();
    auto it = inst._devices.find(type);
    if (it == inst._devices.end()) {
        return nullptr;
    }
    return it->second.get();
}

void DeviceManager::set_default_device(DeviceType type) {
    auto& inst = instance();
    std::lock_guard<std::mutex> lock(_mutex);
    if (inst._devices.find(type) == inst._devices.end()) {
        throw std::runtime_error("Cannot set default device to an unregistered device type: " + std::string(to_string(type)));
    }
    inst._default_device = type;
    std::cout << "Default device set to: " << to_string(type) << std::endl;
}

DeviceType DeviceManager::default_device() {
    auto& inst = instance();
    std::lock_guard<std::mutex> lock(_mutex);
    return inst._default_device;
}

void DeviceManager::register_device(std::unique_ptr<Device> device) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto type = device->type();
    if (_devices.find(type) == _devices.end()) {
        std::cout << to_string(type) << " device registered." << std::endl;
        _devices[type] = std::move(device);
    } else {
        std::cout << to_string(type) << " device already registered." << std::endl;
    }
}

void DeviceManager::init() {
    cpu::register_device();
    #ifdef CPPGRAD_ON_APPLE
        metal::register_device();
    #endif
    set_default_device(DeviceType::CPU);
    // if (_devices.count(DeviceType::METAL)) {
    //     set_default_device(DeviceType::METAL);
    // } else {
    //     set_default_device(DeviceType::CPU);
    // }
}

} // namespace backend
} // namespace cppgrad
