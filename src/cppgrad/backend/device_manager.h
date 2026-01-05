// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <map>
#include <mutex>
#include <memory>
#include "cppgrad/backend/device.h"
#include "cppgrad/executor/executor.h"
#include "cppgrad/backend/allocator.h"

namespace cppgrad {
namespace backend {

class DeviceManager {
public:
    static DeviceManager& instance();
    static Device* device(enum DeviceType type);
    static DeviceType default_device();
    static void set_default_device(enum DeviceType type);

    // Deleted copy constructor & assignment operator (enforce singleton pattern).
    DeviceManager(const DeviceManager&) = delete;
    void operator=(const DeviceManager&) = delete;

    void register_device(std::unique_ptr<Device> device);
    void init();

private:
    DeviceManager() = default;
    std::map<DeviceType, std::unique_ptr<Device>> _devices;
    enum DeviceType _default_device = DeviceType::CPU;
    static std::mutex _mutex;
};

} // namespace backend
} // namespace cppgrad
