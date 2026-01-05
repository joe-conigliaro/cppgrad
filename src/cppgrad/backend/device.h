// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>

namespace cppgrad {
namespace backend {

// Forward declarations.
class Backend;
class Allocator;

enum class DeviceType {
    CPU,
    CUDA,
    METAL
};

inline const char* to_string(DeviceType dt) {
    switch (dt) {
        case DeviceType::CPU:   return "CPU";
        case DeviceType::CUDA:  return "CUDA";
        case DeviceType::METAL: return "METAL";
        default:                return "UNKNOWN";
    }
}

class Device {
public:
    Device(DeviceType type, std::unique_ptr<Backend> backend, std::unique_ptr<Allocator> allocator);
    ~Device();

    DeviceType type() const { return _type; }
    Backend* backend() const { return _backend.get(); }
    Allocator* allocator() const { return _allocator.get(); }

private:
    DeviceType _type;
    std::unique_ptr<Backend> _backend;
    std::unique_ptr<Allocator> _allocator;
};

} // namespace backend
} // namespace cppgrad
