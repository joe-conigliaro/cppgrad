// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/device.h"

#include "cppgrad/backend/allocator.h"
#include "cppgrad/backend/backend.h"

namespace cppgrad::backend {

Device::Device(DeviceType type, std::unique_ptr<Backend> backend, std::unique_ptr<Allocator> allocator)
    : _type(type), _backend(std::move(backend)), _allocator(std::move(allocator)) {}

Device::~Device() = default;

} // namespace cppgrad::backend
