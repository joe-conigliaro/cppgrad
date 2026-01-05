// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/backend/device.h"
#include "cppgrad/backend/dtype.h"
#include "cppgrad/utils/vector.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/ops.h"

namespace cppgrad {
namespace ir {

template <typename T>
inline utils::Ref<Tensor> from_vector(const std::vector<T>& data, const std::vector<size_t>& shape, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device()) {
    constexpr cppgrad::backend::DType dtype = cppgrad::backend::dtype_v<T>;
    static_assert(dtype != cppgrad::backend::DType::UNKNOWN, "from_vector: unsupported vector element type.");
    size_t numel = utils::vector::numel(shape);
    if (data.size() != numel && !(numel == 0 && data.empty())) {
        throw std::runtime_error("from_vector: Data size does not match shape.");
    }
    auto device_obj = cppgrad::backend::DeviceManager::device(device);
    if (!device_obj) {
        throw std::runtime_error("from_vector: device not found.");
    }
    auto buffer = device_obj->allocator()->allocate(data.data(), data.size(), dtype);
    auto t = Tensor::make_leaf(buffer, shape, device, dtype);
    return t;
}

utils::Ref<Tensor> full(const std::vector<size_t>& shape, double fill_value, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);
utils::Ref<Tensor> zeros(const std::vector<size_t>& shape, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);
utils::Ref<Tensor> ones(const std::vector<size_t>& shape, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);

// Like Utilities
utils::Ref<Tensor> zeros_like(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> ones_like(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> full_like(float fill_value, const utils::Ref<const Tensor>& t);

// Random Tensor Creation Utilities
utils::Ref<Tensor> uniform(const std::vector<size_t>& shape, float min = -1.0f, float max = 1.0f, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);
utils::Ref<Tensor> normal(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);

// Rank-0 scalar constant with explicit device/dtype
utils::Ref<Tensor> scalar(double value, cppgrad::backend::DeviceType device = cppgrad::backend::DeviceManager::default_device(), cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32);
// Scalar with device/dtype taken from a reference tensor
utils::Ref<Tensor> scalar_like(float v, const utils::Ref<const Tensor>& ref);

} // namespace ir
} // namespace cppgrad
