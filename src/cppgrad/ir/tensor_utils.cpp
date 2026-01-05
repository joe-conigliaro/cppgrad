// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/ops.h"

namespace cppgrad {
namespace ir {

utils::Ref<Tensor> uniform(const std::vector<size_t>& shape, float min, float max, backend::DeviceType device, backend::DType dtype) {
    return Tensor::make(RandomOp{RandomOpType::UNIFORM, UniformParams{min, max}}, {}, shape, device, dtype);
}

utils::Ref<Tensor> normal(const std::vector<size_t>& shape, float mean, float stddev, backend::DeviceType device, backend::DType dtype) {
    return Tensor::make(RandomOp{RandomOpType::NORMAL, NormalParams{mean, stddev}}, {}, shape, device, dtype);
}

utils::Ref<Tensor> zeros(const std::vector<size_t>& shape, backend::DeviceType device, backend::DType dtype) {
    return full(shape, 0.0, device, dtype);
}

utils::Ref<Tensor> ones(const std::vector<size_t>& shape, backend::DeviceType device, backend::DType dtype) {
    return full(shape, 1.0, device, dtype);
}

utils::Ref<Tensor> full(const std::vector<size_t>& shape, double fill_value, backend::DeviceType device, backend::DType dtype) {
    return Tensor::make(ConstantOp{ConstantOpType::FULL, fill_value}, {}, shape, device, dtype);
}

utils::Ref<Tensor> zeros_like(const utils::Ref<const Tensor>& t) {
    return zeros(t->shape(), t->device(), t->dtype());
}

utils::Ref<Tensor> ones_like(const utils::Ref<const Tensor>& t) {
    return ones(t->shape(), t->device(), t->dtype());
}

utils::Ref<Tensor> full_like(float fill_value, const utils::Ref<const Tensor>& t) {
    return full(t->shape(), fill_value, t->device(), t->dtype());
}

utils::Ref<Tensor> scalar(double value, backend::DeviceType device, backend::DType dtype) {
    auto t = Tensor::make(ConstantOp{ConstantOpType::SCALAR, value}, {}, std::vector<size_t>{}, device, dtype);
    return t;
}

utils::Ref<Tensor> scalar_like(float v, const utils::Ref<const Tensor>& ref) {
    return scalar(v, ref->device(), ref->dtype());
}

} // namespace ir
} // namespace cppgrad
