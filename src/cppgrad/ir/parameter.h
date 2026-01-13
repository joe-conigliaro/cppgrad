// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/device.h"
#include "cppgrad/backend/dtype.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/utils/ref.h"

namespace cppgrad {
namespace ir {

// Create a leaf parameter. Storage can be allocated now or deferred until first assign or eval.
inline utils::Ref<Tensor> parameter(const std::vector<size_t>& shape,
    cppgrad::backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
    cppgrad::backend::DType dtype = cppgrad::backend::DType::FLOAT32, bool allocate_now = true) {
    std::shared_ptr<cppgrad::backend::Buffer> storage = nullptr;
    if (allocate_now) {
        auto* device_obj = cppgrad::backend::DeviceManager::device(device_type);
        if (!device_obj) throw std::runtime_error("parameter: device not found");
        storage = device_obj->allocator()->allocate(cppgrad::utils::vector::numel(shape), dtype);
    } else {
        // deferred allocation
        storage = nullptr;
    }
    auto param = Tensor::make_leaf(storage, shape, device_type, dtype);
    // Leaf parameter must be leaf op; set requires_grad true
    param->set_requires_grad(true);

    if (!param->is_canonical_leaf()) throw std::runtime_error("parameter: non-canonical leaf");

    return param;
}

// Convert any tensor (graph or leaf) into a leaf Parameter.
// Ensures: becomes LeafOp, same shape/device/dtype, storage filled, requires_grad=true.
inline utils::Ref<Tensor> parameterize(const utils::Ref<Tensor>& t) {
    if (!t) throw std::runtime_error("parameterize: null tensor");

    if (t->is_canonical_leaf()) {
        // auto buf = t->schedule();
        auto buf = t->eval();
        if (!buf) throw std::runtime_error("parameterize: canonical leaf without buffer");
        t->set_requires_grad(true);
        return t;
    }

    // Realize once and attach to a new leaf
    // auto buf = t->schedule();
    auto buf = t->eval();
    if (!buf) throw std::runtime_error("parameterize: realization failed (null buffer)");

    auto param = Tensor::make_leaf(buf, t->shape(), t->device_type(), t->dtype());
    param->set_requires_grad(true);

    if (!param->is_canonical_leaf()) throw std::runtime_error("parameterize: non-canonical leaf");

    return param;
}

// Convenience alias
inline utils::Ref<ir::Tensor> trainable_from(const utils::Ref<ir::Tensor>& init) {
    return ir::parameterize(init);
}

} // namespace ir
} // namespace cppgrad
