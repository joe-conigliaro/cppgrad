// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_operators.h"
#include "cppgrad/optim/optim.h"

namespace cppgrad {
namespace optim {

// AdamW: Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
// Update:
//   m_t = beta1 * m_{t-1} + (1-beta1) * g
//   v_t = beta2 * v_{t-1} + (1-beta2) * g^2
//   m̂_t = m_t / (1 - beta1^t)
//   v̂_t = v_t / (1 - beta2^t)
//   p = p - lr * ( m̂_t / (sqrt(v̂_t) + eps) ) - lr * weight_decay * p
//
// Note: Unlike classical L2, AdamW applies decay directly on parameters,
class AdamW : public Adam {
public:
    AdamW(std::vector<utils::Ref<ir::Tensor>> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f)
    : Adam(std::move(params), lr, beta1, beta2, eps, weight_decay) {}

protected:
    // Do not add wd*p into the gradient for AdamW
    utils::Ref<ir::Tensor>
    apply_weight_decay_to_grad(const utils::Ref<ir::Tensor>& g, const utils::Ref<ir::Tensor>& /*p*/, float /*wd*/) override {
        return g;
    }

    // Decoupled weight decay: p ← p − step_core − lr * wd * p
    utils::Ref<ir::Tensor>
    update_parameters(const utils::Ref<ir::Tensor>& p, const utils::Ref<ir::Tensor>& step_core, float wd) override {
        if (wd == 0.0f) return p - step_core;
        auto decay = p * (this->_lr * wd);
        return p - decay - step_core;
    }
};

} // namespace optim
} // namespace cppgrad
