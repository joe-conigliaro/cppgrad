// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <numeric>
#include "cppgrad/ir/tensor_operators.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"

namespace cppgrad {
namespace nn {
namespace functional {

enum class Reduction {
    MEAN,
    SUM,
    NONE
};

inline utils::Ref<ir::Tensor> reduce(const utils::Ref<ir::Tensor>& in, Reduction reduction, const std::vector<int>& axes = {}, bool keep_dims = false) {
    if (reduction == Reduction::NONE) return in;

    // If axes are not specified, reduce over all dimensions
    std::vector<int> reduce_axes = axes;
    if (reduce_axes.empty()) {
        reduce_axes.resize(in->shape().size());
        std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    }

    if (reduction == Reduction::MEAN) return ir::mean(in, reduce_axes, keep_dims);
    if (reduction == Reduction::SUM) return ir::sum(in, reduce_axes, keep_dims);

    throw std::runtime_error("Unhandled reduction type");
}

inline utils::Ref<ir::Tensor> mse_loss(const utils::Ref<ir::Tensor>& y_pred, const utils::Ref<ir::Tensor>& y_true, Reduction reduction = Reduction::MEAN) {
    auto diff = y_pred - y_true;
    auto squared_diff = diff * diff;
    return reduce(squared_diff, reduction);
}

// Standard hinge loss, y_true in {+1, -1}
inline utils::Ref<ir::Tensor> hinge_loss(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& y_true, float margin = 1.0f, Reduction reduction = Reduction::MEAN) {
    // loss = relu(margin - y_true * logits)
    auto loss_per_item = ir::relu(margin - (y_true * logits));
    return reduce(loss_per_item, reduction);
}

// Stable softplus: log(1 + exp(x))
inline utils::Ref<ir::Tensor> softplus(const utils::Ref<ir::Tensor>& x) {
    // This is a stable implementation: softplus(x) = max(0, x) + log(1 + exp(-|x|))
    auto relu_x = ir::relu(x);
    auto abs_x = ir::relu(x) + ir::relu(ir::neg(x));
    auto log_term = ir::log(1.0f + ir::exp(ir::neg(abs_x)));
    return relu_x + log_term;
}

// BCE with logits: targets in {0,1}. Stable formulation:
// loss(z, y) = softplus(z) - z*y
// This is mathematically equivalent to the more complex version but simpler to express.
inline utils::Ref<ir::Tensor> bce_with_logits(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets, Reduction reduction = Reduction::MEAN) {
    auto loss_per_item = softplus(logits) - (logits * targets);
    return reduce(loss_per_item, reduction);
}

// Logistic loss (margin targets): targets in {-1, +1}. Stable softplus form:
// loss(z, y) = softplus(-(y*z))
inline utils::Ref<ir::Tensor> logistic_loss_pm1(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets_pm1, Reduction reduction = Reduction::MEAN) {
    auto neg_yz = ir::neg(targets_pm1 * logits);
    auto loss_per_item = softplus(neg_yz);
    return reduce(loss_per_item, reduction);
}

inline utils::Ref<ir::Tensor> softmax(const utils::Ref<ir::Tensor>& logits, int axis = -1) {
    int nd = static_cast<int>(logits->shape().size());
    if (axis < 0) axis += nd;
    auto m = ir::max(logits, {axis}, true);
    auto z_shifted = logits - m;
    auto exp_z = ir::exp(z_shifted);
    auto denom = ir::sum(exp_z, {axis}, true);
    return exp_z / denom;
}

inline utils::Ref<ir::Tensor> log_softmax(const utils::Ref<ir::Tensor>& logits, int axis = -1) {
    int nd = static_cast<int>(logits->shape().size());
    if (axis < 0) axis += nd;
    auto m = ir::max(logits, {axis}, true);
    auto z_shifted = logits - m;
    auto logsumexp = ir::log(ir::sum(ir::exp(z_shifted), {axis}, true));
    return z_shifted - logsumexp;
}

inline utils::Ref<ir::Tensor> softmax_cross_entropy_with_logits(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets_onehot, Reduction reduction = Reduction::MEAN) {
    int nd = static_cast<int>(logits->shape().size());
    int axis = nd - 1;
    auto lsm = log_softmax(logits, axis);
    auto per_sample_loss = ir::neg(ir::sum(targets_onehot * lsm, {axis}, false));
    return reduce(per_sample_loss, reduction);
}

inline utils::Ref<ir::Tensor> relu(const utils::Ref<ir::Tensor>& input) {
    return ir::relu(input);
}

inline utils::Ref<ir::Tensor> tanh(const utils::Ref<ir::Tensor>& input) {
    return ir::tanh(input);
}

} // namespace functional
} // namespace nn
} // namespace cppgrad
