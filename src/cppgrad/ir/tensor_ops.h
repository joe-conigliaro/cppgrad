// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/ir/tensor.h"

namespace cppgrad {
namespace ir {

utils::Ref<Tensor> assign(const utils::Ref<const Tensor>& other);

// Unary Ops
utils::Ref<Tensor> relu(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> exp(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> log(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> neg(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> tanh(const utils::Ref<const Tensor>& t);
utils::Ref<Tensor> sqrt(const utils::Ref<const Tensor>& t);

// Binary Ops
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> cmp_eq(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> cmp_gt(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> min(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);
utils::Ref<Tensor> max(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);

// Reduction Ops
utils::Ref<Tensor> sum(const utils::Ref<const Tensor>& t, const std::vector<int>& axes = {}, bool keep_dims = false);
utils::Ref<Tensor> max(const utils::Ref<const Tensor>& t, const std::vector<int>& axes = {}, bool keep_dims = false);

// MatMul Op
utils::Ref<Tensor> matmul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b);

// Materialization / Layout Ops
utils::Ref<const Tensor> contiguous(const utils::Ref<const Tensor>& t);

// Movement Ops
utils::Ref<Tensor> reshape_view(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape);
utils::Ref<Tensor> permute(const utils::Ref<const Tensor>& t, const std::vector<size_t>& axes);
utils::Ref<Tensor> transpose(const utils::Ref<const Tensor>& t, size_t dim0, size_t dim1);
utils::Ref<Tensor> broadcast(const utils::Ref<const Tensor>& t, const std::vector<size_t>& shape);
utils::Ref<Tensor> slice(const utils::Ref<const Tensor>& t, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step = {});

// Composite Ops
utils::Ref<Tensor> reshape(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape);
utils::Ref<Tensor> mean(const utils::Ref<const Tensor>& t, const std::vector<int>& axes = {}, bool keep_dims = false);

// Scalar Ops (Tensor, float)
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, float val);
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, float val);
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, float val);
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, float val);
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, float val);

// Scalar Ops (float, Tensor)
utils::Ref<Tensor> add(float val, const utils::Ref<const Tensor>& a);
utils::Ref<Tensor> sub(float val, const utils::Ref<const Tensor>& a);
utils::Ref<Tensor> mul(float val, const utils::Ref<const Tensor>& a);
utils::Ref<Tensor> div(float val, const utils::Ref<const Tensor>& a);
utils::Ref<Tensor> pow(float val, const utils::Ref<const Tensor>& a);

} // namespace ir
} // namespace cppgrad
