// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <numeric>
#include <stdexcept>
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/ops.h"
#include "cppgrad/utils/shape.h"
#include "cppgrad/utils/vector.h"

namespace cppgrad {
namespace ir {

// Helpers

static utils::Ref<Tensor> unary(UnaryOpType op, const utils::Ref<const Tensor>& t) {
  auto out = Tensor::make(UnaryOp{op}, { t }, t->shape(), t->device(), t->dtype());
  out->set_access_meta(AccessMeta::contiguous_from(out->shape()));
  return out;
}

static utils::Ref<Tensor> binary(BinaryOpType op, const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) {
  auto out_shape = utils::shape::get_broadcast_shape(a->shape(), b->shape());
  auto out = Tensor::make(BinaryOp{op}, { a, b }, out_shape, a->device(), a->dtype());
  out->set_access_meta(AccessMeta::contiguous_from(out_shape));
  return out;
}

// Public API

utils::Ref<Tensor> assign(const utils::Ref<const Tensor>& dst, const utils::Ref<const Tensor>& src) {
    if (!dst->is_canonical_leaf()) throw std::runtime_error("assign: dst tensor is not a canonical leaf");
    if (dst->shape() != src->shape()) throw std::runtime_error("assign: shape mismatch");
    if (dst->dtype() != src->dtype()) throw std::runtime_error("assign: dtype mismatch");
    if (dst->device() != src->device()) throw std::runtime_error("assign: device mismatch (use `src.to(dst->device())` first)");
    return Tensor::make(AssignOp{}, {dst, src}, dst->shape(), dst->device(), dst->dtype());
}

// Unary Ops
utils::Ref<Tensor> relu(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::RELU, t); }
utils::Ref<Tensor> exp (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::EXP,  t); }
utils::Ref<Tensor> log (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::LOG,  t); }
utils::Ref<Tensor> neg (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::NEG,  t); }
utils::Ref<Tensor> tanh(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::TANH, t); }
utils::Ref<Tensor> sqrt(const utils::Ref<const Tensor>& t) { return pow(t, 0.5f); }

// Binary Ops
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::ADD, a, b); }
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::SUB, a, b); }
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MUL, a, b); }
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::DIV, a, b); }
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::POW, a, b); }
utils::Ref<Tensor> cmp_eq(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::CMP_EQ, a, b); }
utils::Ref<Tensor> cmp_gt(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::CMP_GT, a, b); }
utils::Ref<Tensor> min(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MIN, a, b); }
utils::Ref<Tensor> max(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MAX, a, b); }

// Reduction Ops
utils::Ref<Tensor> sum(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const auto& in_shape = t->shape();
    const int rank = static_cast<int>(in_shape.size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::vector::normalize_axes(axes_in, rank);
    auto out_shape = utils::shape::get_reduce_shape(in_shape, axes_n, keep_dims);

    return Tensor::make(ReduceOp{ ReduceOpType::SUM, axes_n, keep_dims }, {t}, out_shape, t->device(), t->dtype());
}

utils::Ref<Tensor> max(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const auto& in_shape = t->shape();
    const int rank = static_cast<int>(in_shape.size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::vector::normalize_axes(axes_in, rank);
    auto out_shape = utils::shape::get_reduce_shape(in_shape, axes_n, keep_dims);

    return Tensor::make(ReduceOp{ ReduceOpType::MAX, axes_n, keep_dims }, {t}, out_shape, t->device(), t->dtype());
}

// MatMul Op
utils::Ref<Tensor> matmul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) {
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("matmul: invalid shapes");
    }
    std::vector<size_t> out_shape = { a->shape()[0], b->shape()[1] };
    return Tensor::make(MatMulOp{}, {a, b}, out_shape, a->device(), a->dtype());
}

// Materialization / Layout Ops
utils::Ref<const Tensor> contiguous(const utils::Ref<const Tensor>& t) {
    const auto& am = t->access_meta();
    if (am.contiguous && am.offset == 0) return t;
    return Tensor::make(CopyOp{}, {t}, AccessMeta::contiguous_from(t->shape(), 0), t->device(), t->dtype());
}

// Movement Ops
utils::Ref<Tensor> reshape_view(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape) {
    if (utils::vector::numel(t->shape()) != utils::vector::numel(new_shape)) throw std::runtime_error("reshape_view: numel must match");
    auto am = AccessMeta::reshape_from(t->access_meta(), new_shape);
    return Tensor::make(MovementOp{MovementOpType::RESHAPE, new_shape}, {t}, am, t->device(), t->dtype());
}

utils::Ref<Tensor> permute(const utils::Ref<const Tensor>& t, const std::vector<size_t>& axes) {
    return Tensor::make(MovementOp{MovementOpType::PERMUTE, axes}, {t}, AccessMeta::permute_from(t->access_meta(), axes), t->device(), t->dtype());
}

utils::Ref<Tensor> transpose(const utils::Ref<const Tensor>& t, size_t dim0, size_t dim1) {
    std::vector<size_t> axes(t->shape().size());
    std::iota(axes.begin(), axes.end(), 0);
    std::swap(axes[dim0], axes[dim1]);
    return permute(t, axes);
}

utils::Ref<Tensor> broadcast(const utils::Ref<const Tensor>& t, const std::vector<size_t>& shape) {
    return Tensor::make(MovementOp{MovementOpType::BROADCAST, shape}, {t}, AccessMeta::broadcast_from(t->access_meta(), shape), t->device(), t->dtype());
}

utils::Ref<Tensor> slice(const utils::Ref<const Tensor>& t, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) {
    std::vector<size_t> steps = step.empty() ? std::vector<size_t>(begin.size(), 1) : step;
    return Tensor::make(MovementOp{MovementOpType::SLICE, steps, begin, end}, {t}, AccessMeta::slice_from(t->access_meta(), begin, end, steps), t->device(), t->dtype());
}

// Composite Ops
utils::Ref<Tensor> reshape(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape) {
    if (utils::vector::numel(t->shape()) != utils::vector::numel(new_shape)) throw std::runtime_error("reshape: numel must match");
    if (t->access_meta().contiguous) return reshape_view(t, new_shape);
    return reshape_view(contiguous(t), new_shape);
}

utils::Ref<Tensor> mean(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const int rank = static_cast<int>(t->shape().size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::vector::normalize_axes(axes_in, rank);
    auto summed = sum(t, axes_n, keep_dims);
    size_t reduction_size = cppgrad::utils::shape::get_reduce_count(t->shape(), axes_n);
    return div(summed, static_cast<float>(reduction_size));
}

// Scalar Ops (Tensor, float)
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, float val) { return add(a, scalar_like(val, a)); }
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, float val) { return sub(a, scalar_like(val, a)); }
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, float val) { return mul(a, scalar_like(val, a)); }
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, float val) { return div(a, scalar_like(val, a)); }
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, float val) { return pow(a, scalar_like(val, a)); }

// Scalar Ops (float, Tensor)
utils::Ref<Tensor> add(float val, const utils::Ref<const Tensor>& a) { return add(scalar_like(val, a), a); }
utils::Ref<Tensor> sub(float val, const utils::Ref<const Tensor>& a) { return sub(scalar_like(val, a), a); }
utils::Ref<Tensor> mul(float val, const utils::Ref<const Tensor>& a) { return mul(scalar_like(val, a), a); }
utils::Ref<Tensor> div(float val, const utils::Ref<const Tensor>& a) { return div(scalar_like(val, a), a); }
utils::Ref<Tensor> pow(float val, const utils::Ref<const Tensor>& a) { return pow(scalar_like(val, a), a); }

} // namespace ir
} // namespace cppgrad
