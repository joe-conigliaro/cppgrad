// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <type_traits>
#include <vector>

#include "cppgrad/ir/ops.h"
#include "cppgrad/ir/tensor.h"

// Reverse-mode autograd rules, one free function per differentiable op.
//
// Tensor::backward() owns the driver (tape build, reverse sweep, grad accumulation) and dispatches the
// per-op gradient via `std::visit([&](auto&& op){ return backward_op(op, ctx); }, node->op())`. Each
// rule lives here as an overloaded free function. A differentiable op with no overload is a compile error.
namespace cppgrad::ir {

// Context for a single node's backward rule.
struct BackwardCtx {
    const Tensor *node;                                    // the op's output node
    const std::vector<utils::Ref<const Tensor>> &children; // node->children() (the op's inputs)
    const utils::Ref<Tensor> &grad_this;                   // incoming gradient w.r.t. `node`
};

// Gradient w.r.t. each child, positional (size == children.size(), or empty for no contribution).
using Grads = std::vector<utils::Ref<Tensor>>;

// Per-op rules (defined in autograd.cpp).
Grads backward_op(const UnaryOp &op, const BackwardCtx &ctx);
Grads backward_op(const BinaryOp &op, const BackwardCtx &ctx);
Grads backward_op(const ReduceOp &op, const BackwardCtx &ctx);
Grads backward_op(const MovementOp &op, const BackwardCtx &ctx);
Grads backward_op(const MatMulOp &op, const BackwardCtx &ctx);
Grads backward_op(const CopyOp &op, const BackwardCtx &ctx);
Grads backward_op(const ConcatOp &op, const BackwardCtx &ctx);

// Non-differentiable ops contribute no gradient. Constrained to !is_differentiable_v so that a
// differentiable op lacking an explicit overload above fails to compile (no viable function).
template <class T, std::enable_if_t<!is_differentiable_v<T>, int> = 0>
inline Grads backward_op(const T &, const BackwardCtx &) {
    return {};
}

// Broadcast an upstream gradient up to a parent's shape (singleton dims only; reduction is an error).
// Also used by Tensor::backward()'s accumulation step, hence exported.
utils::Ref<Tensor> unify_to_shape(utils::Ref<Tensor> g, const std::vector<size_t> &parent_shape);

} // namespace cppgrad::ir
