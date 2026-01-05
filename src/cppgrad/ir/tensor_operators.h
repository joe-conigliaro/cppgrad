// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/ir/tensor_ops.h"

namespace cppgrad {
namespace ir {

// Operator Overloads

// Tensor-Tensor Ops
inline utils::Ref<Tensor> operator+(const utils::Ref<Tensor>& a, const utils::Ref<Tensor>& b) { return add(a, b); }
inline utils::Ref<Tensor> operator-(const utils::Ref<Tensor>& a, const utils::Ref<Tensor>& b) { return sub(a, b); }
inline utils::Ref<Tensor> operator*(const utils::Ref<Tensor>& a, const utils::Ref<Tensor>& b) { return mul(a, b); }
inline utils::Ref<Tensor> operator/(const utils::Ref<Tensor>& a, const utils::Ref<Tensor>& b) { return div(a, b); }
inline utils::Ref<Tensor> operator-(const utils::Ref<Tensor>& a) { return neg(a); }

// Tensor-Scalar Ops (Tensor x float)
inline utils::Ref<Tensor> operator+(const utils::Ref<Tensor>& a, float val) { return add(a, val); }
inline utils::Ref<Tensor> operator-(const utils::Ref<Tensor>& a, float val) { return sub(a, val); }
inline utils::Ref<Tensor> operator*(const utils::Ref<Tensor>& a, float val) { return mul(a, val); }
inline utils::Ref<Tensor> operator/(const utils::Ref<Tensor>& a, float val) { return div(a, val); }
// Scalar-Tensor Ops (float x Tensor)
inline utils::Ref<Tensor> operator+(float val, const utils::Ref<Tensor>& a) { return add(val, a); }
inline utils::Ref<Tensor> operator-(float val, const utils::Ref<Tensor>& a) { return sub(val, a); }
inline utils::Ref<Tensor> operator*(float val, const utils::Ref<Tensor>& a) { return mul(val, a); }
inline utils::Ref<Tensor> operator/(float val, const utils::Ref<Tensor>& a) { return div(val, a); }

} // namespace ir
} // namespace cppgrad
