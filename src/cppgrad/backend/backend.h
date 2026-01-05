// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <vector>
#include <cstddef>
#include "cppgrad/backend/copy.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/ir/ops.h"

namespace cppgrad {
namespace backend {

class Buffer;

class Backend {
public:
    virtual ~Backend() = default;

    // Data Ops
    virtual void copy(Buffer& dst, const Buffer& src) const { cppgrad::backend::copy(dst, src); } // Use backend copy util by default
    virtual void fill(Buffer& buf, double value) const = 0;

    // Main Compute Ops
    virtual void unary_op(ir::UnaryOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const = 0;
    virtual void binary_op(ir::BinaryOpType op_type, const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const = 0;
    virtual void reduce_op(ir::ReduceOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<int>& axes, bool keep_dims) const = 0;
    virtual void matmul(const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const = 0;

    // Movement Ops
    // virtual void permute(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& axes) const = 0;
    // virtual void broadcast(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const = 0;
    // virtual void slice_forward(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& begin, const std::vector<size_t>&, const std::vector<size_t>& step) const = 0;
    // virtual void slice_backward_scatter_add(const Buffer& grad_out, const backend::View& vgo, Buffer& grad_in,  const backend::View& vgi, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) const = 0;

    // Generic (materialize a view mapping)
    virtual void copy_view(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) const = 0;

    // Random Ops
    virtual void rand_uniform(Buffer& out, float min, float max) const = 0;
    virtual void rand_normal(Buffer& out, float mean, float stddev) const = 0;
};

} // namespace backend
} // namespace cppgrad
