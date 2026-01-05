// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/backend/backend.h"
#include "cppgrad/backend/view.h"

namespace cppgrad {
namespace backend {
namespace metal {

class MetalBackend final : public Backend {
public:
    // Construct from native (non-owning) Metal device and queue pointers
    MetalBackend(void* native_device, void* native_queue);
    ~MetalBackend();

    // Data Ops
    void fill(Buffer& buf, double value) const override;

    // Main Compute Ops
    void unary_op(ir::UnaryOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const override;
    void binary_op(ir::BinaryOpType op_type, const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const override;
    void reduce_op(ir::ReduceOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<int>& axes, bool keep_dims) const override;
    void matmul(const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const override;

    // Movement
    // void permute(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& axes) const override;
    // void broadcast(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const override;
    // void slice_forward(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) const override;
    // void slice_backward_scatter_add(const Buffer& grad_out, const backend::View& vgo, Buffer& grad_in,  const backend::View& vgi, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) const override;

    // Generic (materialize a view mapping)
    void copy_view(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) const override;

    // Random
    void rand_uniform(Buffer& out, float min, float max) const override;
    void rand_normal(Buffer& out, float mean, float stddev) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace metal
} // namespace backend
} // namespace cppgrad
