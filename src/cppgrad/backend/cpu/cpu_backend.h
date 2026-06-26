// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/backend/backend.h"
#include "cppgrad/backend/view.h"

namespace cppgrad {
namespace backend {
namespace cpu {

class CPUBackend : public Backend {
public:
    // Data Ops
    void fill(Buffer& buf, double value) const override;

    // Main Compute Ops
    void unary_op(ir::UnaryOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const override;
    void binary_op(ir::BinaryOpType op_type, const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const override;
    void reduce_op(ir::ReduceOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<int>& axes, bool keep_dims) const override;
    void matmul(const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const override;
    void quantized_matmul(const Buffer& a, const Buffer& qweight, const std::vector<const Buffer*>& aux,
                          Buffer& out, size_t M, size_t N, size_t K, const ir::QuantParams& params) const override;
    void flash_attention(const Buffer& q, const Buffer& k, const Buffer& v, Buffer& out,
                         size_t B, size_t S, size_t nH, size_t Dh, size_t KV, size_t nKV,
                         float scale, int n_rep, bool causal, size_t q_offset) const override;
    void gather_op(const Buffer& table, const Buffer& indices, Buffer& out, size_t V, size_t D) const override;
    void concat_op(const std::vector<const Buffer*>& inputs, const std::vector<backend::View>& input_views,
                   Buffer& out, const backend::View& out_view, int axis) const override;
    void gather_axis_op(const Buffer& tensor, const backend::View& tv,
                        const Buffer& indices,
                        Buffer& out, const backend::View& ov,
                        int axis) const override;
    void scatter_axis_op(const Buffer& base, const backend::View& bv,
                         const Buffer& values, const backend::View& vv,
                         const Buffer& indices,
                         Buffer& out, const backend::View& ov,
                         int axis) const override;

    // Generic (materialize a view mapping)
    void copy_view(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) const override;

    // Random
    void rand_uniform(Buffer& out, float min, float max) const override;
    void rand_normal(Buffer& out, float mean, float stddev) const override;
};

} // namespace cpu
} // namespace backend
} // namespace cppgrad
