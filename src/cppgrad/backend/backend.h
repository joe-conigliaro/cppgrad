// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/backend/copy.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/ir/ops.h"
#include <cstddef>
#include <vector>

namespace cppgrad {
namespace backend {

class Buffer;

class Backend {
public:
    virtual ~Backend() = default;
    
    // Commit any work the backend has batched but not yet executed (Metal: commit+wait the command
    // buffer, freeing the intermediate output buffers it held). No-op for synchronous backends (CPU).
    // Called at GraphScope boundaries, and explicitly between phases of a long computation (e.g.
    // prefill chunks) to bound the number of resident buffers -- otherwise a long prompt accumulates
    // every chunk's buffers in one uncommitted command buffer until Metal can't satisfy even a tiny
    // allocation ("allocation failed").
    //
    // This stateless hook is enough while a backend's batching state is global
    // (Metal's execution context is a long-lived singleton). If a backend ever
    // needs genuine per-scope state (a command buffer / memory pool / fences
    // created per scope, nested-scope isolation, CPU↔GPU overlap), replace this
    // with a generic opaque per-scope handle so GraphScope can carry backend
    // state without the IR layer knowing about any specific backend:
    //
    //     // backend/scope_context.h (pure C++, no backend-specific headers)
    //     class ScopeContext { public: virtual ~ScopeContext() = default; };
    //     dtor flushes virtual std::unique_ptr<ScopeContext> make_scope_context()
    //     const { return nullptr; }
    //
    //     // GraphScope holds: std::unique_ptr<backend::ScopeContext>
    //     _backend_ctx;
    //     //   ctor: _backend_ctx = dev->backend()->make_scope_context();
    //     null for CPU
    //     //   dtor: _backend_ctx.reset();
    //     RAII flush
    //
    // Metal would return a MetalScopeContext (defined under backend/metal/, NOT
    // in ir/) whose destructor flushes the execution context - same effect as
    // flush_pending() but able to own per-scope resources, and still leaving the
    // IR layer backend-agnostic.
    virtual void flush_pending() const {}

    // Attach a debug label to a buffer (visible in Xcode GPU captures). No-op by default.
    virtual void set_buffer_debug_label(const Buffer& /*buf*/, const char* /*label*/) const {}

    // Data Ops
    virtual void copy(Buffer &dst, const Buffer &src) const { cppgrad::backend::copy(dst, src); } // Use backend copy util by default
    virtual void fill(Buffer &buf, double value) const = 0;
    
    // Main Compute Ops
    virtual void unary_op(ir::UnaryOpType op_type, const Buffer &a, const backend::View &va, Buffer &out, const backend::View &vo) const = 0;
    virtual void binary_op(ir::BinaryOpType op_type, const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb, Buffer &out, const backend::View &vo) const = 0;
    virtual void reduce_op(ir::ReduceOpType op_type, const Buffer &a, const backend::View &va, Buffer &out, const backend::View &vo, const std::vector<int> &axes, bool keep_dims) const = 0;
    virtual void matmul(const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb, Buffer &out, const backend::View &vo) const = 0;

    // Fused flash attention (inference only), contiguous inputs in native layout:
    //   q [B,S,nH,Dh], k,v [B,KV,nKV,Dh] -> out [B,S,nH,Dh].
    // out = softmax(scale * QKᵀ + causal)V via online-softmax over keys (no [S,KV] materialization).
    // query head h reads kv head h/n_rep; if causal, query row s attends keys [0, q_offset+s].
    virtual void flash_attention(const Buffer & /*q*/, const Buffer & /*k*/, const Buffer & /*v*/,
                                 Buffer & /*out*/, size_t /*B*/, size_t /*S*/, size_t /*nH*/, size_t /*Dh*/,
                                 size_t /*KV*/, size_t /*nKV*/, float /*scale*/, int /*n_rep*/,
                                 bool /*causal*/, size_t /*q_offset*/) const {
        throw std::runtime_error("flash_attention: not implemented for this backend");
    }

    // Quantized matmul: out[M,N] = a[M,K] @ dequant(qweight)^T (dequant in kernel). Dispatches on
    // params.scheme internally (one entry point for all schemes, not a virtual per quant type).
    // `aux` holds the scheme's metadata buffers in a scheme-defined order (see ir::QuantScheme):
    // for MLX_AFFINE, aux = {scales, biases}, all contiguous fp32; qweight is contiguous u32 [N,K/pack].
    virtual void quantized_matmul(const Buffer &a, const Buffer &qweight,
                                  const std::vector<const Buffer*> &aux, Buffer &out,
                                  size_t M, size_t N, size_t K, const ir::QuantParams &params) const = 0;

    // Gather: table[V, D] (float), indices[N] (int32) -> out[N, D] (float)
    // N = indices.numel(). Backend just processes flat elements; caller shapes the output tensor.
    virtual void gather_op(const Buffer &table, const Buffer &indices, Buffer &out, size_t V, size_t D) const = 0;

    // Gather along axis: tensor with View, indices[N] int32 -> out with same rank but axis dim replaced by N
    virtual void gather_axis_op(const Buffer &tensor, const backend::View &tv,
                                 const Buffer &indices,
                                 Buffer &out, const backend::View &ov,
                                 int axis) const = 0;

    // Scatter along axis: base + values at indices -> out (same shape as base)
    virtual void scatter_axis_op(const Buffer &base, const backend::View &bv,
                                  const Buffer &values, const backend::View &vv,
                                  const Buffer &indices,
                                  Buffer &out, const backend::View &ov,
                                  int axis) const = 0;

    // Concat: concatenate two tensors along axis
    virtual void concat_op(const std::vector<const Buffer*>& inputs, const std::vector<backend::View>& input_views,
                           Buffer &out, const backend::View &out_view, int axis) const = 0;
    
    // Movement ops (reshape/permute/broadcast/slice) are zero-copy: they produce a strided
    // backend::View that downstream ops consume directly (see ir/access_meta.h + the executor's
    // MovementOp aliasing). The only materialization is copy_view below (for ir::contiguous()).

    // Generic (materialize a view mapping)
    virtual void copy_view(const Buffer &src, const backend::View &vs, Buffer &dst, const backend::View &vd) const = 0;
    
    // Random Ops
    virtual void rand_uniform(Buffer &out, float min, float max) const = 0;
    virtual void rand_normal(Buffer &out, float mean, float stddev) const = 0;
};

} // namespace backend
} // namespace cppgrad
