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
    
    // Commit any work the backend has batched but not yet executed. Called at
    // GraphScope boundaries. No-op for synchronous backends (e.g. CPU); the Metal
    // backend flushes its execution context so scope work completes at scope end.
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
    
    // Data Ops
    virtual void copy(Buffer &dst, const Buffer &src) const { cppgrad::backend::copy(dst, src); } // Use backend copy util by default
    virtual void fill(Buffer &buf, double value) const = 0;
    
    // Main Compute Ops
    virtual void unary_op(ir::UnaryOpType op_type, const Buffer &a, const backend::View &va, Buffer &out, const backend::View &vo) const = 0;
    virtual void binary_op(ir::BinaryOpType op_type, const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb, Buffer &out, const backend::View &vo) const = 0;
    virtual void reduce_op(ir::ReduceOpType op_type, const Buffer &a, const backend::View &va, Buffer &out, const backend::View &vo, const std::vector<int> &axes, bool keep_dims) const = 0;
    virtual void matmul(const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb, Buffer &out, const backend::View &vo) const = 0;
    
    // Movement Ops
    // virtual void permute(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& axes) const = 0;
    // virtual void broadcast(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const = 0;
    // virtual void slice_forward(const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<size_t>& begin, const std::vector<size_t>&, const std::vector<size_t>& step) const = 0; virtual void
    // slice_backward_scatter_add(const Buffer& grad_out, const backend::View& vgo, Buffer& grad_in,  const backend::View& vgi, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) const = 0;
    
    // Generic (materialize a view mapping)
    virtual void copy_view(const Buffer &src, const backend::View &vs, Buffer &dst, const backend::View &vd) const = 0;
    
    // Random Ops
    virtual void rand_uniform(Buffer &out, float min, float max) const = 0;
    virtual void rand_normal(Buffer &out, float mean, float stddev) const = 0;
};

} // namespace backend
} // namespace cppgrad
