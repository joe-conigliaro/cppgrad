// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Metal/Metal.h>
#include "cppgrad/backend/metal/metal_backend.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/dtype.h"
#include "cppgrad/backend/metal/metal_execution_context.h"
#include "cppgrad/backend/metal/metal_kernel_cache.h"
#include "cppgrad/backend/metal/metal_shared_structs.h"
#include "cppgrad/backend/metal/metal_utils.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/utils/rng.h"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cppgrad {
namespace backend {
namespace metal {

// Pack backend::View -> View32
static inline void pack_view32(const backend::View &v, View32 &out) {
    out.rank = static_cast<unsigned short>(v.rank);
    out.pad = 0;
    out.offset = static_cast<unsigned int>(v.offset);
    out.flags = static_cast<unsigned int>(v.flags);
    for (int i = 0; i < 8; ++i) {
        out.shape[i] = (i < static_cast<int>(v.rank))
            ? static_cast<unsigned int>(v.shape[i])
            : 0u;
        out.strides[i] = (i < static_cast<int>(v.rank))
            ? static_cast<unsigned int>(v.strides[i])
            : 0u;
    }
}

static inline bool same_shape(const backend::View &a, const backend::View &b) {
    if (a.rank != b.rank)
        return false;
    for (uint32_t i = 0; i < a.rank; ++i)
        if (a.shape[i] != b.shape[i])
            return false;
    return true;
}

static inline uint32_t next_u32_from_global() {
    auto &gen = cppgrad::utils::global_rng();
    return static_cast<uint32_t>(gen());
}

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unique_ptr<metal::MetalKernelCache> cache;
    MetalExecutionContext *exec_ctx = nullptr; // non-owning; set at construction

    Impl(void *native_device, void *native_queue, MetalExecutionContext *ec)
    : device((__bridge id<MTLDevice>)native_device),
    queue((__bridge id<MTLCommandQueue>)native_queue),
    cache(std::make_unique<metal::MetalKernelCache>(device)), exec_ctx(ec) {
    }

// Threadgroup width for a 1D (one-thread-per-element) dispatch.
static NSUInteger linear_tg(id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger tg_width = pso.threadExecutionWidth;
    if (tg_width == 0)
        tg_width = 64;
    NSUInteger tg = MIN(tg_width * 4, [pso maxTotalThreadsPerThreadgroup]);
    tg = MIN(tg, n);
    if (tg == 0)
        tg = 1;
    return tg;
}

// Configure a 1D (one-thread-per-element) dispatch.
static void set_linear(ComputeWork &work, NSUInteger n) {
    work.useThreadgroups = false;
    work.grid = MTLSizeMake(n, 1, 1);
    work.threadsPerThreadgroup = MTLSizeMake(linear_tg(work.pso, n), 1, 1);
}

// Submit a work item: record it for batched execution (async, exec_ctx set)
// or encode + commit + wait immediately (sync).
void encode_submit(ComputeWork &work) const {
    if (exec_ctx) {
        exec_ctx->submit_compute(std::move(work));
        return;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode_work(enc, work);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Synchronous device-to-device copy (for identity copy path). The source may
// have been produced by batched compute still pending in the execution
// context, so flush it first - otherwise this blit (on a different command
// buffer) races ahead and reads stale data.
void sync_copy(Buffer &dst, const Buffer &src) const {
    if (dst.size_bytes() == 0)
        return;
    if (exec_ctx)
        exec_ctx->flush();
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:as_mtl(src)
    sourceOffset:0
    toBuffer:as_mtl(dst)
    destinationOffset:0
    size:(NSUInteger)dst.size_bytes()];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void submit_fill(Buffer &buf, double value) const {
    if (buf.size_bytes() == 0)
        return;
    if (buf.dtype() != backend::DType::FLOAT32) {
        throw std::runtime_error(
        std::string("MetalBackend::submit_fill: unsupported dtype ") +
        to_string(buf.dtype()));
    }
    float value_f32 = static_cast<float>(value);
    ComputeWork work;
    work.pso = cache->get("fill");
    work.buffers.push_back({as_mtl(buf), 0});
    work.add_bytes(1, &value_f32, sizeof(float));
    set_linear(work, buf.numel());
    encode_submit(work);
}

void submit_unary_op(ir::UnaryOpType op_type, const Buffer &a,
const backend::View &va, Buffer &out,
const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    UnaryParams P{};
    pack_view32(va, P.in_v);
    pack_view32(vo, P.out_v);
    P.n = (unsigned int)out.numel();
    P.op = (unsigned short)op_type;

    ComputeWork work;
    work.pso = cache->get("unary_view_f32");
    work.buffers.push_back({as_mtl(a), 0});
    work.buffers.push_back({as_mtl(out), 0});
    work.add_bytes(2, &P, sizeof(P));
    set_linear(work, out.numel());
    encode_submit(work);
}

void submit_binary_op(ir::BinaryOpType op_type, const Buffer &a,
const backend::View &va, const Buffer &b,
const backend::View &vb, Buffer &out,
const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    BinaryParams P{};
    pack_view32(va, P.a_v);
    pack_view32(vb, P.b_v);
    pack_view32(vo, P.o_v);
    P.n = (unsigned int)out.numel();
    P.op = (unsigned short)op_type;

    ComputeWork work;
    work.pso = cache->get("binary_view_f32");
    work.buffers.push_back({as_mtl(a), 0});
    work.buffers.push_back({as_mtl(b), 0});
    work.buffers.push_back({as_mtl(out), 0});
    work.add_bytes(3, &P, sizeof(P));
    set_linear(work, out.numel());
    encode_submit(work);
}

void submit_reduce_op(ir::ReduceOpType op_type, const Buffer &a,
const backend::View &va, Buffer &out,
const backend::View &vo, const std::vector<int> &axes,
bool keep_dims) const {
    if (out.size_bytes() == 0)
        return;
    const unsigned short op = (op_type == ir::ReduceOpType::MAX)
    ? (unsigned short)1
    : (unsigned short)0;

    bool last_only = (axes.size() == 1) &&
    ((axes[0] == (int)va.rank - 1) || (axes[0] == -1));
    bool last_contig = va.last_axis_contiguous();

    if (last_only && last_contig && vo.rank == va.rank - (keep_dims ? 0 : 1)) {
        ReduceFastParams P{};
        pack_view32(va, P.in_v);
        pack_view32(vo, P.out_v);
        P.inner = (unsigned int)va.shape[va.rank - 1];
        P.op = op;

        ComputeWork work;
        work.pso = cache->get("reduce_last_axis_f32");
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(2, &P, sizeof(P));

        // One threadgroup per output row; threads in the group cooperatively
        // reduce the (contiguous) last axis using threadgroup memory.
        NSUInteger tg_width = work.pso.threadExecutionWidth;
        tg_width =
            MIN(tg_width, (NSUInteger)[work.pso maxTotalThreadsPerThreadgroup]);
        tg_width = MIN(tg_width, (NSUInteger)128);
        if (tg_width == 0)
            tg_width = 1;
        work.useThreadgroups = true;
        work.grid = MTLSizeMake((NSUInteger)out.numel(), 1, 1); // #threadgroups
        work.threadsPerThreadgroup = MTLSizeMake(tg_width, 1, 1);
        work.threadgroupMemoryLength = tg_width * sizeof(float);
        encode_submit(work);
        return;
    }

    ReduceGeneralParams P{};
    pack_view32(va, P.in_v);
    pack_view32(vo, P.out_v);
    P.op = op;
    P.pad6 = 0;
    P.out_total = (unsigned int)out.numel();
    for (int i = 0; i < 8; ++i)
        P.is_reduce_axis[i] = 0;
    for (int ax : axes) {
        int aidx = ax;
        if (aidx < 0)
            aidx += (int)va.rank;
        if (aidx < 0 || aidx >= (int)va.rank)
        throw std::runtime_error("Metal reduce_op: axis out of range");
        P.is_reduce_axis[aidx] = 1;
            }

            ComputeWork work;
            work.pso = cache->get("reduce_general_f32");
                work.buffers.push_back({as_mtl(a), 0});
                work.buffers.push_back({as_mtl(out), 0});
                work.add_bytes(2, &P, sizeof(P));
        set_linear(work, out.numel());
    encode_submit(work);
}

void submit_matmul(const Buffer &a, const backend::View &va, const Buffer &b,
const backend::View &vb, Buffer &out,
const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    if (va.rank != 2 || vb.rank != 2 || vo.rank != 2)
        throw std::runtime_error("Metal matmul: rank-2 views required");

    MatmulParams P{};
    pack_view32(va, P.a_v);
    pack_view32(vb, P.b_v);
    pack_view32(vo, P.o_v);
    P.M = static_cast<unsigned int>(va.shape[0]);
    P.K = static_cast<unsigned int>(va.shape[1]);
    P.N = static_cast<unsigned int>(vb.shape[1]);

    const bool tiny = (P.M < 8 || P.N < 8 || P.K < 8);
    const bool fast_packed = va.is_rowmaj_nn_2d() && vo.is_rowmaj_nn_2d();
    const bool nn_layout = fast_packed && vb.is_rowmaj_nn_2d();
    const bool tn_layout = fast_packed && vb.is_rowmaj_tn_2d();

    ComputeWork work;
    work.buffers.push_back({as_mtl(a), 0});
    work.buffers.push_back({as_mtl(b), 0});
    work.buffers.push_back({as_mtl(out), 0});
    work.add_bytes(3, &P, sizeof(P));

    if (tiny || (!nn_layout && !tn_layout)) {
        // Naive kernel: one thread per output element, 2D grid (N x M).
        work.pso = cache->get("matmul_view_f32");
        NSUInteger w = work.pso.threadExecutionWidth;
        if (w == 0)
            w = 32;
        NSUInteger h = [work.pso maxTotalThreadsPerThreadgroup] / w;
        if (h == 0)
            h = 1;
        work.useThreadgroups = false;
        work.grid = MTLSizeMake(P.N, P.M, 1);
        work.threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    } else {
        // Tiled kernel: grid is in threadgroups (one per TMxTN output tile),
        // threadgroup size matches the tile (TM=TN=16, hardcoded in the kernel).
        constexpr unsigned TM = 16, TN = 16;
        work.pso =
            cache->get(nn_layout ? "matmul_tiled_f32" : "matmul_tiled_tn_f32");
        work.useThreadgroups = true;
        work.grid = MTLSizeMake((P.N + TN - 1) / TN, (P.M + TM - 1) / TM, 1);
        work.threadsPerThreadgroup = MTLSizeMake(TN, TM, 1);
    }
    encode_submit(work);
}

void submit_copy_view(const Buffer &src, const backend::View &vs, Buffer &dst,
const backend::View &vd) const {
    if (dst.size_bytes() == 0)
        return;

    // Fast path: identity mapping => synchronous blit.
    if (vs.is_identity() && vd.is_identity() && same_shape(vs, vd)) {
        sync_copy(dst, src);
        return;
    }

    CopyViewParams P{};
    pack_view32(vs, P.src_v);
    pack_view32(vd, P.dst_v);
    P.n = (unsigned int)dst.numel();

    ComputeWork work;
    work.pso = cache->get("copy_view_f32");
    work.buffers.push_back({as_mtl(src), 0});
    work.buffers.push_back({as_mtl(dst), 0});
    work.add_bytes(2, &P, sizeof(P));
    set_linear(work, dst.numel());
    encode_submit(work);
}

void submit_rand_uniform(Buffer &out, float min, float max) const {
    if (out.size_bytes() == 0)
        return;
    uint32_t seed = next_u32_from_global();
    ComputeWork work;
    work.pso = cache->get("rand_uniform");
    work.buffers.push_back({as_mtl(out), 0});
    work.add_bytes(1, &min, sizeof(float));
    work.add_bytes(2, &max, sizeof(float));
    work.add_bytes(3, &seed, sizeof(uint32_t));
    set_linear(work, out.numel());
    encode_submit(work);
}

void submit_rand_normal(Buffer &out, float mean, float stddev) const {
    if (out.size_bytes() == 0)
        return;
    uint32_t seed = next_u32_from_global();
    uint32_t out_numel = (uint32_t)out.numel();
    ComputeWork work;
    work.pso = cache->get("rand_normal");
    work.buffers.push_back({as_mtl(out), 0});
    work.add_bytes(1, &mean, sizeof(float));
    work.add_bytes(2, &stddev, sizeof(float));
    work.add_bytes(3, &seed, sizeof(uint32_t));
    work.add_bytes(4, &out_numel, sizeof(uint32_t));
    set_linear(work, out.numel());
    encode_submit(work);
}
};

MetalBackend::MetalBackend(void *native_device, void *native_queue,
    MetalExecutionContext *ec)
: _impl(std::make_unique<Impl>(native_device, native_queue, ec)) {}
MetalBackend::~MetalBackend() = default;

void MetalBackend::flush_pending() const {
    if (_impl->exec_ctx)
        _impl->exec_ctx->flush();
}

// Fill (sync wrapper: uses async submit + await when in async mode).
void MetalBackend::fill(Buffer &buf, double value) const {
    if (buf.size_bytes() == 0)
        return;
    _impl->submit_fill(buf, value);
}

// Random
void MetalBackend::rand_uniform(Buffer &out, float min, float max) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_rand_uniform(out, min, max);
}

void MetalBackend::rand_normal(Buffer &out, float mean, float stddev) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_rand_normal(out, mean, stddev);
}

// Unary (stride-aware)
void MetalBackend::unary_op(ir::UnaryOpType op_type, const Buffer &a,
const backend::View &va, Buffer &out,
const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_unary_op(op_type, a, va, out, vo);
}

// Binary (stride-aware)
void MetalBackend::binary_op(ir::BinaryOpType op_type, const Buffer &a,
const backend::View &va, const Buffer &b,
const backend::View &vb, Buffer &out,
const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_binary_op(op_type, a, va, b, vb, out, vo);
}

// Reduce (fast path + general)
void MetalBackend::reduce_op(ir::ReduceOpType op_type, const Buffer &a,
const backend::View &va, Buffer &out,
const backend::View &vo,
const std::vector<int> &axes,
bool keep_dims) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_reduce_op(op_type, a, va, out, vo, axes, keep_dims);
}

// Matmul
void MetalBackend::matmul(const Buffer &a, const backend::View &va,
const Buffer &b, const backend::View &vb, Buffer &out,
const backend::View &vo) const {
    _impl->submit_matmul(a, va, b, vb, out, vo);
}

// Copy view
void MetalBackend::copy_view(const Buffer &src, const backend::View &vs,
Buffer &dst, const backend::View &vd) const {
    if (dst.size_bytes() == 0)
        return;
    _impl->submit_copy_view(src, vs, dst, vd);
}

} // namespace metal
} // namespace backend
} // namespace cppgrad
