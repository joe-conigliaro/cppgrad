// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Metal/Metal.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <functional>
#include "cppgrad/backend/metal/metal_backend.h"
#include "cppgrad/backend/metal/metal_kernel_cache.h"
#include "cppgrad/backend/metal/metal_shared_structs.h"
#include "cppgrad/backend/metal/metal_utils.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/backend/dtype.h"
#include "cppgrad/utils/rng.h"

namespace cppgrad {
namespace backend {
namespace metal {

// Pack backend::View -> View32
static inline void pack_view32(const backend::View& v, View32& out) {
    out.rank   = static_cast<unsigned short>(v.rank);
    out.pad    = 0;
    out.offset = static_cast<unsigned int>(v.offset);
    out.flags  = static_cast<unsigned int>(v.flags);
    for (int i=0;i<8;++i) {
        out.shape[i]   = (i < static_cast<int>(v.rank)) ? static_cast<unsigned int>(v.shape[i])   : 0u;
        out.strides[i] = (i < static_cast<int>(v.rank)) ? static_cast<unsigned int>(v.strides[i]) : 0u;
    }
}

static inline bool same_shape(const backend::View& a, const backend::View& b) {
    if (a.rank != b.rank) return false;
    for (uint32_t i=0;i<a.rank;++i) if (a.shape[i] != b.shape[i]) return false;
    return true;
}

static inline uint32_t next_u32_from_global() {
    auto& gen = cppgrad::utils::global_rng();
    return static_cast<uint32_t>(gen());
}

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unique_ptr<metal::MetalKernelCache> cache;

    Impl(void* native_device, void* native_queue)
    : device((__bridge id<MTLDevice>)native_device),
      queue((__bridge id<MTLCommandQueue>)native_queue),
      cache(std::make_unique<metal::MetalKernelCache>(device)) {}

    // 1D dispatch helper
    void dispatch_1d(id<MTLComputePipelineState> pso,
                     id<MTLCommandBuffer> cb,
                     size_t n,
                     const std::function<void(id<MTLComputeCommandEncoder>)>& set_buffers) const {
        if (n == 0) return;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        set_buffers(enc);

        // NSUInteger tg_width = pso.threadExecutionWidth;
        // if (tg_width == 0) tg_width = 128;
        // NSUInteger max_tg = [pso maxTotalThreadsPerThreadgroup];
        // if (tg_width > max_tg) tg_width = max_tg;
        // if (tg_width > n) tg_width = (NSUInteger)n;
        // if (tg_width == 0) tg_width = 1;

        // MTLSize threadsPerThreadgroup = MTLSizeMake(tg_width, 1, 1);
        // MTLSize threadsPerGrid        = MTLSizeMake((NSUInteger)n, 1, 1);
        // [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        // [enc endEncoding];

        NSUInteger tg_width = pso.threadExecutionWidth;
        if (tg_width == 0) tg_width = 64;
        // Prefer 128 or 256 if allowed and beneficial
        NSUInteger desired = tg_width * 4; // 4 warps/wavefronts
        NSUInteger max_tg = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger tg = MIN((NSUInteger)desired, max_tg);
        tg = MIN(tg, (NSUInteger)n);
        if (tg == 0) tg = 1;

        MTLSize threadsPerThreadgroup = MTLSizeMake(tg, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake((NSUInteger)n, 1, 1);
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
    }
};

MetalBackend::MetalBackend(void* native_device, void* native_queue)
: _impl(std::make_unique<Impl>(native_device, native_queue)) {}
MetalBackend::~MetalBackend() = default;

// Fill
void MetalBackend::fill(Buffer& buf, double value) const {
    if (buf.size_bytes() == 0) return;
    if (buf.dtype() != backend::DType::FLOAT32) {
        throw std::runtime_error(std::string("MetalBackend::fill: unsupported dtype ") + to_string(buf.dtype()));
    }
    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("fill");
    float value_f32 = value;
    _impl->dispatch_1d(pso, cb, buf.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(buf) offset:0 atIndex:0];
        [enc setBytes:&value_f32 length:sizeof(float) atIndex:1];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

// Random
void MetalBackend::rand_uniform(Buffer& out, float min, float max) const {
    if (out.size_bytes() == 0) return;
    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("rand_uniform");
    uint32_t seed = next_u32_from_global();
    _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:0];
        [enc setBytes:&min  length:sizeof(float)    atIndex:1];
        [enc setBytes:&max  length:sizeof(float)    atIndex:2];
        [enc setBytes:&seed length:sizeof(uint32_t) atIndex:3];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

void MetalBackend::rand_normal(Buffer& out, float mean, float stddev) const {
    if (out.size_bytes() == 0) return;
    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("rand_normal");
    uint32_t seed = next_u32_from_global();
    uint out_numel = (uint)out.numel();
    _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:0];
        [enc setBytes:&mean      length:sizeof(float)    atIndex:1];
        [enc setBytes:&stddev    length:sizeof(float)    atIndex:2];
        [enc setBytes:&seed      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&out_numel length:sizeof(uint)     atIndex:4];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

// Unary (stride-aware)
void MetalBackend::unary_op(ir::UnaryOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const {
    if (out.size_bytes() == 0) return;

    // Fast: identity mapping (same shape, row-major, offset 0), and op is a no-op? None of our unary ops are true no-ops,
    // but relu/log/exp/tanh/neg all need compute. So only fast path here is when input and output views are identical and
    // we can skip packing-laden 2D path by using the generic 1D kernel anyway. No blit unless it were a no-op.
    // However, if shapes/strides/offset match, we can treat n as contiguous and just dispatch 1D kernel we already have.

    UnaryParams P{}; // default
    pack_view32(va, P.in_v);
    pack_view32(vo, P.out_v);
    P.n  = (unsigned int)out.numel();
    P.op = (unsigned short)op_type;

    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("unary_view_f32");

    _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
        [enc setBytes:&P length:sizeof(P) atIndex:2];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

// Binary (stride-aware)
void MetalBackend::binary_op(ir::BinaryOpType op_type, const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const {
    if (out.size_bytes() == 0) return;
    BinaryParams P{};
    pack_view32(va, P.a_v);
    pack_view32(vb, P.b_v);
    pack_view32(vo, P.o_v);
    P.n  = (unsigned int)out.numel();
    P.op = (unsigned short)op_type;

    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("binary_view_f32");
    _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(b)   offset:0 atIndex:1];
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:2];
        [enc setBytes:&P length:sizeof(P) atIndex:3];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

// Reduce (fast path + general)
void MetalBackend::reduce_op(ir::ReduceOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<int>& axes, bool keep_dims) const {
    if (out.size_bytes() == 0) return;
    const unsigned short op = (op_type == ir::ReduceOpType::MAX) ? (unsigned short)1 : (unsigned short)0;

    // Fast path: reduce last axis only and last axis contiguous
    bool last_only = (axes.size() == 1) && ((axes[0] == (int)va.rank - 1) || (axes[0] == -1));
    bool last_contig = va.last_axis_contiguous(); // stride[last] == 1

    if (last_only && last_contig && vo.rank == va.rank - (keep_dims ? 0 : 1)) {
        ReduceFastParams P{};
        pack_view32(va, P.in_v);
        pack_view32(vo, P.out_v);
        P.inner = (unsigned int)va.shape[va.rank - 1];
        P.op    = op;

        id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
        id<MTLComputePipelineState> pso = _impl->cache->get("reduce_last_axis_f32");

        NSUInteger tg_width = pso.threadExecutionWidth;
        NSUInteger max_tg = [pso maxTotalThreadsPerThreadgroup];
        tg_width = std::min<NSUInteger>(tg_width, max_tg);
        tg_width = std::min<NSUInteger>(tg_width, 128u);

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
        [enc setBytes:&P length:sizeof(P) atIndex:2];
        [enc setThreadgroupMemoryLength:(tg_width * sizeof(float)) atIndex:0];

        size_t outer = out.numel(); // one TG per output element row
        MTLSize numTG = MTLSizeMake((NSUInteger)outer, 1, 1);
        MTLSize threadsPerTG = MTLSizeMake(tg_width, 1, 1);
        [enc dispatchThreadgroups:numTG threadsPerThreadgroup:threadsPerTG];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return;
    }

    // Optional fast path: reduce-all and contiguous => 1 pass reduction in general kernel still okay.
    // We'll let the general kernel handle it; adding a dedicated 1D kernel is possible but not necessary.

    // General path
    ReduceGeneralParams P{};
    pack_view32(va, P.in_v);
    pack_view32(vo, P.out_v);
    P.op = op;
    P.pad6 = 0;
    P.out_total = (unsigned int)out.numel();
    for (int i=0;i<8;++i) P.is_reduce_axis[i] = 0;
    for (int ax : axes) {
        int aidx = ax; if (aidx < 0) aidx += (int)va.rank;
        if (aidx < 0 || aidx >= (int)va.rank) throw std::runtime_error("Metal reduce_op: axis out of range");
        P.is_reduce_axis[aidx] = 1;
    }

    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("reduce_general_f32");
    _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
        [enc setBytes:&P length:sizeof(P) atIndex:2];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

// Matmul (stride-aware rank-2)
// void MetalBackend::matmul(const Buffer& a, const backend::View& va,
//                           const Buffer& b, const backend::View& vb,
//                           Buffer& out, const backend::View& vo) const {
//     if (out.size_bytes() == 0) return;
//     if (va.rank != 2 || vb.rank != 2 || vo.rank != 2)
//         throw std::runtime_error("Metal matmul: rank-2 views required");

//     MatmulParams P{};
//     pack_view32(va, P.a_v);
//     pack_view32(vb, P.b_v);
//     pack_view32(vo, P.o_v);
//     P.M = (unsigned int)va.shape[0];
//     P.K = (unsigned int)va.shape[1];
//     P.N = (unsigned int)vb.shape[1];

//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
//     id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
//     id<MTLComputePipelineState> pso = _impl->cache->get("matmul_view_f32");
//     [enc setComputePipelineState:pso];
//     [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
//     [enc setBuffer:metal::as_mtl(b)   offset:0 atIndex:1];
//     [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:2];
//     [enc setBytes:&P length:sizeof(P) atIndex:3];

//     // 2D grid (N x M)
//     MTLSize grid = MTLSizeMake(P.N, P.M, 1);
//     NSUInteger w = pso.threadExecutionWidth;
//     NSUInteger h = [pso maxTotalThreadsPerThreadgroup] / w;
//     if (h == 0) h = 1;
//     MTLSize tg = MTLSizeMake(w, h, 1);

//     [enc dispatchThreads:grid threadsPerThreadgroup:tg];
//     [enc endEncoding];
//     [cb commit];
//     [cb waitUntilCompleted];
// }
// void MetalBackend::matmul(const Buffer& a, const backend::View& va,
//                           const Buffer& b, const backend::View& vb,
//                           Buffer& out, const backend::View& vo) const {
//     if (out.size_bytes() == 0) return;
//     if (va.rank != 2 || vb.rank != 2 || vo.rank != 2)
//         throw std::runtime_error("Metal matmul: rank-2 views required");

//     // Pack params
//     MatmulParams P{};
//     pack_view32(va, P.a_v);
//     pack_view32(vb, P.b_v);
//     pack_view32(vo, P.o_v);
//     P.M = static_cast<unsigned int>(va.shape[0]);
//     P.K = static_cast<unsigned int>(va.shape[1]);
//     P.N = static_cast<unsigned int>(vb.shape[1]);

//     // Tunables must match kernel #defines
//     constexpr unsigned TM = 16;
//     constexpr unsigned TN = 16;

//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];

//     // Simple heuristic: for tiny sizes, naive may be cheaper
//     const bool use_naive = (P.M < 8 || P.N < 8 || P.K < 8);
//     id<MTLComputePipelineState> pso =
//         _impl->cache->get(use_naive ? "matmul_view_f32" : "matmul_tiled_f32");

//     id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
//     [enc setComputePipelineState:pso];
//     [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
//     [enc setBuffer:metal::as_mtl(b)   offset:0 atIndex:1];
//     [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:2];
//     [enc setBytes:&P length:sizeof(P) atIndex:3];

//     if (use_naive) {
//         // Naive kernel: 2D threads per output element
//         MTLSize grid = MTLSizeMake(P.N, P.M, 1);
//         NSUInteger w = pso.threadExecutionWidth;
//         if (w == 0) w = 32;
//         NSUInteger maxThreads = [pso maxTotalThreadsPerThreadgroup];
//         NSUInteger h = maxThreads / w;
//         if (h == 0) h = 1;
//         MTLSize tg = MTLSizeMake(w, h, 1);
//         [enc dispatchThreads:grid threadsPerThreadgroup:tg];
//     } else {
//         // Tiled kernel: grid is in threadgroups (tiles), not threads
//         NSUInteger groupsX = (P.N + TN - 1) / TN;
//         NSUInteger groupsY = (P.M + TM - 1) / TM;

//         MTLSize grid = MTLSizeMake(groupsX, groupsY, 1);
//         MTLSize tg   = MTLSizeMake(TN, TM, 1);
//         [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
//     }

//     [enc endEncoding];
//     [cb commit];
//     [cb waitUntilCompleted];
// }
void MetalBackend::matmul(const Buffer& a, const backend::View& va,
                          const Buffer& b, const backend::View& vb,
                          Buffer& out, const backend::View& vo) const {
    if (out.size_bytes() == 0) return;
    if (va.rank != 2 || vb.rank != 2 || vo.rank != 2)
        throw std::runtime_error("Metal matmul: rank-2 views required");

    // Pack once
    MatmulParams P{};
    pack_view32(va, P.a_v);
    pack_view32(vb, P.b_v);
    pack_view32(vo, P.o_v);
    P.M = static_cast<unsigned int>(va.shape[0]);
    P.K = static_cast<unsigned int>(va.shape[1]);
    P.N = static_cast<unsigned int>(vb.shape[1]);

    // Heuristic: tiny sizes => naive
    const bool tiny = (P.M < 8 || P.N < 8 || P.K < 8);

    // Layout flags decide which tiled PSO to use
    const bool fast_packed = va.is_rowmaj_nn_2d() && vo.is_rowmaj_nn_2d();
    const bool nn_layout = fast_packed && vb.is_rowmaj_nn_2d();
    const bool tn_layout = fast_packed && vb.is_rowmaj_tn_2d();

    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];

    // 1) Naive fallback (tiny or no recognized layout)
    if (tiny || (!nn_layout && !tn_layout)) {
        id<MTLComputePipelineState> pso = _impl->cache->get("matmul_view_f32");
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(b)   offset:0 atIndex:1];
        [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:2];
        [enc setBytes:&P length:sizeof(P) atIndex:3];

        // One thread per output element; pick a generic tg size
        MTLSize grid = MTLSizeMake(P.N, P.M, 1);
        NSUInteger w = pso.threadExecutionWidth; if (w == 0) w = 32;
        NSUInteger maxThreads = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger h = maxThreads / w; if (h == 0) h = 1;
        MTLSize tg = MTLSizeMake(w, h, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return;
    }

    // 2) Tiled path (NN or TN) — unified dispatch, only the PSO name differs
    const char* pso_name = nn_layout ? "matmul_tiled_f32" : "matmul_tiled_tn_f32";
    id<MTLComputePipelineState> pso = _impl->cache->get(pso_name);

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
    [enc setBuffer:metal::as_mtl(b)   offset:0 atIndex:1];
    [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:2];
    [enc setBytes:&P length:sizeof(P) atIndex:3];

    // Same tiling as the kernels use (TM=16, TN=16)
    constexpr unsigned TM = 16;
    constexpr unsigned TN = 16;
    NSUInteger groupsX = (P.N + TN - 1) / TN;
    NSUInteger groupsY = (P.M + TM - 1) / TM;
    MTLSize grid = MTLSizeMake(groupsX, groupsY, 1);
    MTLSize tg   = MTLSizeMake(TN, TM, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
}

// Broadcast
// void MetalBackend::broadcast(const Buffer& a, const backend::View& va,
//                              Buffer& out, const backend::View& vo) const {
//     if (out.size_bytes() == 0) return;

//     // Identity mapping: same shape, row-major, offset 0 => direct blit
//     if (va.is_contiguous() && vo.is_contiguous() && va.is_offset_zero() && vo.is_offset_zero() && same_shape(va, vo)) {
//         // Same device and same sizes guaranteed
//         copy(out, a); // GPU blit
//         return;
//     }

//     BroadcastParams P{};
//     pack_view32(va, P.in_v);
//     pack_view32(vo, P.out_v);
//     P.n = (unsigned int)out.numel();
//     P.pad3 = 0;

//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
//     id<MTLComputePipelineState> pso = _impl->cache->get("broadcast_view_f32");
//     _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
//         [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
//         [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
//         [enc setBytes:&P length:sizeof(P) atIndex:2];
//     });
//     [cb commit];
//     [cb waitUntilCompleted];
// }

// Permute
// void MetalBackend::permute(const Buffer& a, const backend::View& va,
//                            Buffer& out, const backend::View& vo,
//                            const std::vector<size_t>& axes) const {
//     if (out.size_bytes() == 0) return;

//     bool identity_axes = true;
//     for (size_t i=0;i<axes.size();++i) if (axes[i] != i) { identity_axes=false; break; }
//     if (identity_axes && va.is_contiguous() && vo.is_contiguous() && va.is_offset_zero() && vo.is_offset_zero() && same_shape(va, vo)) {
//         copy(out, a); // blit
//         return;
//     }

//     if (axes.size() != va.rank) throw std::runtime_error("Metal permute: axes size must match input rank");

//     PermuteParams P{};
//     pack_view32(va, P.in_v);
//     pack_view32(vo, P.out_v);
//     P.n = (unsigned int)out.numel();
//     for (size_t i=0;i<axes.size() && i<8;++i) P.axes[i] = (unsigned short)axes[i];

//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
//     id<MTLComputePipelineState> pso = _impl->cache->get("permute_view_f32");
//     _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
//         [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
//         [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
//         [enc setBytes:&P length:sizeof(P) atIndex:2];
//     });
//     [cb commit];
//     [cb waitUntilCompleted];
// }

// // Slice forward
// void MetalBackend::slice_forward(const Buffer& a, const backend::View& va,
//                                  Buffer& out, const backend::View& vo,
//                                  const std::vector<size_t>& begin,
//                                  const std::vector<size_t>& end,
//                                  const std::vector<size_t>& step) const {
//     if (out.size_bytes() == 0) return;
//     if (begin.size() != va.rank) throw std::runtime_error("Metal slice_forward: begin rank mismatch");

//     bool is_identity = va.is_contiguous() && vo.is_contiguous() && va.is_offset_zero() && vo.is_offset_zero() && same_shape(va, vo);
//     if (is_identity) {
//         bool begin_zero = true, step_one = true;
//         for (size_t i=0;i<begin.size();++i) if (begin[i] != 0) { begin_zero=false; break; }
//         for (size_t i=0;i<step.size();++i) if (step[i] != 1) { step_one=false; break; }
//         if (begin_zero && (step.empty() || step_one)) {
//             copy(out, a); // blit
//             return;
//         }
//     }

//     // Fallback to general view copy using slice_forward_view_f32 kernel (forward path)
//     SliceParams P{};
//     pack_view32(va, P.in_v);
//     pack_view32(vo, P.out_v);
//     P.n = (unsigned int)out.numel();
//     for (size_t i=0;i<begin.size() && i<8;++i) P.begin[i] = (unsigned int)begin[i];
//     for (size_t i=0;i<step.size()  && i<8;++i) P.step[i]  = (unsigned int)step[i];

//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
//     id<MTLComputePipelineState> pso = _impl->cache->get("slice_forward_view_f32");
//     _impl->dispatch_1d(pso, cb, out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
//         [enc setBuffer:metal::as_mtl(a)   offset:0 atIndex:0];
//         [enc setBuffer:metal::as_mtl(out) offset:0 atIndex:1];
//         [enc setBytes:&P length:sizeof(P) atIndex:2];
//     });
//     [cb commit];
//     [cb waitUntilCompleted];
// }

// // Slice backward scatter-add
// void MetalBackend::slice_backward_scatter_add(const Buffer& grad_out, const backend::View& vgo,
//                                               Buffer& grad_in,  const backend::View& vgi,
//                                               const std::vector<size_t>& begin,
//                                               const std::vector<size_t>& /*end*/,
//                                               const std::vector<size_t>& step) const {
//     if (grad_out.size_bytes() == 0) return;
//     if (begin.size() != vgi.rank) throw std::runtime_error("Metal slice_backward_scatter_add: begin rank mismatch");

//     SliceParams P{};
//     pack_view32(vgi, P.in_v);
//     pack_view32(vgo, P.out_v);
//     P.n = (unsigned int)grad_out.numel();
//     for (size_t i=0;i<begin.size() && i<8;++i) {
//         P.begin[i] = (unsigned int)begin[i];
//         P.step[i]  = (unsigned int)((i < step.size()) ? step[i] : 1);
//     }
//     id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
//     id<MTLComputePipelineState> pso = _impl->cache->get("slice_backward_scatter_add_view_f32");
//     _impl->dispatch_1d(pso, cb, grad_out.numel(), [&](id<MTLComputeCommandEncoder> enc) {
//         [enc setBuffer:metal::as_mtl(grad_out) offset:0 atIndex:0];
//         [enc setBuffer:metal::as_mtl(grad_in)  offset:0 atIndex:1];
//         [enc setBytes:&P length:sizeof(P) atIndex:2];
//     });
//     [cb commit];
//     [cb waitUntilCompleted];
// }

// Copy view
void MetalBackend::copy_view(const Buffer& src, const backend::View& vs,
                             Buffer& dst, const backend::View& vd) const {
    if (dst.size_bytes() == 0) return;

    // If both views are identity (row-major with offset 0) and same shape, just blit:
    if (vs.is_identity() && vd.is_identity() && same_shape(vs, vd)) {
        copy(dst, src);
        return;
    }

    CopyViewParams P{};
    pack_view32(vs, P.src_v);
    pack_view32(vd, P.dst_v);
    P.n = (unsigned int)dst.numel();

    id<MTLCommandBuffer> cb = [_impl->queue commandBuffer];
    id<MTLComputePipelineState> pso = _impl->cache->get("copy_view_f32");
    _impl->dispatch_1d(pso, cb, dst.numel(), [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:metal::as_mtl(src) offset:0 atIndex:0];
        [enc setBuffer:metal::as_mtl(dst) offset:0 atIndex:1];
        [enc setBytes:&P length:sizeof(P) atIndex:2];
    });
    [cb commit];
    [cb waitUntilCompleted];
}

} // namespace metal
} // namespace backend
} // namespace cppgrad
