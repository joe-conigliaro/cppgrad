// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/metal/metal_backend.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/metal/metal_execution_context.h"
#include "cppgrad/backend/metal/metal_kernel_cache.h"
#include "cppgrad/backend/metal/metal_shared_structs.h"
#include "cppgrad/backend/metal/metal_utils.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/common/dtype.h"
#include "cppgrad/utils/rng.h"
#import <Metal/Metal.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cppgrad::backend::metal {

// Pack backend::View -> View32
static inline void pack_view32(const backend::View &v, View32 &out) {
    out.rank = static_cast<unsigned short>(v.rank);
    out.pad = 0;
    out.offset = static_cast<unsigned int>(v.offset);
    out.flags = static_cast<unsigned int>(v.flags);
    for (int i = 0; i < 8; ++i) {
        out.shape[i] = (i < static_cast<int>(v.rank)) ? static_cast<unsigned int>(v.shape[i]) : 0u;
        out.strides[i] = (i < static_cast<int>(v.rank)) ? static_cast<unsigned int>(v.strides[i]) : 0u;
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
        : device((__bridge id<MTLDevice>)native_device), queue((__bridge id<MTLCommandQueue>)native_queue),
          cache(std::make_unique<metal::MetalKernelCache>(device)), exec_ctx(ec) {}

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
        if (buf.dtype() != common::DType::FLOAT32) {
            throw std::runtime_error(std::string("MetalBackend::submit_fill: unsupported dtype ") +
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

    void submit_unary_op(ir::UnaryOpType op_type, const Buffer &a, const backend::View &va, Buffer &out,
                         const backend::View &vo) const {
        if (out.size_bytes() == 0)
            return;
        UnaryParams P{};
        pack_view32(va, P.in_v);
        pack_view32(vo, P.out_v);
        P.n = (unsigned int)out.numel();
        P.op = (unsigned short)op_type;

        ComputeWork work;
        work.pso = cache->get(out.dtype() == common::DType::BFLOAT16 ? "unary_view_bf16" : "unary_view_f32");
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(2, &P, sizeof(P));
        set_linear(work, out.numel());
        encode_submit(work);
    }

    void submit_binary_op(ir::BinaryOpType op_type, const Buffer &a, const backend::View &va, const Buffer &b,
                          const backend::View &vb, Buffer &out, const backend::View &vo) const {
        if (out.size_bytes() == 0)
            return;
        BinaryParams P{};
        pack_view32(va, P.a_v);
        pack_view32(vb, P.b_v);
        pack_view32(vo, P.o_v);
        P.n = (unsigned int)out.numel();
        P.op = (unsigned short)op_type;

        ComputeWork work;
        work.pso = cache->get(out.dtype() == common::DType::BFLOAT16 ? "binary_view_bf16" : "binary_view_f32");
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(b), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(3, &P, sizeof(P));
        set_linear(work, out.numel());
        encode_submit(work);
    }

    void submit_reduce_op(ir::ReduceOpType op_type, const Buffer &a, const backend::View &va, Buffer &out,
                          const backend::View &vo, const std::vector<int> &axes, bool keep_dims) const {
        if (out.size_bytes() == 0)
            return;
        const unsigned short op = (op_type == ir::ReduceOpType::MAX) ? (unsigned short)1 : (unsigned short)0;

        bool last_only = (axes.size() == 1) && ((axes[0] == (int)va.rank - 1) || (axes[0] == -1));
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
            tg_width = MIN(tg_width, (NSUInteger)[work.pso maxTotalThreadsPerThreadgroup]);
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

    void submit_matmul(const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb, Buffer &out,
                       const backend::View &vo) const {
        if (out.size_bytes() == 0)
            return;
        if (va.rank < 2 || vb.rank < 2 || vo.rank < 2)
            throw std::runtime_error("Metal matmul: views must have rank >= 2");
        if (va.rank != vb.rank || va.rank != vo.rank)
            throw std::runtime_error("Metal matmul: all views must have same rank");

        const unsigned rank = va.rank;
        const unsigned M = va.shape[rank - 2];
        const unsigned K = va.shape[rank - 1];
        const unsigned N = vb.shape[rank - 1];

        // Compute batch count and per-batch strides
        unsigned batch = 1;
        for (unsigned d = 0; d < rank - 2; ++d)
            batch *= va.shape[d];

        // Collapsed batch stride: element offset per unit of linear batch index
        unsigned a_batch_stride = 0, b_batch_stride = 0, o_batch_stride = 0;
        for (unsigned d = 0; d < rank - 2; ++d) {
            unsigned tail = 1;
            for (unsigned e = d + 1; e < rank - 2; ++e)
                tail *= va.shape[e];
            a_batch_stride += va.strides[d] * tail;
            b_batch_stride += vb.strides[d] * tail;
            o_batch_stride += vo.strides[d] * tail;
        }

        MatmulParams P{};
        pack_view32(va, P.a_v);
        pack_view32(vb, P.b_v);
        pack_view32(vo, P.o_v);
        P.M = M;
        P.K = K;
        P.N = N;
        P.batch = batch;
        P.a_batch_stride = a_batch_stride;
        P.b_batch_stride = b_batch_stride;
        P.o_batch_stride = o_batch_stride;
        // NOTE: do NOT clobber a_v/b_v/o_v strides. The naive view kernel decomposes the linear batch
        // index over the leading dims using the full (unmodified) shape/strides, and uses strides
        // [rank-2]/[rank-1] for the matrix. A single collapsed batch stride is wrong for >1 batch dim.

        const bool tiny = (P.M < 8 || P.N < 8 || P.K < 8);
        // For N-D views, fall back to naive kernel (safe for arbitrary strides).
        // Tiled kernels are only used for proper 2D rank-2 views.
        const bool is_2d = (rank == 2);
        const bool fast_packed = is_2d && va.is_rowmaj_nn_2d() && vo.is_rowmaj_nn_2d();
        const bool nn_layout = fast_packed && vb.is_rowmaj_nn_2d();
        const bool tn_layout = fast_packed && vb.is_rowmaj_tn_2d();
        // bf16 weights use the generic naive kernel (the tiled kernels are fp32-only).
        const bool b_bf16 = (b.dtype() == common::DType::BFLOAT16);

        ComputeWork work;
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(b), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(3, &P, sizeof(P));

        if (b_bf16 || tiny || (!nn_layout && !tn_layout)) {
            work.pso = cache->get(b_bf16 ? "matmul_view_bf16w" : "matmul_view_f32");
            NSUInteger w = work.pso.threadExecutionWidth;
            if (w == 0)
                w = 32;
            NSUInteger h = [work.pso maxTotalThreadsPerThreadgroup] / w;
            if (h == 0)
                h = 1;
            work.useThreadgroups = false;
            work.grid = MTLSizeMake(P.N, P.M, batch);
            work.threadsPerThreadgroup = MTLSizeMake(w, h, 1);
        } else {
            constexpr unsigned TM = 16, TN = 16;
            work.pso = cache->get(nn_layout ? "matmul_tiled_f32" : "matmul_tiled_tn_f32");
            work.useThreadgroups = true;
            work.grid = MTLSizeMake((P.N + TN - 1) / TN, (P.M + TM - 1) / TM, batch);
            work.threadsPerThreadgroup = MTLSizeMake(TN, TM, 1);
        }
        encode_submit(work);
    }

    void submit_copy_view(const Buffer &src, const backend::View &vs, Buffer &dst, const backend::View &vd) const {
        if (dst.size_bytes() == 0)
            return;

        // A dtype-converting copy (e.g. fp32 activations -> bf16 KV cache) needs the elementwise
        // kernel; a raw byte blit would reinterpret the bits. Only same-dtype copies may take the blit.
        const bool convert = (src.dtype() != dst.dtype());

        // Fast path: identity mapping => synchronous blit (same-dtype only).
        if (!convert && vs.is_identity() && vd.is_identity() && same_shape(vs, vd)) {
            sync_copy(dst, src);
            return;
        }

        // Pick the (src,dst) dtype-specialized kernel. Only fp32<->bf16 conversion is wired up so far.
        const char *kernel_name = "copy_view_f32";
        if (convert) {
            if (src.dtype() == common::DType::FLOAT32 && dst.dtype() == common::DType::BFLOAT16)
                kernel_name = "copy_view_f32_to_bf16";
            else if (src.dtype() == common::DType::BFLOAT16 && dst.dtype() == common::DType::FLOAT32)
                kernel_name = "copy_view_bf16_to_f32";
            else
                throw std::runtime_error("copy_view: unsupported dtype conversion");
        } else if (dst.dtype() == common::DType::BFLOAT16) {
            kernel_name = "copy_view_bf16"; // same-dtype bf16 strided copy (slice/reshape/materialize)
        }

        CopyViewParams P{};
        pack_view32(vs, P.src_v);
        pack_view32(vd, P.dst_v);
        // Iterate over the destination view region, not the destination buffer. For an in-place
        // write into a sub-region of a larger buffer (e.g. cache_update into a preallocated
        // [1,max_len,nKV,D] KV cache) dst.numel() is the whole buffer, which would launch excess
        // threads whose out-of-shape coords index past the source buffer -> GPU page fault.
        P.n = (unsigned int)vd.numel;

        ComputeWork work;
        work.pso = cache->get(kernel_name);
        work.buffers.push_back({as_mtl(src), 0});
        work.buffers.push_back({as_mtl(dst), 0});
        work.add_bytes(2, &P, sizeof(P));
        set_linear(work, vd.numel);
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

    void submit_matmul_quant(const Buffer &a, const Buffer &qweight, const Buffer &scales, const Buffer &biases,
                             Buffer &out, size_t M, size_t N, size_t K, int group_size) const {
        if (out.size_bytes() == 0)
            return;
        uint32_t n = (uint32_t)N, k = (uint32_t)K, gs = (uint32_t)group_size;
        ComputeWork work;

        // Decode path (M == 1): coalesced simdgroup GEMV -- one 32-lane threadgroup per output column.
        if (M == 1) {
            work.pso = cache->get("matmul_quant_gemv_f32");
            work.buffers.push_back({as_mtl(a), 0});
            work.buffers.push_back({as_mtl(qweight), 0});
            work.buffers.push_back({as_mtl(scales), 0});
            work.buffers.push_back({as_mtl(biases), 0});
            work.buffers.push_back({as_mtl(out), 0});
            work.add_bytes(5, &n, sizeof(uint32_t));
            work.add_bytes(6, &k, sizeof(uint32_t));
            work.add_bytes(7, &gs, sizeof(uint32_t));
            work.useThreadgroups = true;
            work.grid = MTLSizeMake(N, 1, 1);                   // one threadgroup per output column
            work.threadsPerThreadgroup = MTLSizeMake(32, 1, 1); // one simdgroup
            encode_submit(work);
            return;
        }

        // Prefill / general path (M > 1): register-blocked, threadgroup-memory-tiled GEMM
        // (matmul_quant_gemm_tiled_f32) -- 64x64 tile, 256 threads x 4x4 micro-tile, weight tile
        // dequantized ONCE into shared memory and reused across all 64 rows. (A simdgroup_matrix variant
        // was tried and dropped: the 8-bit GEMM is dequant/memory-access bound, not matmul-bound, so the
        // hardware matrix units gave no speedup, and a half-precision path lost too much accuracy.)
        constexpr uint32_t QT_BM = 64, QT_BN = 64, QT_RM = 4, QT_RN = 4; // must match #defines in the .metal kernel
        uint32_t m = (uint32_t)M;
        // Pick the activation-dtype variant. Weights are always 8-bit; only A (in) and OUT vary, for the
        // bf16 FFN activation path (gate/up: f32->bf16; down: bf16->f32). fp32->fp32 is the default.
        const bool a_bf16 = (a.dtype() == common::DType::BFLOAT16);
        const bool out_bf16 = (out.dtype() == common::DType::BFLOAT16);
        const char *qgemm = "matmul_quant_gemm_tiled_f32";
        if (!a_bf16 && out_bf16)
            qgemm = "matmul_quant_gemm_tiled_f32a_bf16o";
        else if (a_bf16 && !out_bf16)
            qgemm = "matmul_quant_gemm_tiled_bf16a_f32o";
        else if (a_bf16 && out_bf16)
            throw std::runtime_error("quant GEMM: bf16->bf16 variant not built");
        work.pso = cache->get(qgemm);
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(qweight), 0});
        work.buffers.push_back({as_mtl(scales), 0});
        work.buffers.push_back({as_mtl(biases), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(5, &m, sizeof(uint32_t));
        work.add_bytes(6, &n, sizeof(uint32_t));
        work.add_bytes(7, &k, sizeof(uint32_t));
        work.add_bytes(8, &gs, sizeof(uint32_t));
        work.useThreadgroups = true;
        work.grid = MTLSizeMake((N + QT_BN - 1) / QT_BN, (M + QT_BM - 1) / QT_BM, 1); // (N-tile, M-tile)
        work.threadsPerThreadgroup = MTLSizeMake(QT_BN / QT_RN, QT_BM / QT_RM, 1);    // 16x16 = 256
        encode_submit(work);
    }

    void submit_flash_attention(const Buffer &q, const Buffer &k, const Buffer &v, Buffer &out, size_t B, size_t S,
                                size_t nH, size_t Dh, size_t KV, size_t nKV, float scale, int n_rep, bool causal,
                                size_t q_offset) const {
        if (out.size_bytes() == 0)
            return;
        FlashParams P;
        P.B = (uint)B;
        P.S = (uint)S;
        P.nH = (uint)nH;
        P.Dh = (uint)Dh;
        P.KV = (uint)KV;
        P.nKV = (uint)nKV;
        P.n_rep = (uint)n_rep;
        P.causal = causal ? 1u : 0u;
        P.q_offset = (uint)q_offset;
        P.scale = scale;
        // bf16 KV cache uses the bf16-read variant (fp32 accumulate); fp32 cache uses the f32 kernel.
        const bool kv_bf16 = (k.dtype() == common::DType::BFLOAT16);
        ComputeWork work;
        work.pso = cache->get(kv_bf16 ? "flash_attention_bf16kv" : "flash_attention_f32");
        work.buffers.push_back({as_mtl(q), 0});
        work.buffers.push_back({as_mtl(k), 0});
        work.buffers.push_back({as_mtl(v), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(4, &P, sizeof(FlashParams));
        work.useThreadgroups = true;
        work.grid = MTLSizeMake(nH, S, B);                  // one threadgroup per (head, query, batch)
        work.threadsPerThreadgroup = MTLSizeMake(32, 1, 1); // one simdgroup splits head_dim
        encode_submit(work);
    }

    void submit_rms_norm(const Buffer &x, const Buffer &w, Buffer &out, size_t rows, size_t D, float eps) const {
        if (out.size_bytes() == 0)
            return;
        ComputeWork work;
        work.pso = cache->get("rms_norm_f32");
        work.buffers.push_back({as_mtl(x), 0});
        work.buffers.push_back({as_mtl(w), 0});
        work.buffers.push_back({as_mtl(out), 0});
        uint32_t d_u32 = (uint32_t)D;
        work.add_bytes(3, &d_u32, sizeof(uint32_t));
        work.add_bytes(4, &eps, sizeof(float));
        NSUInteger tpg = MIN((NSUInteger)work.pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        tpg = MIN(tpg, (NSUInteger)D);
        if (tpg == 0)
            tpg = 1;
        work.useThreadgroups = true;
        work.grid = MTLSizeMake(rows, 1, 1); // one threadgroup per row
        work.threadsPerThreadgroup = MTLSizeMake(tpg, 1, 1);
        work.threadgroupMemoryLength = tpg * sizeof(float); // smem for the reduction
        encode_submit(work);
    }

    void submit_pairwise_decay(const Buffer &G, Buffer &out, size_t BH, size_t L) const {
        if (out.size_bytes() == 0)
            return;
        ComputeWork work;
        work.pso = cache->get("pairwise_decay_f32");
        work.buffers.push_back({as_mtl(G), 0});
        work.buffers.push_back({as_mtl(out), 0});
        uint32_t l_u32 = (uint32_t)L;
        work.add_bytes(2, &l_u32, sizeof(uint32_t));
        set_linear(work, BH * L * L); // one thread per output element
        encode_submit(work);
    }

    void submit_delta_decay_mask(const Buffer &scores, const Buffer &Dexp, const Buffer &beta, Buffer &out, size_t BH,
                                 size_t L, bool strict, bool apply_beta) const {
        if (out.size_bytes() == 0)
            return;
        ComputeWork work;
        work.pso = cache->get("delta_decay_mask_f32");
        work.buffers.push_back({as_mtl(scores), 0});
        work.buffers.push_back({as_mtl(Dexp), 0});
        work.buffers.push_back({as_mtl(beta), 0});
        work.buffers.push_back({as_mtl(out), 0});
        uint32_t l_u32 = (uint32_t)L, st = strict ? 1u : 0u, ab = apply_beta ? 1u : 0u;
        work.add_bytes(4, &l_u32, sizeof(uint32_t));
        work.add_bytes(5, &st, sizeof(uint32_t));
        work.add_bytes(6, &ab, sizeof(uint32_t));
        set_linear(work, BH * L * L);
        encode_submit(work);
    }

    void submit_fma(const Buffer &a, const Buffer &b, const Buffer &c, Buffer &out, size_t n, size_t b_group) const {
        if (out.size_bytes() == 0)
            return;
        ComputeWork work;
        work.pso = cache->get("fma_f32");
        work.buffers.push_back({as_mtl(a), 0});
        work.buffers.push_back({as_mtl(b), 0});
        work.buffers.push_back({as_mtl(c), 0});
        work.buffers.push_back({as_mtl(out), 0});
        uint32_t n_u32 = (uint32_t)n, g_u32 = (uint32_t)b_group;
        work.add_bytes(4, &n_u32, sizeof(uint32_t));
        work.add_bytes(5, &g_u32, sizeof(uint32_t));
        set_linear(work, n);
        encode_submit(work);
    }

    void submit_gather_op(const Buffer &table, const Buffer &indices, Buffer &out, size_t V, size_t D) const {
        if (out.size_bytes() == 0)
            return;
        const size_t N = indices.numel();
        ComputeWork work;
        // The embedding table may be bf16 (dequantized weights kept compact) -> fp32 output.
        const bool table_bf16 = (table.dtype() == common::DType::BFLOAT16);
        work.pso = cache->get(table_bf16 ? "gather_bf16_f32" : "gather_f32");
        work.buffers.push_back({as_mtl(table), 0});
        work.buffers.push_back({as_mtl(indices), 0});
        work.buffers.push_back({as_mtl(out), 0});
        uint32_t v_u32 = static_cast<uint32_t>(V);
        uint32_t d_u32 = static_cast<uint32_t>(D);
        work.add_bytes(3, &v_u32, sizeof(uint32_t));
        work.add_bytes(4, &d_u32, sizeof(uint32_t));
        set_linear(work, N);
        encode_submit(work);
    }

    void submit_concat_op(const std::vector<const Buffer *> &inputs, const std::vector<backend::View> &input_views,
                          Buffer &out, const backend::View &out_view, int axis) const {
        if (out.size_bytes() == 0)
            return;
        const size_t rank = out_view.rank;
        const uint32_t u_axis = static_cast<uint32_t>(axis < 0 ? axis + static_cast<int>(rank) : axis);

        // Build CumParams
        struct {
            View32 in_views[2];
            View32 out_v;
            uint32_t n;
            uint32_t num_inputs;
            uint32_t axis;
            uint32_t cum_sizes[3];
        } P{};

        pack_view32(input_views[0], P.in_views[0]);
        pack_view32(input_views[1], P.in_views[1]);
        pack_view32(out_view, P.out_v);
        P.n = static_cast<uint32_t>(out.numel());
        P.num_inputs = static_cast<uint32_t>(inputs.size());
        P.axis = u_axis;
        P.cum_sizes[0] = 0;
        P.cum_sizes[1] = input_views[0].shape[axis < 0 ? axis + static_cast<int>(rank) : axis];
        P.cum_sizes[2] = P.cum_sizes[1] + input_views[1].shape[axis < 0 ? axis + static_cast<int>(rank) : axis];

        ComputeWork work;
        work.pso = cache->get("concat_f32");
        work.buffers.push_back({as_mtl(*inputs[0]), 0});
        work.buffers.push_back({as_mtl(*inputs[1]), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(3, &P, sizeof(P));
        set_linear(work, out.numel());
        encode_submit(work);
    }

    void submit_gather_axis_op(const Buffer &tensor, const backend::View &tv, const Buffer &indices, Buffer &out,
                               const backend::View &ov, int axis) const {
        if (out.numel() == 0)
            return;

        struct {
            View32 tensor_v;
            View32 out_v;
            uint32_t n;
            uint32_t axis;
        } P{};

        pack_view32(tv, P.tensor_v);
        pack_view32(ov, P.out_v);
        P.n = static_cast<uint32_t>(ov.numel);
        P.axis = static_cast<uint32_t>(axis);

        ComputeWork work;
        work.pso = cache->get("gather_axis_f32");
        work.buffers.push_back({as_mtl(tensor), 0});
        work.buffers.push_back({as_mtl(indices), 0});
        work.buffers.push_back({as_mtl(out), 0});
        work.add_bytes(3, &P, sizeof(P));
        set_linear(work, ov.numel);
        encode_submit(work);
    }

    void submit_scatter_axis_op(const Buffer &base, const backend::View &bv, const Buffer &values,
                                const backend::View &vv, const Buffer &indices, Buffer &out, const backend::View &ov,
                                int axis) const {
        if (out.numel() == 0)
            return;

        // Step 1: copy base to output
        {
            struct {
                View32 base_v;
                View32 values_v;
                View32 out_v;
                uint32_t nval;
                uint32_t axis;
            } P{};
            pack_view32(bv, P.base_v);
            pack_view32(ov, P.out_v);

            ComputeWork work;
            work.pso = cache->get("scatter_base_copy_f32");
            work.buffers.push_back({as_mtl(base), 0});
            work.buffers.push_back({as_mtl(out), 0});
            work.add_bytes(2, &P, sizeof(P));
            set_linear(work, bv.numel);
            encode_submit(work);
        }

        // Step 2: scatter values at indexed positions
        {
            struct {
                View32 base_v;
                View32 values_v;
                View32 out_v;
                uint32_t nval;
                uint32_t axis;
            } P{};
            pack_view32(bv, P.base_v);
            pack_view32(vv, P.values_v);
            pack_view32(ov, P.out_v);
            P.nval = static_cast<uint32_t>(vv.numel);
            P.axis = static_cast<uint32_t>(axis);

            ComputeWork work;
            work.pso = cache->get("scatter_axis_f32");
            work.buffers.push_back({as_mtl(base), 0});
            work.buffers.push_back({as_mtl(values), 0});
            work.buffers.push_back({as_mtl(indices), 0});
            work.buffers.push_back({as_mtl(out), 0});
            work.add_bytes(4, &P, sizeof(P));
            set_linear(work, vv.numel);
            encode_submit(work);
        }
    }
};

MetalBackend::MetalBackend(void *native_device, void *native_queue, MetalExecutionContext *ec)
    : _impl(std::make_unique<Impl>(native_device, native_queue, ec)) {}
MetalBackend::~MetalBackend() = default;

void MetalBackend::flush_pending() const {
    if (_impl->exec_ctx)
        _impl->exec_ctx->flush();
}

void MetalBackend::set_buffer_debug_label(const Buffer &buf, const char *label) const {
    id<MTLBuffer> mb = as_mtl(buf);
    if (mb && label)
        mb.label = [NSString stringWithUTF8String:label];
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
void MetalBackend::unary_op(ir::UnaryOpType op_type, const Buffer &a, const backend::View &va, Buffer &out,
                            const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_unary_op(op_type, a, va, out, vo);
}

// Binary (stride-aware)
void MetalBackend::binary_op(ir::BinaryOpType op_type, const Buffer &a, const backend::View &va, const Buffer &b,
                             const backend::View &vb, Buffer &out, const backend::View &vo) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_binary_op(op_type, a, va, b, vb, out, vo);
}

// Reduce (fast path + general)
void MetalBackend::reduce_op(ir::ReduceOpType op_type, const Buffer &a, const backend::View &va, Buffer &out,
                             const backend::View &vo, const std::vector<int> &axes, bool keep_dims) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_reduce_op(op_type, a, va, out, vo, axes, keep_dims);
}

// Matmul
void MetalBackend::matmul(const Buffer &a, const backend::View &va, const Buffer &b, const backend::View &vb,
                          Buffer &out, const backend::View &vo) const {
    _impl->submit_matmul(a, va, b, vb, out, vo);
}

void MetalBackend::flash_attention(const Buffer &q, const Buffer &k, const Buffer &v, Buffer &out, size_t B, size_t S,
                                   size_t nH, size_t Dh, size_t KV, size_t nKV, float scale, int n_rep, bool causal,
                                   size_t q_offset) const {
    _impl->submit_flash_attention(q, k, v, out, B, S, nH, Dh, KV, nKV, scale, n_rep, causal, q_offset);
}

void MetalBackend::rms_norm(const Buffer &x, const Buffer &w, Buffer &out, size_t rows, size_t D, float eps) const {
    _impl->submit_rms_norm(x, w, out, rows, D, eps);
}

void MetalBackend::pairwise_decay(const Buffer &G, Buffer &out, size_t BH, size_t L) const {
    _impl->submit_pairwise_decay(G, out, BH, L);
}

void MetalBackend::delta_decay_mask(const Buffer &scores, const Buffer &Dexp, const Buffer &beta, Buffer &out,
                                    size_t BH, size_t L, bool strict, bool apply_beta) const {
    _impl->submit_delta_decay_mask(scores, Dexp, beta, out, BH, L, strict, apply_beta);
}

void MetalBackend::fma(const Buffer &a, const Buffer &b, const Buffer &c, Buffer &out, size_t n, size_t b_group) const {
    _impl->submit_fma(a, b, c, out, n, b_group);
}

void MetalBackend::quantized_matmul(const Buffer &a, const Buffer &qweight, const std::vector<const Buffer *> &aux,
                                    Buffer &out, size_t M, size_t N, size_t K, const ir::QuantParams &params) const {
    if ((int)aux.size() != ir::aux_buffer_count(params.scheme))
        throw std::runtime_error("MetalBackend::quantized_matmul: wrong aux buffer count for scheme");
    switch (params.scheme) {
    case ir::QuantScheme::MLX_AFFINE:
        // TODO(4-bit): the kernels (matmul_quant_*_f32) hardcode 8-bit unpacking (4 codes/u32).
        // For bits==4 (8 codes/u32, pack_factor 8) add a code-width branch in the unpack (shift by
        // 4*(k&7), mask 0xF) -- the dequant math + {scales,biases} layout are otherwise identical.
        if (params.bits != 8)
            throw std::runtime_error(
                "MetalBackend::quantized_matmul: MLX_AFFINE only supports bits=8 (4-bit kernel is TODO)");
        _impl->submit_matmul_quant(a, qweight, *aux[0], *aux[1], out, M, N, K, params.group_size);
        return;
    }
    throw std::runtime_error("MetalBackend::quantized_matmul: unsupported quant scheme");
}

// Copy view
void MetalBackend::copy_view(const Buffer &src, const backend::View &vs, Buffer &dst, const backend::View &vd) const {
    if (dst.size_bytes() == 0)
        return;
    _impl->submit_copy_view(src, vs, dst, vd);
}

void MetalBackend::gather_op(const Buffer &table, const Buffer &indices, Buffer &out, size_t V, size_t D) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_gather_op(table, indices, out, V, D);
}

void MetalBackend::concat_op(const std::vector<const Buffer *> &inputs, const std::vector<backend::View> &input_views,
                             Buffer &out, const backend::View &out_view, int axis) const {
    if (out.size_bytes() == 0)
        return;
    _impl->submit_concat_op(inputs, input_views, out, out_view, axis);
}

void MetalBackend::gather_axis_op(const Buffer &tensor, const backend::View &tv, const Buffer &indices, Buffer &out,
                                  const backend::View &ov, int axis) const {
    _impl->submit_gather_axis_op(tensor, tv, indices, out, ov, axis);
}

void MetalBackend::scatter_axis_op(const Buffer &base, const backend::View &bv, const Buffer &values,
                                   const backend::View &vv, const Buffer &indices, Buffer &out, const backend::View &ov,
                                   int axis) const {
    _impl->submit_scatter_axis_op(base, bv, values, vv, indices, out, ov, axis);
}

} // namespace cppgrad::backend::metal
