// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "cppgrad/backend/cpu/cpu_backend.h"
#include "cppgrad/backend/cpu/cpu_kernels.h"
#include "cppgrad/backend/cpu/cpu_quant_kernels.h"
#include "cppgrad/backend/cpu/dtype_dispatch.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/utils/rng.h"

namespace cppgrad::backend::cpu {

// RNG helpers
static inline uint32_t lcg(uint32_t x) { return 1664525u * x + 1013904223u; }
static inline float u01_from_state(uint32_t s) { return float(s & 0xFFFFFFu) / float(0xFFFFFFu); }
static inline uint32_t next_u32_from_global() { auto& gen = cppgrad::utils::global_rng(); return static_cast<uint32_t>(gen()); }

static inline bool same_shape(const backend::View& a, const backend::View& b) {
    if (a.rank != b.rank) return false;
    for (uint32_t i=0;i<a.rank;++i) if (a.shape[i] != b.shape[i]) return false;
    return true;
}

// Buffer operations
void CPUBackend::fill(Buffer& buf, double value) const {
    cpu::dispatch_dtype(buf.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::fill_kernel<T>(buf, static_cast<T>(value));
    }));
}

// Random generation
void CPUBackend::rand_uniform(Buffer& out, float min, float max) const {
    // CPU RNG is independent of dtype today; we write as float to the buffer type.
    // For integer types, we convert by static_cast<T>.
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        T* data = static_cast<T*>(out.data());
        const uint32_t n = static_cast<uint32_t>(out.numel());
        if (n == 0) return;
        const uint32_t seed = next_u32_from_global();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t s = lcg(seed ^ (i + 1u));
            float r = u01_from_state(s);
            float v = min + r * (max - min);
            data[i] = static_cast<T>(v);
        }
    }));
}

void CPUBackend::rand_normal(Buffer& out, float mean, float stddev) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        T* data = static_cast<T*>(out.data());
        const uint32_t n = static_cast<uint32_t>(out.numel());
        if (n == 0) return;
        const uint32_t seed = next_u32_from_global();
        for (uint32_t i = 0; i < n; i += 2) {
            uint32_t s1 = lcg(seed ^ (i + 1u));
            uint32_t s2 = lcg(seed ^ (i + 2u));
            float u1 = std::max(u01_from_state(s1), 1e-7f);
            float u2 = std::max(u01_from_state(s2), 1e-7f);
            float r = std::sqrt(-2.0f * std::log(u1));
            float th = 2.0f * float(M_PI) * u2;
            float z0 = r * std::cos(th), z1 = r * std::sin(th);
            data[i] = static_cast<T>(mean + stddev * z0);
            if (i + 1 < n) data[i + 1] = static_cast<T>(mean + stddev * z1);
        }
    }));
}

// Compute ops (view-aware)

void CPUBackend::unary_op(ir::UnaryOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        auto fn = [&](T x)->T {
            switch(op_type) {
                case ir::UnaryOpType::RELU: return x > T(0) ? x : T(0);
                case ir::UnaryOpType::EXP:  return static_cast<T>(std::exp(static_cast<double>(x)));
                case ir::UnaryOpType::LOG:  return static_cast<T>(std::log(static_cast<double>(x)));
                case ir::UnaryOpType::NEG:  return -x;
                case ir::UnaryOpType::TANH: return static_cast<T>(std::tanh(static_cast<double>(x)));
                case ir::UnaryOpType::SIN:  return static_cast<T>(std::sin(static_cast<double>(x)));
                case ir::UnaryOpType::COS:  return static_cast<T>(std::cos(static_cast<double>(x)));
                case ir::UnaryOpType::SILU: { double s = 1.0/(1.0+std::exp(-static_cast<double>(x))); return static_cast<T>(static_cast<double>(x)*s); }
                case ir::UnaryOpType::SIGMOID: return static_cast<T>(1.0/(1.0+std::exp(-static_cast<double>(x))));
            }
            return x;
        };
        // Global fast path (flat)
        if (va.is_identity() && vo.is_identity() && same_shape(va, vo)) {
            cpu::unary_op_kernel<T>(a, out, fn); // flat loop
            return;
        }
        cpu::unary_view_kernel<T>(a, va, out, vo, fn);
    }));
}

void CPUBackend::binary_op(ir::BinaryOpType op_type, const Buffer& a, const backend::View& va, const Buffer& b, const backend::View& vb, Buffer& out, const backend::View& vo) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        auto fn = [&](T x, T y)->T {
            switch (op_type) {
                case ir::BinaryOpType::ADD:    return x + y;
                case ir::BinaryOpType::SUB:    return x - y;
                case ir::BinaryOpType::MUL:    return x * y;
                case ir::BinaryOpType::DIV:    return x / y;
                case ir::BinaryOpType::POW:    return static_cast<T>(std::pow(static_cast<double>(x), static_cast<double>(y)));
                case ir::BinaryOpType::CMP_EQ: return T(x == y);
                case ir::BinaryOpType::CMP_GT: return T(x > y);
                case ir::BinaryOpType::MIN:    return std::min(x, y);
                case ir::BinaryOpType::MAX:    return std::max(x, y);
            }
            return T(0);
        };
        if (va.is_identity() && vb.is_identity() && vo.is_identity() && same_shape(va, vb) && same_shape(va, vo)) {
            // build std::vector<size_t> once for row-major flat helpers
            std::vector<size_t> shape; shape.reserve(va.rank);
            for (uint32_t i=0;i<va.rank;++i) shape.push_back(va.shape[i]);
            cpu::binary_op_kernel<T>(a, b, out, fn, shape, shape, shape);
            return;
        }
        cpu::binary_view_kernel<T>(a, va, b, vb, out, vo, fn);
    }));
}

void CPUBackend::reduce_op(ir::ReduceOpType op_type, const Buffer& a, const backend::View& va, Buffer& out, const backend::View& vo, const std::vector<int>& axes, bool keep_dims) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        if (op_type == ir::ReduceOpType::SUM) {
            auto init = [](){ return T(0); };
            auto acc  = [](T& acc, T v){ acc += v; };
            cpu::reduce_view_kernel<T,T>(a, va, out, vo, axes, keep_dims, init, acc);
        } else {
            auto init = [](){ return -std::numeric_limits<T>::infinity(); };
            auto acc  = [](T& acc, T v){ acc = std::max(acc, v); };
            cpu::reduce_view_kernel<T,T>(a, va, out, vo, axes, keep_dims, init, acc);
        }
    }));
}

void CPUBackend::quantized_matmul(const Buffer& a, const Buffer& qweight,
                                  const std::vector<const Buffer*>& aux, Buffer& out,
                                  size_t M, size_t N, size_t K, const ir::QuantParams& params) const {
    if ((int)aux.size() != ir::aux_buffer_count(params.scheme))
        throw std::runtime_error("CPUBackend::quantized_matmul: wrong aux buffer count for scheme");
    switch (params.scheme) {
        case ir::QuantScheme::MLX_AFFINE: {
            // TODO(4-bit): matmul_quant_affine_f32 hardcodes 8-bit unpacking (4 codes/u32). For bits==4
            // (8 codes/u32, pack_factor 8) add a code-width branch in the unpack; dequant math and the
            // {scales,biases} layout are identical.
            if (params.bits != 8)
                throw std::runtime_error("CPUBackend::quantized_matmul: MLX_AFFINE only supports bits=8 (4-bit kernel is TODO)");
            const Buffer& scales = *aux[0];
            const Buffer& biases = *aux[1];
            cpu::matmul_quant_affine_f32(static_cast<const float*>(a.data()),
                                         static_cast<const uint32_t*>(qweight.data()),
                                         static_cast<const float*>(scales.data()),
                                         static_cast<const float*>(biases.data()),
                                         static_cast<float*>(out.data()), M, N, K, params.group_size);
            return;
        }
    }
    throw std::runtime_error("CPUBackend::quantized_matmul: unsupported quant scheme");
}

// Flash attention (online softmax over keys, no [S,KV] materialization). Inputs contiguous, native
// layout: q [B,S,nH,Dh], k,v [B,KV,nKV,Dh] -> out [B,S,nH,Dh]. One independent (b,s,h) row per task.
void CPUBackend::flash_attention(const Buffer& q, const Buffer& k, const Buffer& v, Buffer& out,
                                 size_t B, size_t S, size_t nH, size_t Dh, size_t KV, size_t nKV,
                                 float scale, int n_rep, bool causal, size_t q_offset) const {
    const float* Q = static_cast<const float*>(q.data());
    const float* K = static_cast<const float*>(k.data());
    const float* V = static_cast<const float*>(v.data());
    float* O = static_cast<float*>(out.data());
    const size_t rows = B * S * nH;
    cpu::parallel_for((size_t)0, rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            size_t h = r % nH, s = (r / nH) % S, b = r / (nH * S);
            size_t kv = h / (size_t)n_rep;
            const float* qp = Q + ((b * S + s) * nH + h) * Dh;
            float* op = O + ((b * S + s) * nH + h) * Dh;
            size_t jmax = causal ? std::min(KV, q_offset + s + 1) : KV;
            float m = -std::numeric_limits<float>::infinity(), l = 0.0f;
            std::vector<float> acc(Dh, 0.0f);
            for (size_t j = 0; j < jmax; ++j) {
                const float* kp = K + ((b * KV + j) * nKV + kv) * Dh;
                float sij = 0.0f;
                for (size_t d = 0; d < Dh; ++d) sij += qp[d] * kp[d];
                sij *= scale;
                float m_new = std::max(m, sij);
                float corr = std::exp(m - m_new);   // m=-inf first iter -> 0
                float p = std::exp(sij - m_new);
                l = l * corr + p;
                const float* vp = V + ((b * KV + j) * nKV + kv) * Dh;
                for (size_t d = 0; d < Dh; ++d) acc[d] = acc[d] * corr + p * vp[d];
                m = m_new;
            }
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (size_t d = 0; d < Dh; ++d) op[d] = acc[d] * inv;
        }
    });
}

void CPUBackend::matmul(const Buffer& a, const backend::View& va,
                        const Buffer& b, const backend::View& vb,
                        Buffer& out, const backend::View& vo) const {
    // Mixed precision: large model weights live in memory as bfloat16 (half of fp32) while
    // activations / accumulation stay float32. The IR sets the matmul output dtype from the
    // activation (a), so a bf16 weight b with an fp32 a yields an fp32 out here.
    const bool b_bf16 = (b.dtype() == common::DType::BFLOAT16 &&
                         out.dtype() == common::DType::FLOAT32);

    if (va.rank == 2 && vb.rank == 2 && vo.rank == 2) {
        if (b_bf16) {
            cpu::matmul_view_kernel_f32_bf16(a, va, b, vb, out, vo);
        } else {
            cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
                using T = typename decltype(tag)::type;
                cpu::matmul_view_kernel<T>(a, va, b, vb, out, vo);
            }));
        }
        return;
    }
    // N-D batched matmul: collapse leading dims into single batch, loop over batches
    if (va.rank != vb.rank || va.rank != vo.rank || va.rank < 3)
        throw std::runtime_error("CPU matmul: all views must have same rank >= 3 for batched matmul");

    const int rank = static_cast<int>(va.rank);
    const size_t M = va.shape[rank - 2];
    const size_t K = va.shape[rank - 1];
    const size_t N = vb.shape[rank - 1];

    // Iterate over the OUTPUT's batch dims (the broadcast result). For each linear batch
    // index we recover the per-dim indices and sum idx[d]*strides[d] for each operand -
    // broadcast dims carry stride 0, so this is correct for any number of batch dims and for
    // A/B broadcasting. (A single collapsed "bi * batch_stride" is only valid when there is
    // exactly one batch dim; it silently mis-indexes for rank >= 4, e.g. attention's
    // [batch, heads, seq, dim] matmuls, reading out of bounds.)
    const int nb = rank - 2;  // number of batch dims
    size_t batch_count = 1;
    for (int d = 0; d < nb; ++d) batch_count *= vo.shape[d];

    // Build the rank-2 sub-views for batch index bi (shared by both the fp32 and the
    // mixed-precision paths).
    auto build_subviews = [&](size_t bi, backend::View& va2, backend::View& vb2, backend::View& vo2) {
        size_t a_off = va.offset, b_off = vb.offset, o_off = vo.offset;
        size_t rem = bi;
        for (int d = nb - 1; d >= 0; --d) {  // row-major decompose over vo batch dims
            size_t idx = rem % vo.shape[d];
            rem /= vo.shape[d];
            a_off += idx * va.strides[d];
            b_off += idx * vb.strides[d];
            o_off += idx * vo.strides[d];
        }
        va2.rank = 2; va2.shape[0] = (uint32_t)M; va2.shape[1] = (uint32_t)K;
        va2.strides[0] = va.strides[rank - 2]; va2.strides[1] = va.strides[rank - 1];
        va2.offset = (uint32_t)a_off; va2.flags = 0; va2.numel = M * K;
        vb2.rank = 2; vb2.shape[0] = (uint32_t)K; vb2.shape[1] = (uint32_t)N;
        vb2.strides[0] = vb.strides[rank - 2]; vb2.strides[1] = vb.strides[rank - 1];
        vb2.offset = (uint32_t)b_off; vb2.flags = 0; vb2.numel = K * N;
        vo2.rank = 2; vo2.shape[0] = (uint32_t)M; vo2.shape[1] = (uint32_t)N;
        vo2.strides[0] = vo.strides[rank - 2]; vo2.strides[1] = vo.strides[rank - 1];
        vo2.offset = (uint32_t)o_off; vo2.flags = 0; vo2.numel = M * N;
    };

    if (b_bf16) {
        cpu::parallel_for((size_t)0, batch_count, [&](size_t s, size_t e) {
            for (size_t bi = s; bi < e; ++bi) {
                backend::View va2, vb2, vo2;
                build_subviews(bi, va2, vb2, vo2);
                cpu::matmul_view_kernel_f32_bf16(a, va2, b, vb2, out, vo2);
            }
        });
    } else {
        cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
            using T = typename decltype(tag)::type;
            cpu::parallel_for((size_t)0, batch_count, [&](size_t s, size_t e) {
                for (size_t bi = s; bi < e; ++bi) {
                    backend::View va2, vb2, vo2;
                    build_subviews(bi, va2, vb2, vo2);
                    cpu::matmul_view_kernel<T>(a, va2, b, vb2, out, vo2);
                }
            });
        }));
    }
}

void CPUBackend::gather_op(const Buffer& table, const Buffer& indices, Buffer& out, size_t V, size_t D) const {
    const int32_t* idx = static_cast<const int32_t*>(indices.data());
    const size_t N = indices.numel();

    // Mixed: bf16 weight table -> fp32 output (embedding kept in bf16 to save memory).
    if (table.dtype() == common::DType::BFLOAT16 && out.dtype() == common::DType::FLOAT32) {
        const uint16_t* t_ptr = static_cast<const uint16_t*>(table.data());
        float* o_ptr = static_cast<float*>(out.data());
        cpu::parallel_for((size_t)0, N, [&](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                const int32_t k = idx[i];
                if (k < 0 || static_cast<size_t>(k) >= V) {
                    for (size_t j = 0; j < D; ++j) o_ptr[i * D + j] = 0.0f;
                } else {
                    for (size_t j = 0; j < D; ++j) o_ptr[i * D + j] = cpu::bf16_to_f32(t_ptr[k * D + j]);
                }
            }
        });
        return;
    }

    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* t_ptr = static_cast<const T*>(table.data());
        T* o_ptr = static_cast<T*>(out.data());

        cpu::parallel_for((size_t)0, N, [&](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                const int32_t k = idx[i];
                if (k < 0 || static_cast<size_t>(k) >= V) {
                    // Out of bounds: zero out the row
                    for (size_t j = 0; j < D; ++j) o_ptr[i * D + j] = T(0);
                } else {
                    for (size_t j = 0; j < D; ++j) o_ptr[i * D + j] = t_ptr[k * D + j];
                }
            }
        });
    }));
}

void CPUBackend::concat_op(const std::vector<const Buffer*>& inputs, const std::vector<backend::View>& input_views,
                           Buffer& out, const backend::View& out_view, int axis) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::concat_kernel<T>(inputs, input_views, out, out_view, axis);
    }));
}

void CPUBackend::gather_axis_op(const Buffer& tensor, const backend::View& tv,
                                 const Buffer& indices,
                                 Buffer& out, const backend::View& ov,
                                 int axis) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::gather_axis_kernel<T>(tensor, tv, indices, out, ov, axis);
    }));
}

void CPUBackend::scatter_axis_op(const Buffer& base, const backend::View& bv,
                                  const Buffer& values, const backend::View& vv,
                                  const Buffer& indices,
                                  Buffer& out, const backend::View& ov,
                                  int axis) const {
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::scatter_axis_kernel<T>(base, bv, values, vv, indices, out, ov, axis);
    }));
}


// Generic view copy

void CPUBackend::copy_view(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) const {
    // Dtype-converting copy (e.g. fp32 activations -> bf16 KV cache). Only fp32<->bf16 wired up.
    if (src.dtype() != dst.dtype()) {
        if (src.dtype() == common::DType::FLOAT32 && dst.dtype() == common::DType::BFLOAT16)
            cpu::copy_view_convert_kernel<float, common::bfloat16>(src, vs, dst, vd);
        else if (src.dtype() == common::DType::BFLOAT16 && dst.dtype() == common::DType::FLOAT32)
            cpu::copy_view_convert_kernel<common::bfloat16, float>(src, vs, dst, vd);
        else
            throw std::runtime_error("copy_view: unsupported dtype conversion");
        return;
    }
    // Single fast path: dense row-major on both sides, same logical shape.
    if (vs.is_contiguous() && vd.is_contiguous() && same_shape(vs, vd)) {
        const size_t item  = common::size(dst.dtype());
        const size_t bytes = vd.numel * item;
        const uint8_t* sp = static_cast<const uint8_t*>(src.data()) + (size_t)vs.offset * item;
        uint8_t*       dp = static_cast<uint8_t*>(dst.data())       + (size_t)vd.offset * item;
        if (bytes) std::memcpy(dp, sp, bytes);
        return;
    }
    // Same-dtype bf16 strided copy (slice / reshape / materialize): pure bit-movement, so move it as
    // uint16 (dispatch_dtype has no float arithmetic type for bf16, nor does it need one here).
    if (dst.dtype() == common::DType::BFLOAT16) {
        cpu::copy_view_kernel<uint16_t>(src, vs, dst, vd);
        return;
    }
    // Fallback: typed elementwise mapping
    cpu::dispatch_dtype(dst.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::copy_view_kernel<T>(src, vs, dst, vd);
    }));
}

} // namespace cppgrad::backend::cpu
