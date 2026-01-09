// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cstring>
#include "cppgrad/backend/cpu/cpu_backend.h"
#include "cppgrad/backend/cpu/cpu_kernels.h"
#include "cppgrad/backend/cpu/dtype_dispatch.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/utils/rng.h"

namespace cppgrad {
namespace backend {
namespace cpu {

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

void CPUBackend::matmul(const Buffer& a, const backend::View& va,
                        const Buffer& b, const backend::View& vb,
                        Buffer& out, const backend::View& vo) const {
    // if (a.dtype() != out.dtype() || b.dtype() != out.dtype())
    //     throw std::runtime_error("matmul: dtype mismatch");
    cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::matmul_view_kernel<T>(a, va, b, vb, out, vo);
    }));
}

// Movement ops

// void CPUBackend::permute(const Buffer& a, const backend::View& va,
//                          Buffer& out, const backend::View& vo,
//                          const std::vector<size_t>& axes) const {
//     bool identity_axes = true;
//     for (size_t i=0;i<axes.size();++i) if (axes[i] != i) { identity_axes = false; break; }
//     if (identity_axes && va.is_contiguous() && vo.is_contiguous() && same_shape(va, vo) &&
//         va.offset == 0 && vo.offset == 0 && out.size_bytes() == a.size_bytes()) {
//         if (out.size_bytes()) std::memcpy(out.data(), a.data(), out.size_bytes());
//         return;
//     }
//     cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
//         using T = typename decltype(tag)::type;
//         cpu::permute_view_kernel<T>(a, va, out, vo, axes);
//     }));
// }

// void CPUBackend::broadcast(const Buffer& a, const backend::View& va,
//                            Buffer& out, const backend::View& vo) const {
//     // Identity broadcast -> memcpy
//     if (va.is_contiguous() && vo.is_contiguous() && same_shape(va, vo) &&
//         va.offset == 0 && vo.offset == 0 && out.size_bytes() == a.size_bytes()) {
//         if (out.size_bytes()) std::memcpy(out.data(), a.data(), out.size_bytes());
//         return;
//     }
//     cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
//         using T = typename decltype(tag)::type;
//         cpu::broadcast_view_kernel<T>(a, va, out, vo);
//     }));
// }

// void CPUBackend::slice_forward(const Buffer& a, const backend::View& va,
//                                Buffer& out, const backend::View& vo,
//                                const std::vector<size_t>& begin,
//                                const std::vector<size_t>& /*end*/,
//                                const std::vector<size_t>& step) const {
//     bool begin_zero = std::all_of(begin.begin(), begin.end(), [](size_t x){ return x == 0; });
//     bool step_one   = step.empty() || std::all_of(step.begin(), step.end(), [](size_t x){ return x == 1; });
//     if (begin_zero && step_one &&
//         va.is_contiguous() && vo.is_contiguous() && same_shape(va, vo) &&
//         va.offset == 0 && vo.offset == 0 && out.size_bytes() == a.size_bytes()) {
//         if (out.size_bytes()) std::memcpy(out.data(), a.data(), out.size_bytes());
//         return;
//     }

//     cpu::dispatch_dtype(out.dtype(), make_templated([&](auto tag) {
//         using T = typename decltype(tag)::type;
//         cpu::slice_forward_view_kernel<T>(a, va, out, vo, begin);
//     }));
// }

// void CPUBackend::slice_backward_scatter_add(const Buffer& grad_out, const backend::View& vgo,
//                                             Buffer& grad_in,  const backend::View& vgi,
//                                             const std::vector<size_t>& begin,
//                                             const std::vector<size_t>& /*end*/,
//                                             const std::vector<size_t>& /*step*/) const {
//     cpu::dispatch_dtype(grad_in.dtype(), make_templated([&](auto tag) {
//         using T = typename decltype(tag)::type;
//         cpu::slice_backward_scatter_add_view_kernel<T>(grad_out, vgo, grad_in, vgi, begin);
//     }));
// }

// Generic view copy

void CPUBackend::copy_view(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) const {
    // Single fast path: dense row-major on both sides, same logical shape.
    if (vs.is_contiguous() && vd.is_contiguous() && same_shape(vs, vd)) {
        const size_t item  = backend::size(dst.dtype());
        const size_t bytes = vd.numel * item;
        const uint8_t* sp = static_cast<const uint8_t*>(src.data()) + (size_t)vs.offset * item;
        uint8_t*       dp = static_cast<uint8_t*>(dst.data())       + (size_t)vd.offset * item;
        if (bytes) std::memcpy(dp, sp, bytes);
        return;
    }
    // Fallback: typed elementwise mapping
    cpu::dispatch_dtype(dst.dtype(), make_templated([&](auto tag) {
        using T = typename decltype(tag)::type;
        cpu::copy_view_kernel<T>(src, vs, dst, vd);
    }));
}

} // namespace cpu
} // namespace backend
} // namespace cppgrad
