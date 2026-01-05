// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <set>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <functional>

#include "cppgrad/backend/cpu/thread_runtime.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/utils/shape.h"
#include "cppgrad/backend/view.h"

namespace cppgrad {
namespace backend {
namespace cpu {

// Helpers

template<typename T> inline T* ptr(Buffer& buf) { return static_cast<T*>(buf.data()); }
template<typename T> inline const T* ptr(const Buffer& buf) { return static_cast<const T*>(buf.data()); }

// Contiguous kernels

template<typename T>
inline void fill_kernel(Buffer& buf, float value) {
    const size_t n = buf.size_bytes() / sizeof(T);
    if (!n) return;
    T* data_ptr = ptr<T>(buf);
    const T v = static_cast<T>(value);
    std::fill(data_ptr, data_ptr + n, v);
}

template<typename T, typename Func>
inline void unary_op_kernel(const Buffer& a, Buffer& out, Func op) {
    const size_t n = out.size_bytes() / sizeof(T);
    if (!n) return;
    const T* a_ptr = ptr<const T>(a);
    T* o_ptr = ptr<T>(out);
    for (size_t i = 0; i < n; ++i) o_ptr[i] = op(a_ptr[i]);
}

template<typename T, typename Func>
inline void binary_op_kernel(const Buffer& a, const Buffer& b, Buffer& out, Func op,
                             const std::vector<size_t>& shape_a,
                             const std::vector<size_t>& shape_b,
                             const std::vector<size_t>& out_shape) {
    const T* a_ptr = ptr<const T>(a);
    const T* b_ptr = ptr<const T>(b);
    T* out_ptr = ptr<T>(out);
    const size_t out_numel = out.size_bytes() / sizeof(T);
    if (!out_numel) return;

    // Fast path 1: identical shapes (no broadcast)
    if (shape_a == shape_b) {
        for (size_t i = 0; i < out_numel; ++i) out_ptr[i] = op(a_ptr[i], b_ptr[i]);
        return;
    }
    // Fast path 2: scalar broadcast
    const bool a_scalar = shape_a.empty() || (shape_a.size()==1 && shape_a[0]==1);
    const bool b_scalar = shape_b.empty() || (shape_b.size()==1 && shape_b[0]==1);
    if (a_scalar) {
        const T av = a_ptr[0];
        for (size_t i = 0; i < out_numel; ++i) out_ptr[i] = op(av, b_ptr[i]);
        return;
    }
    if (b_scalar) {
        const T bv = b_ptr[0];
        for (size_t i = 0; i < out_numel; ++i) out_ptr[i] = op(a_ptr[i], bv);
        return;
    }

    // Slow path: general broadcast
    const auto strides_a   = cppgrad::utils::shape::row_major_strides(shape_a);
    const auto strides_b   = cppgrad::utils::shape::row_major_strides(shape_b);
    const auto strides_out = cppgrad::utils::shape::row_major_strides(out_shape);

    std::vector<size_t> idx_out(strides_out.size());
    std::vector<size_t> idx_a(shape_a.size());
    std::vector<size_t> idx_b(shape_b.size());

    // for (size_t i = 0; i < out_numel; ++i) {
    //     cppgrad::utils::shape::coords_from_index_inplace(i, strides_out, idx_out);
    //     // Map to a, b indices (right-aligned broadcast)
    //     for (size_t j = 0; j < shape_a.size(); ++j) {
    //         const size_t out_dim_idx = idx_out.size() - 1 - j;
    //         const size_t a_dim_idx   = shape_a.size() - 1 - j;
    //         idx_a[a_dim_idx] = (shape_a[a_dim_idx] == 1) ? 0 : idx_out[out_dim_idx];
    //     }
    //     for (size_t j = 0; j < shape_b.size(); ++j) {
    //         const size_t out_dim_idx = idx_out.size() - 1 - j;
    //         const size_t b_dim_idx   = shape_b.size() - 1 - j;
    //         idx_b[b_dim_idx] = (shape_b[b_dim_idx] == 1) ? 0 : idx_out[out_dim_idx];
    //     }
    //     size_t flat_a = 0, flat_b = 0;
    //     for (size_t d = 0; d < idx_a.size(); ++d) flat_a += idx_a[d] * strides_a[d];
    //     for (size_t d = 0; d < idx_b.size(); ++d) flat_b += idx_b[d] * strides_b[d];

    //     out_ptr[i] = op(a_ptr[flat_a], b_ptr[flat_b]);
    // }
    cpu::parallel_for((size_t)0, out_numel, [&](size_t s, size_t e) {
        // Thread/task-local scratch
        std::vector<size_t> idx_out(strides_out.size());
        std::vector<size_t> idx_a(shape_a.size());
        std::vector<size_t> idx_b(shape_b.size());

        for (size_t i = s; i < e; ++i) {
            cppgrad::utils::shape::coords_from_index_inplace(i, strides_out, idx_out);

            // Map to a indices (right-aligned broadcast)
            for (size_t j = 0; j < shape_a.size(); ++j) {
                const size_t out_dim_idx = idx_out.size() - 1 - j;
                const size_t a_dim_idx   = shape_a.size() - 1 - j;
                idx_a[a_dim_idx] = (shape_a[a_dim_idx] == 1) ? 0 : idx_out[out_dim_idx];
            }

            // Map to b indices (right-aligned broadcast)
            for (size_t j = 0; j < shape_b.size(); ++j) {
                const size_t out_dim_idx = idx_out.size() - 1 - j;
                const size_t b_dim_idx   = shape_b.size() - 1 - j;
                idx_b[b_dim_idx] = (shape_b[b_dim_idx] == 1) ? 0 : idx_out[out_dim_idx];
            }

            size_t flat_a = 0, flat_b = 0;
            for (size_t d = 0; d < idx_a.size(); ++d) flat_a += idx_a[d] * strides_a[d];
            for (size_t d = 0; d < idx_b.size(); ++d) flat_b += idx_b[d] * strides_b[d];

            out_ptr[i] = op(a_ptr[flat_a], b_ptr[flat_b]);
        }
    });
}

template<typename T>
inline void matmul_kernel(const Buffer& a, const Buffer& b, Buffer& out, const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
    const size_t M = shape_a[0];
    const size_t K = shape_a[1];
    const size_t N = shape_b[1];
    const T* a_ptr = ptr<const T>(a);
    const T* b_ptr = ptr<const T>(b);
    T* out_ptr = ptr<T>(out);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum_val = 0;
            for (size_t k = 0; k < K; ++k) {
                sum_val += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            out_ptr[i * N + j] = sum_val;
        }
    }
}

template<typename T>
inline void sum_kernel(const Buffer& a, Buffer& out, const std::vector<size_t>& in_shape, const std::vector<int>& axes, bool keep_dims) {
    T* out_ptr = ptr<T>(out);
    std::fill(out_ptr, out_ptr + (out.size_bytes() / sizeof(T)), static_cast<T>(0));

    if (axes.empty() || axes.size() == in_shape.size()) {
        const T* a_ptr = ptr<const T>(a);
        size_t n = a.size_bytes() / sizeof(T);
        out_ptr[0] = std::accumulate(a_ptr, a_ptr + n, static_cast<T>(0));
        return;
    }

    const T* a_ptr = ptr<const T>(a);
    auto in_strides = cppgrad::utils::shape::row_major_strides(in_shape);
    size_t n_in = a.size_bytes() / sizeof(T);

    std::vector<size_t> out_shape_calc;
    std::set<int> axes_set(axes.begin(), axes.end());
    if (keep_dims) {
        out_shape_calc = in_shape;
        for (int axis : axes) out_shape_calc[axis] = 1;
    } else {
        for (size_t i = 0; i < in_shape.size(); ++i) {
            if (axes_set.find((int)i) == axes_set.end()) out_shape_calc.push_back(in_shape[i]);
        }
        if (out_shape_calc.empty()) out_shape_calc.push_back(1);
    }
    auto out_strides = cppgrad::utils::shape::row_major_strides(out_shape_calc);

    for (size_t i = 0; i < n_in; ++i) {
        auto multi_index_in = cppgrad::utils::shape::coords_from_index(i, in_strides);
        std::vector<size_t> multi_index_out;
        if (keep_dims) {
            multi_index_out = multi_index_in;
            for (int axis : axes) multi_index_out[(size_t)axis] = 0;
        } else {
            for (size_t j = 0; j < in_shape.size(); ++j) {
                if (axes_set.find((int)j) == axes_set.end()) multi_index_out.push_back(multi_index_in[j]);
            }
            if (multi_index_out.empty()) multi_index_out.push_back(0);
        }
        size_t out_idx = cppgrad::utils::shape::index_from_coords(multi_index_out, out_strides);
        out_ptr[out_idx] += a_ptr[i];
    }
}

template<typename T>
inline void max_kernel(const Buffer& a, Buffer& out, const std::vector<size_t>& in_shape, const std::vector<int>& axes) {
    T* out_ptr = ptr<T>(out);
    std::fill(out_ptr, out_ptr + (out.size_bytes() / sizeof(T)), std::numeric_limits<T>::lowest());

    if (axes.empty() || axes.size() == in_shape.size()) {
        const T* a_ptr = ptr<const T>(a);
        size_t n = a.size_bytes() / sizeof(T);
        if (n > 0) out_ptr[0] = *std::max_element(a_ptr, a_ptr + n);
        return;
    }

    const T* a_ptr = ptr<const T>(a);
    auto in_strides = cppgrad::utils::shape::row_major_strides(in_shape);
    size_t n_in = a.size_bytes() / sizeof(T);

    std::vector<size_t> out_shape_calc = in_shape;
    for(int axis : axes) out_shape_calc[(size_t)axis] = 1;
    auto out_strides = cppgrad::utils::shape::row_major_strides(out_shape_calc);

    for(size_t i = 0; i < n_in; ++i) {
        auto multi_index_in = cppgrad::utils::shape::coords_from_index(i, in_strides);
        std::vector<size_t> multi_index_out = multi_index_in;
        for(int axis : axes) multi_index_out[(size_t)axis] = 0;
        size_t out_idx = cppgrad::utils::shape::index_from_coords(multi_index_out, out_strides);
        out_ptr[out_idx] = std::max(out_ptr[out_idx], a_ptr[i]);
    }
}

// Reduce last axis only (contiguous inner dimension)
// template<typename T, typename AccFunc, typename InitFn>
// inline void reduce_last_axis_kernel(const Buffer& a, Buffer& out,
//                                     const std::vector<size_t>& in_shape,
//                                     AccFunc acc_op, InitFn init_val) {
//     const T* ap = ptr<const T>(a);
//     T* op = ptr<T>(out);
//     const size_t outer = in_shape.size() > 1 ? (a.size_bytes()/sizeof(T))/in_shape.back() : 1;
//     const size_t inner = in_shape.size() ? in_shape.back() : 1;
//     for (size_t i = 0; i < outer; ++i) {
//         T acc = init_val();
//         const T* row = ap + i * inner;
//         for (size_t j = 0; j < inner; ++j) acc_op(acc, row[j]);
//         op[i] = acc;
//     }
// }

template<typename T>
inline void permute_kernel(const Buffer& a, Buffer& out, const std::vector<size_t>& in_shape, const std::vector<size_t>& axes) {
    const T* a_ptr = ptr<const T>(a);
    T* out_ptr = ptr<T>(out);
    size_t n = a.size_bytes() / sizeof(T);
    auto in_strides = cppgrad::utils::shape::row_major_strides(in_shape);
    std::vector<size_t> out_shape(in_shape.size());
    for(size_t i = 0; i < axes.size(); ++i) out_shape[i] = in_shape[axes[i]];
    auto out_strides = cppgrad::utils::shape::row_major_strides(out_shape);

    std::vector<size_t> out_coords;
    for (size_t i = 0; i < n; ++i) {
        cppgrad::utils::shape::coords_from_index_inplace(i, out_strides, out_coords);
        std::vector<size_t> in_coords(axes.size());
        for (size_t j = 0; j < axes.size(); ++j) in_coords[axes[j]] = out_coords[j];
        size_t in_idx = cppgrad::utils::shape::index_from_coords(in_coords, in_strides);
        out_ptr[i] = a_ptr[in_idx];
    }
}

template<typename T>
inline void permute_kernel_fast(const Buffer& a, Buffer& out,
                                const std::vector<size_t>& in_shape,
                                const std::vector<size_t>& axes) {
    bool is_identity = true;
    for (size_t i=0;i<axes.size();++i) if (axes[i] != i) { is_identity=false; break; }
    if (is_identity) {
        std::memcpy(ptr<T>(out), ptr<const T>(a), out.size_bytes());
        return;
    }
    cpu::permute_kernel<T>(a, out, in_shape, axes);
}

template<typename T>
inline void broadcast_kernel(const Buffer& a, Buffer& out, const std::vector<size_t>& in_shape, const std::vector<size_t>& out_shape) {
    const T* a_ptr = ptr<const T>(a);
    T* out_ptr = ptr<T>(out);
    size_t n_out = out.size_bytes() / sizeof(T);
    auto in_strides = cppgrad::utils::shape::row_major_strides(in_shape);
    auto out_strides = cppgrad::utils::shape::row_major_strides(out_shape);

    std::vector<size_t> out_coords;
    for (size_t i = 0; i < n_out; ++i) {
        cppgrad::utils::shape::coords_from_index_inplace(i, out_strides, out_coords);
        std::vector<size_t> in_coords(in_shape.size());
        size_t offset = out_shape.size() - in_shape.size();
        for (size_t j = 0; j < in_shape.size(); ++j) {
            in_coords[j] = (in_shape[j] == 1) ? 0 : out_coords[j + offset];
        }
        size_t in_idx = cppgrad::utils::shape::index_from_coords(in_coords, in_strides);
        out_ptr[i] = a_ptr[in_idx];
    }
}

template<typename T>
inline void broadcast_kernel_fast(const Buffer& a, Buffer& out,
                                  const std::vector<size_t>& in_shape,
                                  const std::vector<size_t>& out_shape) {
    const T* ap = ptr<const T>(a);
    T* op = ptr<T>(out);
    const size_t n = out.size_bytes()/sizeof(T);
    if (!n) return;
    if (in_shape == out_shape) {
        std::memcpy(op, ap, n * sizeof(T));
        return;
    }
    cpu::broadcast_kernel<T>(a, out, in_shape, out_shape);
}

// Direct View helpers

inline size_t index_from_coords(const backend::View& v, const std::vector<size_t>& coords) {
    size_t idx = v.offset;
    for (size_t i=0;i<v.rank;++i) idx += coords[i] * static_cast<size_t>(v.strides[i]);
    return idx;
}
// inline size_t index_from_coords(const backend::View& v, const std::vector<size_t>& coords) {
// #ifndef CPPGRAD_DEBUG
//     for (size_t i = 0; i < v.rank; ++i) {
//         if (v.strides[i] < 0) throw std::runtime_error("negative stride not supported here");
//     }
// #endif
//     size_t idx = (size_t)v.offset;
//     for (size_t i = 0; i < v.rank; ++i) idx += coords[i] * (size_t)v.strides[i];
//     return idx;
// }

inline void coords_from_linear(size_t lin,
                               const std::vector<uint32_t>& shape_u32,
                               std::vector<size_t>& coords) {
    const size_t rank = shape_u32.size();
    coords.assign(rank, 0);
    if (rank == 0) return; // scalar
    size_t rem = lin;
    for (size_t d = rank; d-- > 0; ) {
        const size_t dim = static_cast<size_t>(shape_u32[d]);
        coords[d] = (dim == 0) ? 0 : (rem % dim);
        rem       = (dim == 0) ? rem : (rem / dim);
    }
}

inline void map_out_to_in_coords_broadcast(const std::vector<size_t>& out_coords,
                                           const backend::View& in_v,
                                           std::vector<size_t>& in_coords) {
    const size_t r_out = out_coords.size(), r_in = in_v.rank;
    in_coords.assign(r_in, 0);
    const size_t off = r_out - r_in;
    for (size_t i=0;i<r_in;++i) {
        const size_t oc = out_coords[off + i];
        const size_t dim = static_cast<size_t>(in_v.shape[i]);
        in_coords[i] = (dim == 1) ? 0 : oc;
    }
}

// Stride-aware View kernels (backend::View)

template<typename T, typename UnaryFn>
inline void unary_view_kernel(const Buffer& a, const backend::View& va,
                              Buffer& out, const backend::View& vo,
                              UnaryFn fn) {
    const T* ap = static_cast<const T*>(a.data());
    T* op = static_cast<T*>(out.data());
    const size_t nout = vo.numel;
    if (!nout) return;

    if (va.is_contiguous() && vo.is_contiguous() &&
        va.rank == vo.rank && va.is_offset_zero() && vo.is_offset_zero()) {
        bool same_shape = true;
        for (uint32_t i=0;i<va.rank;++i) if (va.shape[i] != vo.shape[i]) { same_shape=false; break; }
        if (same_shape) {
            for (size_t i=0;i<nout;++i) op[i] = fn(ap[i]);
            return;
        }
    }

    // std::vector<size_t> ocoords, icoords;
    // std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    // for (size_t lin=0; lin<nout; ++lin) {
    //     coords_from_linear(lin, out_shape, ocoords);
    //     map_out_to_in_coords_broadcast(ocoords, va, icoords);
    //     const size_t ai = index_from_coords(va, icoords);
    //     const size_t oi = index_from_coords(vo, ocoords);
    //     op[oi] = fn(ap[ai]);
    // }
    std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    cpu::parallel_for((size_t)0, nout, [&](size_t s, size_t e) {
        std::vector<size_t> ocoords, icoords; // task-local scratch

        for (size_t lin = s; lin < e; ++lin) {
            coords_from_linear(lin, out_shape, ocoords);
            map_out_to_in_coords_broadcast(ocoords, va, icoords);

            const size_t ai = index_from_coords(va, icoords);
            const size_t oi = index_from_coords(vo, ocoords);

            op[oi] = fn(ap[ai]);
        }
    });
}

template<typename T, typename BinaryFn>
inline void binary_view_kernel(const Buffer& a, const backend::View& va,
                               const Buffer& b, const backend::View& vb,
                               Buffer& out, const backend::View& vo,
                               BinaryFn fn) {
    const T* ap = static_cast<const T*>(a.data());
    const T* bp = static_cast<const T*>(b.data());
    T* op = static_cast<T*>(out.data());
    const size_t nout = vo.numel;
    if (!nout) return;

    if (va.is_contiguous() && vb.is_contiguous() && vo.is_contiguous() &&
        va.rank == vb.rank && va.rank == vo.rank &&
        va.is_offset_zero() && vb.is_offset_zero() && vo.is_offset_zero()) {
        bool same_ab = true, same_ao = true;
        for (uint32_t i=0;i<va.rank;++i) {
            if (va.shape[i] != vb.shape[i]) { same_ab=false; break; }
        }
        if (same_ab) {
            for (uint32_t i=0;i<va.rank;++i) {
                if (va.shape[i] != vo.shape[i]) { same_ao=false; break; }
            }
            if (same_ao) {
                for (size_t i=0;i<nout;++i) op[i] = fn(ap[i], bp[i]);
                return;
            }
        }
    }

    // std::vector<size_t> ocoords, acoords, bcoords;
    // std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    // for (size_t lin=0; lin<nout; ++lin) {
    //     coords_from_linear(lin, out_shape, ocoords);
    //     map_out_to_in_coords_broadcast(ocoords, va, acoords);
    //     map_out_to_in_coords_broadcast(ocoords, vb, bcoords);
    //     const size_t ai = index_from_coords(va, acoords);
    //     const size_t bi = index_from_coords(vb, bcoords);
    //     const size_t oi = index_from_coords(vo, ocoords);
    //     op[oi] = fn(ap[ai], bp[bi]);
    // }
    std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    cpu::parallel_for((size_t)0, nout, [&](size_t s, size_t e) {
        std::vector<size_t> ocoords, acoords, bcoords; // task-local

        for (size_t lin = s; lin < e; ++lin) {
            coords_from_linear(lin, out_shape, ocoords);
            map_out_to_in_coords_broadcast(ocoords, va, acoords);
            map_out_to_in_coords_broadcast(ocoords, vb, bcoords);

            const size_t ai = index_from_coords(va, acoords);
            const size_t bi = index_from_coords(vb, bcoords);
            const size_t oi = index_from_coords(vo, ocoords);

            op[oi] = fn(ap[ai], bp[bi]);
        }
    });
}

template<typename T, typename ReduceAcc, typename InitFn, typename AccFn>
inline void reduce_last_axis_kernel_view(
    const Buffer& a, const backend::View& va,
    Buffer& out, const backend::View& vo,
    bool keep_dims,
    InitFn init, AccFn acc_fn
) {
    const T* ap = static_cast<const T*>(a.data());
    T* op = static_cast<T*>(out.data());

    const int rank = (int)va.rank;
    if (rank <= 0) return;

    const size_t inner = (size_t)va.shape[rank - 1];
    if (inner == 0) return;

    const size_t outer = va.numel / inner;

    // Decode outer linear index into coords for dims [0..rank-2]
    // We do this with row-major math because "outer" is the logical iteration space.
    // Then we map coords -> buffer index using backend::index_from_coords (view strides + offset).
    std::vector<uint32_t> outer_shape_u32;
    outer_shape_u32.reserve((size_t)std::max(1, rank - 1));
    for (int d = 0; d < rank - 1; ++d) outer_shape_u32.push_back((uint32_t)va.shape[d]);
    if (outer_shape_u32.empty()) outer_shape_u32.push_back(1);

    std::vector<size_t> ocoords_outer((size_t)std::max(0, rank - 1));
    std::vector<size_t> icoords((size_t)rank);
    std::vector<size_t> ocoords_out((size_t)vo.rank);

    const size_t s_last = (size_t)va.strides[rank - 1];

    for (size_t oi = 0; oi < outer; ++oi) {
        if (rank - 1 > 0) coords_from_linear(oi, outer_shape_u32, ocoords_outer);

        // input coords = outer coords + last axis = 0
        for (int d = 0; d < rank - 1; ++d) icoords[(size_t)d] = ocoords_outer[(size_t)d];
        icoords[(size_t)(rank - 1)] = 0;

        const size_t row_base_in = index_from_coords(va, icoords);

        // output coords follow vo's rank convention
        if (!keep_dims) {
            // vo.rank == rank-1
            for (int d = 0; d < rank - 1; ++d) ocoords_out[(size_t)d] = icoords[(size_t)d];
        } else {
            // vo.rank == rank
            for (int d = 0; d < rank - 1; ++d) ocoords_out[(size_t)d] = icoords[(size_t)d];
            ocoords_out[(size_t)(rank - 1)] = 0;
        }

        const size_t row_base_out = index_from_coords(vo, ocoords_out);

        ReduceAcc acc = init();
        for (size_t j = 0; j < inner; ++j) {
            acc_fn(acc, ap[row_base_in + j * s_last]);
        }
        op[row_base_out] = (T)acc;
    }
}

// template<typename T, typename ReduceAcc, typename InitFn, typename AccFn>
// inline void reduce_view_kernel(const Buffer& a, const backend::View& va,
//                                Buffer& out, const backend::View& vo,
//                                const std::vector<int>& axes,
//                                bool keep_dims,
//                                InitFn init, AccFn acc_fn) {
//     const T* ap = static_cast<const T*>(a.data());
//     T* optr = static_cast<T*>(out.data());
//     const size_t nin = a.size_bytes()/sizeof(T);
//     const size_t nout = out.size_bytes()/sizeof(T);
//     if (!nout) return;

//     for (size_t i=0;i<nout;++i) optr[i] = init();

//     const int rank = static_cast<int>(va.rank);

//     // Normalize and unique-sort axes
//     auto axes_norm = cppgrad::utils::shape::normalize_unique_sorted_axes(axes, (size_t)rank);
//     std::vector<bool> reduce_mask(rank,false);
//     for (int ax: axes_norm) reduce_mask[(size_t)ax] = true;

//     const bool reduce_all = (int)axes_norm.size() == rank;
//     if (reduce_all && va.is_contiguous() && vo.is_contiguous()) {
//         ReduceAcc acc = init();
//         // contiguous slice from offset
//         const size_t base = va.offset;
//         for (size_t i=0;i<nin;++i) acc_fn(acc, ap[base + i]);
//         optr[vo.offset] = static_cast<T>(acc);
//         return;
//     }

//     // Fast path: reduce along last axis only, last-axis contiguous, and vo is rank +/- keep dims shape
//     if (axes_norm.size()==1) {
//         int ax = axes_norm[0] < 0 ? axes_norm[0] + rank : axes_norm[0];
//         if (ax == rank-1 && va.last_axis_contiguous()) {
//             // Build in_shape as size_t from va.shape[]
//             std::vector<size_t> in_shape(va.rank);
//             for (size_t i=0;i<in_shape.size();++i) in_shape[i] = (size_t)va.shape[i];
//             if (keep_dims) {
//                 // Out is same rank with last axis = 1, contiguous in practice
//                 reduce_last_axis_kernel<T>(a, out, in_shape,
//                     [&](T& acc, T v){ acc += v; }, // acc_fn not used here; keep consistent?
//                     [](){ return T(0); }
//                 );
//             } else {
//                 reduce_last_axis_kernel<T>(a, out, in_shape,
//                     [&](T& acc, T v){ acc += v; },
//                     [](){ return T(0); }
//                 );
//             }
//             return;
//         }
//     }

//     std::vector<size_t> icoords(rank), ocoords;
//     // materialize va.shape as vector<uint32_t> and also compute o-shape rank to decode linear
//     std::vector<uint32_t> in_shape_u32(va.shape, va.shape + va.rank);

//     // Build output "rank" explicitly from vo
//     std::vector<uint32_t> out_shape_u32(vo.shape, vo.shape + vo.rank);

//     for (size_t lin=0; lin<nin; ++lin) {
//         coords_from_linear(lin, in_shape_u32, icoords);
//         ocoords.clear();
//         if (keep_dims) {
//             ocoords.resize((size_t)rank);
//             for (int d=0; d<rank; ++d) {
//                 ocoords[(size_t)d] = reduce_mask[(size_t)d] ? 0 : icoords[(size_t)d];
//             }
//         } else {
//             for (int d=0; d<rank; ++d) if (!reduce_mask[(size_t)d]) ocoords.push_back(icoords[(size_t)d]);
//             if (ocoords.empty()) ocoords.push_back(0);
//         }
//         const size_t ai = index_from_coords(va, icoords);
//         const size_t oi = index_from_coords(vo, ocoords);
//         ReduceAcc tmp = optr[oi];
//         acc_fn(tmp, ap[ai]);
//         optr[oi] = static_cast<T>(tmp);
//     }
// }
template<typename T, typename ReduceAcc, typename InitFn, typename AccFn>
inline void reduce_view_kernel(
    const Buffer& a, const backend::View& va,
    Buffer& out, const backend::View& vo,
    const std::vector<int>& axes,
    bool keep_dims,
    InitFn init, AccFn acc_fn
) {
    const T* ap = static_cast<const T*>(a.data());
    T* optr     = static_cast<T*>(out.data());

    const size_t nin  = va.numel;
    const size_t nout = vo.numel;

    if (nout == 0) return;

    // Initialize output.
    for (size_t i = 0; i < nout; ++i) optr[(size_t)vo.offset + i] = (T)init();

    const int rank = (int)va.rank;

    // Normalize & unique-sort axes
    auto axes_norm = cppgrad::utils::shape::normalize_unique_sorted_axes(axes, (size_t)rank);

    std::vector<bool> reduce_mask((size_t)rank, false);
    for (int ax : axes_norm) reduce_mask[(size_t)ax] = true;

    const bool reduce_all = ((int)axes_norm.size() == rank);

    // Reduce-all fast path (contiguous)
    if (reduce_all && va.is_contiguous() && vo.is_contiguous()) {
        ReduceAcc acc = init();
        const size_t base = (size_t)va.offset;
        for (size_t i = 0; i < nin; ++i) acc_fn(acc, ap[base + i]);
        optr[(size_t)vo.offset] = (T)acc;
        return;
    }

    // Fast path: last axis only, last-axis contiguous, and output rank matches expectation
    if (rank > 0 && axes_norm.size() == 1) {
        const int ax = axes_norm[0];
        if (ax == rank - 1 && va.last_axis_contiguous()) {
            const int expected_out_rank = keep_dims ? rank : (rank - 1);
            if ((int)vo.rank == expected_out_rank) {
                reduce_last_axis_kernel_view<T, ReduceAcc>(a, va, out, vo, keep_dims, init, acc_fn);
                return;
            }
        }
    }

    // General path (view-correct)
    std::vector<size_t> icoords((size_t)rank);
    std::vector<size_t> ocoords;
    ocoords.reserve((size_t)rank);

    std::vector<uint32_t> in_shape_u32(va.shape, va.shape + va.rank);
    for (size_t lin = 0; lin < nin; ++lin) {
        coords_from_linear(lin, in_shape_u32, icoords);
        ocoords.clear();
        if (keep_dims) {
            ocoords.resize((size_t)rank);
            for (int d = 0; d < rank; ++d) {
                ocoords[(size_t)d] = reduce_mask[(size_t)d] ? 0 : icoords[(size_t)d];
            }
        } else {
            for (int d = 0; d < rank; ++d) if (!reduce_mask[(size_t)d]) ocoords.push_back(icoords[(size_t)d]);
            if (ocoords.empty()) ocoords.push_back(0);
        }
        const size_t ai = index_from_coords(va, icoords);
        const size_t oi = index_from_coords(vo, ocoords);
        ReduceAcc tmp = (ReduceAcc)optr[oi];
        acc_fn(tmp, ap[ai]);
        optr[oi] = (T)tmp;
    }
}

// template<typename T>
// inline void matmul_view_kernel(const Buffer& a, const backend::View& va,
//                                const Buffer& b, const backend::View& vb,
//                                Buffer& out, const backend::View& vo) {
//     const T* ap = static_cast<const T*>(a.data());
//     const T* bp = static_cast<const T*>(b.data());
//     T* op = static_cast<T*>(out.data());

//     if (va.rank != 2 || vb.rank != 2 || vo.rank != 2) {
//         throw std::runtime_error("matmul_view_kernel: rank-2 required");
//     }
//     const size_t M = (size_t)va.shape[0], K = (size_t)va.shape[1];
//     const size_t Kb = (size_t)vb.shape[0], N = (size_t)vb.shape[1];
//     if (K != Kb) throw std::runtime_error("matmul_view_kernel: inner dims mismatch");
//     if (vo.shape[0] != va.shape[0] || vo.shape[1] != vb.shape[1]) throw std::runtime_error("matmul_view_kernel: out shape mismatch");

//     if (va.is_contiguous() && vb.is_contiguous() && vo.is_contiguous() &&
//         va.is_offset_zero() && vb.is_offset_zero() && vo.is_offset_zero()) {
//         for (size_t i=0;i<M;++i) {
//             for (size_t j=0;j<N;++j) {
//                 T sum = 0;
//                 const size_t arow = i*K;
//                 const size_t bcol = j;
//                 for (size_t k=0;k<K;++k) sum += ap[arow + k] * bp[k*N + bcol];
//                 op[i*N + j] = sum;
//             }
//         }
//         return;
//     }

//     for (size_t i=0;i<M;++i) {
//         for (size_t j=0;j<N;++j) {
//             T sum = 0;
//             for (size_t k=0;k<K;++k) {
//                 const size_t ai = va.offset + i*(size_t)va.strides[0] + k*(size_t)va.strides[1];
//                 const size_t bi = vb.offset + k*(size_t)vb.strides[0] + j*(size_t)vb.strides[1];
//                 sum += ap[ai] * bp[bi];
//             }
//             const size_t oi = vo.offset + i*(size_t)vo.strides[0] + j*(size_t)vo.strides[1];
//             op[oi] = sum;
//         }
//     }
// }
// template<typename T>
// inline void matmul_view_kernel(const Buffer& a, const backend::View& va,
//                                const Buffer& b, const backend::View& vb,
//                                Buffer& out, const backend::View& vo) {
//     const T* ap = static_cast<const T*>(a.data());
//     const T* bp = static_cast<const T*>(b.data());
//     T* op = static_cast<T*>(out.data());

//     if (va.rank != 2 || vb.rank != 2 || vo.rank != 2) {
//         throw std::runtime_error("matmul_view_kernel: rank-2 required");
//     }
//     const size_t M  = (size_t)va.shape[0];
//     const size_t K  = (size_t)va.shape[1];
//     const size_t Kb = (size_t)vb.shape[0];
//     const size_t N  = (size_t)vb.shape[1];
//     if (K != Kb) throw std::runtime_error("matmul_view_kernel: inner dims mismatch");
//     if (vo.shape[0] != va.shape[0] || vo.shape[1] != vb.shape[1]) {
//         throw std::runtime_error("matmul_view_kernel: out shape mismatch");
//     }

//     const bool fast_flat_indexing_ok =
//         va.is_rowmaj_nn_2d() && vo.is_rowmaj_nn_2d() &&
//         va.is_offset_zero()  && vb.is_offset_zero()  && vo.is_offset_zero();

//     // Fast path 1: A,B row-major NN, output row-major identity
//     if (fast_flat_indexing_ok && vb.is_rowmaj_nn_2d()) {
//         for (size_t i=0; i<M; ++i) {
//             const size_t arow = i * K;
//             const size_t crow = i * N;
//             for (size_t j=0; j<N; ++j) {
//                 T sum = 0;
//                 for (size_t k=0; k<K; ++k) {
//                     sum += ap[arow + k] * bp[k * N + j];
//                 }
//                 op[crow + j] = sum;
//             }
//         }
//     }
//     // Fast path 2: A row-major NN, B row-major TN (B strides [1,K]), output row-major identity
//     else if (fast_flat_indexing_ok && vb.is_rowmaj_tn_2d()) {
//         for (size_t i=0; i<M; ++i) {
//             const size_t arow = i * K;
//             const size_t crow = i * N;
//             for (size_t j=0; j<N; ++j) {
//                 T sum = 0;
//                 const size_t bcol = j * K; // contiguous along K
//                 for (size_t k=0; k<K; ++k) {
//                     sum += ap[arow + k] * bp[bcol + k];
//                 }
//                 op[crow + j] = sum;
//             }
//         }
//     }
//     // Generic stride-aware path
//     else {
//         for (size_t i = 0; i < M; ++i) {
//             const size_t a_row = (size_t)va.offset + i*(size_t)va.strides[0];
//             const size_t o_row = (size_t)vo.offset + i*(size_t)vo.strides[0];
//             for (size_t j = 0; j < N; ++j) {
//                 const size_t b_col = (size_t)vb.offset + j*(size_t)vb.strides[1];
//                 T sum = 0;
//                 for (size_t k = 0; k < K; ++k) {
//                     sum += ap[a_row + k*(size_t)va.strides[1]] * bp[b_col + k*(size_t)vb.strides[0]];
//                 }
//                 op[o_row + j*(size_t)vo.strides[1]] = sum;
//             }
//         }
//     }
// }

template<typename T>
inline void matmul_view_kernel(const Buffer& a, const backend::View& va,
                               const Buffer& b, const backend::View& vb,
                               Buffer& out, const backend::View& vo) {
    const T* ap = static_cast<const T*>(a.data());
    const T* bp = static_cast<const T*>(b.data());
    T*       op = static_cast<T*>(out.data());

    if (va.rank != 2 || vb.rank != 2 || vo.rank != 2)
        throw std::runtime_error("matmul_view_kernel: rank-2 required");

    const size_t M  = (size_t)va.shape[0];
    const size_t K  = (size_t)va.shape[1];
    const size_t Kb = (size_t)vb.shape[0];
    const size_t N  = (size_t)vb.shape[1];

    if (K != Kb) throw std::runtime_error("matmul_view_kernel: inner dims mismatch");
    if (vo.shape[0] != va.shape[0] || vo.shape[1] != vb.shape[1])
        throw std::runtime_error("matmul_view_kernel: out shape mismatch");

    // Packed layout preconditions (no stride math inside inner loops).
    // NOTE: no offset==0 requirement anymore; we just shift base pointers.
    const bool fast_packed = va.is_rowmaj_nn_2d() && vo.is_rowmaj_nn_2d();

    // // Fast path 1: NN (A NN, B NN, Out NN)
    // if (fast_packed && vb.is_rowmaj_nn_2d()) {
    //     const T* ap0 = ap + (size_t)va.offset;
    //     const T* bp0 = bp + (size_t)vb.offset;
    //     T*       op0 = op + (size_t)vo.offset;

    //     // Tuneable: small j-tile. 32/64 are typical.
    //     constexpr size_t BJ = 64;

    //     cpu::parallel_for(0, M, [&](size_t i0, size_t i1) {
    //         for (size_t i = i0; i < i1; ++i) {
    //             const size_t arow = i * K;
    //             const size_t crow = i * N;

    //             for (size_t jb = 0; jb < N; jb += BJ) {
    //                 const size_t jend = std::min(N, jb + BJ);

    //                 for (size_t j = jb; j < jend; ++j) {
    //                     T sum = 0;
    //                     // bp0[k*N + j] is strided by N -> not great, but correct.
    //                     for (size_t k = 0; k < K; ++k) {
    //                         sum += ap0[arow + k] * bp0[k * N + j];
    //                     }
    //                     op0[crow + j] = sum;
    //                 }
    //             }
    //         }
    //     });

    //     return;
    // }
    // Fast path 1: NN (A NN, B NN, Out NN)
    if (fast_packed && vb.is_rowmaj_nn_2d()) {
        const T* ap0 = ap + (size_t)va.offset;
        const T* bp0 = bp + (size_t)vb.offset;
        T*       op0 = op + (size_t)vo.offset;

        // Tune these. Start here on M2:
        constexpr size_t BJ = 128;  // j tile
        constexpr size_t BK = 64;   // k tile

        cpu::parallel_for((size_t)0, M, [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                T* crow = op0 + i * N;

                // Initialize output row once
                for (size_t j = 0; j < N; ++j) crow[j] = T(0);

                const T* arow = ap0 + i * K;

                for (size_t kk = 0; kk < K; kk += BK) {
                    const size_t kend = std::min(K, kk + BK);

                    for (size_t jb = 0; jb < N; jb += BJ) {
                        const size_t jend = std::min(N, jb + BJ);

                        for (size_t k = kk; k < kend; ++k) {
                            const T a = arow[k];
                            const T* brow = bp0 + k * N; // contiguous across j

                            for (size_t j = jb; j < jend; ++j) {
                                crow[j] += a * brow[j];
                            }
                        }
                    }
                }
            }
        });

        return;
    }

    // Fast path 2: TN (A NN, B TN, Out NN)
    // B is laid out as strides [1, K], so bp0[j*K + k] is contiguous over k.
    if (fast_packed && vb.is_rowmaj_tn_2d()) {
        const T* ap0 = ap + (size_t)va.offset;
        const T* bp0 = bp + (size_t)vb.offset;
        T*       op0 = op + (size_t)vo.offset;

        cpu::parallel_for(0, M, [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                const size_t arow = i * K;
                const size_t crow = i * N;

                for (size_t j = 0; j < N; ++j) {
                    const size_t bcol = j * K; // contiguous along K
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k) {
                        sum += ap0[arow + k] * bp0[bcol + k];
                    }
                    op0[crow + j] = sum;
                }
            }
        });

        return;
    }

    // Generic stride-aware path
    for (size_t i = 0; i < M; ++i) {
        const size_t a_row = (size_t)va.offset + i*(size_t)va.strides[0];
        const size_t o_row = (size_t)vo.offset + i*(size_t)vo.strides[0];

        for (size_t j = 0; j < N; ++j) {
            const size_t b_col = (size_t)vb.offset + j*(size_t)vb.strides[1];
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += ap[a_row + k*(size_t)va.strides[1]] *
                       bp[b_col + k*(size_t)vb.strides[0]];
            }
            op[o_row + j*(size_t)vo.strides[1]] = sum;
        }
    }
}

template<typename T>
inline void permute_view_kernel(const Buffer& a, const backend::View& va,
                                Buffer& out, const backend::View& vo,
                                const std::vector<size_t>& axes) {
    const T* ap = static_cast<const T*>(a.data());
    T* op = static_cast<T*>(out.data());
    const size_t nout = out.size_bytes()/sizeof(T);
    if (!nout) return;

    bool identity = true;
    for (size_t i=0;i<axes.size();++i) if (axes[i]!=i) { identity=false; break; }
    if (identity && va.is_contiguous() && vo.is_contiguous() &&
        va.rank == vo.rank && va.is_offset_zero() && vo.is_offset_zero()) {
        bool same = true;
        for (uint32_t i=0;i<va.rank;++i) if (va.shape[i]!=vo.shape[i]) { same=false; break; }
        if (same) {
            std::memcpy(op, ap, out.size_bytes());
            return;
        }
    }

    std::vector<size_t> ocoords, icoords(axes.size());
    std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    for (size_t lin=0; lin<nout; ++lin) {
        coords_from_linear(lin, out_shape, ocoords);
        for (size_t d=0; d<axes.size(); ++d) icoords[axes[d]] = ocoords[d];
        const size_t ai = index_from_coords(va, icoords);
        const size_t oi = index_from_coords(vo, ocoords);
        op[oi] = ap[ai];
    }
}

template<typename T>
inline void broadcast_view_kernel(const Buffer& a, const backend::View& va,
                                  Buffer& out, const backend::View& vo) {
    const T* ap = static_cast<const T*>(a.data());
    T* op = static_cast<T*>(out.data());
    const size_t nout = out.size_bytes()/sizeof(T);
    if (!nout) return;

    if (va.is_contiguous() && vo.is_contiguous() &&
        va.rank == vo.rank && va.is_offset_zero() && vo.is_offset_zero()) {
        bool same = true;
        for (uint32_t i=0;i<va.rank;++i) if (va.shape[i]!=vo.shape[i]) { same=false; break; }
        if (same) {
            std::memcpy(op, ap, out.size_bytes());
            return;
        }
    }

    std::vector<size_t> ocoords, icoords;
    std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    for (size_t lin=0; lin<nout; ++lin) {
        coords_from_linear(lin, out_shape, ocoords);
        map_out_to_in_coords_broadcast(ocoords, va, icoords);
        const size_t ai = index_from_coords(va, icoords);
        const size_t oi = index_from_coords(vo, ocoords);
        op[oi] = ap[ai];
    }
}

template<typename T>
inline void slice_forward_view_kernel(const Buffer& a, const backend::View& va,
                                      Buffer& out, const backend::View& vo,
                                      const std::vector<size_t>& begin) {
    const T* ap = static_cast<const T*>(a.data());
    T* op = static_cast<T*>(out.data());
    const size_t nout = out.size_bytes()/sizeof(T);
    if (!nout) return;

    std::vector<size_t> ocoords, icoords(begin.size());
    std::vector<uint32_t> out_shape(vo.shape, vo.shape + vo.rank);
    for (size_t lin=0; lin<nout; ++lin) {
        coords_from_linear(lin, out_shape, ocoords);
        for (size_t d=0; d<begin.size(); ++d) icoords[d] = begin[d] + ocoords[d];
        const size_t ai = index_from_coords(va, icoords);
        const size_t oi = index_from_coords(vo, ocoords);
        op[oi] = ap[ai];
    }
}

template<typename T>
inline void slice_backward_scatter_add_view_kernel(const Buffer& grad_out, const backend::View& vgo,
                                                   Buffer& grad_in,  const backend::View& vgi,
                                                   const std::vector<size_t>& begin) {
    const T* gop = static_cast<const T*>(grad_out.data());
    T* gip = static_cast<T*>(grad_in.data());
    const size_t nout = grad_out.size_bytes()/sizeof(T);
    if (!nout) return;

    std::vector<size_t> ocoords, icoords(begin.size());
    std::vector<uint32_t> out_shape(vgo.shape, vgo.shape + vgo.rank);
    for (size_t lin=0; lin<nout; ++lin) {
        coords_from_linear(lin, out_shape, ocoords);
        for (size_t d=0; d<begin.size(); ++d) icoords[d] = begin[d] + ocoords[d];
        const size_t gi = index_from_coords(vgi, icoords);
        const size_t go = index_from_coords(vgo, ocoords);
        gip[gi] += gop[go];
    }
}

template<typename T>
inline void copy_view_kernel(const Buffer& src, const backend::View& vs, Buffer& dst, const backend::View& vd) {
    const T* sp = static_cast<const T*>(src.data());
    T*       dp = static_cast<T*>(dst.data());
    const size_t nout = vd.numel;
    if (!nout) return;

    std::vector<size_t> ocoords, icoords;
    std::vector<uint32_t> out_shape(vd.shape, vd.shape + vd.rank);
    for (size_t lin=0; lin<nout; ++lin) {
        coords_from_linear(lin, out_shape, ocoords);
        map_out_to_in_coords_broadcast(ocoords, vs, icoords);
        const size_t si = index_from_coords(vs, icoords);
        const size_t di = index_from_coords(vd, ocoords);
        dp[di] = sp[si];
    }
    // TODO: fixed array helpers.
    // // Fixed-size coordinate buffers (rank <= kMaxRank)
    // uint32_t ocoords[backend::kMaxRank];
    // uint32_t icoords[backend::kMaxRank];

    // for (size_t lin = 0; lin < nout; ++lin) {
    //     coords_from_linear_fixed(lin, vd, ocoords); // out coords
    //     map_out_to_in_coords_broadcast_fixed(ocoords, vd, vs, icoords); // in coords (broadcast-aware)

    //     const size_t si = index_from_coords_fixed(vs, icoords);
    //     const size_t di = index_from_coords_fixed(vd, ocoords);

    //     dp[di] = sp[si];
    // }
}


} // namespace cpu
} // namespace backend
} // namespace cppgrad
