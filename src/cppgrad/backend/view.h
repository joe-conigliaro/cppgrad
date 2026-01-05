// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/utils/shape.h"

namespace cppgrad {
namespace backend {

constexpr int kMaxRank = 8;

// Bit flags for fast paths
enum ViewFlags : uint32_t {
    // General
    VIEW_CONTIGUOUS          = 1u << 0,  // dense row-major layout for this shape (offset may be non-zero)
    // VIEW_ROW_MAJOR           = 1u << 1,  // strides equal row-major for shape (offset may be non-zero)
    VIEW_IDENTITY            = 1u << 2,  // row-major and offset==0
    VIEW_OFFSET_ZERO         = 1u << 3,  // offset == 0
    VIEW_HAS_ZERO_STRIDE     = 1u << 4,  // any stride == 0 (broadcast axis present)
    VIEW_LAST_AXIS_CONTIG    = 1u << 5,  // stride[last] == 1 (rank > 0)
    VIEW_RANK0               = 1u << 6,  // rank == 0 (scalar)
    VIEW_SINGLETON           = 1u << 7,  // product(shape) == 1

    // 2D layout classification (for matmul planning)
    VIEW2D_LAYOUT_SHIFT      = 8,
    VIEW2D_LAYOUT_MASK       = 0xFu << VIEW2D_LAYOUT_SHIFT,
    VIEW2D_ROWMAJ_NN         = 1u << VIEW2D_LAYOUT_SHIFT, // strides [K,1] (inner contiguous)
    VIEW2D_ROWMAJ_TN         = 2u << VIEW2D_LAYOUT_SHIFT  // strides [1,K] (transpose-like)
};

struct View {
    uint32_t rank = 0;
    uint32_t shape[kMaxRank]   = {0};
    uint32_t strides[kMaxRank] = {0};
    uint32_t offset = 0;
    uint32_t flags  = 0;
    size_t   numel  = 1;

    // General fast path
    bool is_contiguous()        const { return (flags & VIEW_CONTIGUOUS)       != 0; }
    // bool is_row_major()         const { return (flags & VIEW_ROW_MAJOR)        != 0; }
    bool is_identity()          const { return (flags & VIEW_IDENTITY)         != 0; }
    bool is_offset_zero()       const { return (flags & VIEW_OFFSET_ZERO)      != 0; }
    bool has_zero_stride()      const { return (flags & VIEW_HAS_ZERO_STRIDE)  != 0; }
    bool last_axis_contiguous() const { return (flags & VIEW_LAST_AXIS_CONTIG) != 0; }
    bool is_rank0()             const { return (flags & VIEW_RANK0)            != 0; }
    bool is_singleton()         const { return (flags & VIEW_SINGLETON)        != 0; }
    // 2D helpers
    bool is_rowmaj_nn_2d()      const { return (rank == 2) && ((flags & VIEW2D_ROWMAJ_NN) != 0); }
    bool is_rowmaj_tn_2d()      const { return (rank == 2) && ((flags & VIEW2D_ROWMAJ_TN) != 0); }

    static View from(const ir::AccessMeta& acc) {
        View v{};
        v.rank = static_cast<uint32_t>(acc.shape.size());
        if (v.rank > kMaxRank) throw std::runtime_error("View::from: rank exceeds kMaxRank");

        // Copy shape/strides; compute zero-stride/broadcast
        for (uint32_t i=0;i<v.rank;++i) {
            v.shape[i]   = static_cast<uint32_t>(acc.shape[i]);
            v.strides[i] = static_cast<uint32_t>(acc.strides[i]);
            if (v.strides[i] == 0) v.flags |= VIEW_HAS_ZERO_STRIDE;
        }
        v.offset = static_cast<uint32_t>(acc.offset);

        if (acc.contiguous) v.flags |= VIEW_CONTIGUOUS;
        if (v.offset == 0)  v.flags |= VIEW_OFFSET_ZERO;
        if (v.rank == 0)    v.flags |= VIEW_RANK0;

        // Row-major and identity
        // If AccessMeta reports contiguous, the strides match row-major for this shape.
        // Otherwise, fall back to an explicit row-major stride equality check (offset may be non-zero).
        // bool is_row_major = acc.contiguous || (cppgrad::utils::shape::row_major_strides(acc.shape) == acc.strides);
        // NOTE: for now rely on acc.contiguous, but add debug check. If we find it throws we can investigate.
        bool is_row_major = acc.contiguous;
        #ifdef CPPGRAD_DEBUG
            const bool is_row_major_strides = (cppgrad::utils::shape::row_major_strides(acc.shape) == acc.strides);
            if (is_row_major_strides != acc.contiguous) {
                throw std::runtime_error("AccessMeta.contiguous is stale/inconsistent");
            }
        #endif
        if (is_row_major) {
            // v.flags |= VIEW_ROW_MAJOR;
            if (v.offset == 0) v.flags |= VIEW_IDENTITY;
        }

        // last axis contiguous
        if (v.rank > 0 && v.strides[v.rank-1] == 1) v.flags |= VIEW_LAST_AXIS_CONTIG;

        // singleton
        size_t numel = 1;
        // for (uint32_t i=0;i<v.rank;++i) numel *= std::max<uint32_t>(1, v.shape[i]);
        for (uint32_t i = 0; i < v.rank; ++i) numel *= (size_t)v.shape[i];
        // TODO: use stored numel in backends instead of re-calculating
        v.numel = numel;
        if (numel == 1) v.flags |= VIEW_SINGLETON;

        // 2D layout tags
        if (v.rank == 2) {
            const uint32_t M = v.shape[0], N = v.shape[1];
            const uint32_t s0 = v.strides[0], s1 = v.strides[1];
            if (s1 == 1 && s0 == N)      v.flags |= VIEW2D_ROWMAJ_NN;
            else if (s0 == 1 && s1 == M) v.flags |= VIEW2D_ROWMAJ_TN;
        }
        return v;
    }
};

} // namespace backend
} // namespace cppgrad
