// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "vector"
#include "cppgrad/utils/shape.h"

namespace cppgrad {
namespace ir {

// Access metadata (logical view of buffer)
struct AccessMeta {
    static AccessMeta from(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset = 0) {
        AccessMeta m;
        m.shape = std::move(shape);
        m.strides = std::move(strides);
        m.offset = offset;
        m.contiguous = cppgrad::utils::shape::is_row_major_contiguous(m.shape, m.strides);
        return m;
    }

    static AccessMeta contiguous_from(std::vector<size_t> shape, size_t offset = 0) {
        AccessMeta m;
        m.shape = std::move(shape);
        m.strides = cppgrad::utils::shape::row_major_strides(m.shape);
        m.offset = offset;
        m.contiguous = true;
        return m;
    }

    static AccessMeta reshape_from(const AccessMeta& p, const std::vector<size_t>& new_shape) {
        if (cppgrad::utils::vector::numel(p.shape) != cppgrad::utils::vector::numel(new_shape))
            throw std::runtime_error("reshape_from: numel mismatch");

        if (!p.contiguous)
            throw std::runtime_error("reshape_from: non-dense view reshape not supported (materialize first)");

        AccessMeta a;
        a.shape = new_shape;
        a.strides = cppgrad::utils::shape::row_major_strides(a.shape);
        a.offset = p.offset;
        a.contiguous = true;
        return a;
    }

    static AccessMeta permute_from(const AccessMeta& p, const std::vector<size_t>& perm) {
        if (perm.size() != p.shape.size()) throw std::invalid_argument("permute: rank mismatch");
        std::vector<char> seen(perm.size(), 0);
        for (size_t i=0;i<perm.size();++i) {
            if (perm[i] >= perm.size() || seen[perm[i]]) throw std::invalid_argument("permute: invalid permutation");
            seen[perm[i]] = 1;
        }
        AccessMeta a;
        a.shape.resize(perm.size());
        a.strides.resize(perm.size());
        for (size_t i=0;i<perm.size();++i) {
            a.shape[i]   = p.shape[perm[i]];
            a.strides[i] = p.strides[perm[i]];
        }
        a.offset = p.offset;
        a.recompute_contiguity();
        return a;
    }

    static AccessMeta broadcast_from(const AccessMeta& p, const std::vector<size_t>& out_shape) {
        const size_t r_in = p.shape.size();
        const size_t r_out = out_shape.size();

        AccessMeta a;
        a.shape   = out_shape;
        a.strides.resize(r_out);
        a.offset  = p.offset;

        // Right-align p.shape against out_shape
        for (size_t i = 0; i < r_out; ++i) {
            size_t out_dim = out_shape[r_out - 1 - i];
            size_t in_dim  = (i < r_in) ? p.shape[r_in - 1 - i] : 1;
            size_t in_stride = (i < r_in) ? p.strides[r_in - 1 - i] : 0;

            if (in_dim == out_dim) {
                a.strides[r_out - 1 - i] = in_stride;
            } else if (in_dim == 1 && out_dim > 1) {
                a.strides[r_out - 1 - i] = 0; // broadcast
            } else {
                throw std::runtime_error("broadcast_from: incompatible shapes");
            }
        }
        a.recompute_contiguity();
        return a;
    }

    static AccessMeta slice_from(const AccessMeta& p, const std::vector<size_t>& begin,
                                 const std::vector<size_t>& end, const std::vector<size_t>& step) {
        const size_t R = p.shape.size();
        if (begin.size()!=R || end.size()!=R || step.size()!=R)
            throw std::invalid_argument("slice_from: rank mismatch");

        AccessMeta a;
        a.shape.resize(R);
        a.strides.resize(R);
        size_t new_offset = p.offset;

        for (size_t d=0; d<R; ++d) {
            if (begin[d] > end[d] || end[d] > p.shape[d]) throw std::out_of_range("slice_from: invalid begin/end");
            if (step[d] == 0) throw std::invalid_argument("slice_from: step=0");
            // Compute size along d
            size_t len = (end[d] > begin[d]) ? ((end[d] - begin[d] + step[d] - 1) / step[d]) : 0;
            a.shape[d]   = len;
            a.strides[d] = p.strides[d] * step[d];
            new_offset  += begin[d] * p.strides[d];
        }
        a.offset = new_offset;
        a.recompute_contiguity();
        return a;
    }

    void recompute_contiguity() {
        contiguous = cppgrad::utils::shape::is_row_major_contiguous(shape, strides);
    }

    std::vector<size_t> shape;     // logical dims
    std::vector<size_t> strides;   // in elements
    size_t offset = 0;             // in elements
    bool contiguous = true;
};

} // namespace ir
} // namespace cppgrad
