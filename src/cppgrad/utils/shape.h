// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <vector>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include "cppgrad/utils/vector.h"

namespace cppgrad {
namespace utils {
namespace shape {

inline std::vector<size_t> get_broadcast_shape(const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape) {
    auto max_rank = std::max(a_shape.size(), b_shape.size());
    std::vector<size_t> out_shape(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim_a = (i < a_shape.size()) ? a_shape[a_shape.size() - 1 - i] : 1;
        size_t dim_b = (i < b_shape.size()) ? b_shape[b_shape.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Shapes are not broadcastable.");
        }
        out_shape[max_rank - 1 - i] = std::max(dim_a, dim_b);
    }
    return out_shape;
}

inline std::vector<size_t> get_reduce_shape(const std::vector<size_t>& in_shape, const std::vector<int>& axes, bool keep_dims) {
    // If axes is empty, it means reduce over all dimensions.
    if (axes.empty()) {
        if (keep_dims) {
            return std::vector<size_t>(in_shape.size(), 1); // e.g., {2, 3} -> {1, 1}
        } else {
            return {1}; // Full reduction to a scalar tensor
        }
    }

    std::vector<size_t> out_shape;
    std::vector<bool> axes_to_reduce(in_shape.size(), false);
    for (int axis : axes) {
        int current_axis = axis < 0 ? axis + static_cast<int>(in_shape.size()) : axis;
        if (current_axis < 0 || current_axis >= static_cast<int>(in_shape.size())) {
            throw std::runtime_error("get_reduce_shape: axis out of bounds");
        }
        axes_to_reduce[current_axis] = true;
    }

    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (axes_to_reduce[i]) {
            if (keep_dims) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(in_shape[i]);
        }
    }

    // If the result of a reduction is a scalar (e.g., reducing {5} by axis {0}),
    // the shape should be {1}, not {}.
    if (out_shape.empty()) {
        return {1};
    }

    return out_shape;
}

inline std::vector<int> get_reduce_axes(const std::vector<size_t>& larger_shape, const std::vector<size_t>& smaller_shape) {
    std::vector<int> axes;
    int rank_diff = larger_shape.size() - smaller_shape.size();
    // Axes that exist in the larger shape but not the smaller one
    for (int i = 0; i < rank_diff; ++i) {
        axes.push_back(i);
    }
    // Axes that were broadcast from 1 to something > 1
    for (size_t i = 0; i < smaller_shape.size(); ++i) {
        size_t L = larger_shape[i + rank_diff];
        size_t S = smaller_shape[i];
        if (S == 1 && L > 1) {
            axes.push_back(static_cast<int>(i + rank_diff));
        } else if (S != L) {
            throw std::runtime_error("get_reduce_axes: shapes not broadcast-compatible");
        }
    }
    return axes;
}

inline size_t get_reduce_count(const std::vector<size_t>& in_shape, const std::vector<int>& axes) {
    // If axes empty, treat as "reduce all dims"
    if (axes.empty()) {
        return cppgrad::utils::vector::numel(in_shape);
    }
    const int rank = static_cast<int>(in_shape.size());
    size_t count = 1;
    for (int ax : axes) {
        int a = ax < 0 ? ax + rank : ax;
        if (a < 0 || a >= rank) throw std::runtime_error("get_reduce_count: axis out of bounds");
        count *= in_shape[static_cast<size_t>(a)];
    }
    return count;
}

inline std::vector<size_t> permute(const std::vector<size_t>& in_shape, const std::vector<size_t>& axes) {
    std::vector<size_t> out_shape(in_shape.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        out_shape[i] = in_shape[axes[i]];
    }
    return out_shape;
}

// Calculates row-major strides for a given shape.
// For a shape {A, B, C}, the strides are {B*C, C, 1}.
inline std::vector<size_t> row_major_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {}; // A scalar has no strides.
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0; ) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Row-major contiguous (dense) layout, independent of storage offset.
inline bool is_row_major_contiguous(const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    return row_major_strides(shape) == strides;
}

// Row-major contiguous (dense) layout + identity/base-aligned (offset == 0).
inline bool is_row_major_identity(const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset) {
    return offset == 0 && is_row_major_contiguous(shape, strides);
}

// Allocating: flat index -> coordinates
// Converts a flat index to N-dimensional coordinates using pre-calculated row-major strides.
inline std::vector<size_t> coords_from_index(size_t idx, const std::vector<size_t>& strides) {
    if (strides.empty()) return {idx}; // scalar convention
    std::vector<size_t> coords(strides.size());
    size_t rem = idx;
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t s = strides[i];
        coords[i] = s ? (rem / s) : 0;
        rem -= coords[i] * s;
    }
    return coords;
}

// No-alloc: flat index -> coordinates (writes into coords, which must be sized or will be resized)
inline void coords_from_index_inplace(size_t idx, const std::vector<size_t>& strides, std::vector<size_t>& coords) {
    coords.resize(strides.size());
    size_t rem = idx;
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t s = strides[i];
        coords[i] = s ? (rem / s) : 0;
        rem -= coords[i] * s;
    }
}

// coordinates -> flat index
inline size_t index_from_coords(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t idx = 0;
    for (size_t i = 0; i < coords.size(); ++i) idx += coords[i] * strides[i];
    return idx;
}

// Map possibly-negative axes to [0, rank). Throws on OOB.
inline std::vector<int> normalize_axes(const std::vector<int>& axes, size_t rank) {
    std::vector<int> out;
    out.reserve(axes.size());
    for (int ax : axes) {
        int a = ax < 0 ? ax + static_cast<int>(rank) : ax;
        if (a < 0 || a >= static_cast<int>(rank))
            throw std::runtime_error("normalize_axes: axis out of bounds");
        out.push_back(a);
    }
    return out;
}

// Sort and dedup when order doesn’t matter (e.g., reductions, sum over multiple axes).
inline std::vector<int> unique_sorted_axes(std::vector<int> axes_norm) {
    std::sort(axes_norm.begin(), axes_norm.end());
    axes_norm.erase(std::unique(axes_norm.begin(), axes_norm.end()), axes_norm.end());
    return axes_norm;
}

// Convenience: normalize + unique + sorted in one call.
inline std::vector<int> normalize_unique_sorted_axes(const std::vector<int>& axes, size_t rank) {
    return unique_sorted_axes(normalize_axes(axes, rank));
}

} // namespace shape
} // namespace utils
} // namespace cppgrad
