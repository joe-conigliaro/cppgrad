// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <vector>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace cppgrad {
namespace utils {
namespace vector {

namespace detail {
template<typename T>
inline std::string vec_to_string_impl(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i=0; i<v.size(); ++i) { oss << v[i]; if (i+1 < v.size()) oss << ","; }
    oss << "]";
    return oss.str();
}
} // namespace detail

inline std::string to_string(const std::vector<int>& v) {
    return detail::vec_to_string_impl(v);
}

inline std::string to_string(const std::vector<size_t>& v) {
    return detail::vec_to_string_impl(v);
}

inline size_t numel(const std::vector<size_t>& shape) {
    // A scalar has 1 element
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

inline std::vector<int> normalize_axes(const std::vector<int>& axes, int rank) {
    std::vector<int> ax = axes;
    for (int& a : ax) { if (a < 0) a += rank; }
    // Clamp and validate after normalization
    for (int a : ax) {
        if (a < 0 || a >= rank) {
            throw std::runtime_error("normalize_axes: axis out of bounds");
        }
    }
    std::sort(ax.begin(), ax.end());
    ax.erase(std::unique(ax.begin(), ax.end()), ax.end());
    return ax;
}


} // namespace vector
} // namespace utils
} // namespace cppgrad
