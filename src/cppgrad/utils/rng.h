// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <random>
#include <cstdint>

namespace cppgrad {
namespace utils {

// Single global RNG for reproducibility.
inline std::mt19937& global_rng() {
  static std::mt19937 rng(42u);
  return rng;
}

} // namespace utils
} // namespace cppgrad
