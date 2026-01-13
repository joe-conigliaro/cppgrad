// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

namespace cppgrad {
namespace ir {

struct GradMode {
    inline static thread_local bool enabled = true;
};

struct NoGradScope {
    bool prev;
    NoGradScope() : prev(GradMode::enabled) { GradMode::enabled = false; }
    ~NoGradScope() { GradMode::enabled = prev; }
};

} // namespace ir
} // namespace cppgrad
