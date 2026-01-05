// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/utils/arena.h"

namespace cppgrad {
namespace ir {

class Tensor;

// Manages arena allocation and batched realization for a compute scope
class GraphContext {
public:
    static GraphContext& instance(); // thread-local singleton
    static bool active() noexcept { return s_scope_depth > 0; }

    size_t generation() const noexcept { return _generation; }

    // Allocate a node in the arena (raw pointer)
    template<typename T, typename... Args>
    T* alloc(Args&&... args) {
        return _arena.alloc<T>(std::forward<Args>(args)...);
    }

    // Allocate a tensor node and return a `utils::Ref<ir::Tensor>`
    template<typename... Args>
    utils::Ref<ir::Tensor> make_node(Args&&... args) {
        auto* p = _arena.alloc<ir::Tensor>(std::forward<Args>(args)...);
        return utils::Ref<ir::Tensor>(p);
    }

    // Schedule a node for realization; no-op if already realized or already scheduled in this generation.
    void schedule_realization(const utils::Ref<const Tensor>& t);

    // Realize all scheduled nodes via the interpreter
    void realize_all();

    // Reset the arena and schedule (invalidates arena-backed nodes)
    void reset();

    // Debug
    bool arena_owns(const void* p) const noexcept { return _arena.owns(p); }

private:
    GraphContext() = default;
    GraphContext(const GraphContext&) = delete;
    GraphContext& operator=(const GraphContext&) = delete;

    utils::Arena _arena;
    std::vector<utils::Ref<const Tensor>> _to_realize;

    size_t _generation = 1; // start at 1. 0 reserved for heap tensors.
    static thread_local int s_scope_depth;

    friend class AutoGraphScope;
};

// RAII-scoped graph lifetime
class AutoGraphScope {
public:
    AutoGraphScope();
    ~AutoGraphScope();

    AutoGraphScope(const AutoGraphScope&) = delete;
    AutoGraphScope& operator=(const AutoGraphScope&) = delete;

    static void flush();
};

} // namespace ir
} // namespace cppgrad
