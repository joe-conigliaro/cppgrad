// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/executor/interpreter/interpreter_executor.h"
namespace cppgrad {
namespace ir {

GraphContext& GraphContext::instance() {
    static thread_local GraphContext ctx;
    return ctx;
}

thread_local int GraphContext::s_scope_depth = 0;

void GraphContext::schedule_realization(const utils::Ref<const Tensor>& t) {
    if (!t) return;

    // Already realized => nothing to do.
    if (t->realized_buffer()) return;

    if (t->_last_enqueued_token == _generation) return;
    t->_last_enqueued_token = _generation;

    _to_realize.push_back(t);
}

void GraphContext::realize_all() {
    if (_to_realize.empty()) return;

    executor::interpreter::InterpreterExecutor executor;
    executor.realize_many(_to_realize);

    _to_realize.clear();
}

void GraphContext::reset() {
    _arena.reset();
    _to_realize.clear();

    _generation++;
    // overflow wrap around (unlikely to be reached)
    if (_generation == 0) _generation = 1;
}

GraphScope::GraphScope() {
    if (GraphContext::s_scope_depth == 0) {
        GraphContext::instance().reset();
    }
    GraphContext::s_scope_depth++;
}

GraphScope::~GraphScope() {
    GraphContext::s_scope_depth--;
    if (GraphContext::s_scope_depth == 0) {
        GraphContext::instance().realize_all();
        GraphContext::instance().reset();
    }
}

void GraphScope::flush() {
    GraphContext::instance().realize_all();
}

} // namespace ir
} // namespace cppgrad
