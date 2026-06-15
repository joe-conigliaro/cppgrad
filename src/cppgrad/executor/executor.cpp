// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/backend/device.h"
#include "cppgrad/executor/executor.h"

namespace cppgrad {
namespace executor {

void Executor::build_schedule_dfs(
    const utils::Ref<const ir::Tensor>& t,
    std::unordered_set<utils::Ref<const ir::Tensor>>& visited,
    std::vector<utils::Ref<const ir::Tensor>>& order) {
    if (!t || visited.count(t)) return;
    visited.insert(t);
    // An already-realized node is an immutable value: treat it as a scheduling
    // boundary (like a leaf) so we neither recompute it nor walk its history.
    // This turns repeated realize() of overlapping graphs from O(whole-graph)
    // into O(new-work) -- e.g. autoregressive decode reusing the previous step's
    // KV nodes, or backward reusing realized forward activations.
    if (t->realized_buffer()) {
        order.push_back(t);
        return;
    }
    for (const auto& ch : t->children()) {
        build_schedule_dfs(ch, visited, order);
    }
    order.push_back(t);
}

std::vector<Executor::DeviceSchedule>
Executor::build_device_schedules(
    const std::vector<utils::Ref<const ir::Tensor>>& outs) {
    std::unordered_map<backend::DeviceType, std::vector<utils::Ref<const ir::Tensor>>> outs_by_dev;
    for (const auto& o : outs) if (o) outs_by_dev[o->device_type()].push_back(o);

    std::vector<DeviceSchedule> schedules;
    schedules.reserve(outs_by_dev.size());

    for (auto& [dev, dev_outs] : outs_by_dev) {
        std::unordered_set<utils::Ref<const ir::Tensor>> visited;
        std::vector<utils::Ref<const ir::Tensor>> ord;
        ord.reserve(dev_outs.size() * 4);
        for (const auto& out : dev_outs) build_schedule_dfs(out, visited, ord);
        schedules.push_back(DeviceSchedule{dev, std::move(ord)});
    }
    return schedules;
}

} // namespace executor
} // namespace cppgrad
