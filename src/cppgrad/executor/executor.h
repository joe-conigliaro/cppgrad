// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include <unordered_set>
#include "cppgrad/backend/device.h"
#include "cppgrad/utils/ref.h"

// Forward declarations.
namespace cppgrad { namespace ir { class Tensor; class AccessMeta; } }

namespace cppgrad {
// Forward declaration.
namespace backend { class Buffer; }
namespace executor {

class Executor {
public:
    virtual ~Executor() = default;

    virtual std::shared_ptr<backend::Buffer> realize(const utils::Ref<const ir::Tensor>& out) = 0;
    virtual void realize_many(const std::vector<utils::Ref<const ir::Tensor>>& outs) = 0;

protected:
    struct DeviceSchedule {
        backend::DeviceType device;
        std::vector<utils::Ref<const ir::Tensor>> schedule; // topo order
    };

    static void build_schedule_dfs(
        const utils::Ref<const ir::Tensor>& t,
        std::unordered_set<utils::Ref<const ir::Tensor>>& visited,
        std::vector<utils::Ref<const ir::Tensor>>& order);

    static std::vector<DeviceSchedule> build_device_schedules(
        const std::vector<utils::Ref<const ir::Tensor>>& outs);
};

} // namespace executor
} // namespace cppgrad
