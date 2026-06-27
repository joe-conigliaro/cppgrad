// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/executor/executor.h"
#include "cppgrad/utils/ref.h"

// Forward declaration.
namespace cppgrad::backend { class Buffer; }
namespace cppgrad::executor::interpreter {

class InterpreterExecutor : public cppgrad::executor::Executor {
public:
    InterpreterExecutor() = default;

    // Realizes a lazy computation graph into a concrete buffer.
    std::shared_ptr<backend::Buffer> realize(const utils::Ref<const ir::Tensor>& out) override;
    void realize_many(const std::vector<utils::Ref<const ir::Tensor>>& outs) override;

    void realize_scheduled(const std::vector<DeviceSchedule>& schedules);
};

} // namespace cppgrad::executor::interpreter
