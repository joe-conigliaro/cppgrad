// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>

#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/tensor.h"

namespace cppgrad::optim {

class Optimizer {
  public:
    Optimizer(std::vector<utils::Ref<ir::Tensor>> params, float lr) : _params(params), _lr(lr) {}

    virtual ~Optimizer() = default;

    void zero_grad() {
        for (auto &p : _params) {
            p->zero_grad();
        }
    }

    virtual void step() = 0;

  protected:
    std::vector<utils::Ref<ir::Tensor>> _params;
    float _lr;
};

} // namespace cppgrad::optim
