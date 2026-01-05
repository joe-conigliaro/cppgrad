// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/optim/optim.h"
#include "cppgrad/ir/tensor_operators.h"

namespace cppgrad {
namespace optim {

class SGD : public Optimizer {
public:
    using Optimizer::Optimizer;

    void step() override {
        for (auto& p : _params) {
            if (p->grad()) {
                // Build the lazy graph for the update.
                auto updated_p = p - (p->grad() * _lr);

                // Use set_parameter_data/copy_into_parameter.
                // auto updated_p_buffer = updated_p->eval();
                // p->set_parameter_data(updated_p_buffer);
                // p->set_requires_grad(true);

                // Use AssignOp graph node.
                p->assign(updated_p)->schedule(); p->set_requires_grad(true);
            }
        }
    }

};

} // namespace optim
} // namespace cppgrad
