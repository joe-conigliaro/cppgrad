// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <vector>
#include <iomanip>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/optim/sgd.h"

using namespace cppgrad;

int main() {
    backend::DeviceManager::instance().init();

    // Data: x in R^{N,1}, y = 2x + 3
    auto x = ir::from_vector<float>({0, 1, 2, 3}, {4, 1});
    auto y = ir::from_vector<float>({3, 5, 7, 9}, {4, 1});

    // Trainable parameters (canonical leaf tensors)
    auto w = ir::parameter({1, 1});
    auto b = ir::parameter({1, 1});

    optim::SGD opt({w, b}, /*lr=*/0.1f);

    for (int step = 0; step < 100; ++step) {
        // One scope per step: builds a graph, then batch-realizes at scope exit.
        ir::GraphScope scope;

        // Forward: yhat = x*w + b
        auto yhat = ir::add(ir::mul(x, w), b);

        // Loss: mean((yhat - y)^2)
        auto diff = ir::sub(yhat, y);
        auto loss = ir::mean(ir::mul(diff, diff));

        opt.zero_grad();
        loss->backward();
        opt.step();

        if (step == 0 || (step + 1) % 10 == 0) {
            // `item()` forces realization of 'loss'
            std::cout << "step " << step+1
                      << " loss=" << std::fixed << std::setprecision(6) << loss->item<float>() << "\n";
        }
    }

    return 0;
}
