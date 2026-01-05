// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/nn/mlp.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/activations.h"

namespace cppgrad {
namespace nn {

MLP::MLP(int in_features, int hidden_size, int out_features) {
    // Register the layers in order. The base class forward pass will handle them.
    register_modules({
        std::make_shared<Linear>(in_features, hidden_size),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(hidden_size, hidden_size),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(hidden_size, out_features)
    });
}

MLP::~MLP() = default;

} // namespace nn
} // namespace cppgrad
