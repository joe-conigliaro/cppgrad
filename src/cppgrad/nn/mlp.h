// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/nn/module.h"
#include "cppgrad/nn/activations.h"

namespace cppgrad::nn {

class MLP : public Module {
public:
    MLP(int in_features, int hidden_size, int out_features);
    ~MLP() override;
};

} // namespace cppgrad:nn
