// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/nn/module.h"

namespace cppgrad {
namespace nn {


class ReLU : public Module {
public:
    ReLU() = default;

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) override {
        return functional::relu(input);
    }
};

class Tanh : public Module {
public:
    Tanh() = default;

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) override {
        return functional::tanh(input);
    }
};


} // namespace nn
} // namespace cppgrad
