// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <tuple>
#include <cstddef>
#include "cppgrad/nn/module.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/backend/device_manager.h"

namespace cppgrad {
namespace nn {

enum class Init { Default, KaimingUniform, KaimingNormal, XavierUniform, XavierNormal };

class Linear : public Module {
public:
    utils::Ref<ir::Tensor> weight;
    utils::Ref<ir::Tensor> bias;

    Linear(size_t in_features,
           size_t out_features,
           bool use_bias = true,
           Init init = Init::Default,
           backend::DeviceType device = backend::DeviceManager::default_device()) {

        auto [w_min, w_max, use_uniform, stddev] = limits_for_init(init, in_features, out_features);

        // Initializers as graphs
        auto w_init = use_uniform
            ? ir::uniform({in_features, out_features}, w_min, w_max, device)
            : ir::normal({in_features, out_features}, 0.0f, stddev, device);

        utils::Ref<ir::Tensor> b_init;
        if (use_bias) {
            float bias_bound = 1.0f / std::sqrt(static_cast<float>(in_features));
            b_init = ir::uniform({1, out_features}, -bias_bound, bias_bound, device);
        }

        // Convert initializers into canonical leaf parameters
        weight = ir::parameterize(w_init);
        bias = use_bias ? ir::parameterize(b_init) : nullptr;

        // Register
        register_parameter("weight", weight);
        if (use_bias) register_parameter("bias", bias);
    }

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) override {
        auto output = ir::matmul(input, weight);
        if (bias) output = ir::add(output, bias);
        return output;
    }

private:
    static std::tuple<float, float, bool, float>
    limits_for_init(Init init, size_t fan_in, size_t fan_out) {
        float fi = static_cast<float>(fan_in);
        float fo = static_cast<float>(fan_out);
        switch (init) {
            case Init::Default:
            case Init::KaimingUniform: {
                float limit = std::sqrt(6.0f / fi);
                return {-limit, limit, true, 0.0f};
            }
            case Init::KaimingNormal: {
                float stddev = std::sqrt(2.0f / fi);
                return {0.0f, 0.0f, false, stddev};
            }
            case Init::XavierUniform: {
                float limit = std::sqrt(6.0f / (fi + fo));
                return {-limit, limit, true, 0.0f};
            }
            case Init::XavierNormal: {
                float stddev = std::sqrt(2.0f / (fi + fo));
                return {0.0f, 0.0f, false, stddev};
            }
        }
        float limit = std::sqrt(6.0f / fi);
        return {-limit, limit, true, 0.0f};
    }
};

} // namespace nn
} // namespace cppgrad
