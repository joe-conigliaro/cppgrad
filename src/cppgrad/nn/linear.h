// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <cstddef>
#include <tuple>
#include "cppgrad/nn/module.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/ops.h" // ir::QuantParams
#include "cppgrad/backend/device_manager.h"

namespace cppgrad::nn {

enum class Init { Default, KaimingUniform, KaimingNormal, XavierUniform, XavierNormal };

namespace detail {

inline std::tuple<float, float, bool, float>
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

} // namespace detail

/* Linear - a dense OR quantized linear projection behind one forward(x).
 *
 *   dense    : weight [in, out] (+ optional bias)         -> ir::matmul(x, weight)
 *   quantized: qweight [out, in/pack], scales/biases      -> ir::quantized_matmul(...)
 *
 * Whether a layer is quantized is *data*, not a type: the same Linear is built once, then either
 * trained / dense-loaded (weight) or quantized-loaded (qweight/scales/biases) at checkpoint time.
 * forward() branches on the `quantized` flag - a single predictable branch in front of a matmul,
 * so it is free in practice. This keeps containers (GatedFFN, blocks, whole models) untemplated and
 * lets a model switch dense<->quantized at load time without recompiling.
 *
 * Construction:
 *   Linear(in, out)                    - dense, random-init, trainable (MLP / training)
 *   Linear(in, out, use_bias=false)    - dense, no bias (e.g. LLM projections)
 *   Linear(in, out, ..., lazy=true)    - deferred storage, no init; the loader fills either the
 *                                        dense weight or the quantized triple after construction.
 */
class Linear : public Module {
public:
    // Dense weight [in, out] (matmul convention) + optional bias [1, out]. Used when !quantized.
    utils::Ref<ir::Tensor> weight;
    utils::Ref<ir::Tensor> bias;

    // Quantized weights (MLX affine etc.): qweight [out, in/pack_factor], scales/biases [out, groups].
    utils::Ref<ir::Tensor> qweight, scales, biases;
    ir::QuantParams params;
    bool quantized = false;   // set by the loader (or a quantize pass) after construction

    Linear(size_t in_features,
           size_t out_features,
           bool use_bias = true,
           Init init = Init::Default,
           backend::DeviceType device_type = backend::DeviceManager::default_device_type(),
           bool lazy = false)
        : _use_bias(use_bias)
    {
        if (lazy) {
            // Deferred (unallocated) storage, no random init: weights arrive from a checkpoint.
            // The dense weight is created so the dense-load path can rebind it; a quantized load
            // instead fills qweight/scales/biases and flips `quantized`, leaving this unrealized.
            weight = ir::parameter({in_features, out_features}, device_type, common::DType::FLOAT32, false);
            if (use_bias) bias = ir::parameter({1, out_features}, device_type, common::DType::FLOAT32, false);
        } else {
            auto [w_min, w_max, use_uniform, stddev] =
                detail::limits_for_init(init, in_features, out_features);
            auto w_init = use_uniform
                ? ir::uniform({in_features, out_features}, w_min, w_max, device_type)
                : ir::normal({in_features, out_features}, 0.0f, stddev, device_type);
            weight = ir::parameterize(w_init);
            if (use_bias) {
                float bias_bound = 1.0f / std::sqrt(static_cast<float>(in_features));
                bias = ir::parameterize(ir::uniform({1, out_features}, -bias_bound, bias_bound, device_type));
            }
        }

        register_parameter("weight", weight);
        if (use_bias) register_parameter("bias", bias);
    }

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) override {
        if (quantized) return ir::quantized_matmul(input, qweight, {scales, biases}, params);
        auto output = ir::matmul(input, weight);
        if (bias) output = ir::add(output, bias);
        return output;
    }

private:
    bool _use_bias = true;
};

} // namespace cppgrad::nn
