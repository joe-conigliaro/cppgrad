// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include "cppgrad/nn/functional.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/module.h"

namespace cppgrad {
namespace nn {

/* GatedFFN - gated feed-forward network: inner_act(x @ gate_proj) * (x @ up_proj) @ down_proj
 *
 * The standard FFN pattern in modern LLMs. Unlike a sequential MLP, GatedFFN has a branching
 * compute graph: the input is projected through both gate_proj and up_proj in parallel, the
 * gate path is activated, the two are multiplied elementwise, then projected down. Projections
 * are bias-free (the convention for gated FFNs).
 *
 * Variants (differ only in the inner activation):
 *   SwiGLU (silu) - Llama 2/3/4, Mistral, Mixtral, Qwen, Gemma (most common)
 *   GeGLU  (gelu) - T5, Flan-T5, PaLM variants
 *   ReGLU  (relu) - some efficient architectures
 *
 * Like Linear, dense-vs-quantized is decided at load time (per projection), not by type. Pass
 * lazy=true for checkpoint loading (deferred storage; the loader fills each projection).
 *
 * Parameters:
 *   in_features  : input dimension
 *   hidden_size  : intermediate (hidden) dimension (gate and up projection width)
 *   out_features : output dimension (down projection width; often == in_features)
 *   inner_act    : which activation to apply to the gate path
 */
class GatedFFN : public Module {
public:
    enum class InnerAct {
        SILU,  // SwiGLU - silu(gate) * up
        GELU,  // GeGLU - gelu(gate) * up
        RELU,  // ReGLU - relu(gate) * up
    };

    std::shared_ptr<Linear> gate_proj;
    std::shared_ptr<Linear> up_proj;
    std::shared_ptr<Linear> down_proj;

    GatedFFN(size_t in_features, size_t hidden_size, size_t out_features,
             InnerAct inner_act = InnerAct::SILU,
             Init init = Init::Default,
             backend::DeviceType device_type = backend::DeviceManager::default_device_type(),
             bool lazy = false)
        : _inner_act(inner_act)
    {
        gate_proj = std::make_shared<Linear>(in_features, hidden_size, /*use_bias=*/false, init, device_type, lazy);
        up_proj   = std::make_shared<Linear>(in_features, hidden_size, /*use_bias=*/false, init, device_type, lazy);
        down_proj = std::make_shared<Linear>(hidden_size, out_features, /*use_bias=*/false, init, device_type, lazy);

        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
    }

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) override {
        auto gate = (*gate_proj)(input);
        auto up   = (*up_proj)(input);

        auto activated = [&]() -> utils::Ref<ir::Tensor> {
            switch (_inner_act) {
                case InnerAct::SILU: return functional::silu(gate);
                case InnerAct::GELU: return functional::gelu(gate);
                case InnerAct::RELU: return functional::relu(gate);
            }
            return functional::silu(gate);
        }();

        return (*down_proj)(activated * up);
    }

private:
    InnerAct _inner_act;
};

} // namespace nn
} // namespace cppgrad
