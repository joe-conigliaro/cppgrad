// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// KV / recurrent-state cache equivalence.
//
// Cached greedy decoding (Qwen3Model::generate -- one prefill + one-token decode steps, with a
// K/V cache on full-attention layers and a recurrent + conv state cache on linear-attention
// layers) must produce exactly the same tokens as a naive O(n^2) full recompute (calling the
// non-cached forward() on the growing sequence each step). This holds for any weights, so a tiny
// random hybrid model (3 linear + 1 full attention layer) suffices.
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "tests/helpers.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::Qwen3Model;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::LayerType;

// Deterministic materialized random tensor in [-scale, scale] (uniform() re-draws per eval).
static utils::Ref<ir::Tensor> rand_tensor(const std::vector<size_t>& shape, float scale,
                                          uint32_t& seed, backend::DeviceType dev) {
    size_t n = 1; for (auto d : shape) n *= d;
    std::vector<float> v(n);
    for (auto& f : v) { seed = seed * 1664525u + 1013904223u; f = (((seed >> 8) / float(1u << 24)) * 2.0f - 1.0f) * scale; }
    return ir::from_vector<float>(v, shape, dev);
}

static int argmax_last_row(const utils::Ref<ir::Tensor>& logits, size_t len, size_t V) {
    auto lv = logits->to_vector<float>();           // [1, len, V]
    const float* row = lv.data() + (len - 1) * V;
    int best = 0; float bv = row[0];
    for (size_t i = 1; i < V; ++i) if (row[i] > bv) { bv = row[i]; best = (int)i; }
    return best;
}

int main() {
    try {
        TEST_HEADER("KV cache: cached generate == full recompute");
        auto dev = backend::DeviceManager::default_device_type();

        Qwen3Config cfg = Qwen3Config::get_27b_qwen3_6();
        cfg.hidden_size            = 32;
        cfg.num_hidden_layers      = 4;
        cfg.intermediate_size      = 64;
        cfg.vocab_size             = 50;
        cfg.num_attention_heads    = 4;
        cfg.num_key_value_heads    = 2;
        cfg.head_dim               = 8;
        cfg.linear_num_key_heads   = 2;
        cfg.linear_key_head_dim    = 4;
        cfg.linear_num_value_heads = 4;
        cfg.linear_value_head_dim  = 4;
        cfg.linear_conv_kernel_dim = 4;
        cfg.partial_rotary_factor  = 0.5;
        cfg.layer_types = {LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION,
                           LayerType::LINEAR_ATTENTION, LayerType::FULL_ATTENTION};

        Qwen3Model model(cfg, dev);

        // Give the embedding / head / final norm real (materialized) values so logits are not
        // degenerate; block weights keep their (frozen) random init.
        uint32_t seed = 999u;
        model.embedding_weight  = rand_tensor({(size_t)cfg.vocab_size, (size_t)cfg.hidden_size}, 0.5f, seed, dev);
        model.lm_head->weight   = rand_tensor({(size_t)cfg.hidden_size, (size_t)cfg.vocab_size}, 0.5f, seed, dev);
        model.final_norm_weight = rand_tensor({(size_t)cfg.hidden_size}, 0.1f, seed, dev);

        std::vector<int32_t> prompt = {3, 7, 1, 9, 2};
        const int N = 6;
        const size_t V = (size_t)cfg.vocab_size;

        // Reference: naive full recompute each step.
        std::vector<int32_t> ids = prompt, ref;
        for (int i = 0; i < N; ++i) {
            auto in = ir::from_vector<int32_t>(ids, {1, ids.size()}, dev);
            int next = argmax_last_row(model.forward(in), ids.size(), V);
            ref.push_back(next);
            ids.push_back(next);
        }

        // Cached.
        std::vector<int32_t> cached = model.generate(prompt, N);

        EXPECT_TRUE(cached.size() == (size_t)N, "cached produced N tokens");
        bool match = (cached == ref);
        EXPECT_TRUE(match, "cached tokens == full-recompute tokens");
        std::cout << "  reference:";
        for (int t : ref) std::cout << " " << t;
        std::cout << "\n  cached:   ";
        for (int t : cached) std::cout << " " << t;
        std::cout << "\n";

        if (g_fail_count == 0) { std::cout << "\nALL TESTS PASSED (qwen3 kv cache)\n"; return 0; }
        std::cerr << "\nTESTS FAILED (qwen3 kv cache): " << g_fail_count << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
