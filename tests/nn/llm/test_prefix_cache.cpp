// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Prefix KV-cache reuse == cold prefill. The shared decode runtime
// (nn::llm::generate_with_prefix_cache) reuses the longest cached EXACT token prefix across requests,
// prefilling only the new suffix. Reuse is an optimization, so for a prompt that EXTENDS a previously
// cached sequence, generation must be bit-identical (greedy) to generating that same prompt on a
// fresh cache. Validates KV reuse (full-attn) AND recurrent-state reuse (linear-attn) on a tiny
// random hybrid model. CPU + Metal (default device).
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/nn/llm/decode_model.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::Qwen3Model;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::LayerType;
namespace llm = cppgrad::nn::llm;

static utils::Ref<ir::Tensor> rand_tensor(const std::vector<size_t>& shape, float scale,
                                          uint32_t& seed, backend::DeviceType dev) {
    size_t n = 1; for (auto d : shape) n *= d;
    std::vector<float> v(n);
    for (auto& f : v) { seed = seed*1664525u + 1013904223u; f = (((seed>>8)/float(1u<<24))*2.f-1.f)*scale; }
    return ir::from_vector<float>(v, shape, dev);
}

// Greedy generate via the shared prefix-cache runtime; returns the generated tokens.
static std::vector<int32_t> gen(Qwen3Model& m, llm::PrefixCacheSession& sess,
                                const std::vector<int32_t>& prompt, int N) {
    std::unordered_set<int32_t> stop;  // empty: produce exactly N tokens
    return llm::generate_with_prefix_cache(m, sess, prompt, N,
                                           [](int32_t){ return true; }, stop, llm::SamplingParams{});
}

int main() {
    try {
        TEST_HEADER("prefix KV-cache reuse == cold prefill");
        auto dev = backend::DeviceManager::default_device_type();

        Qwen3Config cfg = Qwen3Config::get_27b();
        cfg.hidden_size=32; cfg.num_hidden_layers=4; cfg.intermediate_size=64; cfg.vocab_size=50;
        cfg.num_attention_heads=4; cfg.num_key_value_heads=2; cfg.head_dim=8;
        cfg.linear_num_key_heads=2; cfg.linear_key_head_dim=4; cfg.linear_num_value_heads=4;
        cfg.linear_value_head_dim=4; cfg.linear_conv_kernel_dim=4; cfg.partial_rotary_factor=0.5;
        cfg.layer_types = {LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION,
                           LayerType::LINEAR_ATTENTION, LayerType::FULL_ATTENTION};

        Qwen3Model model(cfg, dev);
        uint32_t seed = 999u;
        model.embedding_weight  = rand_tensor({(size_t)cfg.vocab_size, (size_t)cfg.hidden_size}, 0.5f, seed, dev);
        model.lm_head->weight   = rand_tensor({(size_t)cfg.hidden_size, (size_t)cfg.vocab_size}, 0.5f, seed, dev);
        model.final_norm_weight = rand_tensor({(size_t)cfg.hidden_size}, 0.1f, seed, dev);

        // Turn 1: a session generates from prompt1 (populates its cache with prompt1 + gen1).
        std::vector<int32_t> prompt1 = {3, 7, 1, 9, 2, 4, 8};
        llm::PrefixCacheSession sess;
        sess.capacity_hint = 64;   // (server sets this to max context so reuse survives growth)
        auto gen1 = gen(model, sess, prompt1, 5);

        // Turn 2: the new prompt = prompt1 + gen1 + a new user suffix (the conversation continued).
        // The cached sequence (prompt1+gen1) is an exact prefix => reuse kicks in, only the suffix
        // is prefilled.
        std::vector<int32_t> suffix = {5, 0, 6, 1};
        std::vector<int32_t> prompt2 = prompt1;
        prompt2.insert(prompt2.end(), gen1.begin(), gen1.end());
        prompt2.insert(prompt2.end(), suffix.begin(), suffix.end());

        auto cached = gen(model, sess, prompt2, 6);
        size_t reused = sess.reused, prefilled = sess.prefilled;

        // Reference: same prompt2 on a FRESH session (cold prefill, reuse=0).
        llm::PrefixCacheSession fresh;
        auto ref = gen(model, fresh, prompt2, 6);

        std::cout << "  turn2 prompt len=" << prompt2.size()
                  << "  reused=" << reused << " prefilled=" << prefilled
                  << " (fresh reused=" << fresh.reused << ")\n";
        std::cout << "  cached:"; for (int t : cached) std::cout << " " << t; std::cout << "\n";
        std::cout << "  cold  :"; for (int t : ref)    std::cout << " " << t; std::cout << "\n";

        EXPECT_TRUE(reused > 0, "turn 2 reused a cached prefix");
        EXPECT_TRUE(reused == prompt1.size() + gen1.size(), "reused == full cached length (prompt1+gen1)");
        EXPECT_TRUE(fresh.reused == 0, "fresh session reused nothing");
        EXPECT_TRUE(cached == ref, "prefix-cached generation == cold generation (KV + linear state reuse)");

        if (g_fail_count == 0) { std::cout << "\nALL TESTS PASSED (prefix cache)\n"; return 0; }
        std::cerr << "\nTESTS FAILED (prefix cache): " << g_fail_count << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
