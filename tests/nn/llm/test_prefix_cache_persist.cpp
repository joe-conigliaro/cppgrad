// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Prefix-cache persistence round-trip: a session saved to disk and reloaded must reproduce the same
// generation as a cold prefill -- i.e. the serialized KV (full-attn) + recurrent state (linear-attn)
// is restored exactly. This is what lets a fixed system prompt be prefilled once on a machine and
// reused across server restarts. Tiny random hybrid model; CPU + Metal (default device).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>
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
    for (auto& f : v) { seed = seed*1664525u+1013904223u; f = (((seed>>8)/float(1u<<24))*2.f-1.f)*scale; }
    return ir::from_vector<float>(v, shape, dev);
}
static std::vector<int32_t> gen(Qwen3Model& m, llm::PrefixCacheSession& s,
                                const std::vector<int32_t>& prompt, int N) {
    std::unordered_set<int32_t> stop;
    return llm::generate_with_prefix_cache(m, s, prompt, N, [](int32_t){ return true; }, stop, {});
}

int main() {
    try {
        TEST_HEADER("prefix cache persistence: saved+reloaded == cold prefill");
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

        std::vector<int32_t> prompt1 = {3, 7, 1, 9, 2, 4, 8};
        std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                           + "/cppgrad_prefix_cache_test.bin";

        // Session A: prefill prompt1 (populates cache), then SAVE to disk.
        llm::PrefixCacheSession a; a.capacity_hint = 64;
        auto gen1 = gen(model, a, prompt1, 5);
        bool saved = llm::save_prefix_cache(model, a, path);
        EXPECT_TRUE(saved, "save_prefix_cache wrote the file");

        // The continued conversation: prompt2 extends what A's cache holds (prompt1 + gen1).
        std::vector<int32_t> prompt2 = prompt1;
        prompt2.insert(prompt2.end(), gen1.begin(), gen1.end());
        for (int t : {5, 0, 6, 1}) prompt2.push_back(t);

        // Session B: fresh, LOAD from disk, then continue. Reuse should restore A's exact cache.
        llm::PrefixCacheSession b; b.capacity_hint = 64;
        bool loaded = llm::load_prefix_cache(model, b, path);
        EXPECT_TRUE(loaded, "load_prefix_cache restored the session");
        EXPECT_TRUE(b.tokens == a.tokens, "restored token sequence matches");
        auto from_disk = gen(model, b, prompt2, 6);
        size_t reused = b.reused;

        // Reference: cold prefill of the same prompt2 on a fresh (empty) session.
        llm::PrefixCacheSession cold; cold.capacity_hint = 64;
        auto cold_gen = gen(model, cold, prompt2, 6);

        std::printf("  reused %zu/%zu tokens from disk-loaded cache\n", reused, prompt2.size());
        std::printf("  from-disk:"); for (int t : from_disk) std::printf(" %d", t); std::printf("\n");
        std::printf("  cold     :"); for (int t : cold_gen)  std::printf(" %d", t); std::printf("\n");

        EXPECT_TRUE(reused == prompt1.size() + gen1.size(), "reused the full restored prefix");
        EXPECT_TRUE(from_disk == cold_gen, "disk-restored cache generation == cold prefill");

        std::remove(path.c_str());
        if (g_fail_count == 0) { std::printf("\nALL TESTS PASSED (prefix cache persist)\n"); return 0; }
        std::fprintf(stderr, "\nTESTS FAILED (prefix cache persist): %d\n", g_fail_count); return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nEXCEPTION: %s\n", e.what()); return 2;
    }
}
