// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Chunked prefill with per-chunk commit == single-pass / full recompute (docs/decode-runtime.md
// step 1 acceptance test). Drives the executor's "committed in-place effects" design: prefilling a
// prompt in small chunks while committing (realize + detach) the recurrent state between chunks must
// produce exactly the tokens of (a) the concat-cache reference and (b) a single-pass (no per-chunk
// commit) in-place prefill. A tiny random hybrid model (3 linear + 1 full attention) suffices; the
// equivalence holds for any weights.
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::Qwen3Model;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::LayerType;

static utils::Ref<ir::Tensor> rand_tensor(const std::vector<size_t>& shape, float scale,
                                          uint32_t& seed, backend::DeviceType dev) {
    size_t n = 1; for (auto d : shape) n *= d;
    std::vector<float> v(n);
    for (auto& f : v) { seed = seed * 1664525u + 1013904223u; f = (((seed >> 8) / float(1u << 24)) * 2.0f - 1.0f) * scale; }
    return ir::from_vector<float>(v, shape, dev);
}

static Qwen3Config tiny_cfg() {
    Qwen3Config cfg = Qwen3Config::get_27b();
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
    return cfg;
}

static void init_model(Qwen3Model& model, const Qwen3Config& cfg, backend::DeviceType dev) {
    uint32_t seed = 999u;
    model.embedding_weight  = rand_tensor({(size_t)cfg.vocab_size, (size_t)cfg.hidden_size}, 0.5f, seed, dev);
    model.lm_head->weight   = rand_tensor({(size_t)cfg.hidden_size, (size_t)cfg.vocab_size}, 0.5f, seed, dev);
    model.final_norm_weight = rand_tensor({(size_t)cfg.hidden_size}, 0.1f, seed, dev);
}

// Greedy full recompute (no cache) -- the ground-truth reference, like test_qwen3_kv_cache.
static int argmax_last_row(const utils::Ref<ir::Tensor>& logits, size_t len, size_t V) {
    auto lv = logits->to_vector<float>();
    const float* row = lv.data() + (len - 1) * V;
    int best = 0; float bv = row[0];
    for (size_t i = 1; i < V; ++i) if (row[i] > bv) { bv = row[i]; best = (int)i; }
    return best;
}

static void run_device(backend::DeviceType dev) {
    auto cfg = tiny_cfg();
    const size_t V = (size_t)cfg.vocab_size;
    std::cout << "\n[" << backend::to_string(dev) << "]\n";

    // ONE model instance: block weights are random-initialized in the ctor, so all variants must
    // share the same model (re-instantiating would draw different block weights).
    Qwen3Model model(cfg, dev);
    init_model(model, cfg, dev);

    // A prompt long enough to span several chunks at CHUNK=3.
    std::vector<int32_t> prompt = {3, 7, 1, 9, 2, 4, 8, 6, 5, 0, 1, 7};
    const int N = 6;

    // Reference: naive full recompute each step (ground truth, cache-independent).
    std::vector<int32_t> ref, ids = prompt;
    for (int i = 0; i < N; ++i) {
        auto in = ir::from_vector<int32_t>(ids, {1, ids.size()}, dev);
        int next = argmax_last_row(model.forward(in), ids.size(), V);
        ref.push_back(next);
        ids.push_back(next);
    }

    // In-place, single-pass prefill (no per-chunk commit): whole prompt in one chunk.
    model.inplace_kv = true;
    setenv("CPPGRAD_PREFILL_CHUNK", "1024", 1);   // >= prompt len => single chunk
    std::vector<int32_t> inplace_single = model.generate(prompt, N);

    // In-place, chunked prefill with per-chunk commit (CHUNK=3): realize + detach the recurrent
    // state between chunks -- the case docs/decode-runtime.md targets.
    setenv("CPPGRAD_PREFILL_CHUNK", "3", 1);
    std::vector<int32_t> chunked_commit = model.generate(prompt, N);

    auto show = [](const char* tag, const std::vector<int32_t>& v) {
        std::cout << "  " << tag; for (int t : v) std::cout << " " << t; std::cout << "\n";
    };
    show("reference (recompute)   :", ref);
    show("in-place single-pass    :", inplace_single);
    show("in-place chunked+commit :", chunked_commit);

    EXPECT_TRUE(inplace_single  == ref, "in-place single-pass == recompute");
    EXPECT_TRUE(chunked_commit  == ref, "in-place chunked+commit == recompute");
}

int main() {
    try {
        TEST_HEADER("prefill: chunked per-chunk commit == single-pass / recompute");
        backend::DeviceManager::instance().init();
        run_device(backend::DeviceType::CPU);
        if (backend::DeviceManager::default_device_type() == backend::DeviceType::METAL)
            run_device(backend::DeviceType::METAL);

        if (g_fail_count == 0) { std::cout << "\nALL TESTS PASSED (qwen3 prefill commit)\n"; return 0; }
        std::cerr << "\nTESTS FAILED (qwen3 prefill commit): " << g_fail_count << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
