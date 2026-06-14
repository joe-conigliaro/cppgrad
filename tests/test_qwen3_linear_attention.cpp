// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Qwen3.5/3.6 gated delta-rule linear attention (GatedDeltaNet).
//
// The mixer is implemented as the exact sequential recurrence applied over the whole
// sequence, so a full-sequence prefill must be bit-for-bit identical to stepping one
// token at a time while threading the recurrent state and the conv state. That equivalence
// is the property a recurrent KV-cache relies on, so we assert it directly here (it holds
// for any weights, so random init suffices).
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_block.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "tests/helpers.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::Qwen3Block;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::LayerType;

static void test_prefill_equals_stepwise() {
    TEST_HEADER("linear attention: prefill == stepwise (true sequential scan)");
    auto dev = backend::DeviceManager::default_device_type();

    // Tiny GatedDeltaNet config (n_v / n_kv = 2 -> exercises q/k head replication).
    Qwen3Config cfg = Qwen3Config::get_27b_qwen3_6();
    cfg.hidden_size            = 16;
    cfg.linear_num_key_heads   = 2;
    cfg.linear_key_head_dim    = 4;
    cfg.linear_num_value_heads = 4;
    cfg.linear_value_head_dim  = 4;
    cfg.linear_conv_kernel_dim = 4;

    Qwen3Block block(LayerType::LINEAR_ATTENTION, cfg, dev);

    const size_t S = 6, H = 16;
    // Deterministic, materialized input (uniform() re-draws on each realization).
    std::vector<float> xs(S * H);
    uint32_t seed = 12345u;
    for (auto& f : xs) { seed = seed * 1664525u + 1013904223u; f = ((seed >> 8) / float(1u << 24)) * 2.0f - 1.0f; }
    auto x = ir::from_vector(xs, {1, S, H}, dev);

    // Prefill: whole sequence at once.
    utils::Ref<ir::Tensor> st, cv;
    auto full_v = block.forward_linear_cached(x, nullptr, nullptr, st, cv)->to_vector<float>();

    // Decode: one token at a time, threading recurrent + conv state.
    utils::Ref<ir::Tensor> state = nullptr, conv = nullptr;
    std::vector<float> step_v;
    step_v.reserve(S * H);
    for (size_t t = 0; t < S; ++t) {
        auto xt = ir::slice(x, {0, t, 0}, {1, t + 1, H});
        utils::Ref<ir::Tensor> ns, nc;
        auto ov = block.forward_linear_cached(xt, state, conv, ns, nc)->to_vector<float>();
        state = ns; conv = nc;
        for (float f : ov) step_v.push_back(f);
    }

    EXPECT_TRUE(full_v.size() == step_v.size(), "output element count matches");
    double max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < full_v.size(); ++i) {
        if (!std::isfinite(full_v[i]) || !std::isfinite(step_v[i])) finite = false;
        max_abs = std::max(max_abs, std::fabs((double)full_v[i] - (double)step_v[i]));
    }
    EXPECT_TRUE(finite, "outputs are finite");
    EXPECT_TRUE(max_abs < 1e-4, "prefill matches stepwise (max_abs_diff < 1e-4)");
    std::cout << "  max_abs_diff = " << max_abs << "\n";
}

int main() {
    try {
        test_prefill_equals_stepwise();
        if (g_fail_count == 0) { std::cout << "\nALL TESTS PASSED (qwen3 linear attention)\n"; return 0; }
        std::cerr << "\nTESTS FAILED (qwen3 linear attention): " << g_fail_count << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
