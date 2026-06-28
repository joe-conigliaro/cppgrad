// Copyright (c) 2026 Joe Conigliaro
//
// Batched (B>1) non-cached forward validation: forwarding a batch of sequences must produce, per
// row, the SAME logits as forwarding each sequence alone at B=1. Uses a tiny random-init model
// (no checkpoint) so it runs fast and deterministically. This is step 1 of the batched/ragged
// decode work - validating the core forward at B>1 before touching the KV cache or scheduler.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::LayerType;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::Qwen3Model;

// For a tiny random-init model, check that a B=2 forward produces, per row, the same logits as two
// separate B=1 forwards. Returns max abs diff across both rows.
static double check_batched(const Qwen3Config &cfg, backend::DeviceType device) {
    Qwen3Model model(cfg, device, /*lazy_weights=*/false); // random init, deterministic
    const size_t V = (size_t)cfg.vocab_size;
    std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> b = {7, 11, 13, 2, 9, 4};
    const size_t S = a.size(), SV = S * V;

    auto fwd = [&](const std::vector<int32_t> &ids, size_t B) {
        ir::NoGradScope ng;
        return model.forward(ir::from_vector<int32_t>(ids, {B, ids.size() / B}, device))->to_vector<float>();
    };
    auto la = fwd(a, 1), lb = fwd(b, 1);
    std::vector<int32_t> batch = a;
    batch.insert(batch.end(), b.begin(), b.end());
    auto lab = fwd(batch, 2);

    double d = 0;
    for (size_t i = 0; i < SV; ++i) {
        d = std::max(d, std::abs((double)la[i] - lab[i]));
        d = std::max(d, std::abs((double)lb[i] - lab[SV + i]));
    }
    return d;
}

int main() {
    backend::DeviceManager::instance().init();
    auto device = backend::DeviceManager::default_device_type();

    // Tiny all-full-attention config (Qwen3 style).
    Qwen3Config full{
        64,    2,       128, 256, 4096, // hidden, layers, intermediate, vocab, max_pos
        4,     2,       16,             // attn_heads, kv_heads, head_dim
        0,     0,       0,   0,   0,    // linear attention (unused)
        1e-6,  1000000,                 // rms_norm_eps, rope_theta
        false, {},      1.0,            // mrope
        false, "",                      // attn_output_gate
        {},    0,                       // layer_types, full_attention_interval
    };

    // Tiny mixed config (Qwen3.6 style: 3 linear + 1 full attention).
    Qwen3Config mixed{
        64,
        4,
        128,
        256,
        4096, // hidden, layers, intermediate, vocab, max_pos
        4,
        2,
        16, // attn_heads, kv_heads, head_dim
        16,
        2,
        16,
        4,
        4, // linear: key_dim, key_heads, val_dim, val_heads, conv_kernel
        1e-6,
        1000000, // rms_norm_eps, rope_theta
        false,
        {},
        1.0, // mrope (full rotary to avoid section deps)
        true,
        "swish", // attn_output_gate
        {LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION,
         LayerType::FULL_ATTENTION},
        4,
    };

    double df = check_batched(full, device);
    double dm = check_batched(mixed, device);
    printf("full-attention  B=2 vs B=1: max_diff=%g\n", df);
    printf("mixed (lin+full) B=2 vs B=1: max_diff=%g\n", dm);

    bool ok = (df < 1e-4 && dm < 1e-4);
    printf("\n%s\n", ok ? "ALL TESTS PASSED (batched forward == per-sequence forward, full + mixed)"
                        : "FAILED (batched forward differs from per-sequence)");
    return ok ? 0 : 1;
}
