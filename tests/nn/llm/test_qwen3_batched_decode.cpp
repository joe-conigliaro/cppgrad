// Copyright (c) 2026 Joe Conigliaro
//
// Batched (B>1) cached decode equivalence: decoding a batch of sequences with the KV / recurrent
// cache must produce, per row, the same tokens as a non-cached full recompute. Validates the
// batched decode path (concat-mode KV cache for full attention + recurrent state for linear
// attention) at B>1. Tiny random-init models, no checkpoint.

#include <cstdio>
#include <string>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::LayerType;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::Qwen3Model;

// Returns true if batched cached decode == batched recompute for `B` sequences of prompt length P,
// decoding N tokens. Prints the first mismatch if any.
static bool check(const std::string &name, const Qwen3Config &cfg, backend::DeviceType device) {
    Qwen3Model model(cfg, device, /*lazy_weights=*/false); // random init, deterministic
    const size_t B = 2, P = 4, N = 8, V = (size_t)cfg.vocab_size;
    const int L = cfg.num_hidden_layers;

    // Two distinct prompts of equal length, laid out [B, P] row-major.
    std::vector<int32_t> prompts = {1, 2, 3, 4, 5, 6, 7, 8};

    // --- batched cached decode (concat KV cache) ---
    std::vector<int32_t> seqs(B * N); // [B, N] generated tokens
    {
        ir::NoGradScope ng;
        std::vector<Qwen3Model::LayerCache> caches((size_t)L);                // null -> concat grows them
        auto logits = model.forward_cached_batched(prompts, B, P, 0, caches); // [B,P,V]
        std::vector<int32_t> cur(B);
        for (size_t b = 0; b < B; ++b) {
            cur[b] = model.greedy_at(logits, b, P - 1);
            seqs[b * N + 0] = cur[b];
        }
        size_t pos = P;
        for (size_t k = 1; k < N; ++k) {
            auto lg = model.forward_cached_batched(cur, B, 1, pos, caches); // [B,1,V]
            for (size_t b = 0; b < B; ++b) {
                cur[b] = model.greedy_at(lg, b, 0);
                seqs[b * N + k] = cur[b];
            }
            ++pos;
        }
    }

    // --- recompute: forward() on the full [B, P+N] sequence; argmax at pos P+k-1 predicts token k ---
    std::vector<int32_t> full(B * (P + N));
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < P; ++i)
            full[b * (P + N) + i] = prompts[b * P + i];
        for (size_t k = 0; k < N; ++k)
            full[b * (P + N) + P + k] = seqs[b * N + k];
    }
    bool ok = true;
    {
        ir::NoGradScope ng;
        auto logits = model.forward(ir::from_vector<int32_t>(full, {B, P + N}, device)); // [B,P+N,V]
        for (size_t b = 0; b < B && ok; ++b) {
            for (size_t k = 0; k < N; ++k) {
                int32_t rc = model.greedy_at(logits, b, P + k - 1); // recompute's predicted token k
                if (rc != seqs[b * N + k]) {
                    printf("  [%s] mismatch at row %zu step %zu: cached=%d recompute=%d\n", name.c_str(), b, k,
                           seqs[b * N + k], rc);
                    ok = false;
                    break;
                }
            }
        }
    }
    printf("[%s] B=%zu cached decode vs recompute: %s\n", name.c_str(), B, ok ? "MATCH" : "MISMATCH");
    return ok;
}

int main() {
    backend::DeviceManager::instance().init();
    auto device = backend::DeviceManager::default_device_type();

    Qwen3Config full{
        64, 2, 128, 256, 4096, 4, 2, 16, 0, 0, 0, 0, 0, 1e-6, 1000000, false, {}, 1.0, false, "", {}, 0,
    };
    Qwen3Config mixed{
        64,
        4,
        128,
        256,
        4096,
        4,
        2,
        16,
        16,
        2,
        16,
        4,
        4,
        1e-6,
        1000000,
        false,
        {},
        1.0,
        true,
        "swish",
        {LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION, LayerType::LINEAR_ATTENTION,
         LayerType::FULL_ATTENTION},
        4,
    };

    bool ok = true;
    ok &= check("full ", full, device);
    ok &= check("mixed", mixed, device);
    printf("\n%s\n", ok ? "ALL TESTS PASSED (batched B>1 cached decode == recompute)"
                        : "FAILED (batched cached decode differs from recompute)");
    return ok ? 0 : 1;
}
