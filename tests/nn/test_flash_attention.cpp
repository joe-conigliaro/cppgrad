// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Fused flash attention == gqa_attention (repeat_kv + SDPA reference), to fp tolerance. Flash streams
// over keys with online softmax (no [S,KV] materialization) and masks causally by position; it must
// reproduce the materialized causal GQA attention. Covers n_rep=1/>1, S=1 (decode), prefill blocks at
// an offset (KV>S), B>1. CPU + Metal.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/nn/functional.h"

using namespace cppgrad;
static uint32_t s = 7u;
static float rnd() {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) / float(1u << 24)) * 2.f - 1.f;
}
static utils::Ref<ir::Tensor> rt(const std::vector<size_t> &shp, backend::DeviceType d) {
    size_t n = 1;
    for (auto x : shp)
        n *= x;
    std::vector<float> v(n);
    for (auto &x : v)
        x = rnd();
    return ir::from_vector<float>(v, shp, d);
}
// Additive causal mask [1,1,S,KV]: query i (abs pos (KV-S)+i) attends keys [0, (KV-S)+i].
static utils::Ref<ir::Tensor> causal_mask(size_t S, size_t KV, backend::DeviceType d) {
    std::vector<float> m(S * KV, 0.f);
    size_t off = KV - S;
    for (size_t i = 0; i < S; ++i)
        for (size_t j = off + i + 1; j < KV; ++j)
            m[i * KV + j] = -1e9f;
    return ir::from_vector<float>(m, {1, 1, S, KV}, d);
}

static bool run(backend::DeviceType dev, size_t B, size_t S, size_t KV, size_t nKV, size_t n_rep, size_t Dh = 8) {
    ir::NoGradScope ng;
    size_t nH = nKV * n_rep, off = KV - S;
    auto q = rt({B, S, nH, Dh}, dev);
    auto k = rt({B, KV, nKV, Dh}, dev);
    auto v = rt({B, KV, nKV, Dh}, dev);

    auto ref = nn::functional::gqa_attention(q, k, v, causal_mask(S, KV, dev), n_rep)->to_vector<float>();
    auto got = nn::functional::flash_attention(q, k, v, n_rep, /*causal=*/true, /*q_offset=*/off)->to_vector<float>();

    float w = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        w = std::max(w, std::fabs(ref[i] - got[i]));
    bool ok = w < 1e-3f;
    std::printf("  [%s] B=%zu S=%zu KV=%zu nKV=%zu n_rep=%zu Dh=%zu off=%zu : diff=%.2e %s\n", backend::to_string(dev),
                B, S, KV, nKV, n_rep, Dh, off, w, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    backend::DeviceManager::instance().init();
    std::vector<backend::DeviceType> devs = {backend::DeviceType::CPU};
    if (backend::DeviceManager::default_device_type() == backend::DeviceType::METAL)
        devs.push_back(backend::DeviceType::METAL);
    std::printf("=== flash_attention == gqa_attention (causal) ===\n");
    bool ok = true;
    for (auto d : devs) {
        ok &= run(d, 1, 5, 5, 2, 1);   // MHA prefill, q_offset=0
        ok &= run(d, 1, 5, 5, 2, 2);   // GQA prefill
        ok &= run(d, 1, 1, 9, 2, 4);   // decode (S=1), KV=9 -> q_offset=8
        ok &= run(d, 1, 7, 12, 4, 2);  // prefill block at offset (KV>S)
        ok &= run(d, 2, 3, 3, 3, 2);   // B=2
        ok &= run(d, 1, 16, 40, 2, 3); // longer KV
        // Real-model head dims (Dh not a multiple of the 32-lane simdgroup; per-lane acc[] striding).
        ok &= run(d, 1, 8, 32, 4, 2, /*Dh=*/128);
        ok &= run(d, 1, 8, 32, 4, 2, /*Dh=*/256); // Qwen3.6 head_dim
        ok &= run(d, 1, 1, 40, 4, 2, /*Dh=*/256); // decode at Dh=256
    }
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED (flash attention)" : "FAILED (flash attention)");
    return ok ? 0 : 1;
}
