// Copyright (c) 2026 Joe Conigliaro
//
// Repro for the chunked in-place cache bug: repeated multi-token (S>1) cache_update writes at
// increasing offsets, with a readback between writes (mirroring speculative-decode verify).
// Compares the cache read-view against a hand-maintained reference. Runs on CPU and Metal.
//
// Fast (no model) so the chunked-cache fix can be iterated quickly.

#include <cstdio>
#include <vector>
#include <numeric>

#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/backend/device_manager.h"

using namespace cppgrad;

// Block mask [1,1,S,start+S]: new token i attends cols 0..start+i (mirrors qwen3_model make_block_mask).
static utils::Ref<ir::Tensor> block_mask(size_t S, size_t start, backend::DeviceType dev) {
    size_t KV = start + S;
    std::vector<float> m(S * KV, 0.0f);
    for (size_t i = 0; i < S; ++i)
        for (size_t j = start + i + 1; j < KV; ++j) m[i * KV + j] = -1e9f;
    return ir::from_vector<float>(m, {1, 1, S, KV}, dev);
}

// Write `S` rows (each H*D floats) at row offset `start` into a preallocated [1, L, H, D] cache,
// reading the [0, start+S) prefix back each time; compare to a reference buffer. Returns max abs diff.
static float run_case(backend::DeviceType dev, size_t L, size_t H, size_t D, std::vector<int> block_sizes) {
    ir::NoGradScope no_grad;
    auto cache = ir::parameter({1, L, H, D}, dev, common::DType::FLOAT32, true);
    cache->set_requires_grad(false);
    // zero-init the cache via an initial full write
    std::vector<float> zeros(L * H * D, 0.0f);
    ir::cache_update(cache, ir::from_vector<float>(zeros, {1, L, H, D}, dev), /*axis=*/1, 0);

    std::vector<float> ref(L * H * D, 0.0f);
    size_t pos = 0;
    float worst = 0.0f;
    float fill = 1.0f;
    for (int S : block_sizes) {
        // distinct values per write so corruption is visible
        std::vector<float> vals((size_t)S * H * D);
        for (auto& v : vals) v = fill++;
        // reference
        std::copy(vals.begin(), vals.end(), ref.begin() + pos * H * D);

        auto rv = ir::cache_update(cache, ir::from_vector<float>(vals, {1, (size_t)S, H, D}, dev),
                                   /*axis=*/1, pos);  // read-view [0, pos+S)
        auto got = rv->to_vector<float>();             // forces realization + readback
        size_t end = (pos + S) * H * D;
        for (size_t i = 0; i < end; ++i) worst = std::max(worst, std::abs(got[i] - ref[i]));

        // Mirror the model: read the FULL [0,end) prefix through a downstream op (matmul) and
        // read THAT back. Reshape cache prefix [1,end,H,D] -> [end, H*D], multiply by identity.
        size_t rows = pos + S, cols = H * D;
        auto flat = ir::reshape(rv, {rows, cols});
        std::vector<float> eye(cols * cols, 0.0f);
        for (size_t i = 0; i < cols; ++i) eye[i * cols + i] = 1.0f;
        auto prod = ir::matmul(flat, ir::from_vector<float>(eye, {cols, cols}, dev));  // == flat
        auto pg = prod->to_vector<float>();
        for (size_t i = 0; i < rows * cols; ++i) worst = std::max(worst, std::abs(pg[i] - ref[i]));
        pos += S;
    }
    return worst;
}

// Per round: write a K/V block at `pos` and run SDPA(Qblock, Kfull, Vfull, block_mask) over the
// growing INCREMENTAL cache, and compare to the same SDPA over a FRESH cache built by writing the
// entire [0,end) history in one shot. This is exactly the model discrepancy (incremental verify vs
// single-block) that the speculative test exposed. Returns max abs diff between the two.
static float run_case_attn(backend::DeviceType dev, size_t H, size_t D, std::vector<int> block_sizes) {
    ir::NoGradScope no_grad;
    size_t total = 0; for (int s : block_sizes) total += s;
    auto kc = ir::parameter({1, total, H, D}, dev, common::DType::FLOAT32, true); kc->set_requires_grad(false);
    auto vc = ir::parameter({1, total, H, D}, dev, common::DType::FLOAT32, true); vc->set_requires_grad(false);

    std::vector<float> khist, vhist;   // full K/V history (for the fresh-cache comparison)
    size_t pos = 0; float fill = 1.0f, worst = 0.0f;
    utils::Ref<ir::Tensor> last_kf, last_vf;  // mimic qwen3_block's persistent `_last_k = k_full`
    for (int S : block_sizes) {
        std::vector<float> kb((size_t)S*H*D), vb((size_t)S*H*D), qb((size_t)S*H*D);
        for (auto& x : kb) x = std::sin(fill++);
        for (auto& x : vb) x = std::cos(fill++);
        for (auto& x : qb) x = std::sin(fill++ * 0.5f);
        khist.insert(khist.end(), kb.begin(), kb.end());
        vhist.insert(vhist.end(), vb.begin(), vb.end());
        size_t end = pos + S;

        auto q = ir::from_vector<float>(qb, {1,(size_t)S,H,D}, dev);
        auto mask = block_mask(S, pos, dev);

        // incremental cache (hold read-views across rounds, like the model's _last_k/_last_v)
        auto kf = ir::cache_update(kc, ir::from_vector<float>(kb, {1,(size_t)S,H,D}, dev), 1, pos);
        auto vf = ir::cache_update(vc, ir::from_vector<float>(vb, {1,(size_t)S,H,D}, dev), 1, pos);
        last_kf = kf; last_vf = vf;
        auto inc = nn::functional::scaled_dot_product_attention(q, kf, vf, mask)->to_vector<float>();

        // fresh cache (write whole [0,end) history at once)
        auto kc2 = ir::parameter({1, end, H, D}, dev, common::DType::FLOAT32, true); kc2->set_requires_grad(false);
        auto vc2 = ir::parameter({1, end, H, D}, dev, common::DType::FLOAT32, true); vc2->set_requires_grad(false);
        auto kf2 = ir::cache_update(kc2, ir::from_vector<float>(khist, {1,end,H,D}, dev), 1, 0);
        auto vf2 = ir::cache_update(vc2, ir::from_vector<float>(vhist, {1,end,H,D}, dev), 1, 0);
        auto fresh = nn::functional::scaled_dot_product_attention(q, kf2, vf2, mask)->to_vector<float>();

        for (size_t i = 0; i < inc.size(); ++i) worst = std::max(worst, std::abs(inc[i] - fresh[i]));
        pos += S;
    }
    return worst;
}

int main() {
    backend::DeviceManager::instance().init();
    std::vector<backend::DeviceType> devs = {backend::DeviceType::CPU};
    if (backend::DeviceManager::default_device_type() == backend::DeviceType::METAL)
        devs.push_back(backend::DeviceType::METAL);

    bool ok = true;
    for (auto dev : devs) {
        const char* name = backend::to_string(dev);
        // S=1 repeated (the decode path; expected to pass)
        float d1 = run_case(dev, 32, 2, 4, {1, 1, 1, 1, 1, 1, 1, 1});
        // S>1 repeated at increasing offsets (the chunked/spec path; the suspected bug)
        float dN = run_case(dev, 32, 2, 4, {4, 4, 4, 4, 4});
        // mixed (prefill block then S>1 chunks)
        float dM = run_case(dev, 32, 2, 4, {5, 4, 3, 4});
        // incremental-cache SDPA vs fresh-cache SDPA over the growing prefix (the model's pattern)
        float aN = run_case_attn(dev, 2, 4, {4, 4, 4, 4, 4});
        float a1 = run_case_attn(dev, 2, 4, {1, 1, 1, 1, 1, 1});
        printf("[%s] cache_update S=1:%g S=4:%g mixed:%g | SDPA incr-vs-fresh  S=1:%g  S=4:%g\n",
               name, d1, dN, dM, a1, aN);
        if (d1 > 1e-6f || dN > 1e-6f || dM > 1e-6f || a1 > 1e-4f || aN > 1e-4f) ok = false;
    }
    printf("\n%s\n", ok ? "ALL TESTS PASSED (chunked cache_update)" : "FAILED (chunked cache_update corrupts)");
    return ok ? 0 : 1;
}
