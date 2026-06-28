// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Chunked-parallel gated delta-rule scan == sequential recurrence.
//
// nn::functional::gated_delta_scan_chunked must reproduce the exact sequential GatedDeltaNet
// recurrence (the one in qwen3_block.h forward_linear_attention) to within fp tolerance, for any
// sub-chunk size and sequence length -- including S not a multiple of the sub-chunk, S=1 (decode),
// and a non-null incoming state. Tiny random tensors; CPU and Metal.
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

static uint32_t g_seed = 1234567u;
static float frand(float lo, float hi) {
    g_seed = g_seed * 1664525u + 1013904223u;
    float u = (g_seed >> 8) / float(1u << 24);
    return lo + u * (hi - lo);
}
static utils::Ref<ir::Tensor> rnd(const std::vector<size_t> &shape, float lo, float hi, backend::DeviceType dev) {
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    std::vector<float> v(n);
    for (auto &x : v)
        x = frand(lo, hi);
    return ir::from_vector<float>(v, shape, dev);
}

// Sequential reference, operating on [BH,S,*] (matches qwen3_block.h:504-520 exactly).
static utils::Ref<ir::Tensor> seq_scan(const utils::Ref<ir::Tensor> &q, const utils::Ref<ir::Tensor> &k,
                                       const utils::Ref<ir::Tensor> &v, const utils::Ref<ir::Tensor> &decay,
                                       const utils::Ref<ir::Tensor> &beta, const utils::Ref<ir::Tensor> &state_in,
                                       utils::Ref<ir::Tensor> &state_out) {
    auto dev = q->device_type();
    size_t BH = q->shape()[0], S = q->shape()[1], dk = q->shape()[2], dv = v->shape()[2];
    utils::Ref<ir::Tensor> state = state_in ? state_in : ir::zeros({BH, dk, dv}, dev);
    std::vector<utils::Ref<ir::Tensor>> outs;
    for (size_t t = 0; t < S; ++t) {
        auto qt = ir::reshape(ir::slice(q, {0, t, 0}, {BH, t + 1, dk}), {BH, dk});
        auto kt = ir::reshape(ir::slice(k, {0, t, 0}, {BH, t + 1, dk}), {BH, dk});
        auto vt = ir::reshape(ir::slice(v, {0, t, 0}, {BH, t + 1, dv}), {BH, dv});
        auto dt = ir::reshape(ir::slice(decay, {0, t}, {BH, t + 1}), {BH, 1, 1});
        auto bt = ir::reshape(ir::slice(beta, {0, t}, {BH, t + 1}), {BH, 1});
        auto kt_c = ir::reshape_view(kt, {BH, dk, 1});
        state = state * dt;
        auto kv = ir::sum(state * kt_c, {1}); // [BH,dv]
        auto delta = (vt - kv) * bt;
        state = state + kt_c * ir::reshape_view(delta, {BH, 1, dv});
        auto qt_c = ir::reshape_view(qt, {BH, dk, 1});
        outs.push_back(ir::reshape(ir::sum(state * qt_c, {1}), {BH, 1, dv}));
    }
    state_out = state;
    utils::Ref<ir::Tensor> o = outs[0];
    for (size_t t = 1; t < S; ++t)
        o = ir::concat(o, outs[t], 1);
    return o;
}

static float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    float w = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        w = std::max(w, std::abs(a[i] - b[i]));
    return w;
}

static bool run_case(backend::DeviceType dev, size_t BH, size_t dk, size_t dv, size_t S, size_t chunk, bool with_state,
                     float gmax = 0.3f, float tol = 1e-3f) {
    ir::NoGradScope ng;
    auto q = rnd({BH, S, dk}, -1.f, 1.f, dev);
    auto k = rnd({BH, S, dk}, -1.f, 1.f, dev);
    auto v = rnd({BH, S, dv}, -1.f, 1.f, dev);
    // decay = exp(g), g in [-gmax,-0.01]; beta in (0,1). Larger gmax => faster decay => bigger
    // exp(-cumsum(g)) within a chunk (the numerical-stability stressor).
    auto decay = ir::exp(ir::neg(rnd({BH, S}, 0.01f, gmax, dev)));
    auto beta = rnd({BH, S}, 0.1f, 0.9f, dev);
    utils::Ref<ir::Tensor> st_in = with_state ? rnd({BH, dk, dv}, -0.5f, 0.5f, dev) : nullptr;

    utils::Ref<ir::Tensor> so_seq, so_chunk;
    auto o_seq = seq_scan(q, k, v, decay, beta, st_in, so_seq);
    auto o_chunk = nn::functional::gated_delta_scan_chunked(q, k, v, decay, beta, st_in, so_chunk, chunk);

    float od = max_abs_diff(o_seq->to_vector<float>(), o_chunk->to_vector<float>());
    float sd = max_abs_diff(so_seq->to_vector<float>(), so_chunk->to_vector<float>());
    bool ok = (od < tol && sd < tol);
    std::printf("  [%s] BH=%zu dk=%zu dv=%zu S=%3zu chunk=%2zu state=%d gmax=%.1f : o_diff=%.2e state_diff=%.2e %s\n",
                backend::to_string(dev), BH, dk, dv, S, chunk, (int)with_state, gmax, od, sd, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    backend::DeviceManager::instance().init();
    std::vector<backend::DeviceType> devs = {backend::DeviceType::CPU};
    if (backend::DeviceManager::default_device_type() == backend::DeviceType::METAL)
        devs.push_back(backend::DeviceType::METAL);

    std::printf("=== gated delta rule: chunked == sequential ===\n");
    bool ok = true;
    for (auto dev : devs) {
        for (size_t chunk : {4u, 16u, 64u}) {
            for (size_t S : {1u, 3u, 5u, 16u, 17u, 64u, 130u}) {
                ok &= run_case(dev, 6, 4, 4, S, chunk, /*with_state=*/false);
                ok &= run_case(dev, 6, 4, 4, S, chunk, /*with_state=*/true);
            }
        }
        // asymmetric head dims (dk != dv), like the real model (key_head != val_head possible)
        ok &= run_case(dev, 8, 8, 4, 70, 32, true);
        ok &= run_case(dev, 8, 4, 8, 70, 32, true);

        // Numerical-stability stress: stronger per-token decay (bigger gmax) blows up exp(-cumsum(g))
        // within a chunk. Find where each sub-chunk size stays accurate. Looser tol (1e-2) since the
        // de-decay/re-decay loses precision; we only need token-argmax-stable accuracy in the model.
        std::printf("  -- stability stress (strong decay) --\n");
        for (size_t chunk : {16u, 32u, 64u})
            for (float gmax : {0.5f, 1.0f, 2.0f})
                ok &= run_case(dev, 6, 8, 8, 128, chunk, true, gmax, 1e-2f);

        // Decay UNDERFLOW: g up to -200 => decay = exp(g) underflows to 0 (the real-model case that
        // produced all-garbage via log(0) = -inf -> NaN). Must stay finite and match the sequential
        // reference (which handles decay==0 exactly). Loose tol: near-zero decay annihilates old
        // state, so only the current token's contribution survives -- both forms agree there.
        std::printf("  -- decay underflow (g -> -200, decay -> 0) --\n");
        for (size_t chunk : {16u, 32u, 64u})
            ok &= run_case(dev, 6, 8, 8, 96, chunk, true, /*gmax=*/200.0f, /*tol=*/1e-2f);
    }
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED (gated delta chunked)" : "FAILED (gated delta chunked)");
    return ok ? 0 : 1;
}
