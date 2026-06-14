// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// End-to-end ir::matmul_quant (QuantizedMatMulOp) on every available device (CPU + Metal),
// validated against a host dequant-then-matmul reference.
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;

int main() {
    TEST_HEADER("ir::matmul_quant on each device == host dequant+matmul");
    const size_t M = 4, N = 6, K = 128;
    const int gs = 64;
    const size_t G = K / gs, Kp = K / 4;

    uint32_t seed = 3u;
    auto rnd = [&]{ seed = seed * 1664525u + 1013904223u; return (seed >> 8) / float(1u << 24); };
    std::vector<float> A(M * K);
    std::vector<uint8_t> q(N * K);
    std::vector<float> scales(N * G), biases(N * G);
    for (auto& a : A) a = rnd() * 2.f - 1.f;
    for (auto& v : q) v = (uint8_t)(rnd() * 256.f);
    for (auto& s : scales) s = rnd() * 0.05f + 0.001f;
    for (auto& b : biases) b = rnd() * 0.2f - 0.1f;

    std::vector<uint32_t> qw(N * Kp, 0);
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            qw[n * Kp + (k >> 2)] |= (uint32_t)q[n * K + k] << (8 * (k & 3));

    std::vector<float> ref(M * N, 0.f);
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (size_t k = 0; k < K; ++k)
                acc += A[m * K + k] * (scales[n * G + k / gs] * (float)q[n * K + k] + biases[n * G + k / gs]);
            ref[m * N + n] = acc;
        }

    for (auto devt : {backend::DeviceType::CPU, backend::DeviceType::METAL}) {
        if (!backend::DeviceManager::device(devt)) continue;
        std::string dn = backend::to_string(devt);
        auto At = ir::from_vector<float>(A, {M, K}, devt);
        auto QWt = ir::from_vector<uint32_t>(qw, {N, Kp}, devt);
        auto St = ir::from_vector<float>(scales, {N, G}, devt);
        auto Bt = ir::from_vector<float>(biases, {N, G}, devt);
        ir::QuantParams qp{ir::QuantScheme::MLX_AFFINE_U8, 8, gs, 4};
        auto out = ir::quantized_matmul(At, QWt, St, Bt, qp)->to_vector<float>();
        double maxabs = 0; for (size_t i = 0; i < M * N; ++i) maxabs = std::max(maxabs, std::fabs((double)out[i] - ref[i]));
        EXPECT_TRUE(maxabs < 1e-3, (dn + ": matmul_quant matches reference").c_str());
        std::cout << "  " << dn << " max_abs_diff = " << maxabs << "\n";
        std::cout << "    out[:6] ="; for (int i=0;i<6;++i) std::cout << " " << out[i]; std::cout << "\n";
        std::cout << "    ref[:6] ="; for (int i=0;i<6;++i) std::cout << " " << ref[i]; std::cout << "\n";
    }

    if (g_fail_count == 0) { std::cout << "\nALL TESTS PASSED (ir matmul_quant)\n"; return 0; }
    std::cerr << "\nTESTS FAILED (ir matmul_quant): " << g_fail_count << "\n";
    return 1;
}
