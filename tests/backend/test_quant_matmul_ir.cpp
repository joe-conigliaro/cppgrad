// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// End-to-end ir::matmul_quant (QuantizedMatMulOp) on every available device (CPU + Metal),
// validated against a host dequant-then-matmul reference.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;

// One (M,N,K) case on a device: build a random MLX-affine-quantized weight, run ir::quantized_matmul,
// compare to a host dequant+matmul reference. M==1 hits the GEMV kernel; M>1 the tiled GEMM (which
// processes blocks of QGEMM_MB=8 rows -- so M = 4/8/13/37/256 exercise the <block, ==block, remainder,
// and production-size paths).
static void run_case(backend::DeviceType devt, size_t M, size_t N, size_t K) {
    const int gs = 64;
    const size_t G = K / gs, Kp = K / 4;
    uint32_t seed = 3u + (uint32_t)(M * 131 + N * 17 + K);
    auto rnd = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / float(1u << 24);
    };
    std::vector<float> A(M * K);
    std::vector<uint8_t> q(N * K);
    std::vector<float> scales(N * G), biases(N * G);
    for (auto &a : A)
        a = rnd() * 2.f - 1.f;
    for (auto &v : q)
        v = (uint8_t)(rnd() * 256.f);
    for (auto &s : scales)
        s = rnd() * 0.05f + 0.001f;
    for (auto &b : biases)
        b = rnd() * 0.2f - 0.1f;

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

    std::string dn = backend::to_string(devt);
    auto At = ir::from_vector<float>(A, {M, K}, devt);
    auto QWt = ir::from_vector<uint32_t>(qw, {N, Kp}, devt);
    auto St = ir::from_vector<float>(scales, {N, G}, devt);
    auto Bt = ir::from_vector<float>(biases, {N, G}, devt);
    ir::QuantParams qp{ir::QuantScheme::MLX_AFFINE, 8, gs, 4};
    auto out = ir::quantized_matmul(At, QWt, {St, Bt}, qp)->to_vector<float>();
    double maxabs = 0;
    for (size_t i = 0; i < M * N; ++i)
        maxabs = std::max(maxabs, std::fabs((double)out[i] - ref[i]));
    char msg[96];
    std::snprintf(msg, sizeof(msg), "%s M=%zu N=%zu K=%zu matches reference", dn.c_str(), M, N, K);
    EXPECT_TRUE(maxabs < 1e-3, msg);
    std::cout << "  " << dn << " M=" << M << " N=" << N << " K=" << K << " max_abs_diff = " << maxabs << "\n";
}

int main() {
    TEST_HEADER("ir::matmul_quant on each device == host dequant+matmul");
    for (auto devt : {backend::DeviceType::CPU, backend::DeviceType::METAL}) {
        if (!backend::DeviceManager::device(devt))
            continue;
        for (size_t M : {(size_t)1, (size_t)4, (size_t)8, (size_t)13, (size_t)37, (size_t)256})
            run_case(devt, M, /*N=*/6, /*K=*/128);
        run_case(devt, 64, 80, 256); // larger N,K, multiple row-blocks
    }
    if (g_fail_count == 0) {
        std::cout << "\nALL TESTS PASSED (ir matmul_quant)\n";
        return 0;
    }
    std::cerr << "\nTESTS FAILED (ir matmul_quant): " << g_fail_count << "\n";
    return 1;
}
