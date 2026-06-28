// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// MLX affine 8-bit quantized matmul (dequant-in-kernel) must equal dequantize-then-matmul.
// Self-contained (synthetic weights), so it guards the kernel math without needing a checkpoint.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cppgrad/backend/cpu/cpu_quant_kernels.h"
#include "tests/helpers.h"

using namespace cppgrad;

int main() {
    TEST_HEADER("quantized matmul: dequant-in-kernel == dequantize-then-matmul");
    const size_t M = 3, N = 5, K = 128;
    const int gs = 64;
    const size_t G = K / gs, Kp = K / 4;

    uint32_t seed = 7u;
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
        v = (uint8_t)(rnd() * 256.f); // codes in [0,255]
    for (auto &s : scales)
        s = rnd() * 0.05f + 0.001f;
    for (auto &b : biases)
        b = rnd() * 0.2f - 0.1f;

    // Pack codes 4-per-uint32 (little-endian), layout [N, K/4].
    std::vector<uint32_t> qw(N * Kp, 0);
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            qw[n * Kp + (k >> 2)] |= (uint32_t)q[n * K + k] << (8 * (k & 3));

    // Reference: dequantize then matmul (independent triple loop).
    std::vector<float> ref(M * N, 0.f);
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (size_t k = 0; k < K; ++k) {
                float w = scales[n * G + k / gs] * (float)q[n * K + k] + biases[n * G + k / gs];
                acc += A[m * K + k] * w;
            }
            ref[m * N + n] = acc;
        }

    // Kernel under test.
    std::vector<float> out(M * N, 0.f);
    backend::cpu::matmul_quant_affine_f32(A.data(), qw.data(), scales.data(), biases.data(), out.data(), M, N, K, gs);

    double max_abs = 0.0;
    for (size_t i = 0; i < M * N; ++i)
        max_abs = std::max(max_abs, std::fabs((double)out[i] - ref[i]));
    EXPECT_TRUE(max_abs < 1e-3, "quant matmul matches dequant-then-matmul");
    std::cout << "  max_abs_diff = " << max_abs << "\n";

    if (g_fail_count == 0) {
        std::cout << "\nALL TESTS PASSED (quant matmul)\n";
        return 0;
    }
    std::cerr << "\nTESTS FAILED (quant matmul): " << g_fail_count << "\n";
    return 1;
}
