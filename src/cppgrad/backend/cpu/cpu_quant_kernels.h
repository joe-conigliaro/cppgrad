// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// MLX affine 8-bit quantized matmul (dequantize-in-kernel).
//
// Lets the model keep weights in their compact 8-bit form (~27GB for the 27B instead of ~54GB
// dequantized to bf16) and dequantize per-element during the matmul. This is the CPU reference /
// correctness oracle for the (future) Metal quantized GEMM.
//
// MLX affine quantization: a weight row n is split into groups of `group_size` along the input
// dim K; each group g has a bf16 scale and bias, and each weight is an unsigned 8-bit code q:
//     w[n,k] = scale[n,g]*q[n,k] + bias[n,g],   g = k / group_size
// The 8-bit codes are packed 4-per-uint32 (little-endian byte order) -> qweight is [N, K/4] u32.
// Weights are stored [N=out_features, K=in_features] (PyTorch/MLX nn.Linear layout), so the matmul
// computes  out = A @ Wᵀ :   out[m,n] = sum_k A[m,k] * w[n,k].
#include <cstddef>
#include <cstdint>

namespace cppgrad::backend::cpu {

// out [M,N] = A [M,K] @ dequant(qweight)ᵀ, with per-group affine dequant.
//   A        : [M, K]            row-major fp32 activations
//   qweight  : [N, K/4]          row-major uint32, 4 unsigned 8-bit codes per word
//   scales   : [N, K/group_size] row-major fp32 (convert bf16->fp32 at the call site)
//   biases   : [N, K/group_size] row-major fp32
//   out      : [M, N]            row-major fp32
inline void matmul_quant_affine_f32(const float *A, const uint32_t *qweight, const float *scales, const float *biases,
                                    float *out, size_t M, size_t N, size_t K, int group_size) {
    const size_t G = K / static_cast<size_t>(group_size); // groups per row
    const size_t Kp = K / 4;                              // packed words per row
    for (size_t n = 0; n < N; ++n) {
        const uint32_t *wrow = qweight + n * Kp;
        const float *srow = scales + n * G;
        const float *brow = biases + n * G;
        for (size_t m = 0; m < M; ++m) {
            const float *arow = A + m * K;
            float acc = 0.0f;
            for (size_t g = 0; g < G; ++g) {
                const float s = srow[g], b = brow[g];
                const size_t k0 = g * static_cast<size_t>(group_size);
                float ga = 0.0f, gq = 0.0f; // sum(a) and sum(a*q) within the group
                for (int j = 0; j < group_size; ++j) {
                    const size_t k = k0 + static_cast<size_t>(j);
                    const uint32_t word = wrow[k >> 2];
                    const uint8_t q = static_cast<uint8_t>((word >> (8 * (k & 3))) & 0xFFu);
                    const float a = arow[k];
                    gq += a * static_cast<float>(q);
                    ga += a;
                }
                acc += s * gq + b * ga; // sum_k a*(s*q + b), factored per group
            }
            out[m * N + n] = acc;
        }
    }
}

} // namespace cppgrad::backend::cpu
