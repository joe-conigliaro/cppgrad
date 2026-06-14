// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Metal quantized-matmul kernels (dequantize-in-kernel). Compiled into the same default.metallib
// as metal_kernels.metal; kept separate as quant is a distinct, growing concern (more schemes /
// a tiled GEMM to come). Backend dispatch picks the kernel by name per QuantScheme.
#include <metal_stdlib>
using namespace metal;

// MLX affine 8-bit quantized matmul: out[M,N] = A[M,K] @ dequant(W)^T.
// A [M,K] fp32; QW [N,K/4] u32 (4 unsigned 8-bit codes/word); S,B [N,K/GS] fp32. One thread per out.
kernel void matmul_quant_f32(device const float*    A  [[buffer(0)]],
                             device const uint32_t* QW [[buffer(1)]],
                             device const float*    S  [[buffer(2)]],
                             device const float*    B  [[buffer(3)]],
                             device float*          OUT[[buffer(4)]],
                             constant uint32_t& M  [[buffer(5)]],
                             constant uint32_t& N  [[buffer(6)]],
                             constant uint32_t& K  [[buffer(7)]],
                             constant uint32_t& GS [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= M * N) return;
    uint m = gid / N, n = gid % N;
    uint G = K / GS, Kp = K / 4;
    device const uint32_t* wrow = QW + (ulong)n * Kp;
    device const float*    srow = S  + (ulong)n * G;
    device const float*    brow = B  + (ulong)n * G;
    device const float*    arow = A  + (ulong)m * K;
    float acc = 0.0f;
    for (uint g = 0; g < G; ++g) {
        float s = srow[g], b = brow[g];
        uint k0 = g * GS;
        float ga = 0.0f, gq = 0.0f;
        for (uint j = 0; j < GS; ++j) {
            uint k = k0 + j;
            uint word = wrow[k >> 2];
            uint q = (word >> (8 * (k & 3))) & 0xFFu;
            float a = arow[k];
            gq += a * (float)q;
            ga += a;
        }
        acc += s * gq + b * ga;
    }
    OUT[gid] = acc;
}
