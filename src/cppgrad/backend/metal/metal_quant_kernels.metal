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

// One packed word -> 4 codes contributing to a GEMV dot product:
//   sum_i a_i*(s*q_i + b) = s*dot(a,q) + b*sum(a),  with all 4 codes in one quant group g=w/wpg.
inline float quant_gemv_word(uint w,
                             device const uint32_t* wrow,
                             device const float4*   A4,
                             device const float*    srow,
                             device const float*    brow,
                             uint wpg) {
    uint  word = wrow[w];
    uint  g    = w / wpg;
    float s = srow[g], b = brow[g];
    float4 a = A4[w];
    float4 q = float4(float( word        & 0xFFu),
                      float((word >> 8)  & 0xFFu),
                      float((word >> 16) & 0xFFu),
                      float((word >> 24) & 0xFFu));
    return s * dot(a, q) + b * (a.x + a.y + a.z + a.w);
}

// Decode GEMV (M == 1): one simdgroup (32 lanes) per output column n, cooperative reduction over K.
// Each lane reads ONE distinct packed word (4 codes) per step -> fully coalesced, no redundant
// loads (the earlier `k=lane; k+=32` had 4 lanes hammering the same word, k>>2). Weights are the
// decode bottleneck (~85% of memory traffic), so reading each weight word exactly once, coalesced,
// is what matters. A is loaded as float4 (K is a multiple of 4); 4 MACs per loaded word.
//   per word w (codes at k = 4w..4w+3, all in one quant group g):
//     sum_i a_i*(s*q_i + b) = s*dot(a,q) + b*sum(a)
kernel void matmul_quant_gemv_f32(device const float*    A  [[buffer(0)]],   // [1, K]
                                  device const uint32_t* QW [[buffer(1)]],   // [N, K/4]
                                  device const float*    S  [[buffer(2)]],   // [N, K/GS]
                                  device const float*    B  [[buffer(3)]],
                                  device float*          OUT[[buffer(4)]],   // [1, N]
                                  constant uint32_t& N  [[buffer(5)]],
                                  constant uint32_t& K  [[buffer(6)]],
                                  constant uint32_t& GS [[buffer(7)]],
                                  uint n    [[threadgroup_position_in_grid]],
                                  uint lane [[thread_position_in_threadgroup]],
                                  uint nlanes [[threads_per_threadgroup]]) {
    if (n >= N) return;
    uint Kp = K / 4;                       // packed words per row
    uint wpg = GS / 4;                     // words per quant group (GS is a multiple of 4)
    device const uint32_t* wrow = QW + (ulong)n * Kp;
    device const float*    srow = S  + (ulong)n * (K / GS);
    device const float*    brow = B  + (ulong)n * (K / GS);
    device const float4*   A4   = (device const float4*)A;
    float acc = 0.0f;
    for (uint w = lane; w < Kp; w += nlanes) {
        acc += quant_gemv_word(w, wrow, A4, srow, brow, wpg);
    }
    acc = simd_sum(acc);            // 32-lane threadgroup == one simdgroup
    if (lane == 0) OUT[n] = acc;
}
