// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Metal quantized-matmul kernels (dequantize-in-kernel). Compiled into the same default.metallib
// as metal_kernels.metal; kept separate as quant is a distinct, growing concern (more schemes).
// Backend dispatch (submit_matmul_quant) picks GEMV for decode (M==1) or the tiled GEMM for M>1.
#include <metal_stdlib>
using namespace metal;

// MLX affine 8-bit quantized matmul: out[M,N] = A[M,K] @ dequant(W)^T.
//   A [M,K] fp32; QW [N,K/4] u32 (4 unsigned 8-bit codes/word); S,B [N,K/GS] fp32.
// Two kernels: a GEMV for decode (M==1) and a tiled GEMM for prefill (M>1). Backend dispatch
// (submit_matmul_quant) picks by M.

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

// Register-blocked, threadgroup-memory-tiled quantized GEMM (prefill, M > 1).
// OUT[M,N] = A[M,K] @ dequant(W)^T,  W[N,K] = S*q + B  (MLX affine u8).
// Each threadgroup computes a QT_BM x QT_BN output tile with (QT_BM/QT_RM)x(QT_BN/QT_RN)=16x16=256
// threads, each accumulating a QT_RM x QT_RN register micro-tile over the K dimension in QT_BK steps.
// The weight tile is DEQUANTIZED ONCE into threadgroup memory per K-step and reused across all QT_BM
// rows of the tile -> full batch reuse + high occupancy. (Replaced an earlier naive one-thread-per-
// output kernel and a simdgroup-per-row-block kernel, both ~bandwidth/occupancy bound.)
#define QT_BM 64
#define QT_BN 64
#define QT_BK 16
#define QT_RM 4
#define QT_RN 4
kernel void matmul_quant_gemm_tiled_f32(device const float*    A  [[buffer(0)]],
                                        device const uint32_t* QW [[buffer(1)]],
                                        device const float*    S  [[buffer(2)]],
                                        device const float*    Bb [[buffer(3)]],
                                        device float*          OUT[[buffer(4)]],
                                        constant uint32_t& M  [[buffer(5)]],
                                        constant uint32_t& N  [[buffer(6)]],
                                        constant uint32_t& K  [[buffer(7)]],
                                        constant uint32_t& GS [[buffer(8)]],
                                        uint3 tid [[thread_position_in_threadgroup]],
                                        uint3 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[QT_BM][QT_BK];
    threadgroup float Ws[QT_BK][QT_BN];

    const uint row0 = bid.y * QT_BM;
    const uint col0 = bid.x * QT_BN;
    const uint Kp = K / 4;
    const uint G  = K / GS;
    const uint NT = (QT_BM / QT_RM) * (QT_BN / QT_RN);   // 256 threads
    const uint lt = tid.y * (QT_BN / QT_RN) + tid.x;      // linear thread id 0..255

    float acc[QT_RM][QT_RN];
    for (uint i = 0; i < QT_RM; ++i) for (uint j = 0; j < QT_RN; ++j) acc[i][j] = 0.0f;

    const uint ktiles = (K + QT_BK - 1) / QT_BK;
    for (uint t = 0; t < ktiles; ++t) {
        const uint kbase = t * QT_BK;
        // Load A tile [QT_BM][QT_BK] (zero-padded past M/K).
        for (uint idx = lt; idx < QT_BM * QT_BK; idx += NT) {
            uint i = idx / QT_BK, j = idx % QT_BK;
            uint m = row0 + i, k = kbase + j;
            As[i][j] = (m < M && k < K) ? A[(ulong)m * K + k] : 0.0f;
        }
        // Load + dequantize weight tile Ws[QT_BK][QT_BN]. Vectorized: one thread handles one packed
        // u32 word = 4 consecutive K codes for a column, so the word load and the (per-group) scale +
        // bias loads are each done ONCE per 4 codes instead of per code -- 4x fewer global loads, the
        // dequant bottleneck. (Relies on GS % 4 == 0 so all 4 codes share one quant group, as the GEMV
        // path already assumes.) The tile has QT_BN*(QT_BK/4) words; with NT == that count, one each.
        const uint WPC = QT_BK / 4;                       // words per column in the K-tile
        for (uint widx = lt; widx < QT_BN * WPC; widx += NT) {
            uint n = widx / WPC, wj = widx % WPC;
            uint col = col0 + n;
            uint k0 = kbase + wj * 4;
            uint  word = 0u; float s = 0.0f, b = 0.0f;
            bool valid = (col < N && k0 < K);
            if (valid) {
                word = QW[(ulong)col * Kp + (k0 >> 2)];
                uint g = k0 / GS;                         // shared by all 4 codes (GS % 4 == 0)
                s = S[(ulong)col * G + g];
                b = Bb[(ulong)col * G + g];
            }
            for (uint c = 0; c < 4; ++c) {
                uint k = k0 + c;
                Ws[wj * 4 + c][n] = (valid && k < K) ? (s * (float)((word >> (8 * c)) & 0xFFu) + b) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < QT_BK; ++k) {
            float a[QT_RM], w[QT_RN];
            for (uint i = 0; i < QT_RM; ++i) a[i] = As[tid.y * QT_RM + i][k];
            for (uint j = 0; j < QT_RN; ++j) w[j] = Ws[k][tid.x * QT_RN + j];
            for (uint i = 0; i < QT_RM; ++i)
                for (uint j = 0; j < QT_RN; ++j)
                    acc[i][j] += a[i] * w[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < QT_RM; ++i) {
        uint m = row0 + tid.y * QT_RM + i;
        if (m >= M) continue;
        for (uint j = 0; j < QT_RN; ++j) {
            uint n = col0 + tid.x * QT_RN + j;
            if (n < N) OUT[(ulong)m * N + n] = acc[i][j];
        }
    }
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
