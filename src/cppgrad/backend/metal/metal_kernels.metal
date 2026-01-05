// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <metal_stdlib>
using namespace metal;

// Metal-safe POD structs used only in shader compilation unit
namespace mslp {

struct View32 {
    ushort rank;
    ushort pad;
    uint   offset;
    uint   flags;
    uint   shape[8];
    uint   strides[8];
};

struct UnaryParams {
    View32 in_v;
    View32 out_v;
    uint   n;
    ushort op; // 0:relu 1:exp 2:log 3:neg 4:tanh
    ushort pad2;
};

struct BinaryParams {
    View32 a_v;
    View32 b_v;
    View32 o_v;
    uint   n;
    ushort op; // 0:add 1:sub 2:mul 3:div 4:pow 5:eq 6:gt 7:min 8:max
    ushort pad2;
};

struct MatmulParams {
    View32 a_v;
    View32 b_v;
    View32 o_v;
    uint   M, K, N;
};

struct BroadcastParams {
    View32 in_v;
    View32 out_v;
    uint   n;
    uint   pad3;
};

struct PermuteParams {
    View32 in_v;
    View32 out_v;
    uint   n;
    ushort axes[8]; // out_d -> in_d
    ushort pad_axes[8];
};

struct SliceParams {
    View32 in_v;
    View32 out_v;
    uint   n;
    uint   begin[8];
    uint   step[8];
};

struct CopyViewParams {
    View32 src_v;
    View32 dst_v;
    uint   n;
    uint   pad4;
};

struct ReduceFastParams {
    View32 in_v;
    View32 out_v;
    uint   inner; // last axis size
    ushort op;    // 0 sum, 1 max
    ushort pad5;
};

struct ReduceGeneralParams {
    View32 in_v;
    View32 out_v;
    ushort op; // 0 sum, 1 max
    ushort pad6;
    uint   out_total;
    uchar  is_reduce_axis[8]; // 0/1 flags per input axis
    uchar  pad7[8];
};

} // namespace mslp

// Helpers

// Helpers for index and coord decoding with proper address-space overloads

inline uint index_from_coords(thread const uint* coords,
                              thread const mslp::View32& v) {
    uint idx = v.offset;
    for (ushort i=0;i<v.rank;++i) idx += coords[i] * v.strides[i];
    return idx;
}

inline uint index_from_coords(thread const uint* coords,
                              constant mslp::View32& v) {
    uint idx = v.offset;
    for (ushort i=0;i<v.rank;++i) idx += coords[i] * v.strides[i];
    return idx;
}

inline void coords_from_linear(uint lin,
                               thread const uint* shape,
                               ushort rank,
                               thread uint* coords) {
    uint rem = lin;
    for (ushort i = 0; i < rank; ++i) {
        uint stride = 1;
        for (ushort j = i + 1; j < rank; ++j) stride *= shape[j];
        uint c = (stride > 0) ? (rem / stride) : 0;
        coords[i] = c;
        rem -= c * stride;
    }
}

inline void coords_from_linear(uint lin,
                               constant uint* shape,
                               ushort rank,
                               thread uint* coords) {
    uint rem = lin;
    for (ushort i = 0; i < rank; ++i) {
        uint stride = 1;
        for (ushort j = i + 1; j < rank; ++j) stride *= shape[j];
        uint c = (stride > 0) ? (rem / stride) : 0;
        coords[i] = c;
        rem -= c * stride;
    }
}

inline float apply_unary(float x, ushort op) {
    switch (op) {
        case 0: return (x > 0.0f) ? x : 0.0f;
        case 1: return exp(x);
        case 2: return log(x);
        case 3: return -x;
        case 4: return tanh(x);
    }
    return x;
}

inline float apply_binary(float a, float b, ushort op) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow(a, b);
        case 5: return (a == b) ? 1.0f : 0.0f;
        case 6: return (a > b) ? 1.0f : 0.0f;
        case 7: return fmin(a, b);
        case 8: return fmax(a, b);
    }
    return 0.0f;
}

// Fill & Random

kernel void fill(device float* out [[buffer(0)]],
                 constant float& value [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = value;
}

inline uint lcg(uint x) { return 1664525u * x + 1013904223u; }
inline float u01_from_state(uint s) { return float(s & 0xFFFFFFu) / float(0xFFFFFFu); }

kernel void rand_uniform(device float* out [[buffer(0)]],
                         constant float& min_val [[buffer(1)]],
                         constant float& max_val [[buffer(2)]],
                         constant uint& seed [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    uint s = lcg(seed ^ (gid + 1u));
    float r = u01_from_state(s);
    out[gid] = min_val + r * (max_val - min_val);
}

kernel void rand_normal(device float* out [[buffer(0)]],
                        constant float& mean [[buffer(1)]],
                        constant float& stddev [[buffer(2)]],
                        constant uint& seed [[buffer(3)]],
                        constant uint& out_numel [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= out_numel) return;
    if ((gid & 1u) != 0u) return;
    uint s1 = lcg(seed ^ (gid + 1u));
    uint s2 = lcg(seed ^ (gid + 2u));
    float u1 = fmax(u01_from_state(s1), 1e-7f);
    float u2 = fmax(u01_from_state(s2), 1e-7f);
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;
    float z0 = r * cos(theta), z1 = r * sin(theta);
    out[gid] = mean + stddev * z0;
    uint next = gid + 1u;
    if (next < out_numel) out[next] = mean + stddev * z1;
}

// Unary (stride-aware)

kernel void unary_view_f32(device const float* in_buf [[buffer(0)]],
                           device float* out_buf [[buffer(1)]],
                           constant mslp::UnaryParams& P [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;

    uint ocoords[8];
    coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);

    uint icoords[8];
    bool same = (P.in_v.rank == P.out_v.rank);
    if (same) {
        for (ushort i=0;i<P.in_v.rank;++i) if (P.in_v.shape[i] != P.out_v.shape[i]) { same=false; break; }
    }
    if (same) {
        for (ushort i=0;i<P.in_v.rank;++i) icoords[i] = ocoords[i];
    } else {
        ushort r_in = P.in_v.rank, r_out = P.out_v.rank;
        ushort off = (r_out > r_in) ? (r_out - r_in) : 0;
        for (ushort i=0;i<r_in;++i) {
            uint oc = ocoords[off + i];
            icoords[i] = (P.in_v.shape[i] == 1) ? 0 : oc;
        }
    }

    uint ai = index_from_coords(icoords, P.in_v);
    uint oi = index_from_coords(ocoords, P.out_v);
    float x = in_buf[ai];
    out_buf[oi] = apply_unary(x, P.op);
}

// Binary (stride-aware)

kernel void binary_view_f32(device const float* a_buf [[buffer(0)]],
                            device const float* b_buf [[buffer(1)]],
                            device float* out_buf [[buffer(2)]],
                            constant mslp::BinaryParams& P [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;

    uint ocoords[8];
    coords_from_linear(gid, P.o_v.shape, P.o_v.rank, ocoords);

    uint acoords[8], bcoords[8];

    // Map a (broadcast-aware)
    {
        bool same = (P.a_v.rank == P.o_v.rank);
        if (same) {
            for (ushort i=0;i<P.a_v.rank;++i) if (P.a_v.shape[i] != P.o_v.shape[i]) { same=false; break; }
        }
        if (same) {
            for (ushort i=0;i<P.a_v.rank;++i) acoords[i] = ocoords[i];
        } else {
            ushort r_in = P.a_v.rank, r_out = P.o_v.rank;
            ushort off = (r_out > r_in) ? (r_out - r_in) : 0;
            for (ushort i=0; i<r_in; ++i) {
                uint oc = ocoords[off + i];
                acoords[i] = (P.a_v.shape[i] == 1) ? 0 : oc;
            }
        }
    }

    // Map b (broadcast-aware)
    {
        bool same = (P.b_v.rank == P.o_v.rank);
        if (same) {
            for (ushort i=0;i<P.b_v.rank;++i) if (P.b_v.shape[i] != P.o_v.shape[i]) { same=false; break; }
        }
        if (same) {
            for (ushort i=0;i<P.b_v.rank;++i) bcoords[i] = ocoords[i];
        } else {
            ushort r_in = P.b_v.rank, r_out = P.o_v.rank;
            ushort off = (r_out > r_in) ? (r_out - r_in) : 0;
            for (ushort i=0; i<r_in; ++i) {
                uint oc = ocoords[off + i];
                bcoords[i] = (P.b_v.shape[i] == 1) ? 0 : oc;
            }
        }
    }

    uint ai = index_from_coords(acoords, P.a_v);
    uint bi = index_from_coords(bcoords, P.b_v);
    uint oi = index_from_coords(ocoords, P.o_v);

    float x = a_buf[ai], y = b_buf[bi];
    out_buf[oi] = apply_binary(x, y, P.op);
}

// Matmul (stride-aware rank-2)

kernel void matmul_view_f32(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* C [[buffer(2)]],
                            constant mslp::MatmulParams& P [[buffer(3)]],
                            uint2 tid [[thread_position_in_grid]]) {
    uint i = tid.y;
    uint j = tid.x;
    if (i >= P.M || j >= P.N) return;

    float sum = 0.0f;
    for (uint k=0; k<P.K; ++k) {
        uint a_idx = P.a_v.offset + i*P.a_v.strides[0] + k*P.a_v.strides[1];
        uint b_idx = P.b_v.offset + k*P.b_v.strides[0] + j*P.b_v.strides[1];
        sum += A[a_idx] * B[b_idx];
    }
    uint c_idx = P.o_v.offset + i*P.o_v.strides[0] + j*P.o_v.strides[1];
    C[c_idx] = sum;
}

// Tunables
#define TM 16
#define TN 16
#define TK 16

kernel void matmul_tiled_f32(device const float* A [[buffer(0)]],
                             device const float* B [[buffer(1)]],
                             device float* C [[buffer(2)]],
                             constant mslp::MatmulParams& P [[buffer(3)]],
                             uint2 tid  [[thread_position_in_threadgroup]],
                             uint2 bid  [[threadgroup_position_in_grid]]) {
    // Tile origin in output
    uint row0 = bid.y * TM;
    uint col0 = bid.x * TN;

    // Accumulator
    float acc[TM][TN];
    for (uint i=0;i<TM;++i) for (uint j=0;j<TN;++j) acc[i][j] = 0.0f;

    threadgroup float Asub[TM][TK];
    threadgroup float Bsub[TK][TN];

    uint tiles = (P.K + TK - 1) / TK;

    for (uint t=0; t<tiles; ++t) {
        // Each thread loads one element; optionally load 2–4 per thread for bandwidth
        uint a_row = row0 + tid.y;
        uint a_col = t*TK + tid.x;
        if (a_row < P.M && a_col < P.K) {
            uint a_idx = P.a_v.offset + a_row*P.a_v.strides[0] + a_col*P.a_v.strides[1];
            Asub[tid.y][tid.x] = A[a_idx];
        } else {
            Asub[tid.y][tid.x] = 0.0f;
        }

        uint b_row = t*TK + tid.y;
        uint b_col = col0 + tid.x;
        if (b_row < P.K && b_col < P.N) {
            uint b_idx = P.b_v.offset + b_row*P.b_v.strides[0] + b_col*P.b_v.strides[1];
            Bsub[tid.y][tid.x] = B[b_idx];
        } else {
            Bsub[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial
        for (uint k=0;k<TK;++k) {
            float a = Asub[tid.y][k];
            // Broadcast this thread's a to TN lanes by reading Bsub[k][x]
            for (uint n=0;n<TN;++n) {
                acc[tid.y][n] += a * Bsub[k][n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    uint out_row = row0 + tid.y;
    for (uint n=0;n<TN;++n) {
        uint out_col = col0 + n;
        if (out_row < P.M && out_col < P.N) {
            uint c_idx = P.o_v.offset + out_row*P.o_v.strides[0] + out_col*P.o_v.strides[1];
            C[c_idx] = acc[tid.y][n];
        }
    }
}

// Tunables must match host-side expectation
// #define TM 16
// #define TN 16
// #define TK 16

kernel void matmul_tiled_tn_f32(device const float* A [[buffer(0)]],
                                device const float* B [[buffer(1)]],
                                device float* C [[buffer(2)]],
                                constant mslp::MatmulParams& P [[buffer(3)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 bid [[threadgroup_position_in_grid]]) {
    // Tile origin in output
    uint row0 = bid.y * TM;
    uint col0 = bid.x * TN;

    // Accumulator
    float acc[TM][TN];
    for (uint i=0;i<TM;++i) for (uint j=0;j<TN;++j) acc[i][j] = 0.0f;

    threadgroup float Asub[TM][TK];
    threadgroup float Bsub[TK][TN];

    // In TN layout, B’s columns are contiguous: for each output column j,
    // the "column vector" of length K starts at base j*K (in row-major TN strides).
    // We still tile K in TK, but load B col-wise tiles accordingly.

    uint tiles = (P.K + TK - 1) / TK;
    for (uint t=0; t<tiles; ++t) {
        // Load A tile (row-major NN): rows (row0 + tid.y), K chunk (t*TK + tid.x)
        uint a_row = row0 + tid.y;
        uint a_col = t*TK + tid.x;
        if (a_row < P.M && a_col < P.K) {
            uint a_idx = P.a_v.offset + a_row*P.a_v.strides[0] + a_col*P.a_v.strides[1];
            Asub[tid.y][tid.x] = A[a_idx];
        } else {
            Asub[tid.y][tid.x] = 0.0f;
        }

        // Load B tile (TN): columns (col0 .. col0+TN-1), each is contiguous along K
        uint k_base = t*TK;           // start of current K tile
        uint k_off  = tid.y;          // per-thread y loads K
        uint j_off  = tid.x;          // per-thread x loads N
        uint k_idx  = k_base + k_off; // K index within tile
        uint j_idx  = col0 + j_off;   // column j

        if (k_idx < P.K && j_idx < P.N) {
            // TN layout: contiguous in K along column j => base j*K + k
            uint b_idx = P.b_v.offset + j_idx*P.b_v.strides[1] + k_idx*P.b_v.strides[0];
            Bsub[k_off][j_off] = B[b_idx];
        } else {
            Bsub[k_off][j_off] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial
        for (uint k=0;k<TK;++k) {
            float a = Asub[tid.y][k];
            for (uint n=0;n<TN;++n) {
                acc[tid.y][n] += a * Bsub[k][n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    uint out_row = row0 + tid.y;
    for (uint n=0;n<TN;++n) {
        uint out_col = col0 + n;
        if (out_row < P.M && out_col < P.N) {
            uint c_idx = P.o_v.offset + out_row*P.o_v.strides[0] + out_col*P.o_v.strides[1];
            C[c_idx] = acc[tid.y][n];
        }
    }
}

// Broadcast

kernel void broadcast_view_f32(device const float* in_buf [[buffer(0)]],
                               device float* out_buf [[buffer(1)]],
                               constant mslp::BroadcastParams& P [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;

    uint ocoords[8]; coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);
    uint icoords[8];

    bool same = (P.in_v.rank == P.out_v.rank);
    if (same) {
        for (ushort i=0;i<P.in_v.rank;++i) if (P.in_v.shape[i] != P.out_v.shape[i]) { same=false; break; }
    }
    if (same) {
        for (ushort i=0;i<P.in_v.rank;++i) icoords[i] = ocoords[i];
    } else {
        ushort r_in = P.in_v.rank, r_out = P.out_v.rank;
        ushort off = (r_out > r_in) ? (r_out - r_in) : 0;
        for (ushort i=0; i<r_in; ++i) {
            uint oc = ocoords[off + i];
            icoords[i] = (P.in_v.shape[i] == 1) ? 0 : oc;
        }
    }

    uint ai = index_from_coords(icoords, P.in_v);
    uint oi = index_from_coords(ocoords, P.out_v);
    out_buf[oi] = in_buf[ai];
}

// Permute

kernel void permute_view_f32(device const float* in_buf [[buffer(0)]],
                             device float* out_buf [[buffer(1)]],
                             constant mslp::PermuteParams& P [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    uint ocoords[8]; coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);
    uint icoords[8];
    for (ushort out_d=0; out_d<P.out_v.rank; ++out_d) {
        ushort in_d = P.axes[out_d];
        icoords[in_d] = ocoords[out_d];
    }
    uint ai = index_from_coords(icoords, P.in_v);
    uint oi = index_from_coords(ocoords, P.out_v);
    out_buf[oi] = in_buf[ai];
}

// Slice forward/backward

kernel void slice_forward_view_f32(device const float* in_buf [[buffer(0)]],
                                   device float* out_buf [[buffer(1)]],
                                   constant mslp::SliceParams& P [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;
    uint ocoords[8]; coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);
    uint icoords[8];
    for (ushort d=0; d<P.in_v.rank; ++d) {
        uint st = (d < 8) ? P.step[d] : 1u;
        icoords[d] = P.begin[d] + ocoords[d] * st;
    }
    uint ai = index_from_coords(icoords, P.in_v);
    uint oi = index_from_coords(ocoords, P.out_v);
    out_buf[oi] = in_buf[ai];
}

kernel void slice_backward_scatter_add_view_f32(device const float* grad_out [[buffer(0)]],
                                                device float* grad_in [[buffer(1)]],
                                                constant mslp::SliceParams& P [[buffer(2)]],
                                                uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;
    uint ocoords[8]; coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);
    uint icoords[8];
    for (ushort d=0; d<P.in_v.rank; ++d) {
        uint st = (d < 8) ? P.step[d] : 1u;
        icoords[d] = P.begin[d] + ocoords[d] * st;
    }
    uint gi = index_from_coords(icoords, P.in_v);
    uint go = index_from_coords(ocoords, P.out_v);
    grad_in[gi] += grad_out[go];
}

// Copy view

// kernel void copy_view_f32(device const float* src [[buffer(0)]],
//                           device float* dst [[buffer(1)]],
//                           constant mslp::CopyViewParams& P [[buffer(2)]],
//                           uint gid [[thread_position_in_grid]]) {
//     if (gid >= P.n) return;

//     uint ocoords[8]; coords_from_linear(gid, P.dst_v.shape, P.dst_v.rank, ocoords);
//     uint icoords[8];

//     bool same = (P.src_v.rank == P.dst_v.rank);
//     if (same) {
//         for (ushort i=0;i<P.src_v.rank;++i) if (P.src_v.shape[i] != P.dst_v.shape[i]) { same=false; break; }
//     }
//     if (same) {
//         for (ushort i=0;i<P.src_v.rank;++i) icoords[i] = ocoords[i];
//     } else {
//         ushort r_in = P.src_v.rank, r_out = P.dst_v.rank;
//         ushort off = (r_out > r_in) ? (r_out - r_in) : 0;
//         for (ushort i=0; i<r_in; ++i) {
//             uint oc = ocoords[off + i];
//             icoords[i] = (P.src_v.shape[i] == 1) ? 0 : oc;
//         }
//     }

//     uint si = index_from_coords(icoords, P.src_v);
//     uint di = index_from_coords(ocoords, P.dst_v);
//     dst[di] = src[si];
// }
kernel void copy_view_f32(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant mslp::CopyViewParams& P [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;

    uint ocoords[8]; coords_from_linear(gid, P.dst_v.shape, P.dst_v.rank, ocoords);

    bool same = (P.src_v.rank == P.dst_v.rank);
    if (same) {
        for (ushort i=0; i<P.src_v.rank; ++i) if (P.src_v.shape[i] != P.dst_v.shape[i]) { same = false; break; }
    }
    uint si, di;
    if (same) {
        // Fast path: direct coordinate use
        si = P.src_v.offset;
        for (ushort i=0; i<P.src_v.rank; ++i) si += ocoords[i] * P.src_v.strides[i];
    } else {
        // General path: right-align + broadcast dims
        uint icoords[8];
        ushort r_in  = P.src_v.rank;
        ushort r_out = P.dst_v.rank;
        ushort off   = (r_out > r_in) ? (r_out - r_in) : 0;
        for (ushort i=0; i<r_in; ++i) {
            uint oc = ocoords[off + i];
            icoords[i] = (P.src_v.shape[i] == 1) ? 0 : oc;
        }
        si = index_from_coords(icoords, P.src_v);
    }

    di = index_from_coords(ocoords, P.dst_v);
    dst[di] = src[si];
}

// Reduce (fast: last axis)
kernel void reduce_last_axis_f32(device const float* in_buf [[buffer(0)]],
                                 device float* out_buf [[buffer(1)]],
                                 constant mslp::ReduceFastParams& P [[buffer(2)]],
                                 threadgroup float* smem [[threadgroup(0)]],
                                 uint tid  [[thread_position_in_threadgroup]],
                                 uint gidx [[threadgroup_position_in_grid]],
                                 uint tpg  [[threads_per_threadgroup]]) {
    const ushort r_in  = P.in_v.rank;
    const ushort r_out = P.out_v.rank;

    // Input-outer shape (drop the last axis)
    uint in_outer_shape[8] = {1,1,1,1,1,1,1,1};
    for (ushort d = 0; d < r_in - 1; ++d)
        in_outer_shape[d] = P.in_v.shape[d];

    // Decode outer coords against the INPUT-outer shape
    uint ocoords_in[8] = {0,0,0,0,0,0,0,0};
    coords_from_linear(gidx, in_outer_shape, (ushort)(r_in - 1), ocoords_in);

    // Build full input coords (insert last axis = 0)
    uint icoords[8] = {0,0,0,0,0,0,0,0};
    for (ushort d = 0; d < r_in - 1; ++d)
        icoords[d] = ocoords_in[d];
    icoords[r_in - 1] = 0;

    // Compute input/output bases via views
    uint row_base_in  = index_from_coords(icoords, P.in_v);

    uint ocoords_out[8] = {0,0,0,0,0,0,0,0};
    coords_from_linear(gidx, P.out_v.shape, r_out, ocoords_out);
    uint row_base_out = index_from_coords(ocoords_out, P.out_v);

    // Reduce along inner (last axis)
    uint inner = P.in_v.shape[r_in - 1];
    uint s     = P.in_v.strides[r_in - 1];

    float acc = (P.op == 0) ? 0.0f : -FLT_MAX;
    for (uint i = tid; i < inner; i += tpg) {
        float v = in_buf[row_base_in + i * s];
        acc = (P.op == 0) ? (acc + v) : fmax(acc, v);
    }

    // In-threadgroup reduction for any tpg
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Halving loop, power-of-two or not
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = (P.op == 0)
                        ? (smem[tid] + smem[tid + stride])
                        : fmax(smem[tid], smem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        out_buf[row_base_out] = smem[0];
    }
}

// Reduce (general N-D)

kernel void reduce_general_f32(device const float* in_buf [[buffer(0)]],
                               device float* out_buf [[buffer(1)]],
                               constant mslp::ReduceGeneralParams& P [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= P.out_total) return;

    // Decode output coords
    uint ocoords[8]; coords_from_linear(gid, P.out_v.shape, P.out_v.rank, ocoords);

    // Build base input coords for non-reduced axes
    uint icoords_base[8];
    uint out_pos = 0;
    const bool keep_dims = (P.out_v.rank == P.in_v.rank);
    for (ushort d=0; d<P.in_v.rank; ++d) {
        if (P.is_reduce_axis[d] != 0) {
            icoords_base[d] = 0;
        } else {
            uint c = keep_dims ? ocoords[d] : ocoords[out_pos++];
            icoords_base[d] = c;
        }
    }

    // Collect reduced dims
    ushort nred = 0;
    uint red_dims[8]; uint red_sizes[8];
    for (ushort d=0; d<P.in_v.rank; ++d) {
        if (P.is_reduce_axis[d] != 0) { red_dims[nred]=d; red_sizes[nred]=P.in_v.shape[d]; ++nred; }
    }

    float acc = (P.op == 0) ? 0.0f : -FLT_MAX;
    uint idxs[8] = {0};

    while (true) {
        uint icoords[8];
        for (ushort d=0; d<P.in_v.rank; ++d) icoords[d] = icoords_base[d];
        for (ushort i=0; i<nred; ++i) icoords[red_dims[i]] = idxs[i];

        uint ii = index_from_coords(icoords, P.in_v);
        float v = in_buf[ii];
        if (P.op == 0) acc += v; else acc = fmax(acc, v);

        if (nred == 0) break;
        ushort pos = nred;
        while (pos > 0) {
            --pos;
            idxs[pos]++;
            if (idxs[pos] < red_sizes[pos]) break;
            idxs[pos] = 0;
        }
        if (pos == 0 && idxs[0] == 0) break;
    }

    uint oi = index_from_coords(ocoords, P.out_v);
    out_buf[oi] = acc;
}
