// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

namespace cppgrad {
namespace backend {
namespace metal {

struct View32 {
    unsigned short rank;
    unsigned short pad;
    unsigned int   offset;
    unsigned int   flags;
    unsigned int   shape[8];
    unsigned int   strides[8];
};

struct UnaryParams {
    View32 in_v;
    View32 out_v;
    unsigned int   n;
    unsigned short op;
    unsigned short pad2;
};

struct BinaryParams {
    View32 a_v;
    View32 b_v;
    View32 o_v;
    unsigned int   n;
    unsigned short op;
    unsigned short pad2;
};

// Flash attention params (must match the struct in metal_kernels.metal). q [B,S,nH,Dh],
// k,v [B,KV,nKV,Dh] -> out [B,S,nH,Dh]; query head h reads kv head h/n_rep.
struct FlashParams {
    unsigned int B, S, nH, Dh, KV, nKV, n_rep, causal, q_offset;
    float        scale;
};

struct MatmulParams {
    View32 a_v;
    View32 b_v;
    View32 o_v;
    unsigned int M, K, N;
    unsigned int batch;       // number of batch matrices (1 for 2D)
    unsigned int a_batch_stride; // element stride per batch step in A
    unsigned int b_batch_stride; // element stride per batch step in B
    unsigned int o_batch_stride; // element stride per batch step in output
};

struct BroadcastParams {
    View32 in_v;
    View32 out_v;
    unsigned int n;
    unsigned int pad3;
};

struct PermuteParams {
    View32 in_v;
    View32 out_v;
    unsigned int n;
    unsigned short axes[8];
    unsigned short pad_axes[8];
};

struct SliceParams {
    View32 in_v;
    View32 out_v;
    unsigned int n;
    unsigned int begin[8];
    unsigned int step[8];
};

struct CopyViewParams {
    View32 src_v;
    View32 dst_v;
    unsigned int n;
    unsigned int pad4;
};

struct ReduceFastParams {
    View32 in_v;
    View32 out_v;
    unsigned int inner;
    unsigned short op;
    unsigned short pad5;
};

struct ReduceGeneralParams {
    View32 in_v;
    View32 out_v;
    unsigned short op;
    unsigned short pad6;
    unsigned int  out_total;
    unsigned char is_reduce_axis[8];
    unsigned char pad7[8];
};

struct ConcatParams {
    View32  in_views[2];   // up to 2 inputs
    View32  out_v;
    unsigned int n;         // total output elements
    unsigned int num_inputs;
    unsigned int axis;
    unsigned int cum_sizes[3]; // cum_sizes[0]=0, cum_sizes[1]=in0[axis], cum_sizes[2]=in0+in1
};

struct GatherAxisParams {
    View32  tensor_v;
    View32  out_v;
    unsigned int n;          // output elements
    unsigned int axis;
};

struct ScatterAxisParams {
    View32  base_v;
    View32  values_v;
    View32  out_v;
    unsigned int nval;       // value elements to scatter
    unsigned int axis;
};

} // namespace metal
} // namespace backend
} // namespace cppgrad
