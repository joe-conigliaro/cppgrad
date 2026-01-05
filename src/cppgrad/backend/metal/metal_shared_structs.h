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

struct MatmulParams {
    View32 a_v;
    View32 b_v;
    View32 o_v;
    unsigned int M, K, N;
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

} // namespace metal
} // namespace backend
} // namespace cppgrad
