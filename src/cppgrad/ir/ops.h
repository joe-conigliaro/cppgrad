// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <type_traits>

namespace cppgrad {
namespace ir {

enum class ConstantOpType {
    FULL,  // fill a tensor of `shape` with one scalar
    SCALAR // a rank-0 scalar tensor (shape `{}`, `numel=1`)
};
struct ConstantOp { ConstantOpType type; double value = 0.0; };

// Represents a pre-existing buffer. The root of a graph.
struct LeafOp {};

// Copy Operation (produces a new output tensor). Can cross devices.
struct CopyOp {};

struct GatherOp {};  // table[V, D, ...], indices[...] -> output[indices->shape(), D, ...] (gathers along axis 0)
struct GatherAxisOp { int axis; };  // gather elements along arbitrary axis
struct ScatterOp { int axis; };  // scatter values into base at indexed positions along axis

struct ConcatOp { int axis; };  // concatenate tensors along axis

// In-place assignment (mutates existing tensor). Same-device only.
// Used primarily for parameter updates in optimizers (e.g., w -= lr * grad).
struct AssignOp {};

// In-place autoregressive cache append. Writes `values` into a preallocated cache buffer
// at [.., start : start+S, ..] along `axis` and returns a view of the cache covering
// [.., 0 : start+S, ..]. The write and the read-view are a single atomic node, so there is
// no read-after-write hazard on the (hazard-naive) executor, and the in-place write makes
// each decode step O(S) instead of the O(context) copy that ConcatOp incurs per step.
// Inference only (no backward); batch dim must be 1 (autoregressive decode).
struct CacheUpdateOp { int axis; size_t start; };

struct MatMulOp {};

// Quantization scheme descriptor (an op parameter, like RandomParams). The backend dispatches on
// `scheme` internally, so a new scheme (GPTQ, AWQ, ...) is a kernel branch -- not a virtual per type.
enum class QuantScheme {
    MLX_AFFINE_U8,   // MLX 8-bit affine: w = scale*q + bias, unsigned codes, per-group scale/bias
};
struct QuantParams {
    QuantScheme scheme = QuantScheme::MLX_AFFINE_U8;
    int bits        = 8;    // bits per quantized code
    int group_size  = 64;   // codes sharing one (scale, bias), along the K (input) dim
    int pack_factor = 4;    // codes packed per storage word (e.g. 4 u8 per uint32)
};

// Quantized matmul (inference only, no backward): out = A @ dequant(W)^T. The `params` carry the
// quant scheme (+ bits/group_size/packing); the backend dispatches on scheme. Inputs: activation
// [M,K] fp32, packed qweight, and scheme-specific extra buffers (e.g. scales/biases for MLX affine).
struct QuantizedMatMulOp { QuantParams params; };

enum class RandomOpType { UNIFORM, NORMAL };
struct UniformParams { float min = 0.f, max = 0.f; };
struct NormalParams { float mean = 0.f, stddev = 1.f; };

using RandomParams = std::variant<UniformParams, NormalParams>;
struct RandomOp { RandomOpType type; RandomParams params; };

enum class UnaryOpType { RELU, EXP, LOG, NEG, TANH, SIN, COS };
struct UnaryOp { UnaryOpType type; };

enum class BinaryOpType { ADD, SUB, MUL, DIV, POW, CMP_EQ, CMP_GT, MIN, MAX };
struct BinaryOp { BinaryOpType type; };

enum class ReduceOpType { SUM, MAX };
struct ReduceOp { ReduceOpType type; std::vector<int> axes; bool keep_dims = false; };

enum class MovementOpType { RESHAPE, PERMUTE, BROADCAST, SLICE };
struct MovementOp {
    MovementOpType type;
    // For RESHAPE/BROADCAST: arg = shape
    // For PERMUTE: arg = axes
    // For SLICE: arg = slice step
    std::vector<size_t> arg;
    // Slice params optional.
    std::vector<size_t> slice_begin;
    std::vector<size_t> slice_end; // exclusive
};

// The main Op variant,
using Op = std::variant<
    ConstantOp,
    LeafOp,
    CopyOp,
    AssignOp,
    CacheUpdateOp,
    MatMulOp,
    QuantizedMatMulOp,
    RandomOp,
    UnaryOp,
    BinaryOp,
    ReduceOp,
    MovementOp,
    GatherOp,
    GatherAxisOp,
    ScatterOp,
    ConcatOp
>;

inline const char* to_string(const ConstantOp& op)    { return "ConstantOp"; }
inline const char* to_string(const LeafOp& op)        { return "LeafOp"; }
inline const char* to_string(const CopyOp& op)        { return "CopyOp"; }
inline const char* to_string(const GatherOp& op)      { return "GatherOp"; }
inline const char* to_string(const GatherAxisOp& op)  { return "GatherAxisOp"; }
inline const char* to_string(const ScatterOp& op)     { return "ScatterOp"; }
inline const char* to_string(const ConcatOp& op)      { return "ConcatOp"; }
inline const char* to_string(const AssignOp& op)      { return "AssignOp"; }
inline const char* to_string(const CacheUpdateOp& op) { return "CacheUpdateOp"; }
inline const char* to_string(const MatMulOp& op)      { return "MatMulOp"; }
inline const char* to_string(const QuantizedMatMulOp& op) { return "QuantizedMatMulOp"; }
inline const char* to_string(const RandomOp& op) {
    switch (op.type) {
        case RandomOpType::UNIFORM: return "RandomOp:UNIFORM";
        case RandomOpType::NORMAL:  return "RandomOp:NORMAL";
    }
}
inline const char* to_string(const UnaryOp& op) {
    switch (op.type) {
        case UnaryOpType::RELU: return "UnaryOp:RELU";
        case UnaryOpType::EXP:  return "UnaryOp:EXP";
        case UnaryOpType::LOG:  return "UnaryOp:LOG";
        case UnaryOpType::NEG:  return "UnaryOp:NEG";
        case UnaryOpType::TANH: return "UnaryOp:TANH";
        case UnaryOpType::SIN:  return "UnaryOp:SIN";
        case UnaryOpType::COS:  return "UnaryOp:COS";
    }
}
inline const char* to_string(const BinaryOp& op) {
    switch (op.type) {
        case BinaryOpType::ADD:    return "BinaryOp:ADD";
        case BinaryOpType::SUB:    return "BinaryOp:SUB";
        case BinaryOpType::MUL:    return "BinaryOp:MUL";
        case BinaryOpType::DIV:    return "BinaryOp:DIV";
        case BinaryOpType::POW:    return "BinaryOp:POW";
        case BinaryOpType::CMP_EQ: return "BinaryOp:CMP_EQ";
        case BinaryOpType::CMP_GT: return "BinaryOp:CMP_GT";
        case BinaryOpType::MIN:    return "BinaryOp:MIN";
        case BinaryOpType::MAX:    return "BinaryOp:MAX";
    }
}
inline const char* to_string(const ReduceOp& op) {
    switch (op.type) {
        case ReduceOpType::SUM:  return "UnaryOp:REDUCE_SUM";
        case ReduceOpType::MAX:  return "UnaryOp:REDUCE_MAX";
    }
}
inline const char* to_string(const MovementOp& op) {
    switch (op.type) {
        case MovementOpType::RESHAPE:   return "MovementOp:RESHAPE";
        case MovementOpType::PERMUTE:   return "MovementOp:PERMUTE";
        case MovementOpType::BROADCAST: return "MovementOp:BROADCAST";
        case MovementOpType::SLICE:     return "MovementOp:SLICE";
    }
}

inline const char* to_string(const Op& op_v) {
    return std::visit([](const auto& op) -> const char* {
        return to_string(op);
    }, op_v);
}

// Check if `Op` is a start node.
// Comptime.
template <class T>
inline constexpr bool is_start_node_v =
    std::is_same_v<std::decay_t<T>, ConstantOp> ||
    std::is_same_v<std::decay_t<T>, RandomOp>   ||
    std::is_same_v<std::decay_t<T>, LeafOp>;
// Runtime.
inline bool is_start_node(const Op& op_v) {
    return std::visit([](auto&& op) {
        using T = std::decay_t<decltype(op)>;
        return is_start_node_v<T>;
    }, op_v);
}

// Check if `Op` supports autograd (has a backward rule).
// Comptime.
template <class T>
inline constexpr bool is_differentiable_v =
    !std::is_same_v<std::decay_t<T>, ConstantOp> &&
    !std::is_same_v<std::decay_t<T>, RandomOp>   &&
    !std::is_same_v<std::decay_t<T>, LeafOp>     &&
    !std::is_same_v<std::decay_t<T>, AssignOp>  &&
    !std::is_same_v<std::decay_t<T>, CacheUpdateOp> &&
    !std::is_same_v<std::decay_t<T>, GatherOp> &&
    !std::is_same_v<std::decay_t<T>, GatherAxisOp> &&
    !std::is_same_v<std::decay_t<T>, QuantizedMatMulOp> &&
    !std::is_same_v<std::decay_t<T>, ScatterOp>;
// ConcatOp is differentiable (backward: split grad along axis).
// Runtime.
inline bool is_differentiable(const Op& op_v) {
    return std::visit([](auto&& op) {
        using T = std::decay_t<decltype(op)>;
        return is_differentiable_v<T>;
    }, op_v);
}

} // namespace ir
} // namespace cppgrad
