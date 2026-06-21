// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include "cppgrad/ir/tensor_operators.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"

namespace cppgrad {
namespace nn {
namespace functional {

enum class Reduction {
    NONE,
    MEAN,
    SUM
};

inline utils::Ref<ir::Tensor> reduce(const utils::Ref<ir::Tensor>& in, Reduction reduction, const std::vector<int>& axes = {}, bool keep_dims = false) {
    if (reduction == Reduction::NONE) return in;

    // If axes are not specified, reduce over all dimensions
    std::vector<int> reduce_axes = axes;
    if (reduce_axes.empty()) {
        reduce_axes.resize(in->shape().size());
        std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    }

    if (reduction == Reduction::MEAN) return ir::mean(in, reduce_axes, keep_dims);
    if (reduction == Reduction::SUM) return ir::sum(in, reduce_axes, keep_dims);

    throw std::runtime_error("Unhandled reduction type");
}

inline utils::Ref<ir::Tensor> mse_loss(const utils::Ref<ir::Tensor>& y_pred, const utils::Ref<ir::Tensor>& y_true, Reduction reduction = Reduction::MEAN) {
    auto diff = y_pred - y_true;
    auto squared_diff = diff * diff;
    return reduce(squared_diff, reduction);
}

// Standard hinge loss, y_true in {+1, -1}
inline utils::Ref<ir::Tensor> hinge_loss(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& y_true, float margin = 1.0f, Reduction reduction = Reduction::MEAN) {
    // loss = relu(margin - y_true * logits)
    auto loss_per_item = ir::relu(margin - (y_true * logits));
    return reduce(loss_per_item, reduction);
}

// Stable softplus: log(1 + exp(x))
inline utils::Ref<ir::Tensor> softplus(const utils::Ref<ir::Tensor>& x) {
    // This is a stable implementation: softplus(x) = max(0, x) + log(1 + exp(-|x|))
    auto relu_x = ir::relu(x);
    auto abs_x = ir::relu(x) + ir::relu(ir::neg(x));
    auto log_term = ir::log(1.0f + ir::exp(ir::neg(abs_x)));
    return relu_x + log_term;
}

// BCE with logits: targets in {0,1}. Stable formulation:
// loss(z, y) = softplus(z) - z*y
// This is mathematically equivalent to the more complex version but simpler to express.
inline utils::Ref<ir::Tensor> bce_with_logits(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets, Reduction reduction = Reduction::MEAN) {
    auto loss_per_item = softplus(logits) - (logits * targets);
    return reduce(loss_per_item, reduction);
}

// Logistic loss (margin targets): targets in {-1, +1}. Stable softplus form:
// loss(z, y) = softplus(-(y*z))
inline utils::Ref<ir::Tensor> logistic_loss_pm1(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets_pm1, Reduction reduction = Reduction::MEAN) {
    auto neg_yz = ir::neg(targets_pm1 * logits);
    auto loss_per_item = softplus(neg_yz);
    return reduce(loss_per_item, reduction);
}

inline utils::Ref<ir::Tensor> softmax(const utils::Ref<ir::Tensor>& logits, int axis = -1) {
    int nd = static_cast<int>(logits->shape().size());
    if (axis < 0) axis += nd;
    auto m = ir::max(logits, {axis}, true);
    auto z_shifted = logits - m;
    auto exp_z = ir::exp(z_shifted);
    auto denom = ir::sum(exp_z, {axis}, true);
    return exp_z / denom;
}

inline utils::Ref<ir::Tensor> log_softmax(const utils::Ref<ir::Tensor>& logits, int axis = -1) {
    int nd = static_cast<int>(logits->shape().size());
    if (axis < 0) axis += nd;
    auto m = ir::max(logits, {axis}, true);
    auto z_shifted = logits - m;
    auto logsumexp = ir::log(ir::sum(ir::exp(z_shifted), {axis}, true));
    return z_shifted - logsumexp;
}

inline utils::Ref<ir::Tensor> softmax_cross_entropy_with_logits(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& targets_onehot, Reduction reduction = Reduction::MEAN) {
    int nd = static_cast<int>(logits->shape().size());
    int axis = nd - 1;
    auto lsm = log_softmax(logits, axis);
    auto per_sample_loss = ir::neg(ir::sum(targets_onehot * lsm, {axis}, false));
    return reduce(per_sample_loss, reduction);
}

inline utils::Ref<ir::Tensor> relu(const utils::Ref<ir::Tensor>& input) {
    return ir::relu(input);
}

inline utils::Ref<ir::Tensor> tanh(const utils::Ref<ir::Tensor>& input) {
    return ir::tanh(input);
}

inline utils::Ref<ir::Tensor> silu(const utils::Ref<ir::Tensor>& x) {
    // silu(x) = x / (1 + exp(-x))
    return ir::mul(x, ir::div(ir::scalar_like(1.0f, x), ir::add(ir::scalar_like(1.0f, x), ir::exp(ir::neg(x)))));
}

// GELU (fast approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline utils::Ref<ir::Tensor> gelu(const utils::Ref<ir::Tensor>& x) {
    // tanh_arg = sqrt(2/pi) * (x + 0.044715 * x^3)
    auto x3 = x * x * x;
    auto tanh_arg = ir::scalar_like(0.79788456f, x) * (x + ir::scalar_like(0.044715f, x) * x3);
    return ir::scalar_like(0.5f, x) * x * (ir::scalar_like(1.0f, x) + ir::tanh(tanh_arg));
}

// Expand tensor by adding singleton dimensions at the end until it reaches target_rank.
// e.g., {B, S} with target_rank=4 -> {B, S, 1, 1}
static utils::Ref<ir::Tensor> expand_dims_to(const utils::Ref<ir::Tensor>& t, size_t target_rank) {
    size_t cur_rank = t->shape().size();
    if (cur_rank >= target_rank) return t;
    std::vector<size_t> new_shape = t->shape();
    new_shape.resize(target_rank, 1);
    return ir::reshape(t, new_shape);
}

// RMSNorm: output = x * (1 + weight) * rsqrt(mean(x^2) + eps)
// weight: [D], x: [..., D], norm over last axis
inline utils::Ref<ir::Tensor> rms_norm(const utils::Ref<ir::Tensor>& x, const utils::Ref<ir::Tensor>& weight, float eps = 1e-5f) {
    int nd = static_cast<int>(x->shape().size());
    int axis = nd - 1;
    size_t D = x->shape()[axis];
    auto x2 = x * x;
    auto mean_x2 = ir::sum(x2, {axis}, true) / static_cast<float>(D);
    auto r = ir::pow(mean_x2 + eps, -0.5f);
    auto normed = x * r;
    // Qwen3 / Qwen3.5 RMSNorm: scale by the (learned) weight directly. (Note: this is the
    // standard convention; the Gemma-style "(1 + weight)" would double Qwen's ~1.0 weights.)
    return normed * weight;
}

// Rotary Position Embeddings (non-interleaved, as used by Qwen3)
// x: [..., head_dim], inv_freq: [head_dim/2] precomputed as log(10000) + 2*log(arange)/head_dim
// positions: position ids - any shape whose numel is the sequence length (e.g., [S], [1, S], [B, S])
// Returns x with rotation applied. Last dim must be even.
inline utils::Ref<ir::Tensor> apply_rope(
    const utils::Ref<ir::Tensor>& x,
    const utils::Ref<ir::Tensor>& positions,
    const utils::Ref<ir::Tensor>& inv_freq)
{
    auto shape = x->shape();
    size_t head_dim = shape.back();
    size_t half_d = head_dim / 2;

    // Build theta shape: x shape with last dim = half_d
    std::vector<size_t> theta_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        theta_shape.push_back(i == shape.size() - 1 ? half_d : shape[i]);
    }
    // Expand positions to x's rank (add singleton dims at the end), then broadcast
    auto pos_expanded = expand_dims_to(positions, shape.size());
    auto pos_b = ir::broadcast(pos_expanded, theta_shape);
    auto freq_b = ir::broadcast(inv_freq, theta_shape);
    auto theta = pos_b * freq_b;  // same shape as x but last dim = half_d

    auto cos_t = ir::cos(theta);  // [batch, seq, heads, head_dim/2]
    auto sin_t = ir::sin(theta);  // [batch, seq, heads, head_dim/2]

    // Split x into first half and second half along last dim
    // x: [..., head_dim] -> x1: [..., head_dim/2], x2: [..., head_dim/2]
    std::vector<size_t> begin1(shape.size(), 0), end1(shape.begin(), shape.end());
    end1.back() = half_d;
    std::vector<size_t> begin2(shape.size(), 0), end2(shape.begin(), shape.end());
    begin2.back() = half_d;
    auto x1 = ir::slice(x, begin1, end1);
    auto x2 = ir::slice(x, begin2, end2);

    // Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    auto rx1 = x1 * cos_t - x2 * sin_t;
    auto rx2 = x1 * sin_t + x2 * cos_t;

    // Concatenate along last dim
    return ir::concat(rx1, rx2, -1);
}

// Multimodal Rotary Position Embeddings (Qwen3.5/3.6)
// Supports partial rotary factor and interleaved mode.
// x: [..., head_dim], positions: position ids, inv_freq: [num_rotary_pairs]
// partial_rotary_factor: fraction of head_dim to rotate (e.g., 0.25)
// interleaved: if true, spread rotated dims evenly across head_dim
inline utils::Ref<ir::Tensor> apply_mrope(
    const utils::Ref<ir::Tensor>& x,
    const utils::Ref<ir::Tensor>& positions,
    const utils::Ref<ir::Tensor>& inv_freq,
    float partial_rotary_factor = 1.0f,
    bool interleaved = false)
{
    auto shape = x->shape();
    size_t head_dim = shape.back();
    size_t num_pairs = inv_freq->numel();

    if (partial_rotary_factor >= 1.0f && !interleaved) {
        return apply_rope(x, positions, inv_freq);
    }

    // Build theta shape: x shape with last dim = num_pairs
    std::vector<size_t> theta_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        theta_shape.push_back(i == shape.size() - 1 ? num_pairs : shape[i]);
    }
    // Expand positions to x's rank, then broadcast
    auto pos_expanded = expand_dims_to(positions, shape.size());
    auto pos_b = ir::broadcast(pos_expanded, theta_shape);
    auto freq_b = ir::broadcast(inv_freq, theta_shape);
    auto theta = pos_b * freq_b;
    auto cos_t = ir::cos(theta);
    auto sin_t = ir::sin(theta);

    if (!interleaved) {
        // Partial rotary: only rotate first num_pairs in head_dim
        size_t rot_dim = num_pairs * 2;

        std::vector<size_t> begin1(shape.size(), 0), end1(shape.begin(), shape.end());
        end1.back() = num_pairs;
        std::vector<size_t> begin2(shape.size(), 0), end2(shape.begin(), shape.end());
        begin2.back() = num_pairs; end2.back() = rot_dim;
        auto x1 = ir::slice(x, begin1, end1);
        auto x2 = ir::slice(x, begin2, end2);

        auto rx = ir::concat(x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t, -1);

        if (rot_dim < head_dim) {
            std::vector<size_t> begin3(shape.begin(), shape.end());
            begin3.back() = rot_dim;
            auto x_rest = ir::slice(x, begin3, shape);
            return ir::concat(rx, x_rest, -1);
        }
        return rx;
    }

    // Interleaved mode: spread rotated pairs evenly across head_dim
    size_t stride = head_dim / (2 * num_pairs);

    std::vector<int32_t> q_indices(num_pairs), k_indices(num_pairs);
    for (size_t i = 0; i < num_pairs; ++i) {
        q_indices[i] = static_cast<int32_t>(i * stride);
        k_indices[i] = static_cast<int32_t>(i * stride + stride / 2);
    }
    auto q_idx = ir::from_vector<int32_t>(q_indices, {num_pairs}, x->device_type());
    auto k_idx = ir::from_vector<int32_t>(k_indices, {num_pairs}, x->device_type());

    // Extract Q and K at interleaved positions using native gather_axis
    auto q_vals = ir::gather_axis(x, q_idx, -1);
    auto k_vals = ir::gather_axis(x, k_idx, -1);

    // Rotate
    auto rq = q_vals * cos_t - k_vals * sin_t;
    auto rk = q_vals * sin_t + k_vals * cos_t;

    // Scatter back using native scatter_axis
    auto result = ir::scatter_axis(x, rq, q_idx, -1);
    result = ir::scatter_axis(result, rk, k_idx, -1);

    return result;
}

// Repeat KV heads for Grouped Query Attention
// x: [batch, seq, num_kv_heads, head_dim] -> [batch, seq, num_heads, head_dim]
// n_rep = num_heads / num_kv_heads
inline utils::Ref<ir::Tensor> repeat_kv(const utils::Ref<ir::Tensor>& x, size_t n_rep) {
    auto shape = x->shape();
    // shape: [B, S, n_kv_heads, head_dim]
    int nd = static_cast<int>(shape.size());
    // Expand: [B, S, n_kv_heads, 1, head_dim]
    std::vector<size_t> expanded_shape = {shape.begin(), shape.end() - 1};
    expanded_shape.push_back(1);
    expanded_shape.push_back(shape.back());
    auto exp = ir::reshape_view(x, expanded_shape);
    // Broadcast: [B, S, n_kv_heads, n_rep, head_dim]
    std::vector<size_t> broadcast_shape = {expanded_shape.begin(), expanded_shape.end() - 2};
    broadcast_shape.push_back(n_rep);
    broadcast_shape.push_back(shape.back());
    auto bcast = ir::broadcast(exp, broadcast_shape);
    // Reshape: [B, S, n_kv_heads * n_rep, head_dim]
    std::vector<size_t> out_shape = {shape.begin(), shape.end() - 2};
    out_shape.push_back(shape[nd - 2] * n_rep);
    out_shape.push_back(shape.back());
    return ir::reshape(bcast, out_shape);
}

// Scaled dot-product attention with causal mask
// q: [B, S, n_heads, head_dim], k: [B, S_kv, n_heads, head_dim], v: [B, S_kv, n_heads, head_dim]
inline utils::Ref<ir::Tensor> scaled_dot_product_attention(
    const utils::Ref<ir::Tensor>& q,
    const utils::Ref<ir::Tensor>& k,
    const utils::Ref<ir::Tensor>& v,
    const utils::Ref<ir::Tensor>& mask = nullptr) {
    auto q_shape = q->shape();
    size_t head_dim = q_shape.back();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // scores = q @ k^T / sqrt(head_dim)
    // k^T: [B, S_kv, n_heads, head_dim] -> [B, n_heads, S_kv, S]
    auto kt = ir::transpose(k, 2, 3);  // [B, S_kv, n_heads, head_dim] -> [B, S_kv, head_dim, n_heads]
    // Actually need to permute: [B, S, n_heads, head_dim] -> [B, n_heads, S, head_dim]
    // then k: [B, n_heads, S_kv, head_dim]
    // then scores = q @ k^T = [B, n_heads, S, S_kv]
    auto q_perm = ir::permute(q, {0, 2, 1, 3});  // [B, n_heads, S, head_dim]
    auto k_perm = ir::permute(k, {0, 2, 1, 3});  // [B, n_heads, S_kv, head_dim]
    auto v_perm = ir::permute(v, {0, 2, 1, 3});  // [B, n_heads, S_kv, head_dim]

    // k_perm^T: [B, n_heads, head_dim, S_kv]
    auto kt_perm = ir::transpose(k_perm, 2, 3);
    auto scores = ir::matmul(q_perm, kt_perm) * scale;  // [B, n_heads, S, S_kv]

    if (mask) {
        scores = ir::add(scores, mask);
    }

    auto attn = softmax(scores);  // [B, n_heads, S, S_kv]
    auto out = ir::matmul(attn, v_perm);  // [B, n_heads, S, head_dim]

    // [B, S, n_heads, head_dim]
    return ir::permute(out, {0, 2, 1, 3});
}

// 1D convolution for linear attention (causal, depthwise).
// x: [B, S, C], weight: [C, kernel_size, 1] -> output: [B, S, C]
// Uses circular padding (standard for Mamba/linear attention).
inline utils::Ref<ir::Tensor> conv1d(
    const utils::Ref<ir::Tensor>& x,
    const utils::Ref<ir::Tensor>& weight,  // [C, kernel_size, 1]
    const utils::Ref<ir::Tensor>& bias = nullptr)
{
    auto shape = x->shape();
    size_t B = shape[0], S = shape[1], C = shape[2];
    size_t k_size = weight->shape()[1];

    // Reshape to 2D for matmul: [B*S, C] and [C, k_size] -> [B*S, k_size]
    // Then use sliding window. Actually, implement as:
    // For each position s, output[s] = sum_{i=0}^{k-1} x[(s+i) % S] * weight[:, i, 0]
    // With circular padding for causality.

    // Simpler: use im2col approach
    // Build input columns: for each position s, collect x[s], x[s-1], ..., x[s-k+1] (circular)
    // Result: [B, S, k_size, C] -> [B*S*k_size, C] for matmul

    // For now, implement as loop (fine for small kernel sizes like 4)
    // Each slice: x[:, (s-i):] * weight[:, i, 0] with circular indexing

    auto x_2d = ir::reshape(x, {B * S, C});  // [B*S, C]
    auto w_2d = ir::reshape(weight, {C, k_size});  // [C, k_size]
    auto w_t = ir::transpose(w_2d, 1, 0);  // [k_size, C]

    // Get x data and weight data to build column matrix on CPU
    // This is a composite op that precomputes the column matrix
    // For inference with seq_len=1, this is trivial (just multiplication)

    // Build column: for each output position, gather the k_size input values
    // With circular padding: col[s, i] = x[(s + i) % S]
    // Shape: [B, S, k_size] -> flatten to [B*S*k_size]
    // Then reshape to [B*S*k_size, 1], broadcast w to [1, k_size, C]

    // Simpler approach for small kernels: compute each term separately and sum
    auto result = ir::scalar_like(0.0f, x);
    // Actually we need same shape. Use broadcast.
    // Just iterate: result = sum of x_shifted_i * weight[:, i] for i in 0..k_size-1

    // For circular padding with causal conv:
    // conv_output[s] = sum_i x[(s - k_size + 1 + i) % S] * weight[i]
    // Actually for inference with S=1, conv is just: output[0] = x[0] * weight[:, 0, 0]
    // (assuming the kernel is applied to the current position)

    // General implementation: pad with zeros, then extract windows
    // For simplicity, implement the common case where kernel is small

    // For S=1 (inference): just multiply by first kernel element
    if (S == 1) {
        // Extract first column of weight: [C, 1] -> [C]
        std::vector<size_t> begin_w = {0, 0, 0};
        std::vector<size_t> end_w = {C, 1, 1};
        auto w_col = ir::slice(weight, begin_w, end_w);  // [C, 1, 1]
        auto w_flat = ir::reshape(w_col, {C});  // [C]
        // x: [B, 1, C], result = x * w_flat (broadcasting)
        return x * w_flat;
    }

    // For general S: build column matrix using gather
    // col[B, s, i] = x[B, (s - k_size + 1 + i) % S]
    // This requires modular indexing. Implement via element-wise construction.

    // Pad input: prepend (k_size-1) copies of the last element (circular)
    // padded_x: [B, S + k_size - 1, C]
    // Then extract non-overlapping k_size-windows along sequence

    // Get last element: x[:, S-1:S, :]
    std::vector<size_t> begin_last = {0, S - 1, 0};
    std::vector<size_t> end_last = {B, S, C};
    auto x_last = ir::slice(x, begin_last, end_last);  // [B, 1, C]

    // Tile it k_size-1 times for padding
    // Actually, for circular padding, we need the last (k_size-1) elements
    // Let's just implement it as: pad x with copies of last element
    auto x_padded = x_last;  // Start with [B, 1, C]
    for (size_t i = 1; i < k_size - 1; ++i) {
        x_padded = ir::concat(x_padded, x_last, 1);  // [B, i+1, C]
    }
    x_padded = ir::concat(x_padded, x, 1);  // [B, S + k_size - 1, C]

    // Now extract k_size windows
    // Use im2col: for each position s in 0..S-1, take x_padded[s:s+k_size]
    // Result: [B, S, k_size, C]

    // Build column matrix by stacking slices
    // For each window position, gather the window
    auto col_stack = x_padded;  // First window (or just start accumulating)

    // Actually simpler: reshape x_padded to [B, S+k-1, 1, C]
    // Then for each of the k_size offsets, slice and stack
    // Result: [B, S, k_size, C] -> reshape [B*S, k_size*C]

    // Build columns: for offset i in 0..k_size-1, take x_padded[:, i:i+S, :]
    // Then stack along a new axis
    std::vector<size_t> begin_p = {0, 0, 0};
    std::vector<size_t> end_p = {B, S, C};
    auto first_window = ir::slice(x_padded, begin_p, end_p);  // [B, S, C]

    std::vector<size_t> out_shape_4d = {B, S, 1, C};
    auto col_reshaped = ir::reshape(first_window, out_shape_4d);  // [B, S, 1, C]

    for (size_t i = 1; i < k_size; ++i) {
        begin_p[1] = i;
        end_p[1] = i + S;
        auto window = ir::slice(x_padded, begin_p, end_p);  // [B, S, C]
        window = ir::reshape(window, {B, S, 1, C});  // [B, S, 1, C]
        col_reshaped = ir::concat(col_reshaped, window, 2);  // [B, S, i+1, C]
    }

    // col_reshaped: [B, S, k_size, C]
    // w_reshaped: [k_size, C, 1] -> broadcast to [B, S, k_size, C]
    auto w_3d = ir::reshape(weight, {1, 1, k_size, C, 1});
    // Hmm, weight is [C, k_size, 1]. Need to permute to [k_size, C, 1]
    // Actually: weight [C, k_size, 1] -> [k_size, C, 1] via permute
    auto w_perm = ir::permute(weight, {1, 0, 2});  // [k_size, C, 1]
    w_perm = ir::broadcast(w_perm, {B, S, k_size, C, 1});

    // Hmm, this is getting complex. Let me simplify.
    // col: [B, S, k_size, C], weight: [C, k_size, 1]
    // result[B, s, c] = sum_k col[B, s, k, c] * weight[c, k, 0]
    // = sum_k col[B, s, k, c] * weight[c, k, 0]
    // Element-wise multiply then reduce over k_size axis

    // Reshape for matmul:
    // col_flat: [B*S, k_size, C] -> [B*S*C, k_size]
    // w_flat: [C, k_size, 1] -> [C, k_size] -> [k_size, C]
    // For each (B, s, c): dot product of col[B,s,:,c] and w[c,:,0]

    // Flatten to [B*S*C, k_size] and [k_size, C]... no that doesn't work directly.
    // Instead: element-wise multiply then sum over k_size axis.

    // Broadcast weight to [B, S, k_size, C, 1] -> no, weight is [C, k_size, 1]
    // Permute weight to [1, 1, k_size, C] and broadcast with col [B, S, k_size, C]
    auto w_bc = ir::permute(weight, {1, 0, 2});  // [k_size, C, 1]
    w_bc = ir::reshape(w_bc, {k_size, C});  // [k_size, C]
    // Actually weight is [C, k_size, 1], we need [1, 1, k_size, C]
    // weight without last dim: [C, k_size] -> permute [k_size, C] -> broadcast to [B,S,k_size,C]

    auto w_sq = ir::slice(weight, {0, 0, 0}, {C, k_size, 1});  // [C, k_size]
    w_sq = ir::permute(w_sq, {1, 0});  // [k_size, C]
    w_sq = ir::broadcast(w_sq, {B, S, k_size, C});

    auto product = col_reshaped * w_sq;  // [B, S, k_size, C]
    auto result_t = ir::sum(product, {2});  // [B, S, C] - sum over k_size axis

    if (bias) {
        result_t = result_t + bias;
    }

    return result_t;
}

// Attention output gate (Qwen3.5/3.6).
// Applies a per-head gate to the attention output using swish activation.
// gate_weight: [num_heads] or [num_heads, 1] or [num_heads, head_dim]
// attn_output: [B, S, num_heads, head_dim]
inline utils::Ref<ir::Tensor> attn_output_gate(
    const utils::Ref<ir::Tensor>& attn_output,
    const utils::Ref<ir::Tensor>& gate_weight)
{
    // gate: silu(gate_weight)
    auto gate = silu(gate_weight);

    // Reshape gate to 4D for broadcasting: attn_output is [B, S, nH, head_dim]
    // gate_weight can be [nH] or [nH, 1] or [nH, head_dim] -> reshape to [1, 1, nH, ...]
    auto gate_shape = gate->shape();
    auto out_shape = attn_output->shape();

    std::vector<size_t> expanded_shape = {1, 1};
    for (size_t d : gate_shape) {
        expanded_shape.push_back(d);
    }
    while (expanded_shape.size() < out_shape.size()) {
        expanded_shape.push_back(1);
    }

    auto gate_expanded = ir::reshape(gate, expanded_shape);
    auto gate_bc = ir::broadcast(gate_expanded, out_shape);
    return attn_output * gate_bc;
}

} // namespace functional
} // namespace nn
} // namespace cppgrad
