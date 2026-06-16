// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include "cppgrad/nn/module.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/nn/llm/qwen/qlinear.h"

namespace cppgrad {
namespace nn {
namespace llm {
namespace qwen {

// Unified Qwen3 transformer block supporting both full attention (Qwen3) and
// linear attention (Qwen3.5/3.6 SSM-style) layers.
//
// The constructor takes a LayerType enum to determine which forward path to
// use at runtime.  Both paths share the parallel (mix) residual pattern:
//   output = x + branch1(norm1(x)) + branch2(norm1(x))
// followed by a post-norm RMSNorm.
class Qwen3Block : public Module {
public:
    // ========== Full-attention weights ==========

    // Norms
    utils::Ref<ir::Tensor> fa_norm1_weight;
    utils::Ref<ir::Tensor> fa_norm2_weight;

    // Attention projections (no bias in Qwen3)
    QLinear fa_q_proj_weight;
    QLinear fa_k_proj_weight;
    QLinear fa_v_proj_weight;
    QLinear fa_o_proj_weight;

    // Per-head k / q RMSNorm (Qwen3.5+)
    utils::Ref<ir::Tensor> fa_k_norm_weight;
    utils::Ref<ir::Tensor> fa_q_norm_weight;

    // FFN projections (SwiGLU: gate + up + down)
    QLinear fa_gate_proj_weight;
    QLinear fa_up_proj_weight;
    QLinear fa_down_proj_weight;

    // ========== Linear-attention weights ==========

    // Norms
    utils::Ref<ir::Tensor> la_norm1_weight;
    utils::Ref<ir::Tensor> la_norm2_weight;

    // Conv1d weight: [qkv_output_dim, linear_conv_kernel_dim, 1]
    utils::Ref<ir::Tensor> la_conv1d_weight;

    // In-projections  (all [output_dim, hidden])
    QLinear la_in_proj_qkv_weight;  // QKV combined
    QLinear la_in_proj_a_weight;    // A for state update
    QLinear la_in_proj_b_weight;    // B for state update
    QLinear la_in_proj_z_weight;    // Gate projection

    // State RMSNorm
    utils::Ref<ir::Tensor> la_norm_weight;

    // Log decay rates: [linear_num_value_heads]
    utils::Ref<ir::Tensor> la_A_log;

    // Softplus bias for the decay gate: [linear_num_value_heads]
    utils::Ref<ir::Tensor> la_dt_bias;

    // Output projection: [hidden, linear_output_dim]
    QLinear la_out_proj_weight;

    // MLP (dense SwiGLU) -- every decoder layer has one, including linear-attention layers.
    QLinear la_gate_proj_weight;
    QLinear la_up_proj_weight;
    QLinear la_down_proj_weight;

    // ========== Constructor ==========

    // lazy_weights=true creates parameters with deferred (unallocated) storage and no random init,
    // for inference where weights are loaded from a checkpoint. This avoids materializing the full
    // fp32 weight set at construction (~108GB for the 27B) before load_from_safetensors replaces it.
    Qwen3Block(LayerType layer_type,
               const Qwen3Config& config,
               backend::DeviceType device_type = backend::DeviceManager::default_device_type(),
               bool lazy_weights = false)
    : _layer_type(layer_type), _config(config), _lazy(lazy_weights)
    {
        if (layer_type == LayerType::FULL_ATTENTION) {
            init_full_attention(device_type);
        } else {
            init_linear_attention(device_type);
        }
    }

    // Forward pass
    //
    // x          : [B, S, H]
    // positions  : [S] or [B, 1, S, 1]  -- position ids
    // inv_freq   : [num_rotary_pairs]    -- precomputed inverse frequencies
    // mask       : attention mask (full-attention only, ignored for linear)
    // state_in   : [B, n_v_heads, key_head_dim, value_head_dim] -- linear attention
    //              state cache (nullptr for full attention)
    //
    // Returns output [B, S, H].
    utils::Ref<ir::Tensor> forward(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& positions,
        const utils::Ref<ir::Tensor>& inv_freq,
        const utils::Ref<ir::Tensor>& mask,
        const utils::Ref<ir::Tensor>& state_in = nullptr)
    {
        if (_layer_type == LayerType::FULL_ATTENTION) {
            return forward_full_attention(x, positions, inv_freq, mask);
        } else {
            return forward_linear_attention(x, positions, inv_freq, state_in);
        }
    }

    // Linear-attention forward with explicit recurrent + conv state in/out (for cached decode
    // and for validating prefill==stepwise equivalence). state_in/conv_state_in may be null.
    utils::Ref<ir::Tensor> forward_linear_cached(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& state_in,
        const utils::Ref<ir::Tensor>& conv_state_in,
        utils::Ref<ir::Tensor>& state_out,
        utils::Ref<ir::Tensor>& conv_state_out)
    {
        auto o = forward_linear_attention(x, nullptr, nullptr, state_in, conv_state_in);
        state_out = _last_linear_state;
        conv_state_out = _last_conv_state;
        return o;
    }

    // Full-attention forward with an in-place preallocated K/V cache. k_cache/v_cache are
    // persistent [1, max_len, n_kv, head_dim] leaves; this step's K/V are written at
    // [.., start_pos:start_pos+S, ..] and attention reads the [0:start_pos+S] prefix. mask is
    // the additive attention mask (causal for prefill, null for single-token decode).
    utils::Ref<ir::Tensor> forward_full_cached(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& positions,
        const utils::Ref<ir::Tensor>& inv_freq,
        const utils::Ref<ir::Tensor>& mask,
        const utils::Ref<ir::Tensor>& k_cache,
        const utils::Ref<ir::Tensor>& v_cache,
        size_t start_pos)
    {
        return forward_full_attention(x, positions, inv_freq, mask, k_cache, v_cache, start_pos);
    }

    LayerType get_layer_type() const { return _layer_type; }
    const Qwen3Config& get_config() const { return _config; }

private:
    LayerType _layer_type;
    Qwen3Config _config;
    bool _lazy = false;

    // ==================== Full-attention init ====================

    void init_full_attention(backend::DeviceType device_type)
    {
        int32_t H = _config.hidden_size;
        int32_t D = _config.head_dim;
        int32_t n_heads = _config.num_attention_heads;
        int32_t n_kv    = _config.num_key_value_heads;
        int32_t I       = _config.intermediate_size;

        // Lazy mode: deferred (unallocated) leaf params, no random init -- weights come from load.
        auto attn_init = [&](std::vector<size_t> shape, float limit) {
            return _lazy ? ir::parameter(shape, device_type, backend::DType::FLOAT32, false)
                         : ir::parameterize(ir::uniform(shape, -limit, limit, device_type));
        };
        auto ones_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, backend::DType::FLOAT32, false)
                         : ir::parameterize(ir::ones(shape, device_type));
        };

        // -- Norms --
        fa_norm1_weight = ones_init({(size_t)H});
        fa_norm2_weight = ones_init({(size_t)H});
        register_parameter("fa_norm1_weight", fa_norm1_weight);
        register_parameter("fa_norm2_weight", fa_norm2_weight);

        // -- Attention projections --
        // Weight shape: [input_features, output_features] (matches Linear::matmul convention)
        float qk_limit = 1.0f / std::sqrt(static_cast<float>(H));
        float o_limit  = 1.0f / std::sqrt(static_cast<float>(n_heads * D));

        // q_proj outputs query AND the output gate (Qwen3.5+ attn_output_gate): [H, 2*nH*D].
        fa_q_proj_weight.weight = attn_init({(size_t)H, (size_t)(2 * n_heads * D)}, qk_limit);
        fa_k_proj_weight.weight = attn_init({(size_t)H, (size_t)n_kv * D}, qk_limit);
        fa_v_proj_weight.weight = attn_init({(size_t)H, (size_t)n_kv * D}, qk_limit);
        fa_o_proj_weight.weight = attn_init({(size_t)n_heads * D, (size_t)H}, o_limit);
        register_parameter("fa_q_proj_weight", fa_q_proj_weight.weight);
        register_parameter("fa_k_proj_weight", fa_k_proj_weight.weight);
        register_parameter("fa_v_proj_weight", fa_v_proj_weight.weight);
        register_parameter("fa_o_proj_weight", fa_o_proj_weight.weight);

        // -- Per-head k/q RMSNorm (head_dim each) --
        fa_k_norm_weight = ones_init({(size_t)D});
        fa_q_norm_weight = ones_init({(size_t)D});
        register_parameter("fa_k_norm_weight", fa_k_norm_weight);
        register_parameter("fa_q_norm_weight", fa_q_norm_weight);

        // -- FFN --
        float ffn_limit  = 1.0f / std::sqrt(static_cast<float>(H));
        float down_limit = 1.0f / std::sqrt(static_cast<float>(I));

        fa_gate_proj_weight.weight = attn_init({(size_t)H, (size_t)I}, ffn_limit);
        fa_up_proj_weight.weight    = attn_init({(size_t)H, (size_t)I}, ffn_limit);
        fa_down_proj_weight.weight = attn_init({(size_t)I, (size_t)H}, down_limit);
        register_parameter("fa_gate_proj_weight", fa_gate_proj_weight.weight);
        register_parameter("fa_up_proj_weight", fa_up_proj_weight.weight);
        register_parameter("fa_down_proj_weight", fa_down_proj_weight.weight);
    }

    // ==================== Linear-attention init ====================

    void init_linear_attention(backend::DeviceType device_type)
    {
        const auto& am = _config;
        int32_t H = am.hidden_size;

        // Derived dimensions
        int32_t n_kv_heads  = am.linear_num_key_heads;
        int32_t key_head    = am.linear_key_head_dim;
        int32_t n_v_heads   = am.linear_num_value_heads;
        int32_t val_head    = am.linear_value_head_dim;
        int32_t conv_k      = am.linear_conv_kernel_dim;

        // GatedDeltaNet projection dims (Qwen3-Next / Qwen3.5):
        //   in_proj_qkv -> [q | k | v], q,k have n_kv_heads of key_head; v has n_v_heads of val_head.
        int32_t key_dim     = n_kv_heads * key_head;   // q and k each
        int32_t val_dim     = n_v_heads * val_head;    // v and z each
        int32_t qkv_out_dim = 2 * key_dim + val_dim;   // q + k + v   (conv is applied over this)

        // a, b are per-value-head scalars (decay-gate input / beta).
        int32_t ab_dim = n_v_heads;

        // z is the output gate, one value-head vector each: n_v_heads * val_head.
        int32_t z_dim = val_dim;

        auto attn_init = [&](std::vector<size_t> shape, float limit) {
            return _lazy ? ir::parameter(shape, device_type, backend::DType::FLOAT32, false)
                         : ir::parameterize(ir::uniform(shape, -limit, limit, device_type));
        };
        auto ones_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, backend::DType::FLOAT32, false)
                         : ir::parameterize(ir::ones(shape, device_type));
        };
        auto zeros_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, backend::DType::FLOAT32, false)
                         : ir::parameterize(ir::zeros(shape, device_type));
        };

        // -- Norms --
        la_norm1_weight = ones_init({(size_t)H});
        la_norm2_weight = ones_init({(size_t)H});
        register_parameter("la_norm1_weight", la_norm1_weight);
        register_parameter("la_norm2_weight", la_norm2_weight);

        // -- Conv1d weight: [qkv_out_dim, conv_k, 1] --
        la_conv1d_weight = ones_init({(size_t)qkv_out_dim, (size_t)conv_k, 1});
        register_parameter("la_conv1d_weight", la_conv1d_weight);

        float lim = 1.0f / std::sqrt(static_cast<float>(H));

        // -- In-projections -- [input_features, output_features]
        la_in_proj_qkv_weight.weight = attn_init({(size_t)H, (size_t)qkv_out_dim}, lim);
        la_in_proj_a_weight.weight    = attn_init({(size_t)H, (size_t)ab_dim}, lim);
        la_in_proj_b_weight.weight    = attn_init({(size_t)H, (size_t)ab_dim}, lim);
        la_in_proj_z_weight.weight    = attn_init({(size_t)H, (size_t)z_dim}, lim);
        register_parameter("la_in_proj_qkv_weight", la_in_proj_qkv_weight.weight);
        register_parameter("la_in_proj_a_weight", la_in_proj_a_weight.weight);
        register_parameter("la_in_proj_b_weight", la_in_proj_b_weight.weight);
        register_parameter("la_in_proj_z_weight", la_in_proj_z_weight.weight);

        // -- Gated RMSNorm over the value head dim: [val_head_dim] --
        la_norm_weight = ones_init({(size_t)val_head});
        register_parameter("la_norm_weight", la_norm_weight);

        // -- A_log / dt_bias: [n_v_heads] (decay gate parameters) --
        la_A_log   = zeros_init({(size_t)n_v_heads});
        la_dt_bias = zeros_init({(size_t)n_v_heads});
        register_parameter("la_A_log", la_A_log);
        register_parameter("la_dt_bias", la_dt_bias);

        // -- Output projection: [input_features, output_features] --
        int32_t out_dim = val_dim;
        float out_limit = 1.0f / std::sqrt(static_cast<float>(H));
        la_out_proj_weight.weight = attn_init({(size_t)out_dim, (size_t)H}, out_limit);
        register_parameter("la_out_proj_weight", la_out_proj_weight.weight);

        // -- MLP (dense SwiGLU), same as full-attention layers --
        int32_t I = am.intermediate_size;
        float ffn_limit  = 1.0f / std::sqrt(static_cast<float>(H));
        float down_limit = 1.0f / std::sqrt(static_cast<float>(I));
        la_gate_proj_weight.weight = attn_init({(size_t)H, (size_t)I}, ffn_limit);
        la_up_proj_weight.weight    = attn_init({(size_t)H, (size_t)I}, ffn_limit);
        la_down_proj_weight.weight = attn_init({(size_t)I, (size_t)H}, down_limit);
        register_parameter("la_gate_proj_weight", la_gate_proj_weight.weight);
        register_parameter("la_up_proj_weight", la_up_proj_weight.weight);
        register_parameter("la_down_proj_weight", la_down_proj_weight.weight);
    }

    // ==================== Full-attention forward ====================

    utils::Ref<ir::Tensor> forward_full_attention(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& positions,
        const utils::Ref<ir::Tensor>& inv_freq,
        const utils::Ref<ir::Tensor>& mask,
        const utils::Ref<ir::Tensor>& k_cache = nullptr,
        const utils::Ref<ir::Tensor>& v_cache = nullptr,
        size_t start_pos = 0)
    {
        const auto& am = _config;
        int32_t H   = am.hidden_size;
        int32_t D   = am.head_dim;
        int32_t nH  = am.num_attention_heads;
        int32_t nKV = am.num_key_value_heads;
        int32_t I   = am.intermediate_size;
        int32_t n_rep = am.get_num_kv_head_repeats();

        auto shape = x->shape();
        size_t B = shape[0], S = shape[1];

        const float eps = static_cast<float>(am.rms_norm_eps);

        // === Token mixer: self-attention (sequential residual) ===
        auto residual = x;
        auto normed = ir::reshape(nn::functional::rms_norm(x, fa_norm1_weight, eps), {B * S, (size_t)H});

        // q_proj produces [query | gate] per head: [B,S,nH,2D] -> split into query, gate.
        auto qg = fa_q_proj_weight.forward(normed);   // [B*S, 2*nH*D]
        qg = ir::reshape(qg, {B, S, (size_t)nH, (size_t)(2 * D)});
        auto q    = ir::slice(qg, {0, 0, 0, 0},          {B, S, (size_t)nH, (size_t)D});
        auto gate = ir::slice(qg, {0, 0, 0, (size_t)D},  {B, S, (size_t)nH, (size_t)(2 * D)});

        auto k = ir::reshape(fa_k_proj_weight.forward(normed), {B, S, (size_t)nKV, (size_t)D});
        auto v = ir::reshape(fa_v_proj_weight.forward(normed), {B, S, (size_t)nKV, (size_t)D});

        // Per-head RMSNorm on Q and K (Qwen3.5+)
        q = nn::functional::rms_norm(q, fa_q_norm_weight, eps);
        k = nn::functional::rms_norm(k, fa_k_norm_weight, eps);

        // Multi-modal RoPE (partial rotary + interleaved)
        q = nn::functional::apply_mrope(q, positions, inv_freq,
                                        static_cast<float>(am.partial_rotary_factor), am.mrope_interleaved);
        k = nn::functional::apply_mrope(k, positions, inv_freq,
                                        static_cast<float>(am.partial_rotary_factor), am.mrope_interleaved);

        // KV cache: write this step's (already-RoPE'd) K/V into the preallocated cache in place
        // at [.., start_pos:start_pos+S, ..], then read back the [0:start_pos+S] prefix. This is
        // O(S) per step (no growing concat copy). With no cache (plain forward) use K/V directly.
        auto k_full = k, v_full = v;
        if (k_cache) {
            k_full = ir::cache_update(k_cache, k, /*axis=*/1, start_pos);   // [1, start_pos+S, nKV, D]
            v_full = ir::cache_update(v_cache, v, /*axis=*/1, start_pos);
        }

        auto k_rep = k_full, v_rep = v_full;
        if (n_rep > 1) {
            k_rep = nn::functional::repeat_kv(k_full, (size_t)n_rep);
            v_rep = nn::functional::repeat_kv(v_full, (size_t)n_rep);
        }

        auto attn_out = nn::functional::scaled_dot_product_attention(q, k_rep, v_rep, mask);  // [B,S,nH,D]

        // Output gate (Qwen3.5+): attn_out * sigmoid(gate), elementwise per head/dim.
        auto gate_sig = ir::div(1.0f, ir::add(ir::exp(ir::neg(gate)), 1.0f));
        attn_out = attn_out * gate_sig;

        attn_out = ir::reshape(attn_out, {B * S, (size_t)(nH * D)});
        attn_out = fa_o_proj_weight.forward(attn_out);          // [B*S, H]
        auto h = residual + ir::reshape(attn_out, {B, S, (size_t)H});

        // === MLP (sequential residual): h + mlp(post_attention_layernorm(h)) ===
        auto m = ir::reshape(nn::functional::rms_norm(h, fa_norm2_weight, eps), {B * S, (size_t)H});
        m = fa_down_proj_weight.forward(nn::functional::silu(fa_gate_proj_weight.forward(m)) * fa_up_proj_weight.forward(m));
        return h + ir::reshape(m, {B, S, (size_t)H});
    }

    // ==================== Linear-attention forward (Gated DeltaNet) ====================
    //
    // Qwen3-Next / Qwen3.5 gated delta-rule linear attention. Implemented as the exact
    // sequential recurrence, applied over the whole sequence -- so the prefill (S>1) result
    // is identical to stepping one token at a time (S==1). That equivalence is what makes a
    // recurrent KV-cache mathematically exact.
    //
    //   qkv  = silu(causal_conv1d(in_proj_qkv(x)));  split -> q,k (n_kv heads), v (n_v heads)
    //   g    = -exp(A_log) * softplus(a + dt_bias)         (per value head)
    //   beta = sigmoid(b)                                  (per value head)
    //   S_t  = exp(g_t) * S_{t-1};   kv = S_t^T k_t;   S_t += k_t ((v_t - kv) * beta_t)^T
    //   o_t  = S_t^T q_t
    //   out  = out_proj( rmsnorm( o * silu(z) ) )
    //
    // positions / inv_freq are unused (the linear path has no RoPE). state_in is the recurrent
    // state [B*n_v, key_head, val_head] carried across cached steps (null -> zero-initialised).
    utils::Ref<ir::Tensor> forward_linear_attention(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& /*positions*/,
        const utils::Ref<ir::Tensor>& /*inv_freq*/,
        const utils::Ref<ir::Tensor>& state_in,
        const utils::Ref<ir::Tensor>& conv_state_in = nullptr)
    {
        const auto& am = _config;
        const size_t H        = (size_t)am.hidden_size;
        const size_t n_kv     = (size_t)am.linear_num_key_heads;
        const size_t key_head = (size_t)am.linear_key_head_dim;
        const size_t n_v      = (size_t)am.linear_num_value_heads;
        const size_t val_head = (size_t)am.linear_value_head_dim;
        const size_t conv_k   = (size_t)am.linear_conv_kernel_dim;
        const float  eps      = (float)am.rms_norm_eps;
        const size_t key_dim  = n_kv * key_head;     // q and k each
        const size_t val_dim  = n_v * val_head;      // v and z each
        const size_t qkv_dim  = 2 * key_dim + val_dim;
        const size_t expand   = n_v / n_kv;          // GQA-style head replication for q,k

        const auto shape = x->shape();
        const size_t B = shape[0], S = shape[1];
        const size_t BH = B * n_v;
        auto dev = x->device_type();

        auto residual = x;
        auto normed = nn::functional::rms_norm(x, la_norm1_weight, eps);
        auto nf = ir::reshape(normed, {B * S, H});

        // -- projections --
        auto qkv = la_in_proj_qkv_weight.forward(nf);   // [B*S, qkv_dim]
        auto a   = la_in_proj_a_weight.forward(nf);     // [B*S, n_v]
        auto b   = la_in_proj_b_weight.forward(nf);     // [B*S, n_v]
        auto z   = la_in_proj_z_weight.forward(nf);     // [B*S, val_dim]

        // -- causal depth-wise conv1d over [q|k|v], then silu --
        // conv weight: [qkv_dim, conv_k, 1].  out[t] = sum_j w[:,j] * in[t-(conv_k-1)+j]
        auto qkv_bsc = ir::reshape(qkv, {B, S, qkv_dim});
        // left-pad (conv_k-1) frames: the cached frames from the previous step, or zeros at t=0.
        auto pad = conv_state_in ? conv_state_in : ir::zeros({B, conv_k - 1, qkv_dim}, dev);
        auto padded = ir::concat(pad, qkv_bsc, 1);          // [B, S+conv_k-1, qkv_dim]
        utils::Ref<ir::Tensor> conv = nullptr;
        for (size_t j = 0; j < conv_k; ++j) {
            auto wj = ir::reshape(ir::slice(la_conv1d_weight, {0, j, 0}, {qkv_dim, j + 1, 1}), {qkv_dim});
            auto xj = ir::slice(padded, {0, j, 0}, {B, j + S, qkv_dim});   // [B, S, qkv_dim]
            auto term = xj * wj;
            conv = conv ? (conv + term) : term;
        }
        conv = nn::functional::silu(conv);                  // [B, S, qkv_dim]
        // remember the last (conv_k-1) input frames (from the padded history, so this is
        // always exactly conv_k-1 frames even when decoding a single token).
        _last_conv_state = ir::slice(padded, {0, S, 0}, {B, S + conv_k - 1, qkv_dim});

        // -- split q,k,v; replicate q,k from n_kv to n_v heads (GQA repeat_interleave) --
        auto q = ir::slice(conv, {0, 0, 0},           {B, S, key_dim});
        auto k = ir::slice(conv, {0, 0, key_dim},     {B, S, 2 * key_dim});
        auto v = ir::slice(conv, {0, 0, 2 * key_dim}, {B, S, qkv_dim});

        auto rep = [&](const utils::Ref<ir::Tensor>& t) {   // [B,S,n_kv,key_head] -> [B,S,n_v,key_head]
            auto t5 = ir::reshape(t, {B, S, n_kv, 1, key_head});
            t5 = ir::broadcast(t5, {B, S, n_kv, expand, key_head});
            return ir::reshape(t5, {B, S, n_v, key_head});
        };
        q = rep(ir::reshape(q, {B, S, n_kv, key_head}));    // [B,S,n_v,key_head]
        k = rep(ir::reshape(k, {B, S, n_kv, key_head}));
        v = ir::reshape(v, {B, S, n_v, val_head});

        // Normalize q,k over head_dim with RMS norm (no weight) and asymmetric scaling, matching
        // Qwen3.5: q = (1/d)*rmsnorm(q), k = (1/sqrt(d))*rmsnorm(k). v is not normalized.
        auto rmsn = [&](const utils::Ref<ir::Tensor>& t) {
            auto ms = ir::sum(t * t, {3}, true) / static_cast<float>(key_head);  // mean over head_dim
            return t * ir::pow(ms + 1e-6f, -0.5f);
        };
        const float inv = 1.0f / std::sqrt(static_cast<float>(key_head));
        q = ir::mul(rmsn(q), inv * inv);
        k = ir::mul(rmsn(k), inv);

        // -- gates (per value head): g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b) --
        auto a3 = ir::reshape(a, {B, S, n_v});
        auto b3 = ir::reshape(b, {B, S, n_v});
        auto sp = nn::functional::softplus(a3 + la_dt_bias);                  // [B,S,n_v]
        auto A  = ir::exp(la_A_log);                                          // [n_v]
        auto decay_all = ir::exp(ir::neg(A * sp));                            // exp(g) in (0,1)
        auto beta_all  = ir::div(1.0f, ir::add(ir::exp(ir::neg(b3)), 1.0f));  // sigmoid(b)

        // -- sequential gated delta rule;  state: [BH, key_head, val_head] --
        utils::Ref<ir::Tensor> state = state_in ? state_in
                                                : ir::zeros({BH, key_head, val_head}, dev);
        std::vector<utils::Ref<ir::Tensor>> outs;
        outs.reserve(S);
        for (size_t t = 0; t < S; ++t) {
            auto qt = ir::reshape(ir::slice(q, {0, t, 0, 0}, {B, t + 1, n_v, key_head}), {BH, key_head});
            auto kt = ir::reshape(ir::slice(k, {0, t, 0, 0}, {B, t + 1, n_v, key_head}), {BH, key_head});
            auto vt = ir::reshape(ir::slice(v, {0, t, 0, 0}, {B, t + 1, n_v, val_head}), {BH, val_head});
            auto dt = ir::reshape(ir::slice(decay_all, {0, t, 0}, {B, t + 1, n_v}), {BH, 1, 1});
            auto bt = ir::reshape(ir::slice(beta_all,  {0, t, 0}, {B, t + 1, n_v}), {BH, 1});

            auto kt_c = ir::reshape_view(kt, {BH, key_head, 1});   // [BH,key,1]
            state = state * dt;                                    // decay         [BH,key,val]
            auto kv_mem = ir::sum(state * kt_c, {1});              // S^T k_t       [BH,val]
            auto delta  = (vt - kv_mem) * bt;                      // (v-kv)*beta   [BH,val]
            auto delta_r = ir::reshape_view(delta, {BH, 1, val_head});
            state = state + kt_c * delta_r;                        // += outer      [BH,key,val]
            auto qt_c = ir::reshape_view(qt, {BH, key_head, 1});
            auto ot = ir::sum(state * qt_c, {1});                  // S^T q_t       [BH,val]
            outs.push_back(ir::reshape(ot, {B, 1, n_v, val_head}));
        }
        _last_linear_state = state;

        utils::Ref<ir::Tensor> o = outs[0];
        for (size_t t = 1; t < S; ++t) o = ir::concat(o, outs[t], 1);  // [B,S,n_v,val_head]

        // -- gated RMSNorm (Qwen3-Next: norm FIRST, then gate by silu(z)), then out_proj --
        o = nn::functional::rms_norm(o, la_norm_weight, eps);       // rmsnorm(core)*weight, over val_head
        auto zg = nn::functional::silu(ir::reshape(z, {B, S, n_v, val_head}));
        o = o * zg;                                                 // * silu(z)  (gate after norm)
        o = ir::reshape(o, {B * S, val_dim});
        o = la_out_proj_weight.forward(o);                      // [B*S, H]
        auto h = residual + ir::reshape(o, {B, S, H});              // sequential residual

        // === MLP (sequential residual): h + mlp(post_attention_layernorm(h)) ===
        auto m = ir::reshape(nn::functional::rms_norm(h, la_norm2_weight, eps), {B * S, H});
        m = la_down_proj_weight.forward(nn::functional::silu(la_gate_proj_weight.forward(m)) * la_up_proj_weight.forward(m));
        return h + ir::reshape(m, {B, S, H});
    }

    // Cached state from the last linear-attention forward pass, for chaining
    // autoregressive steps: the recurrent delta-rule state and the last (conv_k-1)
    // conv input frames. Accessible after forward().
    utils::Ref<ir::Tensor> _last_linear_state;
    utils::Ref<ir::Tensor> _last_conv_state;
};

} // namespace qwen
} // namespace llm
} // namespace nn
} // namespace cppgrad
