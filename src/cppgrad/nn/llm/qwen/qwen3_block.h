// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cppgrad/nn/module.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/gated_ffn.h"

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
    std::shared_ptr<Linear> fa_q_proj;
    std::shared_ptr<Linear> fa_k_proj;
    std::shared_ptr<Linear> fa_v_proj;
    std::shared_ptr<Linear> fa_o_proj;

    // Per-head k / q RMSNorm (Qwen3.5+)
    utils::Ref<ir::Tensor> fa_k_norm_weight;
    utils::Ref<ir::Tensor> fa_q_norm_weight;

    // FFN (SwiGLU: gate + up + down)
    std::shared_ptr<GatedFFN> fa_ffn;

    // ========== Linear-attention weights ==========

    // Norms
    utils::Ref<ir::Tensor> la_norm1_weight;
    utils::Ref<ir::Tensor> la_norm2_weight;

    // Conv1d weight: [qkv_output_dim, linear_conv_kernel_dim, 1]
    utils::Ref<ir::Tensor> la_conv1d_weight;

    // In-projections  (all [output_dim, hidden])
    std::shared_ptr<Linear> la_in_proj_qkv;  // QKV combined
    std::shared_ptr<Linear> la_in_proj_a;    // A for state update
    std::shared_ptr<Linear> la_in_proj_b;    // B for state update
    std::shared_ptr<Linear> la_in_proj_z;    // Gate projection

    // State RMSNorm
    utils::Ref<ir::Tensor> la_norm_weight;

    // Log decay rates: [linear_num_value_heads]
    utils::Ref<ir::Tensor> la_A_log;

    // Softplus bias for the decay gate: [linear_num_value_heads]
    utils::Ref<ir::Tensor> la_dt_bias;

    // Output projection: [hidden, linear_output_dim]
    std::shared_ptr<Linear> la_out_proj;

    // MLP (SwiGLU) -- every decoder layer has one, including linear-attention layers.
    std::shared_ptr<GatedFFN> la_ffn;

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

    // Concat-mode K/V cache (reference path for the in-place repro / correctness comparison).
    utils::Ref<ir::Tensor> forward_full_cached_concat(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& positions,
        const utils::Ref<ir::Tensor>& inv_freq,
        const utils::Ref<ir::Tensor>& mask,
        const utils::Ref<ir::Tensor>& past_k,
        const utils::Ref<ir::Tensor>& past_v,
        utils::Ref<ir::Tensor>& k_out,
        utils::Ref<ir::Tensor>& v_out)
    {
        auto o = forward_full_attention(x, positions, inv_freq, mask, nullptr, nullptr, 0, past_k, past_v);
        k_out = _last_k;
        v_out = _last_v;
        return o;
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
        auto ones_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, common::DType::FLOAT32, false)
                         : ir::parameterize(ir::ones(shape, device_type));
        };
        // Bias-free projections; dense (random) or deferred (lazy) per _lazy. A quantized checkpoint
        // load later fills qweight/scales/biases on each. Weight shape: [in_features, out_features].
        auto proj = [&](size_t in, size_t out) {
            return std::make_shared<Linear>(in, out, /*use_bias=*/false, Init::Default, device_type, _lazy);
        };

        // -- Norms --
        fa_norm1_weight = ones_init({(size_t)H});
        fa_norm2_weight = ones_init({(size_t)H});
        register_parameter("fa_norm1_weight", fa_norm1_weight);
        register_parameter("fa_norm2_weight", fa_norm2_weight);

        // -- Attention projections --
        // q_proj outputs query AND the output gate (Qwen3.5+ attn_output_gate): [H, 2*nH*D].
        fa_q_proj = proj((size_t)H, (size_t)(2 * n_heads * D));
        fa_k_proj = proj((size_t)H, (size_t)n_kv * D);
        fa_v_proj = proj((size_t)H, (size_t)n_kv * D);
        fa_o_proj = proj((size_t)n_heads * D, (size_t)H);
        register_module("fa_q_proj", fa_q_proj);
        register_module("fa_k_proj", fa_k_proj);
        register_module("fa_v_proj", fa_v_proj);
        register_module("fa_o_proj", fa_o_proj);

        // -- Per-head k/q RMSNorm (head_dim each) --
        fa_k_norm_weight = ones_init({(size_t)D});
        fa_q_norm_weight = ones_init({(size_t)D});
        register_parameter("fa_k_norm_weight", fa_k_norm_weight);
        register_parameter("fa_q_norm_weight", fa_q_norm_weight);

        // -- FFN (SwiGLU) --
        fa_ffn = std::make_shared<GatedFFN>((size_t)H, (size_t)I, (size_t)H,
                                            GatedFFN::InnerAct::SILU, Init::Default, device_type, _lazy);
        register_module("fa_ffn", fa_ffn);
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

        auto ones_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, common::DType::FLOAT32, false)
                         : ir::parameterize(ir::ones(shape, device_type));
        };
        auto zeros_init = [&](std::vector<size_t> shape) {
            return _lazy ? ir::parameter(shape, device_type, common::DType::FLOAT32, false)
                         : ir::parameterize(ir::zeros(shape, device_type));
        };
        // Bias-free projections; dense (random) or deferred (lazy). [in_features, out_features].
        auto proj = [&](size_t in, size_t out) {
            return std::make_shared<Linear>(in, out, /*use_bias=*/false, Init::Default, device_type, _lazy);
        };

        // -- Norms --
        la_norm1_weight = ones_init({(size_t)H});
        la_norm2_weight = ones_init({(size_t)H});
        register_parameter("la_norm1_weight", la_norm1_weight);
        register_parameter("la_norm2_weight", la_norm2_weight);

        // -- Conv1d weight: [qkv_out_dim, conv_k, 1] --
        la_conv1d_weight = ones_init({(size_t)qkv_out_dim, (size_t)conv_k, 1});
        register_parameter("la_conv1d_weight", la_conv1d_weight);

        // -- In-projections --
        la_in_proj_qkv = proj((size_t)H, (size_t)qkv_out_dim);
        la_in_proj_a   = proj((size_t)H, (size_t)ab_dim);
        la_in_proj_b   = proj((size_t)H, (size_t)ab_dim);
        la_in_proj_z   = proj((size_t)H, (size_t)z_dim);
        register_module("la_in_proj_qkv", la_in_proj_qkv);
        register_module("la_in_proj_a", la_in_proj_a);
        register_module("la_in_proj_b", la_in_proj_b);
        register_module("la_in_proj_z", la_in_proj_z);

        // -- Gated RMSNorm over the value head dim: [val_head_dim] --
        la_norm_weight = ones_init({(size_t)val_head});
        register_parameter("la_norm_weight", la_norm_weight);

        // -- A_log / dt_bias: [n_v_heads] (decay gate parameters) --
        la_A_log   = zeros_init({(size_t)n_v_heads});
        la_dt_bias = zeros_init({(size_t)n_v_heads});
        register_parameter("la_A_log", la_A_log);
        register_parameter("la_dt_bias", la_dt_bias);

        // -- Output projection --
        la_out_proj = proj((size_t)val_dim, (size_t)H);
        register_module("la_out_proj", la_out_proj);

        // -- MLP (SwiGLU), same as full-attention layers --
        int32_t I = am.intermediate_size;
        la_ffn = std::make_shared<GatedFFN>((size_t)H, (size_t)I, (size_t)H,
                                            GatedFFN::InnerAct::SILU, Init::Default, device_type, _lazy);
        register_module("la_ffn", la_ffn);
    }

    // ==================== Full-attention forward ====================

    utils::Ref<ir::Tensor> forward_full_attention(
        const utils::Ref<ir::Tensor>& x,
        const utils::Ref<ir::Tensor>& positions,
        const utils::Ref<ir::Tensor>& inv_freq,
        const utils::Ref<ir::Tensor>& mask,
        const utils::Ref<ir::Tensor>& k_cache = nullptr,   // in-place mode: preallocated cache leaf
        const utils::Ref<ir::Tensor>& v_cache = nullptr,
        size_t start_pos = 0,
        const utils::Ref<ir::Tensor>& past_k = nullptr,    // concat mode: previous K/V to prepend
        const utils::Ref<ir::Tensor>& past_v = nullptr)
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
        auto qg = fa_q_proj->forward(normed);   // [B*S, 2*nH*D]
        qg = ir::reshape(qg, {B, S, (size_t)nH, (size_t)(2 * D)});
        auto q    = ir::slice(qg, {0, 0, 0, 0},          {B, S, (size_t)nH, (size_t)D});
        auto gate = ir::slice(qg, {0, 0, 0, (size_t)D},  {B, S, (size_t)nH, (size_t)(2 * D)});

        auto k = ir::reshape(fa_k_proj->forward(normed), {B, S, (size_t)nKV, (size_t)D});
        auto v = ir::reshape(fa_v_proj->forward(normed), {B, S, (size_t)nKV, (size_t)D});

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
        } else if (past_k) {                          // concat mode (reference / non-in-place)
            k_full = ir::concat(past_k, k, 1);
            v_full = ir::concat(past_v, v, 1);
        }
        _last_k = k_full;   // for concat-mode caching (forward_full_cached_concat)
        _last_v = v_full;

        // Attention over the (grouped-query) cache. Two equivalent paths:
        //  - gqa_attention (default): broadcasts the nKV cache heads across the n_rep query group as
        //    stride-0 views (no [B,KV,nH,D] repeat) but still materializes the [B,nH,S,KV] scores.
        //  - flash_attention (CPPGRAD_FLASH_ATTN): online-softmax streaming over keys, so the score
        //    matrix is NEVER materialized -> O(1) attention memory (no per-chunk score-transient cap
        //    needed) and removes the prefix-sized transient for both prefill and decode.
        // Both bit-equivalent to repeat_kv + SDPA (tests/test_{gqa,flash}_attention.cpp). All model
        // masks are causal-at-offset, so flash uses causal-by-position with q_offset = KV - S.
        static const bool FLASH = std::getenv("CPPGRAD_FLASH_ATTN") != nullptr;
        utils::Ref<ir::Tensor> attn_out;
        if (FLASH) {
            size_t KVlen = k_full->shape()[1], Slen = q->shape()[1];
            attn_out = nn::functional::flash_attention(q, k_full, v_full, (size_t)n_rep,
                                                       /*causal=*/true, KVlen - Slen);  // [B,S,nH,D]
        } else {
            attn_out = nn::functional::gqa_attention(q, k_full, v_full, mask, (size_t)n_rep);  // [B,S,nH,D]
        }

        // Output gate (Qwen3.5+): attn_out * sigmoid(gate), elementwise per head/dim.
        auto gate_sig = ir::div(1.0f, ir::add(ir::exp(ir::neg(gate)), 1.0f));
        attn_out = attn_out * gate_sig;

        attn_out = ir::reshape(attn_out, {B * S, (size_t)(nH * D)});
        attn_out = fa_o_proj->forward(attn_out);          // [B*S, H]
        auto h = residual + ir::reshape(attn_out, {B, S, (size_t)H});

        // === MLP (sequential residual): h + mlp(post_attention_layernorm(h)) ===
        auto m = ir::reshape(nn::functional::rms_norm(h, fa_norm2_weight, eps), {B * S, (size_t)H});
        m = fa_ffn->forward(m);
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
        auto qkv = la_in_proj_qkv->forward(nf);   // [B*S, qkv_dim]
        auto a   = la_in_proj_a->forward(nf);     // [B*S, n_v]
        auto b   = la_in_proj_b->forward(nf);     // [B*S, n_v]
        auto z   = la_in_proj_z->forward(nf);     // [B*S, val_dim]

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

        // -- gated delta rule via chunked-parallel scan --
        // Bit-equivalent to the per-token sequential recurrence (tests/test_gated_delta_chunked.cpp),
        // but computed with batched matmuls over sub-chunks: O(S/chunk) kernel launches instead of
        // O(S). This is what makes long-prompt prefill of a linear-attention (hybrid) model tractable
        // -- the per-token scan launched ~hundreds of tiny kernels per token per layer. The helper
        // wants the batched [BH,S,d] layout (BH = B*n_v); reorder [B,S,n_v,d] -> [B,n_v,S,d] -> [BH,S,d].
        auto to_bh = [&](const utils::Ref<ir::Tensor>& t, size_t d) {
            return ir::reshape(ir::permute(t, {0, 2, 1, 3}), {BH, S, d});
        };
        auto q_bh     = to_bh(q, key_head);
        auto k_bh     = to_bh(k, key_head);
        auto v_bh     = to_bh(v, val_head);
        auto decay_bh = ir::reshape(ir::permute(decay_all, {0, 2, 1}), {BH, S});  // [B,S,n_v]->[BH,S]
        auto beta_bh  = ir::reshape(ir::permute(beta_all,  {0, 2, 1}), {BH, S});

        // Sub-chunk length for the scan (override CPPGRAD_DELTA_CHUNK). 64 is the usual sweet spot;
        // smaller is more numerically stable (the de-decay step exp(-cumsum(g)) grows within a chunk),
        // larger is fewer kernels.
        static const size_t SCAN_CHUNK = []{
            const char* e = std::getenv("CPPGRAD_DELTA_CHUNK");
            return e ? (size_t)std::max(1, atoi(e)) : (size_t)32;
        }();
        utils::Ref<ir::Tensor> state_out;
        auto o_bh = nn::functional::gated_delta_scan_chunked(
            q_bh, k_bh, v_bh, decay_bh, beta_bh, state_in, state_out, SCAN_CHUNK);  // [BH,S,val_head]
        _last_linear_state = state_out;

        // [BH,S,val_head] -> [B,n_v,S,val_head] -> [B,S,n_v,val_head]
        auto o = ir::permute(ir::reshape(o_bh, {B, n_v, S, val_head}), {0, 2, 1, 3});

        // -- gated RMSNorm (Qwen3-Next: norm FIRST, then gate by silu(z)), then out_proj --
        o = nn::functional::rms_norm(o, la_norm_weight, eps);       // rmsnorm(core)*weight, over val_head
        auto zg = nn::functional::silu(ir::reshape(z, {B, S, n_v, val_head}));
        o = o * zg;                                                 // * silu(z)  (gate after norm)
        o = ir::reshape(o, {B * S, val_dim});
        o = la_out_proj->forward(o);                      // [B*S, H]
        auto h = residual + ir::reshape(o, {B, S, H});              // sequential residual

        // === MLP (sequential residual): h + mlp(post_attention_layernorm(h)) ===
        auto m = ir::reshape(nn::functional::rms_norm(h, la_norm2_weight, eps), {B * S, H});
        m = la_ffn->forward(m);
        return h + ir::reshape(m, {B, S, H});
    }

    // Cached state from the last linear-attention forward pass, for chaining
    // autoregressive steps: the recurrent delta-rule state and the last (conv_k-1)
    // conv input frames. Accessible after forward().
    utils::Ref<ir::Tensor> _last_linear_state;
    utils::Ref<ir::Tensor> _last_conv_state;
    utils::Ref<ir::Tensor> _last_k;   // concat-mode full-attn K/V cache (for the non-in-place path)
    utils::Ref<ir::Tensor> _last_v;
};

} // namespace qwen
} // namespace llm
} // namespace nn
} // namespace cppgrad
