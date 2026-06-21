#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include "cppgrad/nn/module.h"
#include "cppgrad/nn/embedding.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/utils/profiler.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/backend.h"
#include "cppgrad/io/safetensors.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/nn/llm/qwen/qwen3_block.h"

namespace cppgrad {
namespace nn {
namespace llm {
namespace qwen {

struct KVCache {
    utils::Ref<ir::Tensor> k;
    utils::Ref<ir::Tensor> v;
};

// Full Qwen3 / Qwen3.5 transformer model.
//
// Supports both Qwen3 (all full-attention layers) and Qwen3.5/3.6
// (mixed full-attention + linear-attention layers) through the
// Qwen3Config layer_types routing.
class Qwen3Model : public Module {
public:
    utils::Ref<ir::Tensor> embedding_weight;
    utils::Ref<ir::Tensor> final_norm_weight;
    std::shared_ptr<Linear> lm_head;   // dense or quantized (quantized avoids a strided transpose-view GEMV)

    // KV-cache strategy for generate(): true = in-place writes into a preallocated [1,max_len,nKV,D]
    // cache (O(1) append, default). false = concat reference path (O(n) copy per step). Set false
    // via QWEN_KV_CONCAT to cross-check the two paths.
    bool inplace_kv = true;

    // lazy_weights=true: construct all parameters with deferred (unallocated) storage and no random
    // init, so a large checkpoint can be loaded without first materializing the full fp32 weight set
    // (~108GB for the 27B). Use it whenever weights will be loaded via load_from_safetensors.
    Qwen3Model(const Qwen3Config& config,
               backend::DeviceType device_type = backend::DeviceManager::default_device_type(),
               bool lazy_weights = false)
    : _config(config), _device_type(device_type) {

        int32_t H = config.hidden_size;
        int32_t V = config.vocab_size;
        const auto F32 = backend::DType::FLOAT32;

        // Embedding
        embedding_weight = ir::parameter({(size_t)V, (size_t)H}, device_type, F32, !lazy_weights);
        register_parameter("embedding_weight", embedding_weight);

        // LM head
        lm_head = std::make_shared<Linear>((size_t)H, (size_t)V, /*use_bias=*/false,
                                           Init::Default, device_type, lazy_weights);
        register_module("lm_head", lm_head);

        // Final norm
        final_norm_weight = lazy_weights ? ir::parameter({(size_t)H}, device_type, F32, false)
                                         : ir::parameterize(ir::ones({(size_t)H}, device_type));
        register_parameter("final_norm_weight", final_norm_weight);

        // Transformer blocks (each block gets its LayerType from config)
        for (int32_t i = 0; i < config.num_hidden_layers; ++i) {
            auto layer_type = config.get_layer_type(i);
            auto block = std::make_shared<Qwen3Block>(layer_type, config, device_type, lazy_weights);
            _blocks.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        // Precompute inverse frequencies for RoPE / M-RoPE
        _inv_freq = precompute_inv_freq(config);
    }

    // Per-layer autoregressive cache. Full-attention layers use {k,v}: preallocated
    // [1, max_len, n_kv, head_dim] leaves written in place by start_pos (see generate()).
    // Linear-attention layers use {state,conv} (fixed-size recurrent + conv state, chained
    // as graph values). Unused slots stay null.
    struct LayerCache {
        utils::Ref<ir::Tensor> k, v, state, conv;
    };

    // Forward pass: input_ids [B, S] -> logits [B, S, vocab_size]. Non-cached (full recompute);
    // builds a causal mask so full-attention layers are correctly causal.
    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input_ids) override {
        size_t S = input_ids->shape()[1];
        std::vector<LayerCache> caches(_blocks.size());  // all-null: no past
        auto pos = create_position_ids(S);
        auto mask = (S > 1) ? make_causal_mask(S) : nullptr;
        return apply_head(run_layers(input_ids, pos, mask, caches));
    }

    // Generate tokens autoregressively (greedy) with a KV / recurrent-state cache: a single
    // prefill over the prompt, then one-token decode steps. Mathematically identical to full
    // recompute (validated in tests/test_qwen3_kv_cache.cpp), but O(n) instead of O(n^2).
    std::vector<int32_t> generate(std::vector<int32_t> input_ids,
                                   int32_t max_new_tokens = 20) {
        ir::NoGradScope no_grad;   // inference: no autograd; required by in-place cache_update
        if (std::getenv("QWEN_KV_CONCAT")) inplace_kv = false;   // opt out to the concat reference path
        std::vector<int32_t> generated;
        generated.reserve(max_new_tokens);
        std::vector<LayerCache> caches(_blocks.size());

        // In-place mode: preallocate the full-attention K/V cache once: a fixed
        // [1, max_len, n_kv, head_dim] leaf per full-attention layer, written in place each step
        // (O(1) append). Concat mode leaves caches null (filled by growing concat on first step).
        if (inplace_kv) {
            const size_t max_len = input_ids.size() + (size_t)max_new_tokens;
            const size_t nKV = (size_t)_config.num_key_value_heads;
            const size_t Dh  = (size_t)_config.head_dim;
            auto* cap_dev = std::getenv("CPPGRAD_METAL_CAPTURE") ? backend::DeviceManager::device(_device_type) : nullptr;
            for (size_t i = 0; i < _blocks.size(); ++i) {
                if (_blocks[i]->get_layer_type() == LayerType::FULL_ATTENTION) {
                    caches[i].k = ir::parameter({1, max_len, nKV, Dh}, _device_type, backend::DType::FLOAT32, true);
                    caches[i].v = ir::parameter({1, max_len, nKV, Dh}, _device_type, backend::DType::FLOAT32, true);
                    caches[i].k->set_requires_grad(false);
                    caches[i].v->set_requires_grad(false);
                    if (cap_dev) {  // label so the cache is searchable in a GPU capture
                        std::string lk = "kvK_L" + std::to_string(i), lv = "kvV_L" + std::to_string(i);
                        cap_dev->backend()->set_buffer_debug_label(*caches[i].k->realized_buffer(), lk.c_str());
                        cap_dev->backend()->set_buffer_debug_label(*caches[i].v->realized_buffer(), lv.c_str());
                    }
                }
            }
        }

        // QWEN_TIMING=1 -> print prefill time and decode tokens/sec (argmax_at forces a sync each step).
        const bool timing = std::getenv("QWEN_TIMING") != nullptr;
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };

        // -- prefill the prompt (causal) --
        size_t S = input_ids.size();
        auto t_pre = clk::now();
        auto in_t = ir::from_vector<int32_t>(input_ids, {1, S}, _device_type);
        auto h = run_layers(in_t, create_position_ids(S), (S > 1) ? make_causal_mask(S) : nullptr, caches);
        int32_t next = argmax_at(apply_head(h), S - 1);
        generated.push_back(next);
        if (timing) std::fprintf(stderr, "[timing] prefill %zu tok: %.1f ms\n", S, ms(t_pre, clk::now()));

        // CPPGRAD_PROFILE=1: drop prefill stats so the report reflects decode only.
        if (utils::Profiler::enabled()) utils::Profiler::instance().reset();

        // -- decode one token at a time (no mask: the new token sees all cached keys) --
        size_t cur_len = S;
        double decode_ms = 0.0;
        for (int32_t t = 1; t < max_new_tokens; ++t) {
            auto t_dec = clk::now();
            auto in1 = ir::from_vector<int32_t>(std::vector<int32_t>{next}, {1, 1}, _device_type);
            dbg_layers = std::getenv("QWEN_DEBUG") && t == 1;   // collect per-layer magnitudes on step 1
            _dbg_red.clear();
            auto h1 = run_layers(in1, create_position_ids_at((int32_t)cur_len, 1), nullptr, caches, cur_len);
            next = argmax_at(apply_head(h1), 0);  // argmax_at reads back, forcing the step to complete
            if (dbg_layers) {  // read the per-layer sum-of-squares AFTER the step (true batched h)
                for (size_t li = 0; li < _dbg_red.size(); ++li)
                    std::fprintf(stderr, "[dbg] step1 layer %zu  sum(h^2)=%g\n", li, _dbg_red[li]->item<float>());
                _dbg_red.clear();
            }
            if (timing) decode_ms += ms(t_dec, clk::now());
            generated.push_back(next);
            ++cur_len;
        }
        if (timing && max_new_tokens > 1) {
            double per = decode_ms / (max_new_tokens - 1);
            std::fprintf(stderr, "[timing] decode: %.1f ms/tok (%.1f tok/s) over %d tok\n",
                         per, 1000.0 / per, max_new_tokens - 1);
        }
        if (utils::Profiler::enabled()) {
            char title[64];
            std::snprintf(title, sizeof(title), "decode profile (%d steps)", max_new_tokens - 1);
            utils::Profiler::instance().report(stderr, title);
        }
        return generated;
    }

    // Load weights from safetensors files. quantize=true keeps MLX-quantized matmul weights packed
    // (8-bit, ~half the bf16 memory) and routes them through ir::quantized_matmul; embeddings,
    // lm_head and norms are still dequantized to fp32/bf16.
    void load_from_safetensors(const std::vector<std::string>& paths, bool quantize = false) {
        std::map<std::string, utils::Ref<ir::Tensor>> all_tensors;
        for (auto& path : paths) {
            // Quantized path loads raw (no dequant) so the packed weight + scales + biases survive.
            auto tensors = io::load_safetensors(path, _device_type, /*dequantize=*/!quantize);
            for (auto& [name, tensor] : tensors) all_tensors[name] = tensor;
        }
        if (quantize) load_weights_quantized(all_tensors);
        else          load_weights(all_tensors);
    }

    // Load weights from a name->tensor map.
    //
    // Handles both Qwen3 and Qwen3.5/3.6 weight naming conventions:
    // - Qwen3:       model.layers.{i}.self_attn.q_proj.weight  etc.
    // - Qwen3.5 text: model.layers.{i}.self_attn.q_proj.weight  etc.
    // - Qwen3.5 MM:   language_model.model.layers.{i}.self_attn.q_proj.weight etc.
    //
    // For Qwen3.5 models, full-attention layers use standard self_attn / mlp
    // keys (plus q_norm, k_norm, attn_output_gate).
    // Linear-attention layers use linear_attn.* keys.
    void load_weights(const std::map<std::string, utils::Ref<ir::Tensor>>& weights) {
        auto set_weight = [&](const std::string& name, utils::Ref<ir::Tensor>& param) {
            auto it = weights.find(name);
            if (it == weights.end()) {
                std::cerr << "[Qwen3Model] WARNING: missing weight '" << name << "'\n";
                return;
            }
            auto w = it->second;
            if (w->shape() != param->shape()) {
                // Safetensors weights are [out_features, in_features] (PyTorch convention).
                // Our model uses [in_features, out_features] (matmul convention).
                // For 2D tensors where dimensions are swapped, transpose.
                const auto& ws = w->shape();
                const auto& ps = param->shape();
                if (ws.size() == 2 && ps.size() == 2 &&
                    ws[0] == ps[1] && ws[1] == ps[0]) {
                    w = ir::transpose(w, 0, 1);
                } else {
                    w = ir::reshape(w, param->shape());
                }
            }
            // Rebind the parameter to the loaded weight rather than ir::assign (in-place copy).
            // This is inference-only loading, and it lets a bf16 loaded weight replace the
            // pre-created fp32 parameter without a dtype-mismatch (assign requires matching
            // dtypes; the model keeps big weights in bf16 to halve resident memory).
            param = w;
        };

        // Detect naming prefix by checking for the embedding weight.
        // Qwen3.5 multimodal checkpoints use language_model.model.* prefix.
        // Pure text models use model.* prefix.
        bool uses_lm_prefix = weights.count("language_model.model.embed_tokens.weight") > 0;
        std::string prefix;
        if (uses_lm_prefix) {
            prefix = "language_model.model.";
        } else {
            prefix = "model.";
        }

        set_weight(prefix + "embed_tokens.weight", embedding_weight);
        set_weight(prefix + "norm.weight", final_norm_weight);
        // lm_head sits one level above `model.` (e.g. language_model.lm_head.weight), not under it.
        std::string head_prefix = uses_lm_prefix ? "language_model." : "";
        set_weight(head_prefix + "lm_head.weight", lm_head->weight);

        // Per-layer weights
        for (int32_t i = 0; i < _config.num_hidden_layers; ++i) {
            std::string layer_prefix = prefix + "layers." + std::to_string(i) + ".";
            auto& block = _blocks[i];
            auto layer_type = _config.get_layer_type(i);

            if (layer_type == LayerType::FULL_ATTENTION) {
                load_full_attention_layer(layer_prefix, block.get(), set_weight);
            } else {
                load_linear_attention_layer(layer_prefix, block.get(), set_weight);
            }
        }
    }

    // Quantized load: keep matmul weights packed (Linear quantized via ir::quantized_matmul);
    // dequantize embeddings (for gather), lm_head, and the per-tensor norms. Expects the raw map
    // (load_from_safetensors(..., quantize=true) loads with dequantize=false).
    void load_weights_quantized(const std::map<std::string, utils::Ref<ir::Tensor>>& W) {
        auto find = [&](const std::string& n) -> utils::Ref<ir::Tensor> {
            auto it = W.find(n); return it == W.end() ? utils::Ref<ir::Tensor>(nullptr) : it->second;
        };
        // Dequantize base.{weight,scales,biases} -> bf16; fall back to .weight if not quantized.
        auto deq = [&](const std::string& base) -> utils::Ref<ir::Tensor> {
            auto w = find(base + ".weight"), s = find(base + ".scales"), b = find(base + ".biases");
            if (!w) { std::cerr << "[Qwen3Model] WARNING: missing " << base << ".weight\n"; return w; }
            if (s && b && w->dtype() == backend::DType::UINT32)
                return io::dequant_mlx_affine(w, s, b, _device_type);
            return w;
        };
        // Bind a Linear from a packed triple (kept 8-bit).
        auto bind_q = [&](Linear& ql, const std::string& base) {
            auto w = find(base + ".weight"), s = find(base + ".scales"), b = find(base + ".biases");
            if (!w || !s || !b) { std::cerr << "[Qwen3Model] WARNING: missing quant triple " << base << "\n"; return; }
            ql.qweight = w; ql.scales = s; ql.biases = b; ql.quantized = true;
            const size_t Kp = w->shape()[1], groups = s->shape()[1];
            ql.params = ir::QuantParams{ir::QuantScheme::MLX_AFFINE_U8, 8, (int)((Kp * 4) / groups), 4};
        };
        auto bind = [&](utils::Ref<ir::Tensor>& dst, const std::string& n) {
            auto t = find(n); if (!t) { std::cerr << "[Qwen3Model] WARNING: missing " << n << "\n"; return; } dst = t;
        };

        const bool lm = W.count("language_model.model.embed_tokens.weight") > 0;
        const std::string p = lm ? "language_model.model." : "model.";
        const std::string hp = lm ? "language_model." : "";

        embedding_weight = deq(p + "embed_tokens");                  // bf16 [V,H] (gather)
        bind(final_norm_weight, p + "norm.weight");
        bind_q(*lm_head, hp + "lm_head");                            // quantized [V,H/4], contiguous

        for (int32_t i = 0; i < _config.num_hidden_layers; ++i) {
            const std::string lp = p + "layers." + std::to_string(i) + ".";
            auto* blk = _blocks[i].get();
            if (_config.get_layer_type(i) == LayerType::FULL_ATTENTION) {
                bind(blk->fa_norm1_weight, lp + "input_layernorm.weight");
                bind(blk->fa_norm2_weight, lp + "post_attention_layernorm.weight");
                bind(blk->fa_q_norm_weight, lp + "self_attn.q_norm.weight");
                bind(blk->fa_k_norm_weight, lp + "self_attn.k_norm.weight");
                bind_q(*blk->fa_q_proj, lp + "self_attn.q_proj");
                bind_q(*blk->fa_k_proj, lp + "self_attn.k_proj");
                bind_q(*blk->fa_v_proj, lp + "self_attn.v_proj");
                bind_q(*blk->fa_o_proj, lp + "self_attn.o_proj");
                bind_q(*blk->fa_ffn->gate_proj, lp + "mlp.gate_proj");
                bind_q(*blk->fa_ffn->up_proj, lp + "mlp.up_proj");
                bind_q(*blk->fa_ffn->down_proj, lp + "mlp.down_proj");
            } else {
                bind(blk->la_norm1_weight, lp + "input_layernorm.weight");
                bind(blk->la_norm2_weight, lp + "post_attention_layernorm.weight");
                bind(blk->la_conv1d_weight, lp + "linear_attn.conv1d.weight");
                bind(blk->la_norm_weight, lp + "linear_attn.norm.weight");
                bind(blk->la_A_log, lp + "linear_attn.A_log");
                bind(blk->la_dt_bias, lp + "linear_attn.dt_bias");
                bind_q(*blk->la_in_proj_qkv, lp + "linear_attn.in_proj_qkv");
                bind_q(*blk->la_in_proj_a, lp + "linear_attn.in_proj_a");
                bind_q(*blk->la_in_proj_b, lp + "linear_attn.in_proj_b");
                bind_q(*blk->la_in_proj_z, lp + "linear_attn.in_proj_z");
                bind_q(*blk->la_out_proj, lp + "linear_attn.out_proj");
                bind_q(*blk->la_ffn->gate_proj, lp + "mlp.gate_proj");
                bind_q(*blk->la_ffn->up_proj, lp + "mlp.up_proj");
                bind_q(*blk->la_ffn->down_proj, lp + "mlp.down_proj");
            }
        }
    }

    const Qwen3Config& get_config() const { return _config; }
    backend::DeviceType get_device_type() const { return _device_type; }
    Qwen3Block* get_block(size_t i) { return _blocks[i].get(); }

private:
    utils::Ref<ir::Tensor> embed(const utils::Ref<ir::Tensor>& input_ids) {
        return ir::gather(embedding_weight, input_ids);
    }

    // Run all decoder blocks for one step, threading the per-layer cache (updated in place).
    // input_ids [1,S], positions [1,S], mask additive [1,1,S,S_kv] or null. Returns h [1,S,H].
    utils::Ref<ir::Tensor> run_layers(const utils::Ref<ir::Tensor>& input_ids,
                                      const utils::Ref<ir::Tensor>& positions,
                                      const utils::Ref<ir::Tensor>& mask,
                                      std::vector<LayerCache>& caches,
                                      size_t start_pos = 0) {
        auto h = embed(input_ids);
        for (size_t i = 0; i < _blocks.size(); ++i) {
            auto& c = caches[i];
            if (_blocks[i]->get_layer_type() == LayerType::FULL_ATTENTION) {
                if (inplace_kv && c.k) {
                    // In-place: preallocated cache leaf written at start_pos.
                    h = _blocks[i]->forward_full_cached(h, positions, _inv_freq, mask, c.k, c.v, start_pos);
                } else {
                    // Concat: prepend past K/V and store the extended cache (also the forward() path).
                    utils::Ref<ir::Tensor> nk, nv;
                    h = _blocks[i]->forward_full_cached_concat(h, positions, _inv_freq, mask, c.k, c.v, nk, nv);
                    c.k = nk; c.v = nv;
                }
            } else {
                utils::Ref<ir::Tensor> ns, nc;
                h = _blocks[i]->forward_linear_cached(h, c.state, c.conv, ns, nc);
                c.state = ns; c.conv = nc;
            }
            // DEBUG (QWEN_DEBUG): per-layer magnitude as a GRAPH node (sum of squares of h). Realized
            // together with the step (not per-layer) so it observes the TRUE batched h without an
            // intervening flush that would hide the bug. Read after the step in generate().
            if (dbg_layers) _dbg_red.push_back(ir::sum(ir::mul(h, h)));
        }
        return h;
    }
    bool dbg_layers = false;
    std::vector<utils::Ref<ir::Tensor>> _dbg_red;

    // Additive causal mask [1,1,S,S]: 0 on/below the diagonal, large negative above.
    utils::Ref<ir::Tensor> make_causal_mask(size_t S) {
        std::vector<float> m(S * S, 0.0f);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = i + 1; j < S; ++j) m[i * S + j] = -1e9f;
        return ir::from_vector<float>(m, {1, 1, S, S}, _device_type);
    }

    // Argmax over the vocab at sequence position `pos` of logits [1, S, V].
    int32_t argmax_at(const utils::Ref<ir::Tensor>& logits, size_t pos) {
        size_t V = (size_t)_config.vocab_size;
        auto row = ir::reshape(ir::slice(logits, {0, pos, 0}, {1, pos + 1, V}), {V});
        return argmax_last(row);
    }

    utils::Ref<ir::Tensor> apply_head(const utils::Ref<ir::Tensor>& h) {
        auto normed = nn::functional::rms_norm(h, final_norm_weight, static_cast<float>(_config.rms_norm_eps));
        size_t B = normed->shape()[0], S = normed->shape()[1];
        auto h_flat = ir::reshape(normed, {B * S, (size_t)_config.hidden_size});
        auto logits = lm_head->forward(h_flat);
        logits = ir::reshape(logits, {B, S, (size_t)_config.vocab_size});
        return logits;
    }

    utils::Ref<ir::Tensor> create_position_ids(size_t seq_len) {
        std::vector<int32_t> positions(seq_len);
        for (size_t i = 0; i < seq_len; ++i) positions[i] = static_cast<int32_t>(i);
        return ir::from_vector<int32_t>(positions, {1, seq_len}, _device_type);
    }

    utils::Ref<ir::Tensor> create_position_ids_at(int32_t start, size_t count) {
        std::vector<int32_t> positions(count);
        for (size_t i = 0; i < count; ++i) positions[i] = start + static_cast<int32_t>(i);
        return ir::from_vector<int32_t>(positions, {1, count}, _device_type);
    }

    // Precompute inverse frequencies for RoPE / M-RoPE.
    //
    // Qwen3:     num_rotary_pairs = head_dim / 2 (full rotary, partial_rotary_factor = 1.0)
    // Qwen3.5:   num_rotary_pairs = head_dim * partial_rotary_factor / 2 (partial rotary)
    utils::Ref<ir::Tensor> precompute_inv_freq(const Qwen3Config& config) {
        int32_t num_rotary_pairs = static_cast<int32_t>(
            std::round(static_cast<double>(config.head_dim) * config.partial_rotary_factor / 2.0));
        std::vector<float> inv_freq(num_rotary_pairs);
        for (int32_t i = 0; i < num_rotary_pairs; ++i) {
            inv_freq[i] = 1.0f / std::pow(config.rope_theta,
                                          static_cast<double>(i) / num_rotary_pairs);
        }
        return ir::from_vector<float>(inv_freq, {(size_t)num_rotary_pairs}, _device_type);
    }

    // Load weights for a full-attention layer.
    //
    // Standard keys (Qwen3 / Qwen3.5 full-attention):
    //   self_attn.q_proj.weight, self_attn.k_proj.weight, self_attn.v_proj.weight,
    //   self_attn.o_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight,
    //   mlp.down_proj.weight, input_layernorm.weight, post_attention_layernorm.weight
    //
    // Additional Qwen3.5 keys (if present):
    //   self_attn.q_norm.weight, self_attn.k_norm.weight,
    //   self_attn.attn_output_gate.weight
    static void load_full_attention_layer(
            const std::string& p,
            Qwen3Block* block,
            const std::function<void(const std::string&, utils::Ref<ir::Tensor>&)>& set)
    {
        set(p + "input_layernorm.weight",                       block->fa_norm1_weight);
        set(p + "post_attention_layernorm.weight",              block->fa_norm2_weight);
        set(p + "self_attn.q_proj.weight",                      block->fa_q_proj->weight);
        set(p + "self_attn.k_proj.weight",                      block->fa_k_proj->weight);
        set(p + "self_attn.v_proj.weight",                      block->fa_v_proj->weight);
        set(p + "self_attn.o_proj.weight",                      block->fa_o_proj->weight);
        set(p + "self_attn.q_norm.weight",                      block->fa_q_norm_weight);
        set(p + "self_attn.k_norm.weight",                      block->fa_k_norm_weight);
        set(p + "mlp.gate_proj.weight",                         block->fa_ffn->gate_proj->weight);
        set(p + "mlp.up_proj.weight",                           block->fa_ffn->up_proj->weight);
        set(p + "mlp.down_proj.weight",                         block->fa_ffn->down_proj->weight);
    }

    // Load weights for a linear-attention layer (Qwen3.5+ only).
    //
    // Keys:
    //   linear_attn.conv1d.weight, linear_attn.in_proj_qkv.weight,
    //   linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight,
    //   linear_attn.in_proj_z.weight, linear_attn.norm.weight,
    //   linear_attn.A_log, linear_attn.out_proj.weight,
    //   input_layernorm.weight, post_attention_layernorm.weight
    static void load_linear_attention_layer(
            const std::string& p,
            Qwen3Block* block,
            const std::function<void(const std::string&, utils::Ref<ir::Tensor>&)>& set)
    {
        set(p + "input_layernorm.weight",                       block->la_norm1_weight);
        set(p + "post_attention_layernorm.weight",              block->la_norm2_weight);
        set(p + "linear_attn.conv1d.weight",                    block->la_conv1d_weight);
        set(p + "linear_attn.in_proj_qkv.weight",               block->la_in_proj_qkv->weight);
        set(p + "linear_attn.in_proj_a.weight",                 block->la_in_proj_a->weight);
        set(p + "linear_attn.in_proj_b.weight",                 block->la_in_proj_b->weight);
        set(p + "linear_attn.in_proj_z.weight",                 block->la_in_proj_z->weight);
        set(p + "linear_attn.norm.weight",                      block->la_norm_weight);
        set(p + "linear_attn.A_log",                            block->la_A_log);
        set(p + "linear_attn.dt_bias",                          block->la_dt_bias);
        set(p + "linear_attn.out_proj.weight",                  block->la_out_proj->weight);
        set(p + "mlp.gate_proj.weight",                         block->la_ffn->gate_proj->weight);
        set(p + "mlp.up_proj.weight",                           block->la_ffn->up_proj->weight);
        set(p + "mlp.down_proj.weight",                         block->la_ffn->down_proj->weight);
    }

    int32_t argmax_last(const utils::Ref<ir::Tensor>& t) {
        // to_vector handles the device->host copy (CPU or Metal); a manual CPU-allocator
        // copy_device_to_device only works for CPU buffers.
        auto data = t->to_vector<float>();
        if (data.empty()) throw std::runtime_error("argmax_last: empty tensor");
        size_t max_idx = 0;
        float max_val = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int32_t>(max_idx);
    }

    Qwen3Config _config;
    backend::DeviceType _device_type;
    std::vector<std::shared_ptr<Qwen3Block>> _blocks;
    utils::Ref<ir::Tensor> _inv_freq;
};

} // namespace qwen
} // namespace llm
} // namespace nn
} // namespace cppgrad
