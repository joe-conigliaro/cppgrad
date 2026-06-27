#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <unordered_set>
#include <optional>
#include <random>
#include <algorithm>
#include <numeric>

#include "cppgrad/nn/module.h"
#include "cppgrad/nn/llm/decode_model.h"
#include "cppgrad/nn/embedding.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/common/bfloat16.h"
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

// Token sampling config + the model-agnostic decode runtime (cache interface, prefix-cache driver)
// live in the shared header; aliased here so existing qwen::SamplingParams references keep working.
using SamplingParams = cppgrad::nn::llm::SamplingParams;

// Sample a token index from a vocab-sized logit vector under `sp`. Applies top_k,
// then softmax(logits/temperature), then top_p nucleus truncation, then draws.
inline int32_t sample_logits(const std::vector<float>& logits, const SamplingParams& sp,
                             std::mt19937_64& rng) {
    const size_t V = logits.size();
    std::vector<int> idx;
    if (sp.top_k > 0 && (size_t)sp.top_k < V) {
        idx.resize(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + sp.top_k, idx.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize(sp.top_k);
    } else {
        idx.resize(V);
        std::iota(idx.begin(), idx.end(), 0);
    }

    float maxl = logits[idx[0]];
    for (int i : idx) maxl = std::max(maxl, logits[i]);
    const float inv_t = 1.0f / sp.temperature;
    std::vector<float> probs(idx.size());
    double sum = 0.0;
    for (size_t k = 0; k < idx.size(); ++k) {
        float p = std::exp((logits[idx[k]] - maxl) * inv_t);
        probs[k] = p;
        sum += p;
    }
    for (auto& p : probs) p = (float)(p / sum);

    // top_p nucleus: keep the smallest set of highest-prob tokens whose mass >= top_p.
    if (sp.top_p < 1.0f) {
        std::vector<size_t> order(idx.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        double cum = 0.0;
        size_t keep = order.size();
        for (size_t r = 0; r < order.size(); ++r) {
            cum += probs[order[r]];
            if (cum >= sp.top_p) { keep = r + 1; break; }
        }
        std::vector<int> nidx(keep);
        std::vector<float> nprobs(keep);
        double nsum = 0.0;
        for (size_t r = 0; r < keep; ++r) { nidx[r] = idx[order[r]]; nprobs[r] = probs[order[r]]; nsum += nprobs[r]; }
        for (auto& p : nprobs) p = (float)(p / nsum);
        idx.swap(nidx);
        probs.swap(nprobs);
    }

    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    double cum = 0.0;
    for (size_t k = 0; k < idx.size(); ++k) {
        cum += probs[k];
        if (r <= cum) return idx[k];
    }
    return idx.back();
}

// Full Qwen3 / Qwen3.5 transformer model.
//
// Supports both Qwen3 (all full-attention layers) and Qwen3.5/3.6
// (mixed full-attention + linear-attention layers) through the
// Qwen3Config layer_types routing.
class Qwen3Model : public Module, public cppgrad::nn::llm::DecodeModel {
public:
    utils::Ref<ir::Tensor> embedding_weight;
    utils::Ref<ir::Tensor> final_norm_weight;
    std::shared_ptr<Linear> lm_head;   // dense or quantized (quantized avoids a strided transpose-view GEMV)

    // MTP (Multi-Token Prediction) self-speculation module, present when the checkpoint ships
    // language_model.mtp.* (DeepSeek/EAGLE style): one transformer layer + an `fc` that combines the
    // previous hidden state with the next token's embedding; shares embed_tokens + lm_head with the
    // main model. Predicts the token two positions ahead, enabling self-speculation (no draft model).
    bool has_mtp_ = false;
    std::shared_ptr<Linear> mtp_fc_;            // [2H -> H], quantized
    std::shared_ptr<Qwen3Block> mtp_block_;     // one full-attention layer
    utils::Ref<ir::Tensor> mtp_pre_fc_norm_emb_, mtp_pre_fc_norm_hidden_, mtp_norm_;

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
        const auto F32 = common::DType::FLOAT32;

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

    // Allocate the decode KV caches for a `prompt_len + max_new_tokens` run.
    // In-place mode preallocates a fixed [1, max_len, n_kv, head_dim] K/V leaf per
    // full-attention layer (written in place each step, O(1) append); concat mode
    // leaves caches null (filled by a growing concat on the first step).
    // KV-cache element dtype, selected by CPPGRAD_KV_DTYPE (bf16|f32). Read once per process.
    static common::DType kv_cache_dtype() {
        static const common::DType dt = [] {
            const char* s = std::getenv("CPPGRAD_KV_DTYPE");
            if (s && (std::string(s) == "bf16" || std::string(s) == "bfloat16"))
                return common::DType::BFLOAT16;
            return common::DType::FLOAT32;
        }();
        return dt;
    }

    std::vector<LayerCache> alloc_kv_caches(size_t prompt_len, int32_t max_new_tokens) {
        std::vector<LayerCache> caches(_blocks.size());
        if (!inplace_kv) return caches;
        const size_t max_len = prompt_len + (size_t)max_new_tokens;
        const size_t nKV = (size_t)_config.num_key_value_heads;
        const size_t Dh  = (size_t)_config.head_dim;
        // KV-cache dtype: bf16 (CPPGRAD_KV_DTYPE=bf16) halves cache memory + read bandwidth at long
        // context and the persisted cache file, with fp32-accumulate attention so no accuracy loss.
        // The fp32 K/V activations are converted on the cache_update write. Default fp32 for now.
        const common::DType kv_dtype = kv_cache_dtype();
        auto* cap_dev = std::getenv("CPPGRAD_METAL_CAPTURE") ? backend::DeviceManager::device(_device_type) : nullptr;
        for (size_t i = 0; i < _blocks.size(); ++i) {
            if (_blocks[i]->get_layer_type() == LayerType::FULL_ATTENTION) {
                caches[i].k = ir::parameter({1, max_len, nKV, Dh}, _device_type, kv_dtype, true);
                caches[i].v = ir::parameter({1, max_len, nKV, Dh}, _device_type, kv_dtype, true);
                caches[i].k->set_requires_grad(false);
                caches[i].v->set_requires_grad(false);
                if (cap_dev) {  // label so the cache is searchable in a GPU capture
                    std::string lk = "kvK_L" + std::to_string(i), lv = "kvV_L" + std::to_string(i);
                    cap_dev->backend()->set_buffer_debug_label(*caches[i].k->realized_buffer(), lk.c_str());
                    cap_dev->backend()->set_buffer_debug_label(*caches[i].v->realized_buffer(), lv.c_str());
                }
            }
        }
        return caches;
    }

    // ===== DecodeModel (model-agnostic decode runtime / prefix cache) implementation =====
    // Concrete cache: one LayerCache per block (KV for full-attn layers, recurrent + conv state for
    // linear-attn layers). The shared runtime (nn/llm/decode_model.h) treats it as an opaque handle.
    struct Cache : public cppgrad::nn::llm::ModelCache {
        std::vector<LayerCache> layers;
    };

    std::unique_ptr<cppgrad::nn::llm::ModelCache> make_cache(size_t capacity_tokens) override {
        auto c = std::make_unique<Cache>();
        c->layers = alloc_kv_caches(capacity_tokens, 0);   // preallocated to `capacity_tokens` positions
        return c;
    }

    utils::Ref<ir::Tensor> prefill(const std::vector<int32_t>& tokens, size_t start_pos,
                                   cppgrad::nn::llm::ModelCache& cache) override {
        auto& c = static_cast<Cache&>(cache);
        return apply_head(prefill_hidden(tokens, c.layers, start_pos));  // [1, suffix_last_chunk, V]
    }

    utils::Ref<ir::Tensor> decode_step(int32_t token, size_t pos,
                                       cppgrad::nn::llm::ModelCache& cache) override {
        auto& c = static_cast<Cache&>(cache);
        auto in1 = ir::from_vector<int32_t>(std::vector<int32_t>{token}, {1, 1}, _device_type);
        auto h1 = run_layers(in1, create_position_ids_at((int32_t)pos, 1), nullptr, c.layers, pos);
        return apply_head(h1);   // [1,1,V]
    }

    int32_t sample_last(const utils::Ref<ir::Tensor>& logits, const SamplingParams& sampling,
                        std::mt19937_64& rng) override {
        return select_at(logits, logits->shape()[1] - 1, sampling, rng);
    }

    int32_t default_eos() const override { return _config.is_qwen3_5() ? 248044 : 151645; }

    // --- Cache persistence (cross-restart prefix caching). Format per tensor: present-flag, rank,
    // dims, then numel fp32. Full-attn layers store the [0:valid_len] K/V prefix; linear layers store
    // the recurrent + conv state. read_cache writes the K/V back into freshly preallocated leaves. ---
    uint64_t cache_tag() const override {
        uint64_t h = 1469598103934665603ULL;                       // FNV-1a over the config
        auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
        mix((uint64_t)_config.hidden_size);      mix((uint64_t)_config.num_hidden_layers);
        mix((uint64_t)_config.num_attention_heads); mix((uint64_t)_config.num_key_value_heads);
        mix((uint64_t)_config.head_dim);         mix((uint64_t)_config.vocab_size);
        mix((uint64_t)_config.intermediate_size);
        for (auto t : _config.layer_types) mix((uint64_t)t);
        return h;
    }

    void write_cache(const cppgrad::nn::llm::ModelCache& cache, size_t valid_len,
                     std::ostream& os) const override {
        auto& c = static_cast<const Cache&>(cache);
        for (size_t i = 0; i < _blocks.size(); ++i) {
            const auto& lc = c.layers[i];
            if (_blocks[i]->get_layer_type() == LayerType::FULL_ATTENTION) {
                write_kv_prefix(os, lc.k, valid_len);
                write_kv_prefix(os, lc.v, valid_len);
            } else {
                write_tensor(os, lc.state);
                write_tensor(os, lc.conv);
            }
        }
    }

    std::unique_ptr<cppgrad::nn::llm::ModelCache> read_cache(std::istream& is, size_t /*valid_len*/,
                                                            size_t capacity) override {
        ir::NoGradScope no_grad;
        auto c = std::make_unique<Cache>();
        c->layers = alloc_kv_caches(capacity, 0);   // preallocated K/V leaves (full-attn); linear null
        for (size_t i = 0; i < _blocks.size(); ++i) {
            auto& lc = c->layers[i];
            if (_blocks[i]->get_layer_type() == LayerType::FULL_ATTENTION) {
                auto k = read_tensor(is), v = read_tensor(is);     // [1, valid_len, nKV, Dh]
                if (k) ir::cache_update(lc.k, k, 1, 0)->eval();     // commit into the leaf at [0:valid_len]
                if (v) ir::cache_update(lc.v, v, 1, 0)->eval();
            } else {
                lc.state = read_tensor(is);
                lc.conv  = read_tensor(is);
            }
        }
        if (auto* dev = backend::DeviceManager::device(_device_type)) dev->backend()->flush_pending();
        return c;
    }

private:
    // Stable on-disk dtype tags (independent of common::DType enum ordering).
    enum : uint8_t { kTagF32 = 0, kTagBF16 = 1 };

    void write_tensor(std::ostream& os, const utils::Ref<ir::Tensor>& t) const {
        uint8_t present = t ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&present), 1);
        if (!t) return;
        const auto& shp = t->shape();
        uint32_t rank = (uint32_t)shp.size();
        os.write(reinterpret_cast<const char*>(&rank), sizeof rank);
        for (auto d : shp) { uint32_t dd = (uint32_t)d; os.write(reinterpret_cast<const char*>(&dd), sizeof dd); }
        // Persist in the tensor's own dtype (a bf16 cache -> a half-size bf16 file), tagged so the
        // reader rebuilds the right element type.
        if (t->dtype() == common::DType::BFLOAT16) {
            uint8_t tag = kTagBF16; os.write(reinterpret_cast<const char*>(&tag), 1);
            auto data = t->to_vector<common::bfloat16>();
            os.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)(data.size() * sizeof(common::bfloat16)));
        } else {
            uint8_t tag = kTagF32; os.write(reinterpret_cast<const char*>(&tag), 1);
            auto data = t->to_vector<float>();
            os.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
        }
    }
    // Write the [0:valid_len] prefix of a [1, capacity, nKV, Dh] cache leaf.
    void write_kv_prefix(std::ostream& os, const utils::Ref<ir::Tensor>& t, size_t valid_len) const {
        if (!t) { uint8_t z = 0; os.write(reinterpret_cast<const char*>(&z), 1); return; }
        const auto& s = t->shape();
        auto pref = ir::slice(t, {0, 0, 0, 0}, {s[0], valid_len, s[2], s[3]}, {1, 1, 1, 1});
        write_tensor(os, ir::reshape(pref, {s[0], valid_len, s[2], s[3]}));  // make contiguous
    }
    utils::Ref<ir::Tensor> read_tensor(std::istream& is) const {
        uint8_t present = 0; is.read(reinterpret_cast<char*>(&present), 1);
        if (!present || !is) return nullptr;
        uint32_t rank = 0; is.read(reinterpret_cast<char*>(&rank), sizeof rank);
        std::vector<size_t> shp(rank); size_t numel = 1;
        for (uint32_t i = 0; i < rank; ++i) { uint32_t d = 0; is.read(reinterpret_cast<char*>(&d), sizeof d); shp[i] = d; numel *= d; }
        uint8_t tag = kTagF32; is.read(reinterpret_cast<char*>(&tag), 1);
        if (!is) return nullptr;
        if (tag == kTagBF16) {
            std::vector<common::bfloat16> data(numel);
            is.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(numel * sizeof(common::bfloat16)));
            if (!is) return nullptr;
            return ir::from_vector<common::bfloat16>(data, shp, _device_type);
        }
        std::vector<float> data(numel);
        is.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(numel * sizeof(float)));
        if (!is) return nullptr;
        return ir::from_vector<float>(data, shp, _device_type);
    }
public:

    // Forward pass: input_ids [B, S] -> logits [B, S, vocab_size]. Non-cached (full recompute);
    // builds a causal mask so full-attention layers are correctly causal.
    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input_ids) override {
        size_t S = input_ids->shape()[1];
        std::vector<LayerCache> caches(_blocks.size());  // all-null: no past
        auto pos = create_position_ids(S);
        auto mask = (S > 1) ? make_causal_mask(S) : nullptr;
        return apply_head(run_layers(input_ids, pos, mask, caches));
    }

    // Streaming callback type: called for each generated token id. Return false to abort early.
    using TokenCallback = std::function<bool(int32_t token_id)>;

    // Generate tokens autoregressively with a streaming callback: prefill over the prompt,
    // then invoke `callback` for each decoded token. Returns the tokens generated so far
    // (may be shorter than max_new_tokens if callback returns false).
    // `stop_tokens` is a set of token IDs that should trigger early termination (default: EOS).
    std::vector<int32_t> generateStreaming(std::vector<int32_t> input_ids,
                                            int32_t max_new_tokens,
                                            TokenCallback callback,
                                            std::optional<std::vector<int32_t>> stop_tokens,
                                            SamplingParams sampling = {}) {
        ir::NoGradScope no_grad;
        if (std::getenv("QWEN_KV_CONCAT")) inplace_kv = false;
        std::mt19937_64 rng(sampling.seed ? sampling.seed : std::random_device{}());
        std::vector<int32_t> generated;
        generated.reserve(max_new_tokens);
        auto caches = alloc_kv_caches(input_ids.size(), max_new_tokens);

        // Build stop-token lookup for fast O(1) check. Note the default (no explicit
        // stop_tokens) is the raw end-of-text token, NOT the chat turn-end <|im_end|>;
        // chat callers should pass <|im_end|> explicitly (the server does).
        std::unordered_set<int32_t> stop_set;
        if (stop_tokens) {
            for (int32_t t : *stop_tokens) stop_set.insert(t);
        } else {
            // Default EOS / end-of-text: 248044 for Qwen3.5/3.6, 151645 for Qwen3.
            stop_set.insert(_config.is_qwen3_5() ? 248044 : 151645);
        }

        // -- prefill (chunked to bound graph/command-buffer size on long prompts) --
        size_t S = input_ids.size();
        auto h = prefill_hidden(input_ids, caches);
        int32_t next = select_at(apply_head(h), h->shape()[1] - 1, sampling, rng);
        // Stop tokens are never emitted to the callback nor added to `generated`
        // (consistent with the decode loop below).
        if (stop_set.count(next)) return generated;
        if (!callback(next)) return generated;
        generated.push_back(next);

        // -- decode --
        size_t cur_len = S;
        for (int32_t t = 1; t < max_new_tokens; ++t) {
            auto in1 = ir::from_vector<int32_t>(std::vector<int32_t>{next}, {1, 1}, _device_type);
            auto h1 = run_layers(in1, create_position_ids_at((int32_t)cur_len, 1), nullptr, caches, cur_len);
            next = select_at(apply_head(h1), 0, sampling, rng);
            if (stop_set.count(next)) break;
            if (!callback(next)) break;
            generated.push_back(next);
            ++cur_len;
        }
        return generated;
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
        auto caches = alloc_kv_caches(input_ids.size(), max_new_tokens);

        // QWEN_TIMING=1 -> print prefill time and decode tokens/sec (argmax_at forces a sync each step).
        const bool timing = std::getenv("QWEN_TIMING") != nullptr;
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };

        // -- prefill the prompt (causal, chunked to bound graph/command-buffer size) --
        size_t S = input_ids.size();
        auto t_pre = clk::now();
        auto h = prefill_hidden(input_ids, caches);
        int32_t next = argmax_at(apply_head(h), h->shape()[1] - 1);
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
            if (s && b && w->dtype() == common::DType::UINT32)
                return io::dequant_mlx_affine(w, s, b, _device_type);
            return w;
        };
        // Bind a Linear from a packed triple (kept 8-bit).
        auto bind_q = [&](Linear& ql, const std::string& base) {
            auto w = find(base + ".weight"), s = find(base + ".scales"), b = find(base + ".biases");
            if (!w || !s || !b) { std::cerr << "[Qwen3Model] WARNING: missing quant triple " << base << "\n"; return; }
            ql.qweight = w; ql.scales = s; ql.biases = b; ql.quantized = true;
            const size_t Kp = w->shape()[1], groups = s->shape()[1];
            ql.params = ir::QuantParams{ir::QuantScheme::MLX_AFFINE, 8, (int)((Kp * 4) / groups), 4};
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

        // Optional MTP module (language_model.mtp.* / mtp.*): one full-attention layer + fc + norms.
        const std::string mp = lm ? "language_model.mtp." : "mtp.";
        if (W.count(mp + "fc.weight")) {
            const int32_t H = _config.hidden_size;
            has_mtp_ = true;
            mtp_fc_ = std::make_shared<Linear>((size_t)(2 * H), (size_t)H, /*use_bias=*/false,
                                               Init::Default, _device_type, /*lazy=*/true);
            mtp_block_ = std::make_shared<Qwen3Block>(LayerType::FULL_ATTENTION, _config, _device_type, /*lazy=*/true);
            // fc is a DENSE bf16 weight [out=H, in=2H] (no scales/biases). Transpose to the model's
            // [in, out] matmul convention and bind it as a plain (non-quantized) Linear weight.
            if (auto fcw = find(mp + "fc.weight")) mtp_fc_->weight = ir::transpose(fcw, 0, 1);
            else std::cerr << "[Qwen3Model] WARNING: missing " << mp << "fc.weight\n";
            bind(mtp_pre_fc_norm_emb_,    mp + "pre_fc_norm_embedding.weight");
            bind(mtp_pre_fc_norm_hidden_, mp + "pre_fc_norm_hidden.weight");
            bind(mtp_norm_,               mp + "norm.weight");
            auto* blk = mtp_block_.get();
            const std::string lp = mp + "layers.0.";
            bind(blk->fa_norm1_weight,  lp + "input_layernorm.weight");
            bind(blk->fa_norm2_weight,  lp + "post_attention_layernorm.weight");
            bind(blk->fa_q_norm_weight, lp + "self_attn.q_norm.weight");
            bind(blk->fa_k_norm_weight, lp + "self_attn.k_norm.weight");
            bind_q(*blk->fa_q_proj, lp + "self_attn.q_proj");
            bind_q(*blk->fa_k_proj, lp + "self_attn.k_proj");
            bind_q(*blk->fa_v_proj, lp + "self_attn.v_proj");
            bind_q(*blk->fa_o_proj, lp + "self_attn.o_proj");
            bind_q(*blk->fa_ffn->gate_proj, lp + "mlp.gate_proj");
            bind_q(*blk->fa_ffn->up_proj,   lp + "mlp.up_proj");
            bind_q(*blk->fa_ffn->down_proj, lp + "mlp.down_proj");
            printf("[Qwen3Model] MTP module loaded (self-speculation available)\n");
        }
    }

    const Qwen3Config& get_config() const { return _config; }
    backend::DeviceType get_device_type() const { return _device_type; }
    Qwen3Block* get_block(size_t i) { return _blocks[i].get(); }

    // ===== speculative decoding =====
    //
    // Primitives below are public so a speculative driver can drive both the main model and a
    // draft source (a smaller model, or this model's own MTP heads) through the same forward.

    // Allocate decode KV caches with headroom for `n_draft` in-flight speculative positions.
    std::vector<LayerCache> alloc_decode_caches(size_t prompt_len, int32_t max_new_tokens, int n_draft) {
        return alloc_kv_caches(prompt_len, max_new_tokens + n_draft);
    }

    // Persistent caches for speculative decode across requests, so the shared prompt prefix is
    // prefilled once (same idea as PrefixCacheSession but holding the main + draft caches the
    // speculative loop needs). `draft` is the separate draft model's cache (unused for MTP, whose
    // single-layer cache is built during decode and re-warms cheaply).
    struct SpecCacheState {
        std::vector<LayerCache> main, draft;
        std::vector<int32_t>    tokens;        // tokens the caches are valid for
        size_t                  capacity = 0;  // positions allocated
    };

    // Longest reusable exact prefix of `input_ids` given the session's cached tokens (cached must be a
    // full prefix; keep >=1 token to prefill). Returns 0 (and leaves st untouched) when there's nothing
    // to reuse / the conversation outgrew the cache -- callers then (re)allocate.
    size_t spec_reuse_len(const SpecCacheState& st, const std::vector<int32_t>& input_ids,
                          size_t need) const {
        if (st.main.empty() || st.tokens.empty() || st.tokens.size() > input_ids.size() ||
            need > st.capacity ||
            !std::equal(st.tokens.begin(), st.tokens.end(), input_ids.begin()))
            return 0;
        return input_ids.empty() ? 0 : std::min(st.tokens.size(), input_ids.size() - 1);
    }

    // Forward a block of `tokens` at absolute positions [start_pos, start_pos+S), writing K/V in
    // place at start_pos and attending over the [0, start_pos+S) prefix. Returns the pre-final-norm
    // hidden [1,S,H] (apply_head -> logits; MTP self-speculation also conditions on this hidden).
    utils::Ref<ir::Tensor> forward_cached_block_hidden(const std::vector<int32_t>& tokens, size_t start_pos,
                                                       std::vector<LayerCache>& caches) {
        size_t S = tokens.size();
        auto in = ir::from_vector<int32_t>(tokens, {1, S}, _device_type);
        auto mask = (S > 1) ? make_block_mask(S, start_pos) : nullptr;
        return run_layers(in, create_position_ids_at((int32_t)start_pos, S), mask, caches, start_pos);
    }

    // Prefill `input_ids` into `caches` in chunks of `chunk` tokens (default 256, env
    // CPPGRAD_PREFILL_CHUNK), writing K/V at successive offsets and threading the linear-attn state.
    // A single huge-prompt forward builds O(S) linear-attention scan+concat nodes per layer, which on
    // Metal exhausts GPU memory (caught OOM for moderate prompts; a hard SIGBUS for very large ones,
    // e.g. Claude Code's full system+tools context). Chunking bounds each forward's graph /
    // command-buffer size. Returns the pre-final-norm hidden of the LAST chunk [1, last_chunk, H].
    // Prefill input_ids[reuse_len:] into `caches` (already valid for [0, reuse_len)), writing K/V at
    // increasing offsets and threading the linear-attn state. reuse_len>0 is the prefix-cache path:
    // a shared prompt prefix is kept and only the new suffix is prefilled.
    utils::Ref<ir::Tensor> prefill_hidden(const std::vector<int32_t>& input_ids,
                                          std::vector<LayerCache>& caches,
                                          size_t reuse_len = 0) {
        // Read per call (not static) so chunk size can be toggled per run (tests, server). getenv once
        // per prefill is negligible against the forward cost.
        //
        // Default 256. Bounds each forward's graph depth + the full-attention score-matrix transient
        // (the in-place KV read-view grows to the full prefix, so the late-chunk attention is
        // [1,nH,CHUNK,prefix]). Linear-attention layers used to force a much smaller chunk -- their
        // per-token sequential scan launched O(chunk) tiny kernels per layer (~190k per 256-token
        // chunk -> Metal OOM) -- but the scan is now the chunked-parallel form (matmuls over
        // CPPGRAD_DELTA_CHUNK sub-chunks, qwen3_block.h), so the linear path no longer dominates the
        // per-chunk kernel/buffer count and the larger chunk (fewer per-chunk GPU syncs) is fine again.
        // Override: CPPGRAD_PREFILL_CHUNK (lower it if a very long prompt still exhausts GPU memory).
        const size_t CHUNK = []{
            const char* e = std::getenv("CPPGRAD_PREFILL_CHUNK");
            return e ? (size_t)std::max(1, atoi(e)) : (size_t)256;
        }();
        // Chunking bounds each forward's graph DEPTH. Per-chunk commit (docs/decode-runtime.md)
        // additionally bounds total MEMORY: after each non-final chunk, force the chunk's graph to
        // realize (committing its in-place KV writes as effects) and detach the linear-attention
        // recurrent + conv state to fresh leaves. This drops the chunk's transient graph and,
        // crucially, the recurrent-state chain that otherwise transitively retains the whole prefix's
        // buffers until the final readback -- the long-prompt Metal-buffer-exhaustion cause. Correct
        // because commit() relies on the executor's realized=immutable-boundary memoization, so
        // committed cache writes are never re-applied/re-ordered.
        const size_t S = input_ids.size();

        // Adaptive chunk: the full-attention score matrix is [1, nH, chunk, prefix] (and softmax keeps
        // a couple of copies), so the per-chunk transient grows with the prefix length -- at a fixed
        // chunk it eventually OOMs the GPU mid-prompt (observed ~offset 11.5k on the 27B). Cap the
        // chunk so the score-matrix AREA (chunk * (offset+chunk)) stays under a budget, via the exact
        // root cs = (-off + sqrt(off^2 + 4*AREA))/2. Tunable: CPPGRAD_PREFILL_AREA (0 disables).
        // (This does NOT bound the prefix-sized repeated-KV materialization; that's handled in the
        // attention path. Flash-style attention would remove both transients entirely -- future work.)
        // Now that gqa_attention removed the prefix-sized repeated-KV materialization, the score
        // matrix is the only prefix-growing transient and AREA caps it for ALL offsets (it does not
        // grow past the cap), so a larger budget is safe and keeps chunks full longer (faster GEMMs,
        // fewer syncs). 8e6 keeps chunk=256 until ~offset 31k. Raise CPPGRAD_PREFILL_AREA further if
        // memory allows; lower it if a very long prompt still exhausts GPU memory.
        const size_t AREA = []{
            const char* e = std::getenv("CPPGRAD_PREFILL_AREA");
            return e ? (size_t)std::max(0, atoi(e)) : (size_t)8000000;
        }();
        const size_t MIN_CHUNK = 8;

        // Progress logging for long (multi-chunk) prefills: a big prompt's prefill can take minutes
        // (the O(S) linear-attention scan), long enough that a client may time out with no server-side
        // sign of life. Log tokens done / total, rate, and ETA to stderr, throttled to ~1s so the log
        // stays readable. Only for multi-chunk prefills; silence with CPPGRAD_PREFILL_QUIET=1.
        using clk = std::chrono::steady_clock;
        const bool progress = (S > CHUNK) && std::getenv("CPPGRAD_PREFILL_QUIET") == nullptr;
        const auto t_start = clk::now();
        auto last_log = t_start;
        auto secs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() / 1000.0;
        };

        utils::Ref<ir::Tensor> h;
        for (size_t off = reuse_len; off < S; /* off += cs */) {
            size_t cs = std::min(CHUNK, S - off);
            if (AREA && off > 0) {
                double root = (-(double)off + std::sqrt((double)off * off + 4.0 * (double)AREA)) / 2.0;
                size_t cs_area = std::max(MIN_CHUNK, (size_t)root);
                cs = std::min(cs, cs_area);
            }
            std::vector<int32_t> c(input_ids.begin() + off, input_ids.begin() + off + cs);
            auto in = ir::from_vector<int32_t>(c, {1, cs}, _device_type);
            auto mask = (cs > 1) ? make_block_mask(cs, off) : nullptr;
            h = run_layers(in, create_position_ids_at((int32_t)off, cs), mask, caches, off);
            if (off + cs < S) {
                // Realize this chunk (commits its in-place KV cache writes; frees the chunk graph),
                // then detach the recurrent state so the next chunk starts from committed leaves.
                h->eval();
                for (auto& lc : caches) {
                    if (lc.state) lc.state = ir::commit(lc.state);
                    if (lc.conv)  lc.conv  = ir::commit(lc.conv);
                }
                // Flush batched GPU work so the chunk's command buffer commits + completes and
                // RELEASES the transient buffers it was holding. Without this, the executor's async
                // Metal path pins every chunk's intermediate buffers in one uncommitted command
                // buffer for the whole prefill -- so even with the graph detached above, a long
                // prompt accumulates buffers until Metal can't satisfy a tiny allocation. Safe here
                // because commit() already snapshotted the cache/state we keep into leaves; only
                // transients are freed. (No-op on CPU, which is synchronous.) Also a natural GPU
                // sync point, so elapsed time below reflects real completed work.
                if (auto* dev = backend::DeviceManager::device(_device_type))
                    dev->backend()->flush_pending();

                if (progress) {
                    auto now = clk::now();
                    if (secs(last_log, now) >= 1.0 || off + cs + CHUNK >= S) {
                        last_log = now;
                        size_t done = off + cs;
                        double el = secs(t_start, now);
                        double rate = el > 0 ? done / el : 0.0;
                        double eta = rate > 0 ? (S - done) / rate : 0.0;
                        std::fprintf(stderr, "[prefill] %zu/%zu tok (%.1f%%)  %.1f tok/s  elapsed %.0fs  ETA %.0fs\n",
                                     done, S, 100.0 * done / (double)S, rate, el, eta);
                    }
                }
            }
            off += cs;
        }
        if (progress)
            std::fprintf(stderr, "[prefill] %zu/%zu tok done in %.1fs\n", S, S, secs(t_start, clk::now()));
        // CPPGRAD_PROFILE=1: per-op GPU-time / memory-traffic breakdown of PREFILL (QuantizedMatMulOp
        // = the dequant-bound weight GEMMs; MatMulOp = attention + linear-attn scan matmuls; etc.),
        // then reset so any subsequent decode profile is reported separately.
        if (utils::Profiler::enabled()) {
            utils::Profiler::instance().report(stderr, "prefill profile");
            utils::Profiler::instance().reset();
        }
        return h;  // [1, last_chunk, H]
    }

    // As above, returning logits [1,S,vocab].
    utils::Ref<ir::Tensor> forward_cached_block(const std::vector<int32_t>& tokens, size_t start_pos,
                                                std::vector<LayerCache>& caches) {
        return apply_head(forward_cached_block_hidden(tokens, start_pos, caches));
    }

    // Greedy token at block position `pos` of logits [1,S,vocab] (public wrapper for argmax_at).
    int32_t greedy_block_at(const utils::Ref<ir::Tensor>& logits, size_t pos) { return argmax_at(logits, pos); }

    // Batched (B>1) cached forward. The in-place KV cache is B=1 only (its prefix read-view is strided
    // for B>1, which the attention can't reshape); so this drives the CONCAT-mode cache, which is
    // B-generic and produces contiguous K/V. Pass a caches vector of null LayerCaches that grow in
    // place across calls. `tokens` is [B,S] flat; positions and the [1,1,S,KV] mask broadcast over B
    // (uniform-length batch). Returns logits [B,S,vocab]. (Efficient in-place/paged batching is a
    // later optimization.)
    utils::Ref<ir::Tensor> forward_cached_batched(const std::vector<int32_t>& tokens, size_t B, size_t S,
                                                  size_t start_pos, std::vector<LayerCache>& caches) {
        bool saved = inplace_kv; inplace_kv = false;  // force concat path (in-place is B=1 only)
        auto in = ir::from_vector<int32_t>(tokens, {B, S}, _device_type);
        auto mask = (S > 1) ? make_block_mask(S, start_pos) : nullptr;
        auto logits = apply_head(run_layers(in, create_position_ids_at((int32_t)start_pos, S), mask, caches, start_pos));
        inplace_kv = saved;
        return logits;
    }

    // Greedy token at batch row `b`, sequence position `pos`, of logits [B,S,vocab].
    int32_t greedy_at(const utils::Ref<ir::Tensor>& logits, size_t b, size_t pos) {
        size_t S = logits->shape()[1], V = (size_t)_config.vocab_size;
        auto row = ir::reshape(ir::slice(logits, {b, pos, 0}, {b + 1, pos + 1, V}), {V});
        return argmax_last(row);
    }

    // Speculative decoding with a separate draft model. GREEDY => LOSSLESS: the emitted sequence is
    // bit-identical to greedy generateStreaming; the draft model only changes how fast we get there
    // (acceptance rate). For a non-greedy `sampling` we fall back to ordinary generateStreaming
    // (lossless speculative *sampling* is a later addition). `n_draft` is the speculation window.
    //
    // Per step: the draft proposes a block [next, d1..d_{w-1}]; the main model verifies the whole
    // block in ONE forward (positions cur..cur+w-1); we accept the longest prefix whose tokens match
    // the main model's own greedy choice, plus one correction/bonus token. Rejected KV is simply
    // overwritten next step (the cache is preallocated; logical length is tracked by start_pos).
    std::vector<int32_t> generateSpeculative(std::vector<int32_t> input_ids,
                                             int32_t max_new_tokens,
                                             TokenCallback callback,
                                             std::optional<std::vector<int32_t>> stop_tokens,
                                             Qwen3Model& draft,
                                             int n_draft = 4,
                                             SamplingParams sampling = {},
                                             SpecCacheState* st = nullptr) {
        // MTP / speculation disabled (n_draft < 2) or sampling requested -> plain decode.
        // (Lossless speculative *sampling* is a later addition; greedy speculation is lossless
        // w.r.t. the parallel forward but not bit-identical to single-token decode on fp near-ties.)
        if (n_draft < 2 || !sampling.greedy())
            return generateStreaming(std::move(input_ids), max_new_tokens, callback, stop_tokens, sampling);

        ir::NoGradScope no_grad;
        std::unordered_set<int32_t> stop_set;
        if (stop_tokens) for (int32_t t : *stop_tokens) stop_set.insert(t);
        else stop_set.insert(_config.is_qwen3_5() ? 248044 : 151645);

        const size_t S = input_ids.size();
        // Prefix-cache reuse: when `st` is given, persist + reuse the main and draft caches across
        // requests (prefill only the new suffix). Without it, allocate per-call (original behavior).
        // Reference-bind so the loop below is unchanged whichever caches we use.
        std::vector<LayerCache> local_main, local_draft;
        std::vector<LayerCache>& main_caches  = st ? st->main  : local_main;
        std::vector<LayerCache>& draft_caches = st ? st->draft : local_draft;
        size_t reuse = 0;
        const size_t need = S + (size_t)max_new_tokens + (size_t)n_draft;
        if (st) {
            reuse = spec_reuse_len(*st, input_ids, need);
            if (reuse == 0) {
                st->capacity = std::max(need + 8, st->capacity);
                st->main  = alloc_decode_caches(st->capacity, 0, 0);
                st->draft = draft.alloc_decode_caches(st->capacity, 0, 0);
            }
        } else {
            main_caches  = alloc_decode_caches(S, max_new_tokens, n_draft);
            draft_caches = draft.alloc_decode_caches(S, max_new_tokens, n_draft);
        }

        std::vector<int32_t> full_seq = input_ids;  // prompt + committed tokens
        std::vector<int32_t> generated;
        generated.reserve(max_new_tokens);

        // -- prefill both models over the prompt suffix (chunked; reuse the cached prefix) --
        auto ph = prefill_hidden(input_ids, main_caches, reuse);
        int32_t next = argmax_at(apply_head(ph), ph->shape()[1] - 1);
        draft.prefill_hidden(input_ids, draft_caches, reuse);  // prime draft cache (hidden unused)
        size_t cur = S;     // committed length: main cache valid for positions [0, cur)
        size_t dlen = S;    // draft cache committed length

        while ((int)generated.size() < max_new_tokens) {
            if (stop_set.count(next)) break;
            // commit block[0] = next
            if (!callback(next)) return generated;
            generated.push_back(next);
            full_seq.push_back(next);
            if ((int)generated.size() >= max_new_tokens) break;

            int window = std::min(n_draft, max_new_tokens - (int)generated.size() + 1);

            // -- draft: propose block = [next, d1, ..., d_{window-1}] at positions [cur, cur+window) --
            if (dlen < cur) {  // re-sync draft to committed length (cheap; draft is small)
                std::vector<int32_t> gap(full_seq.begin() + dlen, full_seq.begin() + cur);
                draft.forward_cached_block(gap, dlen, draft_caches);
                dlen = cur;
            }
            std::vector<int32_t> block = {next};
            int32_t d = next;
            for (int i = 1; i < window; ++i) {
                d = draft.greedy_block_at(draft.forward_cached_block({d}, cur + i - 1, draft_caches), 0);
                block.push_back(d);
            }

            // -- verify: main model scores the whole block in one forward --
            auto vlog = forward_cached_block(block, cur, main_caches);  // logits [1, |block|, vocab]

            if (std::getenv("QWEN_SPEC_DEBUG")) {  // compare incremental-cache verify to a fresh-cache verify
                auto fresh = alloc_decode_caches(full_seq.size(), (int)block.size() + 1, n_draft);
                forward_cached_block(full_seq, 0, fresh);              // prefill committed prefix
                auto fvlog = forward_cached_block(block, cur, fresh);  // same block, fresh cache
                int d = 0;
                for (size_t j = 0; j < block.size(); ++j) if (argmax_at(vlog, j) != argmax_at(fvlog, j)) d++;
                if (d) std::fprintf(stderr, "[spec-debug] cur=%zu acc-window=%d: %d/%zu verify argmax diffs (incremental vs fresh cache)\n",
                                    cur, window, d, block.size());
            }

            size_t acc = 1;                                  // block[0]=next is already main's greedy
            int32_t corrected = argmax_at(vlog, 0);          // main's greedy for position cur+1
            for (size_t j = 1; j < block.size(); ++j) {
                if (block[j] == corrected) { ++acc; corrected = argmax_at(vlog, j); }
                else break;                                  // reject; `corrected` is the fix for cur+j
            }

            // -- commit accepted draft tokens block[1..acc-1] --
            bool stopped = false;
            for (size_t j = 1; j < acc; ++j) {
                if (stop_set.count(block[j]) || (int)generated.size() >= max_new_tokens) { stopped = true; break; }
                if (!callback(block[j])) return generated;
                generated.push_back(block[j]);
                full_seq.push_back(block[j]);
            }
            cur += acc;
            next = corrected;  // becomes block[0] next round (the correction or, on full accept, the bonus)
            if (stopped) break;
        }
        if (st) st->tokens = full_seq;  // caches now valid for prompt + generated (for next-request reuse)
        return generated;
    }

    // ===== MTP (self-speculative drafter) =====

    bool has_mtp() const { return has_mtp_; }

    // Allocate the MTP block's single-layer KV cache (separate from the main caches).
    std::vector<LayerCache> alloc_mtp_cache(size_t prompt_len, int32_t max_new_tokens) {
        std::vector<LayerCache> mc(1);
        const size_t maxlen = prompt_len + (size_t)max_new_tokens;
        const size_t nKV = (size_t)_config.num_key_value_heads, Dh = (size_t)_config.head_dim;
        mc[0].k = ir::parameter({1, maxlen, nKV, Dh}, _device_type, common::DType::FLOAT32, true);
        mc[0].v = ir::parameter({1, maxlen, nKV, Dh}, _device_type, common::DType::FLOAT32, true);
        mc[0].k->set_requires_grad(false);
        mc[0].v->set_requires_grad(false);
        return mc;
    }

    // One MTP step. `hidden` [1,1,H] is the main model's pre-final-norm hidden for position p; `token`
    // is the token at position p+1. Combines norm(hidden) with norm(embed(token)) via `fc`, runs the
    // MTP transformer layer (writing its KV at `pos`=p+1), and predicts the token at p+2. Returns
    // {predicted token, MTP hidden [1,1,H]} (the hidden chains into the next MTP step).
    // `concat_order`: 1 = [emb, hidden] (validated correct for Qwen3.6 MTP: ~85% 1-step acceptance),
    // 0 = [hidden, emb] (wrong order, ~0%). Default 1.
    std::pair<int32_t, utils::Ref<ir::Tensor>> mtp_step(const utils::Ref<ir::Tensor>& hidden,
                                                        int32_t token, size_t pos,
                                                        std::vector<LayerCache>& mtp_cache,
                                                        int concat_order = 1) {
        const float eps = (float)_config.rms_norm_eps;
        const size_t H = (size_t)_config.hidden_size, V = (size_t)_config.vocab_size;
        auto emb = embed(ir::from_vector<int32_t>(std::vector<int32_t>{token}, {1, 1}, _device_type)); // [1,1,H]
        auto ne = nn::functional::rms_norm(emb, mtp_pre_fc_norm_emb_, eps);
        auto nh = nn::functional::rms_norm(hidden, mtp_pre_fc_norm_hidden_, eps);
        auto combined = (concat_order == 0) ? ir::concat(nh, ne, /*axis=*/2)
                                            : ir::concat(ne, nh, /*axis=*/2);   // [1,1,2H]
        auto x = ir::reshape(mtp_fc_->forward(ir::reshape(combined, {1, 2 * H})), {1, 1, H});
        auto h2 = mtp_block_->forward_full_cached(x, create_position_ids_at((int32_t)pos, 1), _inv_freq,
                                                  nullptr, mtp_cache[0].k, mtp_cache[0].v, pos);
        auto logits = lm_head->forward(ir::reshape(nn::functional::rms_norm(h2, mtp_norm_, eps), {1, H}));
        return {argmax_last(ir::reshape(logits, {V})), h2};
    }

    // Validation probe: greedy-decode `steps` tokens with the main model; at each step run one MTP
    // step and count how often its prediction equals the main model's actual next greedy token. A
    // high rate means the MTP module (architecture, weight mapping, fc concat order) is correct.
    double mtp_self_check(std::vector<int32_t> prompt, int steps, int concat_order = 0) {
        if (!has_mtp_) return -1.0;
        ir::NoGradScope no_grad;
        const size_t H = (size_t)_config.hidden_size, S = prompt.size();
        auto caches = alloc_kv_caches(S, steps + 2);
        auto mtp_cache = alloc_mtp_cache(S, steps + 2);

        auto h = run_layers(ir::from_vector<int32_t>(prompt, {1, S}, _device_type),
                            create_position_ids(S), make_causal_mask(S), caches);
        int32_t next = argmax_at(apply_head(h), S - 1);                     // main token at pos S
        auto hidden = ir::reshape(ir::slice(h, {0, S - 1, 0}, {1, S, H}), {1, 1, H});  // hidden_{S-1}
        size_t cur = S;
        auto [mtp_pred, mtp_h] = mtp_step(hidden, next, cur, mtp_cache, concat_order);  // predicts pos S+1

        int match = 0, total = 0;
        for (int t = 0; t < steps; ++t) {
            auto h1 = run_layers(ir::from_vector<int32_t>(std::vector<int32_t>{next}, {1, 1}, _device_type),
                                 create_position_ids_at((int32_t)cur, 1), nullptr, caches, cur);
            int32_t main_next = argmax_at(apply_head(h1), 0);              // main token at pos cur+1
            ++total; if (mtp_pred == main_next) ++match;
            auto hidden_cur = ir::reshape(h1, {1, 1, H});                  // hidden at pos cur
            next = main_next; ++cur;
            std::tie(mtp_pred, mtp_h) = mtp_step(hidden_cur, next, cur, mtp_cache, concat_order);
        }
        return total ? (double)match / total : -1.0;
    }

    // Self-speculative decoding using the MTP module as the drafter (no separate draft model). Same
    // verify/accept/rollback as generateSpeculative; the draft block is produced by recurrently
    // applying the MTP head to the main model's hidden state. GREEDY => lossless w.r.t. the parallel
    // forward (the MTP cache is only a drafting aid: imperfect maintenance affects acceptance, never
    // correctness, since the main model verifies every token). n_draft<2 or sampling => plain decode.
    // `accept_out` (optional) receives {accepted_draft_tokens, verify_rounds} for speedup measurement.
    std::vector<int32_t> generateSpeculativeMTP(std::vector<int32_t> input_ids, int32_t max_new_tokens,
                                                TokenCallback callback,
                                                std::optional<std::vector<int32_t>> stop_tokens,
                                                int n_draft = 4, SamplingParams sampling = {},
                                                std::pair<int,int>* accept_out = nullptr,
                                                SpecCacheState* st = nullptr) {
        if (!has_mtp_ || n_draft < 2 || !sampling.greedy())
            return generateStreaming(std::move(input_ids), max_new_tokens, callback, stop_tokens, sampling);

        ir::NoGradScope no_grad;
        std::unordered_set<int32_t> stop_set;
        if (stop_tokens) for (int32_t t : *stop_tokens) stop_set.insert(t);
        else stop_set.insert(_config.is_qwen3_5() ? 248044 : 151645);

        const size_t S = input_ids.size(), H = (size_t)_config.hidden_size;
        // Prefix-cache reuse for the MAIN cache (mtp_cache stays per-call: it's a single layer built
        // during decode, never prefilled over the prompt; a fresh one just re-warms drafts, which only
        // affects acceptance rate, not correctness -- the main model verifies). Reference-bind so the
        // loop is unchanged.
        std::vector<LayerCache> local_main;
        std::vector<LayerCache>& main_caches = st ? st->main : local_main;
        size_t reuse = 0;
        const size_t need = S + (size_t)max_new_tokens + (size_t)n_draft;
        if (st) {
            reuse = spec_reuse_len(*st, input_ids, need);
            if (reuse == 0) { st->capacity = std::max(need + 8, st->capacity);
                              st->main = alloc_decode_caches(st->capacity, 0, 0); }
        } else {
            main_caches = alloc_decode_caches(S, max_new_tokens, n_draft);
        }
        auto mtp_cache   = alloc_mtp_cache(S, max_new_tokens + n_draft);
        std::vector<int32_t> generated;
        generated.reserve(max_new_tokens);
        int accepted_drafts = 0, verify_rounds = 0;

        // prefill (chunked; reuse the cached main prefix, only the new suffix is prefilled)
        auto h = prefill_hidden(input_ids, main_caches, reuse);
        size_t lc = h->shape()[1];                                            // last chunk length
        int32_t next = argmax_at(apply_head(h), lc - 1);
        auto hidden_last = ir::reshape(ir::slice(h, {0, lc - 1, 0}, {1, lc, H}), {1, 1, H});  // hidden_{S-1}
        size_t cur = S;

        while ((int)generated.size() < max_new_tokens) {
            if (stop_set.count(next)) break;
            if (!callback(next)) break;
            generated.push_back(next);
            if ((int)generated.size() >= max_new_tokens) break;

            int window = std::min(n_draft, max_new_tokens - (int)generated.size() + 1);

            // -- draft: MTP proposes block = [next, d1, ..., d_{window-1}] conditioning on main hidden --
            std::vector<int32_t> block = {next};
            auto mtp_h = hidden_last;
            int32_t mtp_tok = next;
            for (int i = 1; i < window; ++i) {
                int32_t d;
                std::tie(d, mtp_h) = mtp_step(mtp_h, mtp_tok, cur + i - 1, mtp_cache);
                block.push_back(d);
                mtp_tok = d;
            }

            // -- verify with the main model (one forward); keep hidden states for the next draft --
            auto hb = forward_cached_block_hidden(block, cur, main_caches);
            auto vlog = apply_head(hb);
            ++verify_rounds;
            size_t acc = 1;
            int32_t corrected = argmax_at(vlog, 0);
            for (size_t j = 1; j < block.size(); ++j) {
                if (block[j] == corrected) { ++acc; corrected = argmax_at(vlog, j); }
                else break;
            }

            bool stopped = false;
            for (size_t j = 1; j < acc; ++j) {
                if (stop_set.count(block[j]) || (int)generated.size() >= max_new_tokens) { stopped = true; break; }
                if (!callback(block[j])) return generated;
                generated.push_back(block[j]);
                ++accepted_drafts;
            }
            hidden_last = ir::reshape(ir::slice(hb, {0, acc - 1, 0}, {1, acc, H}), {1, 1, H});  // hidden_{cur+acc-1}
            cur += acc;
            next = corrected;
            if (stopped) break;
        }
        if (accept_out) *accept_out = {accepted_drafts, verify_rounds};
        if (st) {  // main cache now valid for prompt + generated (for next-request prefix reuse)
            st->tokens = input_ids;
            st->tokens.insert(st->tokens.end(), generated.begin(), generated.end());
        }
        return generated;
    }

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

    // Additive mask [1,1,S,start_pos+S] for a block of S new tokens written at start_pos
    // attending over the [0 .. start_pos+S) cached prefix: new token i (absolute position
    // start_pos+i) may attend columns 0..start_pos+i. At start_pos==0 this equals make_causal_mask.
    utils::Ref<ir::Tensor> make_block_mask(size_t S, size_t start_pos) {
        size_t KV = start_pos + S;
        std::vector<float> m(S * KV, 0.0f);
        for (size_t i = 0; i < S; ++i)
            for (size_t j = start_pos + i + 1; j < KV; ++j) m[i * KV + j] = -1e9f;
        return ir::from_vector<float>(m, {1, 1, S, KV}, _device_type);
    }

    // Argmax over the vocab at sequence position `pos` of logits [1, S, V].
    int32_t argmax_at(const utils::Ref<ir::Tensor>& logits, size_t pos) {
        size_t V = (size_t)_config.vocab_size;
        auto row = ir::reshape(ir::slice(logits, {0, pos, 0}, {1, pos + 1, V}), {V});
        return argmax_last(row);
    }

    // Select a token at sequence position `pos`: greedy argmax, or a sampled draw
    // under `sp`. Greedy avoids the full-vocab readback+softmax that sampling needs.
    int32_t select_at(const utils::Ref<ir::Tensor>& logits, size_t pos,
                      const SamplingParams& sp, std::mt19937_64& rng) {
        if (sp.greedy()) return argmax_at(logits, pos);
        size_t V = (size_t)_config.vocab_size;
        auto row = ir::reshape(ir::slice(logits, {0, pos, 0}, {1, pos + 1, V}), {V});
        return sample_logits(row->to_vector<float>(), sp, rng);
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
