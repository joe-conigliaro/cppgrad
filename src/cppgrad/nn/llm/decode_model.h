// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Model-agnostic autoregressive decode runtime.
//
// Like mlx-lm / llama.cpp, the cache, the generate loop, sampling, and prompt (prefix) caching are
// SHARED across model architectures and driven through a small capability interface (DecodeModel).
// A concrete model (Qwen3, and future Llama/Mistral/...) only implements the forward primitives:
// allocate a cache, prefill a token range into it, decode one step, and sample. Everything in this
// header is generic.
#pragma once

#include <memory>
#include <vector>
#include <random>
#include <string>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <istream>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/utils/ref.h"

namespace cppgrad::nn::llm {

// Sampling controls (generic). logits are scaled by 1/temperature and optionally filtered by
// top_k / top_p before a categorical draw. seed == 0 => nondeterministic. temperature <= 0 => greedy.
struct SamplingParams {
    float    temperature = 0.0f;
    float    top_p       = 1.0f;
    int      top_k       = 0;
    uint64_t seed        = 0;
    bool greedy() const { return temperature <= 0.0f; }
};

// Opaque, model-defined cache (KV per attention layer + recurrent state per linear layer, etc.).
// The runtime treats it as a handle and only tracks how many leading tokens it is valid for
// (in PrefixCacheSession). The model up-casts to its concrete cache type in its forward methods.
struct ModelCache {
    virtual ~ModelCache() = default;
};

// Capability interface a model implements to be driven by the shared decode runtime.
class DecodeModel {
public:
    virtual ~DecodeModel() = default;

    // Allocate a cache able to hold `capacity_tokens` positions.
    virtual std::unique_ptr<ModelCache> make_cache(size_t capacity_tokens) = 0;

    // Prefill tokens[start_pos:] into `cache` (which is already valid for positions [0, start_pos)),
    // writing KV / threading recurrent state at increasing positions. Returns logits [1, S_suffix, V]
    // for the prefilled suffix (the last row is the logits for the final prompt token).
    virtual utils::Ref<ir::Tensor> prefill(const std::vector<int32_t>& tokens, size_t start_pos,
                                           ModelCache& cache) = 0;

    // Decode one token at absolute position `pos`; returns logits [1, 1, V].
    virtual utils::Ref<ir::Tensor> decode_step(int32_t token, size_t pos, ModelCache& cache) = 0;

    // Sample a token id from the LAST position of a [1, S, V] logits tensor.
    virtual int32_t sample_last(const utils::Ref<ir::Tensor>& logits, const SamplingParams& sampling,
                                std::mt19937_64& rng) = 0;

    // Default end-of-sequence token id (used when the caller passes no explicit stop set).
    virtual int32_t default_eos() const = 0;

    // --- Cache persistence (for cross-restart prompt-prefix caching) ---
    // A tag identifying the model config, so a saved cache from a different model is rejected.
    virtual uint64_t cache_tag() const = 0;
    // Write the cache contents valid for the first `valid_len` tokens (KV prefix + recurrent state).
    virtual void write_cache(const ModelCache& cache, size_t valid_len, std::ostream& os) const = 0;
    // Allocate a cache of `capacity` positions and read `valid_len` tokens of contents into it.
    virtual std::unique_ptr<ModelCache> read_cache(std::istream& is, size_t valid_len,
                                                   size_t capacity) = 0;
};

// Persistent decode session: the model cache + the exact token sequence it currently represents.
// Held across requests by the server so a shared prompt prefix is prefilled ONCE and reused.
struct PrefixCacheSession {
    std::unique_ptr<ModelCache> cache;
    std::vector<int32_t>        tokens;          // tokens the cache is currently valid for
    size_t                      capacity = 0;    // positions the cache was allocated for
    size_t                      capacity_hint = 0;  // optional min capacity to allocate (e.g. max ctx)
    // Stats from the most recent generate (for logging): tokens reused vs newly prefilled.
    size_t reused = 0, prefilled = 0;
};

// Generate up to `max_new_tokens` tokens for `prompt`, reusing the longest cached EXACT PREFIX.
//
// Reuse policy: if the session's cached tokens are a full prefix of `prompt`, keep them and prefill
// only the new suffix at that offset. Any divergence (a branched/edited conversation) => reset and
// prefill from scratch -- always correct. At least one token is always prefilled so we have logits
// for the first new token. This is what makes repeated requests with a large stable prefix (e.g. a
// fixed system + tools prompt) cheap after the first turn.
//
// `on_token(id) -> bool`: called per generated token; return false to stop early. `stop`: token ids
// that end generation (not emitted). Returns the generated token ids.
template <class OnToken>
inline std::vector<int32_t> generate_with_prefix_cache(
    DecodeModel& model, PrefixCacheSession& sess,
    const std::vector<int32_t>& prompt, int max_new_tokens,
    OnToken&& on_token, const std::unordered_set<int32_t>& stop,
    SamplingParams sampling = {}) {

    ir::NoGradScope no_grad;   // inference: no autograd (required by the in-place KV cache_update)
    std::mt19937_64 rng(sampling.seed ? sampling.seed : std::random_device{}());
    const size_t need = prompt.size() + (size_t)std::max(1, max_new_tokens);

    // Longest reusable exact prefix: cached tokens must be a prefix of `prompt`; keep >=1 token to
    // prefill (we need its logits), and never exceed the allocated capacity.
    size_t reuse = 0;
    if (sess.cache && !sess.tokens.empty() && sess.tokens.size() <= prompt.size() &&
        std::equal(sess.tokens.begin(), sess.tokens.end(), prompt.begin())) {
        reuse = sess.tokens.size();
    }
    if (!prompt.empty()) reuse = std::min(reuse, prompt.size() - 1);

    // (Re)allocate when there's nothing to reuse or the conversation outgrew the cache.
    if (!sess.cache || reuse == 0 || need > sess.capacity) {
        sess.capacity = std::max(need + 8, sess.capacity_hint);
        sess.cache = model.make_cache(sess.capacity);
        sess.tokens.clear();
        reuse = 0;
    }
    sess.reused = reuse;
    sess.prefilled = prompt.size() - reuse;

    auto logits = model.prefill(prompt, reuse, *sess.cache);
    int32_t next = model.sample_last(logits, sampling, rng);

    std::vector<int32_t> generated;
    generated.reserve((size_t)max_new_tokens);
    size_t pos = prompt.size();
    for (int i = 0; i < max_new_tokens; ++i) {
        if (stop.count(next)) break;
        if (!on_token(next)) break;
        generated.push_back(next);
        auto lg = model.decode_step(next, pos, *sess.cache);
        next = model.sample_last(lg, sampling, rng);
        ++pos;
    }

    // The cache now represents prompt + generated.
    sess.tokens = prompt;
    sess.tokens.insert(sess.tokens.end(), generated.begin(), generated.end());
    return generated;
}

// ---- Cross-restart prefix-cache persistence ----
// Save / load a session's cache + token sequence to a single file, so a fixed prompt prefix (e.g. a
// system + tools prompt) is prefilled ONCE on a machine, ever -- a server restart reloads the warm
// cache instead of re-prefilling. The generic header (magic, model tag, capacity, tokens) is written
// here; the model serializes its own cache bytes via write_cache / read_cache. A file whose model tag
// doesn't match the running model is rejected.
static constexpr uint64_t kPrefixCacheMagic = 0x43505846435F3032ULL;  // "CPXFC_02" (v2: per-tensor dtype tag)

inline bool save_prefix_cache(DecodeModel& model, const PrefixCacheSession& sess,
                              const std::string& path) {
    if (!sess.cache || sess.tokens.empty()) return false;
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    const uint64_t magic = kPrefixCacheMagic, tag = model.cache_tag();
    const uint64_t vlen = sess.tokens.size(), cap = sess.capacity;
    os.write(reinterpret_cast<const char*>(&magic), sizeof magic);
    os.write(reinterpret_cast<const char*>(&tag),   sizeof tag);
    os.write(reinterpret_cast<const char*>(&vlen),  sizeof vlen);
    os.write(reinterpret_cast<const char*>(&cap),   sizeof cap);
    os.write(reinterpret_cast<const char*>(sess.tokens.data()),
             (std::streamsize)(vlen * sizeof(int32_t)));
    model.write_cache(*sess.cache, sess.tokens.size(), os);
    return static_cast<bool>(os);
}

// Restore into `sess` (cache + tokens + capacity). Returns false (leaving sess untouched) if the file
// is absent / malformed / from a different model. capacity_hint, if larger, raises the allocation.
inline bool load_prefix_cache(DecodeModel& model, PrefixCacheSession& sess,
                              const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    uint64_t magic = 0, tag = 0, vlen = 0, cap = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof magic);
    is.read(reinterpret_cast<char*>(&tag),   sizeof tag);
    is.read(reinterpret_cast<char*>(&vlen),  sizeof vlen);
    is.read(reinterpret_cast<char*>(&cap),   sizeof cap);
    if (!is || magic != kPrefixCacheMagic || tag != model.cache_tag()) return false;
    std::vector<int32_t> toks(vlen);
    is.read(reinterpret_cast<char*>(toks.data()), (std::streamsize)(vlen * sizeof(int32_t)));
    if (!is) return false;
    const size_t capacity = std::max<size_t>(cap, sess.capacity_hint);
    auto cache = model.read_cache(is, (size_t)vlen, capacity);
    if (!cache || !is) return false;
    sess.cache = std::move(cache);
    sess.tokens = std::move(toks);
    sess.capacity = capacity;
    return true;
}

} // namespace cppgrad::nn::llm
