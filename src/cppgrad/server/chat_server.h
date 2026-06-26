// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// Chat server core: wraps Qwen3Model + BPETokenizer with a format-agnostic
// generation core (prompt build + streaming token state machine with Hermes-style
// <tool_call>...</tool_call> parsing), plus the OpenAI chat-completions surface.
//
// The Anthropic Messages surface (/v1/messages) is layered on top of the same core
// in anthropic_messages.h.
//

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <mutex>
#include <cstdio>
#include <cstdint>
#include <glob.h>

#include "cppgrad/server/openai_types.h"
#include "cppgrad/io/chat_template.h"
#include "cppgrad/io/tokenizer.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/grad_mode.h"

#include <nlohmann/json.hpp>

namespace cppgrad {
namespace server {

using cppgrad::nn::llm::qwen::Qwen3Model;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::SamplingParams;
using cppgrad::io::BPETokenizer;
using cppgrad::io::Qwen3ChatTemplate;
using cppgrad::io::ChatMessage;
using cppgrad::io::ChatTool;

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

// Gather safetensors shard paths from a model directory (or pass a file directly).
inline std::vector<std::string> gather_shards(const std::string& path) {
    if (path.size() > 12 && path.substr(path.size() - 12) == ".safetensors")
        return {path};
    std::vector<std::string> out;
    for (const char* pat : {"/model-*-of-*.safetensors", "/*.safetensors"}) {
        glob_t g;
        if (glob((path + pat).c_str(), 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; ++i) out.push_back(g.gl_pathv[i]);
        globfree(&g);
        if (!out.empty()) break;
    }
    if (out.empty()) out.push_back(path);
    return out;
}

inline std::optional<Qwen3Config> parse_config(const std::string& name) {
    if (name == "0.6b")        return Qwen3Config::get_0_6b();
    if (name == "1.5b")        return Qwen3Config::get_1_5b();
    if (name == "3b")          return Qwen3Config::get_3b();
    if (name == "4b")          return Qwen3Config::get_4b();
    if (name == "7b")          return Qwen3Config::get_7b();
    if (name == "27b_qwen3_6") return Qwen3Config::get_27b_qwen3_6();
    return std::nullopt;
}

// Length of the longest prefix of `s` (in bytes) that is complete, valid UTF-8.
// Used to avoid emitting a partial multi-byte codepoint mid-stream (which would
// make nlohmann::json::dump() throw).
inline size_t utf8_complete_len(const std::string& s) {
    size_t n = s.size();
    if (n == 0) return 0;
    size_t i = n, cont = 0;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) { --i; ++cont; }
    if (i == 0) return n;  // all continuation bytes: nothing better to do
    unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    size_t need = (lead < 0x80) ? 1
                : ((lead >> 5) == 0x6) ? 2
                : ((lead >> 4) == 0xE) ? 3
                : ((lead >> 3) == 0x1E) ? 4
                : 1;
    return (cont + 1 >= need) ? n : (i - 1);
}

// Build a ToolCall from the raw JSON found inside a Hermes <tool_call> block,
// e.g. {"name": "get_weather", "arguments": {"city": "Paris"}}.
inline ToolCall tool_call_from_hermes(const std::string& raw) {
    ToolCall tc;
    tc.id = generate_tool_call_id();
    tc.type = "function";
    try {
        auto j = nlohmann::json::parse(raw);
        tc.function.name = j.value("name", "");
        tc.function.arguments = j.contains("arguments") ? j["arguments"].dump() : "{}";
    } catch (...) {
        tc.function.arguments = "{}";
    }
    return tc;
}

// Extract Hermes <tool_call>{json}</tool_call> blocks from `text`. Returns true if
// any were found; `clean_text` receives the text with the blocks removed/trimmed.
inline bool parse_hermes_tool_calls(const std::string& text, std::vector<ToolCall>& calls,
                                    std::string& clean_text) {
    const std::string open = "<tool_call>", close = "</tool_call>";
    calls.clear();
    clean_text.clear();
    size_t pos = 0;
    while (true) {
        size_t start = text.find(open, pos);
        if (start == std::string::npos) { clean_text += text.substr(pos); break; }
        clean_text += text.substr(pos, start - pos);
        size_t body = start + open.size();
        size_t end = text.find(close, body);
        if (end == std::string::npos) { clean_text += text.substr(start); break; }
        calls.push_back(tool_call_from_hermes(text.substr(body, end - body)));
        pos = end + close.size();
    }
    size_t b = clean_text.find_first_not_of(" \n\r\t");
    size_t e = clean_text.find_last_not_of(" \n\r\t");
    clean_text = (b == std::string::npos) ? "" : clean_text.substr(b, e - b + 1);
    return !calls.empty();
}

inline ChatMessage to_chat_message(const Message& m) {
    ChatMessage cm;
    cm.role = m.role;
    cm.content = m.content;
    cm.tool_call_id = m.tool_call_id;
    for (const auto& tc : m.tool_calls) {
        ChatMessage::ToolCall ctc;
        ctc.id = tc.id;
        ctc.name = tc.function.name;
        try { ctc.arguments = nlohmann::json::parse(tc.function.arguments); }
        catch (...) { ctc.arguments = nlohmann::json::object(); }
        cm.tool_calls.push_back(std::move(ctc));
    }
    return cm;
}

inline ChatTool to_tool_def(const ToolDefinition& t) {
    ChatTool td;
    td.type = t.type;
    td.function.name = t.function.name;
    td.function.description = t.function.description;
    td.function.parameters = t.function.parameters;
    return td;
}

// ---------------------------------------------------------------------------
// ChatServer
// ---------------------------------------------------------------------------

// Result of a blocking (non-streaming) generation.
struct GenText {
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    bool hit_length = false;  // stopped because max_tokens was reached
};

// Summary of a streaming generation (content was delivered via callbacks).
struct GenStream {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int tool_calls = 0;
    bool hit_length = false;
};

class ChatServer {
public:
    bool load_model(const std::string& model_dir,
                    const std::string& config_name = "27b_qwen3_6",
                    bool quantize = true) {
        printf("[ChatServer] Loading model from: %s\n", model_dir.c_str());

        cfg_ = parse_config(config_name);
        if (!cfg_) {
            fprintf(stderr, "[ChatServer] Unknown config: %s\n", config_name.c_str());
            return false;
        }

        backend::DeviceManager::instance().init();
        device_type_ = backend::DeviceManager::default_device_type();
        printf("[ChatServer] Device: %s\n", backend::to_string(device_type_));

        try {
            tokenizer_ = std::make_unique<BPETokenizer>(model_dir + "/tokenizer.json");
            printf("[ChatServer] Tokenizer loaded (vocab=%d)\n", tokenizer_->vocab_size());
        } catch (const std::exception& e) {
            fprintf(stderr, "[ChatServer] Failed to load tokenizer: %s\n", e.what());
            return false;
        }

        // Resolve special-token IDs once (works across Qwen3 / 3.5 / 3.6).
        im_end_id_       = tokenizer_->get_token_id("<|im_end|>");
        eot_id_          = tokenizer_->get_token_id("<|endoftext|>");
        tool_call_start_ = tokenizer_->get_token_id("<tool_call>");
        tool_call_end_   = tokenizer_->get_token_id("</tool_call>");
        stop_tokens_.clear();
        for (int id : {im_end_id_, eot_id_}) if (id >= 0) stop_tokens_.push_back(id);

        model_ = std::make_unique<Qwen3Model>(*cfg_, device_type_, /*lazy_weights=*/true);
        auto shards = gather_shards(model_dir);
        printf("[ChatServer] Loading %zu shards (quant=%d)...\n", shards.size(), quantize);
        model_->load_from_safetensors(shards, quantize);

        loaded_ = true;
        model_name_ = "qwen3-" + config_name;
        printf("[ChatServer] Model loaded: %s\n", model_name_.c_str());
        // Prefix-cache capacity: preallocate the persistent KV cache to this many positions so reuse
        // survives the conversation growing. Set CPPGRAD_KV_CAPACITY to your max context (e.g. 65536)
        // to keep a fixed system+tools prefix cached across turns.
        if (const char* e = std::getenv("CPPGRAD_KV_CAPACITY")) {
            session_.capacity_hint = (size_t)std::max(0, atoi(e));
            spec_session_.capacity = session_.capacity_hint;   // preallocate speculative caches too
            printf("[ChatServer] prefix-cache KV capacity hint = %zu tokens\n", session_.capacity_hint);
        }
        // Cross-restart prefix cache: persist the warm prompt-prefix cache to disk so the (expensive)
        // first prefill of a fixed system+tools prompt is paid once on a machine, ever -- a restart
        // reloads it. Set CPPGRAD_KV_CACHE_FILE=/path. Loaded here (after the capacity hint); saved
        // after the first request below.
        if (const char* f = std::getenv("CPPGRAD_KV_CACHE_FILE")) {
            kv_cache_file_ = f;
            if (nn::llm::load_prefix_cache(*model_, session_, kv_cache_file_)) {
                kv_cache_saved_ = true;   // already warm; no need to re-save it
                printf("[ChatServer] prefix cache restored from %s (%zu tokens)\n",
                       kv_cache_file_.c_str(), session_.tokens.size());
            } else {
                printf("[ChatServer] prefix cache file %s not usable yet (will save after first request)\n",
                       kv_cache_file_.c_str());
            }
        }
        // Auto-enable MTP self-speculation if the checkpoint ships an MTP module (no draft model needed).
        if (model_->has_mtp()) {
            mtp_enabled_ = true;
            printf("[ChatServer] MTP self-speculation ENABLED (n_draft=%d)\n", mtp_n_draft_);
        }
        return true;
    }

    // Configure MTP self-speculation (no-op if the checkpoint has no MTP module). n_draft<2 disables.
    void set_mtp(bool enabled, int n_draft = 4) { mtp_enabled_ = enabled; mtp_n_draft_ = n_draft; }

    // Load an optional smaller draft model to enable speculative (MTP-style) decoding. The draft
    // MUST share the main model's tokenizer/vocab (true across the Qwen3 size family). `n_draft` is
    // the speculation window; <2 disables. Greedy requests then verify draft blocks in one main
    // forward (lossless w.r.t. the parallel forward); sampling requests fall back to plain decode.
    bool load_draft_model(const std::string& dir, const std::string& config_name,
                          int n_draft = 4, bool quantize = true) {
        auto dcfg = parse_config(config_name);
        if (!dcfg) {
            fprintf(stderr, "[ChatServer] Unknown draft config: %s\n", config_name.c_str());
            return false;
        }
        draft_model_ = std::make_unique<Qwen3Model>(*dcfg, device_type_, /*lazy_weights=*/true);
        auto shards = gather_shards(dir);
        printf("[ChatServer] Loading draft model %s (%zu shards)...\n", config_name.c_str(), shards.size());
        draft_model_->load_from_safetensors(shards, quantize);
        n_draft_ = n_draft;
        printf("[ChatServer] Speculative decoding ENABLED (draft=%s, n_draft=%d)\n",
               config_name.c_str(), n_draft);
        return true;
    }

    bool is_loaded() const { return loaded_; }
    bool speculative_enabled() const {
        return (draft_model_ && n_draft_ >= 2) || (mtp_enabled_ && model_ && model_->has_mtp() && mtp_n_draft_ >= 2);
    }
    // Short description of the active decode path (for the startup banner).
    std::string decode_mode() const {
        if (draft_model_ && n_draft_ >= 2) return "speculative (draft model, n_draft=" + std::to_string(n_draft_) + ")";
        if (mtp_enabled_ && model_ && model_->has_mtp() && mtp_n_draft_ >= 2)
            return "speculative (MTP, n_draft=" + std::to_string(mtp_n_draft_) + ")";
        return "plain";
    }
    const std::string& model_name() const { return model_name_; }

    ModelList get_models() const {
        ModelList list;
        ModelInfo info;
        info.id = model_name_;
        info.created = current_timestamp();
        list.data.push_back(info);
        return list;
    }

    // ---- format-agnostic generation core (reused by OpenAI + Anthropic) ----

    std::vector<int32_t> build_prompt(const std::vector<ChatMessage>& msgs,
                                      const std::vector<ChatTool>& tools) {
        return template_.apply_tokens(msgs, tools, *tokenizer_);
    }

    std::vector<int32_t> build_prompt(const ChatCompletionRequest& req) {
        std::vector<ChatMessage> msgs;
        msgs.reserve(req.messages.size());
        for (const auto& m : req.messages) msgs.push_back(to_chat_message(m));
        std::vector<ChatTool> tools;
        tools.reserve(req.tools.size());
        for (const auto& t : req.tools) tools.push_back(to_tool_def(t));
        return build_prompt(msgs, tools);
    }

    // Blocking generation; returns the full decoded text + token counts. The
    // generation lock serializes model use (the model is a single, non-reentrant
    // instance) so the HTTP layer can accept concurrent connections safely.
    GenText generate_text(const std::vector<int32_t>& prompt_ids, int max_tokens,
                          const SamplingParams& sampling = {}) {
        std::lock_guard<std::mutex> lock(gen_mutex_);
        ir::NoGradScope no_grad;
        auto ids = decode(prompt_ids, max_tokens, [](int32_t) { return true; }, sampling);
        return {tokenizer_->decode(ids), (int)prompt_ids.size(), (int)ids.size(),
                (int)ids.size() >= max_tokens};
    }

    // Streaming greedy generation. `on_text(delta)` receives complete-UTF-8 text
    // deltas; `on_tool(tc)` receives each fully-parsed Hermes tool call. The tags
    // <tool_call> / </tool_call> are single special tokens, so detection is exact.
    template <class OnText, class OnTool>
    GenStream stream_generate(const std::vector<int32_t>& prompt_ids, int max_tokens,
                              const SamplingParams& sampling, OnText on_text, OnTool on_tool) {
        std::lock_guard<std::mutex> lock(gen_mutex_);
        bool in_tool = false;
        std::vector<int32_t> content_ids, tool_ids;
        size_t emitted = 0;
        int tool_count = 0;

        auto cb = [&](int32_t id) -> bool {
            if (id == im_end_id_ || id == eot_id_) return true;  // suppress stop tokens
            if (!in_tool && id == tool_call_start_) { in_tool = true; tool_ids.clear(); return true; }
            if (in_tool) {
                if (id == tool_call_end_) {
                    on_tool(tool_call_from_hermes(tokenizer_->decode(tool_ids)));
                    tool_ids.clear();
                    in_tool = false;
                    ++tool_count;
                } else {
                    tool_ids.push_back(id);
                }
                return true;
            }
            content_ids.push_back(id);
            std::string full = tokenizer_->decode(content_ids);
            size_t complete = utf8_complete_len(full);
            if (complete > emitted) {
                on_text(full.substr(emitted, complete - emitted));
                emitted = complete;
            }
            return true;
        };

        ir::NoGradScope no_grad;
        auto ids = decode(prompt_ids, max_tokens, cb, sampling);
        return {(int)prompt_ids.size(), (int)ids.size(), tool_count, (int)ids.size() >= max_tokens};
    }

    // Build SamplingParams from optional request fields. Missing temperature => greedy.
    static SamplingParams make_sampling(std::optional<double> temperature,
                                        std::optional<double> top_p,
                                        std::optional<int> top_k,
                                        std::optional<uint64_t> seed) {
        SamplingParams sp;
        sp.temperature = (float)temperature.value_or(0.0);
        sp.top_p = (float)top_p.value_or(1.0);
        sp.top_k = top_k.value_or(0);
        sp.seed = seed.value_or(0);
        return sp;
    }

    // ---- OpenAI chat-completions surface ----

    ChatCompletionResponse chat_complete(const ChatCompletionRequest& req) {
        ChatCompletionResponse resp;
        resp.id = generate_completion_id();
        resp.created = current_timestamp();
        resp.model = model_name_;
        resp.system_fingerprint = "cppgrad";

        auto r = generate_text(build_prompt(req), req.max_tokens.value_or(512),
                               make_sampling(req.temperature, req.top_p, req.top_k, req.seed));

        Choice choice;
        choice.index = 0;
        if (parse_hermes_tool_calls(r.text, choice.tool_calls, choice.content)) {
            choice.finish_reason = "tool_calls";
        } else {
            choice.content = r.text;
            choice.finish_reason = r.hit_length ? "length" : "stop";
        }
        resp.choices.push_back(std::move(choice));
        resp.usage.prompt_tokens = r.prompt_tokens;
        resp.usage.completion_tokens = r.completion_tokens;
        resp.usage.total_tokens = r.prompt_tokens + r.completion_tokens;
        return resp;
    }

    // `callback` receives complete SSE frames ("data: {...}\n\n"); return false to abort.
    void chat_complete_stream(const ChatCompletionRequest& req,
                              std::function<bool(const std::string&)> callback) {
        std::string request_id = generate_completion_id();
        int64_t created = current_timestamp();
        auto emit = [&](const DeltaChoice& dc) {
            ChatCompletionChunk chunk;
            chunk.id = request_id;
            chunk.created = created;
            chunk.model = model_name_;
            chunk.choices = {dc};
            callback("data: " + nlohmann::json(chunk).dump() + "\n\n");
        };

        { DeltaChoice dc; dc.role = "assistant"; emit(dc); }

        auto r = stream_generate(
            build_prompt(req), req.max_tokens.value_or(512),
            make_sampling(req.temperature, req.top_p, req.top_k, req.seed),
            [&](const std::string& text) { DeltaChoice dc; dc.content = text; emit(dc); },
            [&](const ToolCall& tc) { DeltaChoice dc; dc.tool_calls = {tc}; emit(dc); });

        DeltaChoice last;
        last.finish_reason = r.tool_calls > 0 ? "tool_calls" : (r.hit_length ? "length" : "stop");
        emit(last);
        callback("data: [DONE]\n\n");
    }

private:
    // Pick the decode path: an explicit draft model (generateSpeculative) takes precedence; else the
    // checkpoint's MTP module (generateSpeculativeMTP, no second model); else plain decode. Each
    // speculative path itself falls back to plain decode for non-greedy sampling or n_draft<2.
    std::vector<int32_t> decode(const std::vector<int32_t>& prompt_ids, int max_tokens,
                                Qwen3Model::TokenCallback cb, const SamplingParams& sampling) {
        // Speculative paths also reuse the persistent prefix cache (spec_session_), unless opted out.
        Qwen3Model::SpecCacheState* spec = std::getenv("CPPGRAD_NO_PREFIX_CACHE") ? nullptr : &spec_session_;
        if (draft_model_ && n_draft_ >= 2)
            return model_->generateSpeculative(prompt_ids, max_tokens, std::move(cb), stop_tokens_,
                                               *draft_model_, n_draft_, sampling, spec);
        if (mtp_enabled_ && model_->has_mtp() && mtp_n_draft_ >= 2)
            return model_->generateSpeculativeMTP(prompt_ids, max_tokens, std::move(cb), stop_tokens_,
                                                  mtp_n_draft_, sampling, /*accept_out=*/nullptr, spec);
        if (std::getenv("CPPGRAD_NO_PREFIX_CACHE"))
            return model_->generateStreaming(prompt_ids, max_tokens, std::move(cb), stop_tokens_, sampling);

        // Prefix-cache path: persist the KV / recurrent-state cache across requests and reuse the
        // longest shared token prefix (Claude Code resends a fixed system+tools prompt every turn, so
        // after the first turn most of the prompt is prefilled = 0). Model-agnostic driver.
        std::unordered_set<int32_t> stop(stop_tokens_.begin(), stop_tokens_.end());
        auto gen = nn::llm::generate_with_prefix_cache(
            *model_, session_, prompt_ids, max_tokens,
            [&](int32_t id) { return cb(id); }, stop, sampling);
        std::fprintf(stderr, "[prefix-cache] reused %zu / %zu prompt tokens (prefilled %zu)\n",
                     session_.reused, prompt_ids.size(), session_.prefilled);
        // Persist the warm cache once per run (the first request pays the cold prefill; capturing it
        // means restarts reload it). Saving every request would re-write the whole KV each turn.
        if (!kv_cache_file_.empty() && !kv_cache_saved_ && !session_.tokens.empty()) {
            if (nn::llm::save_prefix_cache(*model_, session_, kv_cache_file_)) {
                kv_cache_saved_ = true;
                std::fprintf(stderr, "[prefix-cache] saved warm cache (%zu tokens) to %s\n",
                             session_.tokens.size(), kv_cache_file_.c_str());
            }
        }
        return gen;
    }

    // Persistent decode session for prefix caching (one conversation at a time; the gen mutex
    // serializes requests). capacity_hint sizes the preallocated KV cache so reuse survives the
    // conversation growing -- set it to the max context you expect (CPPGRAD_KV_CAPACITY, default 0 =
    // size on demand, which loses reuse when the conversation outgrows the last allocation).
    nn::llm::PrefixCacheSession session_;       // plain decode path
    Qwen3Model::SpecCacheState  spec_session_;  // speculative / MTP decode path
    std::string kv_cache_file_;                 // CPPGRAD_KV_CACHE_FILE: persist the warm cache to disk
    bool        kv_cache_saved_ = false;        // saved this run? (save once; reload on restart)

    std::unique_ptr<Qwen3Model> model_;
    std::unique_ptr<Qwen3Model> draft_model_;   // optional draft for speculative decoding
    int n_draft_ = 0;                           // draft-model speculation window; <2 disables
    bool mtp_enabled_ = false;                  // use the checkpoint's MTP module for self-speculation
    int mtp_n_draft_ = 4;                       // MTP speculation window
    std::unique_ptr<BPETokenizer> tokenizer_;
    Qwen3ChatTemplate template_;
    std::optional<Qwen3Config> cfg_;
    backend::DeviceType device_type_ = backend::DeviceType::CPU;
    bool loaded_ = false;
    std::string model_name_;

    int im_end_id_ = -1, eot_id_ = -1, tool_call_start_ = -1, tool_call_end_ = -1;
    std::vector<int32_t> stop_tokens_;  // non-empty (always contains <|im_end|>)
    std::mutex gen_mutex_;              // serializes model use across HTTP worker threads
};

} // namespace server
} // namespace cppgrad
