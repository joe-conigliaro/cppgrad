// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// OpenAI-compatible chat-completions API types + (de)serialization.
//

#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <random>
#include <nlohmann/json.hpp>

namespace cppgrad {
namespace server {

// ---------------------------------------------------------------------------
// Shared sub-objects
// ---------------------------------------------------------------------------

// A tool/function call produced by the assistant.
struct ToolCall {
    std::string id;
    std::string type = "function";
    struct Function {
        std::string name;
        std::string arguments;  // JSON-encoded string (per OpenAI spec)
    } function;
};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

struct Message {
    std::string role;          // "system" | "user" | "assistant" | "tool"
    std::string content;
    std::string name;          // optional
    std::string tool_call_id;  // for role == "tool"
    std::vector<ToolCall> tool_calls;  // for role == "assistant"
};

struct FunctionDefinition {
    std::string name;
    std::string description;
    nlohmann::json parameters = nlohmann::json::object();  // JSON Schema
};

struct ToolDefinition {
    std::string type = "function";
    FunctionDefinition function;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<Message> messages;
    std::vector<ToolDefinition> tools;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<int> max_tokens;
    std::optional<bool> stream;
    std::optional<uint64_t> seed;
    std::vector<std::string> stop;  // 0+ stop strings
};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

struct UsageInfo {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

// A non-streaming choice (carries a full message).
struct Choice {
    int index = 0;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string finish_reason;  // "stop" | "length" | "tool_calls"
};

// A streaming delta choice.
struct DeltaChoice {
    int index = 0;
    std::optional<std::string> role;
    std::optional<std::string> content;
    std::vector<ToolCall> tool_calls;  // each carries its own .index via the array order
    std::optional<std::string> finish_reason;
};

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<Choice> choices;
    UsageInfo usage;
    std::string system_fingerprint;
};

struct ChatCompletionChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created = 0;
    std::string model;
    std::vector<DeltaChoice> choices;
};

struct ModelInfo {
    std::string id;
    std::string object = "model";
    int64_t created = 0;
    std::string owned_by = "cppgrad";
};

struct ModelList {
    std::string object = "list";
    std::vector<ModelInfo> data;
};

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

inline nlohmann::json tool_call_to_json(const ToolCall& tc, std::optional<int> index = std::nullopt) {
    nlohmann::json j = {
        {"id", tc.id},
        {"type", tc.type},
        {"function", {{"name", tc.function.name}, {"arguments", tc.function.arguments}}},
    };
    if (index) j["index"] = *index;
    return j;
}

// --- deserialization (request) ---

inline void from_json(const nlohmann::json& j, ToolCall& tc) {
    tc.id = j.value("id", "");
    tc.type = j.value("type", "function");
    if (j.contains("function")) {
        const auto& fn = j["function"];
        tc.function.name = fn.value("name", "");
        if (fn.contains("arguments")) {
            tc.function.arguments = fn["arguments"].is_string()
                ? fn["arguments"].get<std::string>()
                : fn["arguments"].dump();
        }
    }
}

inline void from_json(const nlohmann::json& j, Message& m) {
    m.role = j.value("role", "");
    if (j.contains("content") && !j["content"].is_null())
        m.content = j["content"].is_string() ? j["content"].get<std::string>() : j["content"].dump();
    m.name = j.value("name", "");
    m.tool_call_id = j.value("tool_call_id", "");
    if (j.contains("tool_calls") && j["tool_calls"].is_array())
        m.tool_calls = j["tool_calls"].get<std::vector<ToolCall>>();
}

inline void from_json(const nlohmann::json& j, ToolDefinition& t) {
    t.type = j.value("type", "function");
    if (j.contains("function")) {
        const auto& fn = j["function"];
        t.function.name = fn.value("name", "");
        t.function.description = fn.value("description", "");
        t.function.parameters = fn.value("parameters", nlohmann::json::object());
    }
}

inline void from_json(const nlohmann::json& j, ChatCompletionRequest& r) {
    r.model = j.value("model", "");
    r.messages = j.value("messages", std::vector<Message>{});
    if (j.contains("tools") && j["tools"].is_array())
        r.tools = j["tools"].get<std::vector<ToolDefinition>>();
    if (j.contains("temperature") && !j["temperature"].is_null()) r.temperature = j["temperature"].get<double>();
    if (j.contains("top_p") && !j["top_p"].is_null()) r.top_p = j["top_p"].get<double>();
    if (j.contains("top_k") && !j["top_k"].is_null()) r.top_k = j["top_k"].get<int>();
    if (j.contains("max_tokens") && !j["max_tokens"].is_null()) r.max_tokens = j["max_tokens"].get<int>();
    if (j.contains("stream") && !j["stream"].is_null()) r.stream = j["stream"].get<bool>();
    if (j.contains("seed") && !j["seed"].is_null()) r.seed = j["seed"].get<uint64_t>();
    if (j.contains("stop") && !j["stop"].is_null()) {
        if (j["stop"].is_string()) r.stop.push_back(j["stop"].get<std::string>());
        else if (j["stop"].is_array())
            for (const auto& s : j["stop"]) if (s.is_string()) r.stop.push_back(s.get<std::string>());
    }
}

// --- serialization (response) ---

inline void to_json(nlohmann::json& j, const UsageInfo& u) {
    j = {{"prompt_tokens", u.prompt_tokens},
         {"completion_tokens", u.completion_tokens},
         {"total_tokens", u.total_tokens}};
}

inline void to_json(nlohmann::json& j, const Choice& c) {
    nlohmann::json msg = {{"role", "assistant"}};
    msg["content"] = c.content.empty() ? nlohmann::json(nullptr) : nlohmann::json(c.content);
    if (!c.tool_calls.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& tc : c.tool_calls) arr.push_back(tool_call_to_json(tc));
        msg["tool_calls"] = arr;
    }
    j = {{"index", c.index}, {"message", msg}, {"finish_reason", c.finish_reason}};
}

inline void to_json(nlohmann::json& j, const DeltaChoice& c) {
    nlohmann::json delta = nlohmann::json::object();
    if (c.role) delta["role"] = *c.role;
    if (c.content) delta["content"] = *c.content;
    if (!c.tool_calls.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (size_t i = 0; i < c.tool_calls.size(); ++i)
            arr.push_back(tool_call_to_json(c.tool_calls[i], (int)i));
        delta["tool_calls"] = arr;
    }
    j = {{"index", c.index}, {"delta", delta}};
    j["finish_reason"] = c.finish_reason ? nlohmann::json(*c.finish_reason) : nlohmann::json(nullptr);
}

inline void to_json(nlohmann::json& j, const ChatCompletionResponse& r) {
    j = {{"id", r.id}, {"object", r.object}, {"created", r.created}, {"model", r.model},
         {"choices", r.choices}, {"usage", r.usage}, {"system_fingerprint", r.system_fingerprint}};
}

inline void to_json(nlohmann::json& j, const ChatCompletionChunk& c) {
    j = {{"id", c.id}, {"object", c.object}, {"created", c.created},
         {"model", c.model}, {"choices", c.choices}};
}

inline void to_json(nlohmann::json& j, const ModelInfo& m) {
    j = {{"id", m.id}, {"object", m.object}, {"created", m.created}, {"owned_by", m.owned_by}};
}

inline void to_json(nlohmann::json& j, const ModelList& m) {
    j = {{"object", m.object}, {"data", m.data}};
}

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------

inline int64_t current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

inline std::string random_suffix(size_t n = 24) {
    static const char* alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, 61);
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) s.push_back(alphabet[dis(gen)]);
    return s;
}

inline std::string generate_completion_id() { return "chatcmpl-" + random_suffix(); }
inline std::string generate_tool_call_id() { return "call_" + random_suffix(); }

} // namespace server
} // namespace cppgrad
