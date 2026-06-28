// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// Anthropic Messages API (`POST /v1/messages`) layered on top of ChatServer.
//
// This is what Claude Code speaks, so implementing it natively lets Claude Code
// (or any Anthropic-SDK client) point ANTHROPIC_BASE_URL at this server directly,
// with no translation proxy. Requests/responses are translated to/from the same
// internal ChatMessage / ChatTool / Hermes tool-call representation the OpenAI
// surface uses, so generation is shared.
//
// Wire format reference:
//   request : { system, messages[content blocks], tools[input_schema], max_tokens, stream }
//   stream  : message_start -> (content_block_start/_delta/_stop)* -> message_delta -> message_stop
//

#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "cppgrad/server/chat_server.h"

namespace cppgrad::server::anthropic {

using cppgrad::io::ChatMessage;
using cppgrad::io::ChatTool;

// Parsed request, already lowered to the server's internal representation.
struct Request {
    std::string model;
    int max_tokens = 1024;
    bool stream = false;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::vector<ChatMessage> messages; // system folded in as a role=="system" message
    std::vector<ChatTool> tools;
};

// Flatten an Anthropic content field (string OR array of blocks) to plain text,
// concatenating any "text" blocks. Used for system + simple message content.
inline std::string flatten_text(const nlohmann::json &content) {
    if (content.is_string())
        return content.get<std::string>();
    std::string out;
    if (content.is_array())
        for (const auto &b : content)
            if (b.value("type", "") == "text")
                out += b.value("text", "");
    return out;
}

// Parse an Anthropic Messages request into the internal Request.
inline Request parse_request(const nlohmann::json &j) {
    Request r;
    r.model = j.value("model", "");
    r.max_tokens = j.value("max_tokens", 1024);
    r.stream = j.value("stream", false);
    if (j.contains("temperature") && !j["temperature"].is_null())
        r.temperature = j["temperature"].get<double>();
    if (j.contains("top_p") && !j["top_p"].is_null())
        r.top_p = j["top_p"].get<double>();
    if (j.contains("top_k") && !j["top_k"].is_null())
        r.top_k = j["top_k"].get<int>();

    // system: string or array of text blocks -> a single leading system message.
    if (j.contains("system")) {
        std::string sys = flatten_text(j["system"]);
        if (!sys.empty()) {
            ChatMessage m;
            m.role = "system";
            m.content = sys;
            r.messages.push_back(std::move(m));
        }
    }

    if (j.contains("tools") && j["tools"].is_array()) {
        for (const auto &t : j["tools"]) {
            ChatTool ct;
            ct.function.name = t.value("name", "");
            ct.function.description = t.value("description", "");
            ct.function.parameters = t.value("input_schema", nlohmann::json::object());
            r.tools.push_back(std::move(ct));
        }
    }

    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto &m : j["messages"]) {
            std::string role = m.value("role", "");
            const auto &content = m.contains("content") ? m["content"] : nlohmann::json("");

            if (content.is_string()) {
                ChatMessage cm;
                cm.role = role;
                cm.content = content.get<std::string>();
                r.messages.push_back(std::move(cm));
                continue;
            }

            // Array of content blocks. Assistant: text + tool_use. User: text + tool_result.
            ChatMessage cm;
            cm.role = role;
            std::vector<ChatMessage> tool_results; // tool_result blocks become role=="tool" turns
            for (const auto &b : content) {
                std::string type = b.value("type", "");
                if (type == "text") {
                    cm.content += b.value("text", "");
                } else if (type == "tool_use") {
                    ChatMessage::ToolCall tc;
                    tc.id = b.value("id", "");
                    tc.name = b.value("name", "");
                    tc.arguments = b.contains("input") ? b["input"] : nlohmann::json::object();
                    cm.tool_calls.push_back(std::move(tc));
                } else if (type == "tool_result") {
                    ChatMessage tr2;
                    tr2.role = "tool";
                    tr2.tool_call_id = b.value("tool_use_id", "");
                    tr2.content = b.contains("content") ? flatten_text(b["content"]) : "";
                    tool_results.push_back(std::move(tr2));
                }
            }
            // Tool results render as their own (user-side) turns ahead of any text.
            for (auto &tr2 : tool_results)
                r.messages.push_back(std::move(tr2));
            if (!cm.content.empty() || !cm.tool_calls.empty())
                r.messages.push_back(std::move(cm));
        }
    }
    return r;
}

inline std::string message_id() { return "msg_" + random_suffix(); }
inline std::string tool_use_id() { return "toolu_" + random_suffix(); }

// ---- non-streaming: build a full Anthropic message response ----
inline nlohmann::json handle_messages(ChatServer &server, const Request &req) {
    auto sampling = ChatServer::make_sampling(req.temperature, req.top_p, req.top_k, std::nullopt);
    auto r = server.generate_text(server.build_prompt(req.messages, req.tools), req.max_tokens, sampling);

    std::vector<ToolCall> calls;
    std::string clean;
    bool has_tools = parse_hermes_tool_calls(r.text, calls, clean);

    nlohmann::json content = nlohmann::json::array();
    if (!clean.empty())
        content.push_back({{"type", "text"}, {"text", clean}});
    for (const auto &tc : calls) {
        nlohmann::json input;
        try {
            input = nlohmann::json::parse(tc.function.arguments);
        } catch (...) {
            input = nlohmann::json::object();
        }
        content.push_back({{"type", "tool_use"}, {"id", tool_use_id()}, {"name", tc.function.name}, {"input", input}});
    }
    if (content.empty())
        content.push_back({{"type", "text"}, {"text", ""}});

    std::string stop_reason = has_tools ? "tool_use" : (r.hit_length ? "max_tokens" : "end_turn");
    return {
        {"id", message_id()},
        {"type", "message"},
        {"role", "assistant"},
        {"model", server.model_name()},
        {"content", content},
        {"stop_reason", stop_reason},
        {"stop_sequence", nullptr},
        {"usage", {{"input_tokens", r.prompt_tokens}, {"output_tokens", r.completion_tokens}}},
    };
}

// ---- streaming: emit Anthropic SSE events ----
// `sse` receives complete frames ("event: <type>\ndata: {...}\n\n"); return false to abort.
inline void handle_messages_stream(ChatServer &server, const Request &req,
                                   std::function<bool(const std::string &)> sse) {
    auto prompt_ids = server.build_prompt(req.messages, req.tools);
    std::string msg_id = message_id();

    auto event = [&](const char *type, const nlohmann::json &data) {
        sse(std::string("event: ") + type + "\ndata: " + data.dump() + "\n\n");
    };

    event("message_start", {{"type", "message_start"},
                            {"message",
                             {{"id", msg_id},
                              {"type", "message"},
                              {"role", "assistant"},
                              {"model", server.model_name()},
                              {"content", nlohmann::json::array()},
                              {"stop_reason", nullptr},
                              {"stop_sequence", nullptr},
                              {"usage", {{"input_tokens", (int)prompt_ids.size()}, {"output_tokens", 0}}}}}});

    // Content blocks are opened lazily; only one is open at a time (text accumulates,
    // tool calls are emitted atomically once their <tool_call> block closes).
    int next_index = 0, text_index = -1;
    bool text_open = false;
    auto open_text = [&]() {
        if (text_open)
            return;
        text_index = next_index++;
        event("content_block_start", {{"type", "content_block_start"},
                                      {"index", text_index},
                                      {"content_block", {{"type", "text"}, {"text", ""}}}});
        text_open = true;
    };
    auto close_text = [&]() {
        if (!text_open)
            return;
        event("content_block_stop", {{"type", "content_block_stop"}, {"index", text_index}});
        text_open = false;
    };

    auto sampling = ChatServer::make_sampling(req.temperature, req.top_p, req.top_k, std::nullopt);
    auto r = server.stream_generate(
        prompt_ids, req.max_tokens, sampling,
        [&](const std::string &text) {
            open_text();
            event("content_block_delta", {{"type", "content_block_delta"},
                                          {"index", text_index},
                                          {"delta", {{"type", "text_delta"}, {"text", text}}}});
        },
        [&](const ToolCall &tc) {
            close_text();
            int idx = next_index++;
            nlohmann::json input;
            try {
                input = nlohmann::json::parse(tc.function.arguments);
            } catch (...) {
                input = nlohmann::json::object();
            }
            event("content_block_start", {{"type", "content_block_start"},
                                          {"index", idx},
                                          {"content_block",
                                           {{"type", "tool_use"},
                                            {"id", tool_use_id()},
                                            {"name", tc.function.name},
                                            {"input", nlohmann::json::object()}}}});
            event("content_block_delta", {{"type", "content_block_delta"},
                                          {"index", idx},
                                          {"delta", {{"type", "input_json_delta"}, {"partial_json", input.dump()}}}});
            event("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
        });

    close_text();
    std::string stop_reason = r.tool_calls > 0 ? "tool_use" : (r.hit_length ? "max_tokens" : "end_turn");
    event("message_delta", {{"type", "message_delta"},
                            {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                            {"usage", {{"output_tokens", r.completion_tokens}}}});
    event("message_stop", {{"type", "message_stop"}});
}

} // namespace cppgrad::server::anthropic
