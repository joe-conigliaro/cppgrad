// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// Qwen3 chat template formatter (Hermes-style tool calling).
//
// Renders OpenAI-style chat messages + tool definitions into the exact prompt
// format Qwen3 / Qwen3.5 / Qwen3.6 were trained on:
//
//   <|im_start|>system
//   {system}
//
//   # Tools
//   ...<tools>{json}</tools>...
//   <tool_call>{"name":...,"arguments":...}</tool_call> ...<|im_end|>
//   <|im_start|>user
//   {content}<|im_end|>
//   <|im_start|>assistant
//
// All the angle-bracket tags (<|im_start|>, <tool_call>, ...) are registered
// *special tokens* in the tokenizer, so BPETokenizer::encode() splits them out
// to their single token IDs automatically -- we just build the plain string and
// encode it once. No hardcoded token IDs live here.
//

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "cppgrad/io/tokenizer.h"

namespace cppgrad::io {

// A single chat message (role + content + optional tool-call metadata).
struct ChatMessage {
    std::string role; // "system" | "user" | "assistant" | "tool"
    std::string content;
    std::string tool_call_id; // for role == "tool"
    struct ToolCall {
        std::string id;
        std::string name;
        nlohmann::json arguments; // parsed object (not a JSON string)
    };
    std::vector<ToolCall> tool_calls; // for role == "assistant"
};

// A tool definition (OpenAI-compatible function schema).
struct ChatTool {
    std::string type = "function";
    struct Function {
        std::string name;
        std::string description;
        nlohmann::json parameters; // JSON Schema object
    } function;
};

// Qwen3 chat-template applier. Stateless; all token IDs are resolved through the
// tokenizer at encode time, so the same instance works for Qwen3 / 3.5 / 3.6.
class Qwen3ChatTemplate {
  public:
    // Build the full prompt string for a conversation. `add_generation_prompt`
    // appends the trailing `<|im_start|>assistant\n` that primes generation.
    std::string apply(const std::vector<ChatMessage> &messages, const std::vector<ChatTool> &tools = {},
                      bool add_generation_prompt = true) const {
        std::ostringstream oss;

        // Find an explicit system message (if any); Qwen3 adds no default system.
        const ChatMessage *system_msg = nullptr;
        for (const auto &m : messages) {
            if (m.role == "system") {
                system_msg = &m;
                break;
            }
        }

        // --- system turn (emitted when there is a system message or tools) ---
        if (system_msg || !tools.empty()) {
            oss << "<|im_start|>system\n";
            if (system_msg)
                oss << system_msg->content;
            if (!tools.empty()) {
                if (system_msg && !system_msg->content.empty())
                    oss << "\n\n";
                oss << "# Tools\n\n"
                       "You may call one or more functions to assist with the user query.\n\n"
                       "You are provided with function signatures within <tools></tools> XML tags:\n"
                       "<tools>";
                for (const auto &t : tools) {
                    nlohmann::json fn = {
                        {"type", t.type},
                        {"function",
                         {
                             {"name", t.function.name},
                             {"description", t.function.description},
                             {"parameters", t.function.parameters},
                         }},
                    };
                    oss << "\n" << fn.dump();
                }
                oss << "\n</tools>\n\n"
                       "For each function call, return a json object with function name and "
                       "arguments within <tool_call></tool_call> XML tags:\n"
                       "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                       "</tool_call>";
            }
            oss << "<|im_end|>\n";
        }

        // --- conversation turns ---
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto &msg = messages[i];
            if (msg.role == "system")
                continue; // already rendered above

            if (msg.role == "tool") {
                // Group consecutive tool results into a single user turn, each
                // wrapped in <tool_response></tool_response>.
                oss << "<|im_start|>user";
                size_t j = i;
                while (j < messages.size() && messages[j].role == "tool") {
                    oss << "\n<tool_response>\n" << messages[j].content << "\n</tool_response>";
                    ++j;
                }
                oss << "<|im_end|>\n";
                i = j - 1;
                continue;
            }

            oss << "<|im_start|>" << msg.role << "\n";
            if (msg.role == "assistant" && !msg.tool_calls.empty()) {
                if (!msg.content.empty())
                    oss << msg.content << "\n";
                for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
                    const auto &tc = msg.tool_calls[k];
                    nlohmann::json call = {{"name", tc.name}, {"arguments", tc.arguments}};
                    oss << "<tool_call>\n" << call.dump() << "\n</tool_call>";
                    if (k + 1 < msg.tool_calls.size())
                        oss << "\n";
                }
            } else {
                oss << msg.content;
            }
            oss << "<|im_end|>\n";
        }

        if (add_generation_prompt)
            oss << "<|im_start|>assistant\n";
        return oss.str();
    }

    // Apply the template and return token IDs directly. The tokenizer splits the
    // registered special tokens (<|im_start|>, <tool_call>, ...) out to their own
    // IDs, so a single encode() of the rendered string yields the right sequence.
    std::vector<int32_t> apply_tokens(const std::vector<ChatMessage> &messages, const std::vector<ChatTool> &tools,
                                      BPETokenizer &tokenizer, bool add_generation_prompt = true) const {
        return tokenizer.encode(apply(messages, tools, add_generation_prompt));
    }
};

} // namespace cppgrad::io
