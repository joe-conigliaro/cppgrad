#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cppgrad {
namespace nn {
namespace llm {
namespace qwen {

enum class LayerType {
    FULL_ATTENTION,
    LINEAR_ATTENTION
};

inline std::string to_string(LayerType t) {
    switch (t) {
        case LayerType::FULL_ATTENTION:     return "full_attention";
        case LayerType::LINEAR_ATTENTION:   return "linear_attention";
    }
    return "unknown";
}

inline LayerType parse_layer_type(const std::string& s) {
    if (s == "full_attention")   return LayerType::FULL_ATTENTION;
    if (s == "linear_attention") return LayerType::LINEAR_ATTENTION;
    throw std::runtime_error("unknown layer type: " + s);
}

struct Qwen3Config {
    // Core
    int32_t hidden_size;
    int32_t num_hidden_layers;
    int32_t intermediate_size;
    int32_t vocab_size;
    int32_t max_position_embeddings;

    // Full attention
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t head_dim;

    // Linear attention (Qwen3.5+)
    int32_t linear_key_head_dim;
    int32_t linear_num_key_heads;
    int32_t linear_value_head_dim;
    int32_t linear_num_value_heads;
    int32_t linear_conv_kernel_dim;

    // Norm / init
    double rms_norm_eps;
    double rope_theta;

    // RoPE (Qwen3.5+)
    bool mrope_interleaved;
    std::vector<int32_t> mrope_section;  // e.g., {11, 11, 10}
    double partial_rotary_factor;

    // Attention output gate (Qwen3.5+)
    bool attn_output_gate;
    std::string output_gate_type;  // "swish"

    // Layer routing (Qwen3.5+)
    std::vector<LayerType> layer_types;
    int32_t full_attention_interval;  // 0 = all full attention (Qwen3), 4 = every 4th layer is full

    // Derived
    int32_t get_num_kv_head_repeats() const { return num_attention_heads / num_key_value_heads; }
    int32_t get_num_kv_head_groups() const { return num_key_value_heads; }

    bool is_qwen3_5() const { return !layer_types.empty(); }

    LayerType get_layer_type(int32_t layer_idx) const {
        if (!layer_types.empty()) {
            return layer_types[layer_idx];
        }
        // Qwen3: all full attention
        return LayerType::FULL_ATTENTION;
    }

    bool is_full_attention_layer(int32_t layer_idx) const {
        return get_layer_type(layer_idx) == LayerType::FULL_ATTENTION;
    }

    bool is_linear_attention_layer(int32_t layer_idx) const {
        return get_layer_type(layer_idx) == LayerType::LINEAR_ATTENTION;
    }

    // --- Qwen3 presets (all full attention) ---

    static Qwen3Config get_0_6b() {
        return Qwen3Config{
            896, 24, 4864, 151936, 131072,  // hidden, layers, intermediate, vocab, max_pos
            14, 2, 64,                       // attn_heads, kv_heads, head_dim
            0, 0, 0, 0, 0,                   // linear attention (unused)
            1e-6, 1000000,                   // rms_norm_eps, rope_theta
            false, {}, 1.0,                  // mrope settings
            false, "",                       // attn_output_gate
            {}, 0,                           // layer_types, full_attention_interval
        };
    }

    static Qwen3Config get_1_5b() {
        auto c = Qwen3Config{1536, 28, 8960, 151936, 131072, 12, 2, 128, 0, 0, 0, 0, 0, 1e-6, 1000000, false, {}, 1.0, false, "", {}, 0};
        return c;
    }

    static Qwen3Config get_3b() {
        auto c = Qwen3Config{2048, 36, 10752, 151936, 131072, 16, 2, 128, 0, 0, 0, 0, 0, 1e-6, 1000000, false, {}, 1.0, false, "", {}, 0};
        return c;
    }

    static Qwen3Config get_4b() {
        auto c = Qwen3Config{3072, 16, 16384, 151936, 131072, 16, 2, 192, 0, 0, 0, 0, 0, 1e-6, 1000000, false, {}, 1.0, false, "", {}, 0};
        return c;
    }

    static Qwen3Config get_7b() {
        auto c = Qwen3Config{3584, 28, 18432, 151936, 131072, 28, 4, 128, 0, 0, 0, 0, 0, 1e-6, 1000000, false, {}, 1.0, false, "", {}, 0};
        return c;
    }

    // --- Qwen3.5/3.6 preset ---

    static Qwen3Config get_27b_qwen3_6() {
        // Qwen3.6-27B pattern: 3 linear_attention + 1 full_attention, repeated
        std::vector<LayerType> layer_types;
        for (int i = 0; i < 64; ++i) {
            if (i % 4 == 3) layer_types.push_back(LayerType::FULL_ATTENTION);
            else layer_types.push_back(LayerType::LINEAR_ATTENTION);
        }
        return Qwen3Config{
            5120, 64, 17408, 248320, 262144,  // hidden, layers, intermediate, vocab, max_pos
            24, 4, 256,                        // attn_heads, kv_heads, head_dim
            128, 16, 128, 48, 4,              // linear: key_dim, key_heads, val_dim, val_heads, conv_kernel
            1e-6, 10000000,                    // rms_norm_eps, rope_theta
            true, {11, 11, 10}, 0.25,         // mrope_interleaved, section, partial_rotary_factor
            true, "swish",                     // attn_output_gate
            layer_types, 4,                    // layer_types, full_attention_interval
        };
    }
};

} // namespace qwen
} // namespace llm
} // namespace nn
} // namespace cppgrad
