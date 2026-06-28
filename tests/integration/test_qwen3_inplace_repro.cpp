// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Minimal reproduction of the in-place KV-cache corruption (quantized decode at scale).
// Builds a small but deep all-full-attention Qwen model with MLX-affine-quantized weights, then
// compares cached decode with the in-place KV cache vs the (correct) concat cache. Small weights
// keep an Xcode GPU capture tractable; many layers reproduce the scale-dependent bug.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "tests/helpers.h"

using namespace cppgrad;
using cppgrad::nn::Linear;
using cppgrad::nn::llm::qwen::LayerType;
using cppgrad::nn::llm::qwen::Qwen3Block;
using cppgrad::nn::llm::qwen::Qwen3Config;
using cppgrad::nn::llm::qwen::Qwen3Model;

static utils::Ref<ir::Tensor> rand_tensor(const std::vector<size_t> &shape, float scale, uint32_t &seed,
                                          backend::DeviceType dev) {
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    std::vector<float> v(n);
    for (auto &f : v) {
        seed = seed * 1664525u + 1013904223u;
        f = (((seed >> 8) / float(1u << 24)) * 2.0f - 1.0f) * scale;
    }
    return ir::from_vector<float>(v, shape, dev);
}

// Quantize a dense Linear (weight [in,out]) to MLX-affine 8-bit in place. The quantized weight is
// W[out,in] = weight^T; per (out-row, group-of-64-along-in): scale=(max-min)/255, bias=min,
// code=round((w-bias)/scale); codes packed 4-per-u32 -> qweight[out, in/4]; scales/biases[out, in/64].
static void quantize_linear(Linear &ql, backend::DeviceType dev) {
    const int gs = 64;
    auto w = ql.weight->to_vector<float>(); // [in, out], row-major
    size_t in = ql.weight->shape()[0], out = ql.weight->shape()[1];
    if (in % gs != 0)
        throw std::runtime_error("quantize_linear: in dim must be multiple of 64");
    size_t G = in / gs, Kp = in / 4;
    std::vector<uint32_t> qw(out * Kp, 0);
    std::vector<float> sc(out * G), bs(out * G);
    for (size_t n = 0; n < out; ++n)
        for (size_t g = 0; g < G; ++g) {
            float mn = 1e30f, mx = -1e30f;
            for (size_t j = 0; j < (size_t)gs; ++j) {
                float val = w[(g * gs + j) * out + n];
                mn = std::min(mn, val);
                mx = std::max(mx, val);
            }
            float scale = (mx - mn) / 255.0f;
            if (scale < 1e-12f)
                scale = 1e-12f;
            sc[n * G + g] = scale;
            bs[n * G + g] = mn;
            for (size_t j = 0; j < (size_t)gs; ++j) {
                size_t k = g * gs + j;
                float val = w[k * out + n];
                int code = (int)std::lround((val - mn) / scale);
                code = std::max(0, std::min(255, code));
                qw[n * Kp + (k >> 2)] |= (uint32_t)code << (8 * (k & 3));
            }
        }
    ql.qweight = ir::from_vector<uint32_t>(qw, {out, Kp}, dev);
    ql.scales = ir::from_vector<float>(sc, {out, G}, dev);
    ql.biases = ir::from_vector<float>(bs, {out, G}, dev);
    ql.quantized = true;
    ql.params = ir::QuantParams{}; // MLX_AFFINE, 8-bit, group_size 64
}

int main() {
    try {
        TEST_HEADER("in-place KV cache repro (quantized, deep)");
        auto dev = backend::DeviceManager::default_device_type();

        const int N = 40; // decode tokens
        std::vector<int32_t> prompt = {3, 7, 1, 9, 2};

        for (int L : {4, 8}) {
            Qwen3Config cfg = Qwen3Config::get_27b();
            cfg.hidden_size = 5120; // matches 27B (isolates per-op size as trigger)
            cfg.num_hidden_layers = L;
            cfg.intermediate_size = 5120;
            cfg.vocab_size = 248320; // real vocab: tests total-bound-memory threshold
            cfg.num_attention_heads = 8;
            cfg.num_key_value_heads = 4; // matches 27B
            cfg.head_dim = 256;          // matches 27B
            // Hybrid like the 27B: every 4th layer (i%4==3) is full-attention, the rest linear.
            cfg.layer_types.resize((size_t)L);
            for (int i = 0; i < L; ++i)
                cfg.layer_types[i] = (i % 4 == 3) ? LayerType::FULL_ATTENTION : LayerType::LINEAR_ATTENTION;

            Qwen3Model model(cfg, dev);
            uint32_t seed = 12345u;
            model.embedding_weight = rand_tensor({(size_t)cfg.vocab_size, (size_t)cfg.hidden_size}, 0.5f, seed, dev);
            model.lm_head->weight = rand_tensor({(size_t)cfg.hidden_size, (size_t)cfg.vocab_size}, 0.5f, seed, dev);
            model.final_norm_weight = rand_tensor({(size_t)cfg.hidden_size}, 0.1f, seed, dev);

            for (int i = 0; i < L; ++i) {
                Qwen3Block *b = model.get_block((size_t)i);
                if (b->get_layer_type() == LayerType::FULL_ATTENTION) {
                    quantize_linear(*b->fa_q_proj, dev);
                    quantize_linear(*b->fa_k_proj, dev);
                    quantize_linear(*b->fa_v_proj, dev);
                    quantize_linear(*b->fa_o_proj, dev);
                    quantize_linear(*b->fa_ffn->gate_proj, dev);
                    quantize_linear(*b->fa_ffn->up_proj, dev);
                    quantize_linear(*b->fa_ffn->down_proj, dev);
                } else {
                    quantize_linear(*b->la_in_proj_qkv, dev);
                    quantize_linear(*b->la_in_proj_a, dev);
                    quantize_linear(*b->la_in_proj_b, dev);
                    quantize_linear(*b->la_in_proj_z, dev);
                    quantize_linear(*b->la_out_proj, dev);
                    quantize_linear(*b->la_ffn->gate_proj, dev);
                    quantize_linear(*b->la_ffn->up_proj, dev);
                    quantize_linear(*b->la_ffn->down_proj, dev);
                }
            }
            quantize_linear(*model.lm_head, dev);

            model.inplace_kv = false;
            auto ref = model.generate(prompt, N);
            model.inplace_kv = true;
            auto inp = model.generate(prompt, N);

            bool match = (ref == inp);
            size_t first_diff = 0;
            while (first_diff < ref.size() && first_diff < inp.size() && ref[first_diff] == inp[first_diff])
                ++first_diff;
            std::cout << "L=" << L << ": " << (match ? "MATCH" : "DIFFER")
                      << (match ? "" : " (first diff at token " + std::to_string(first_diff) + ")") << "\n";
            if (!match) {
                std::cout << "  concat   :";
                for (int t : ref)
                    std::cout << " " << t;
                std::cout << "\n";
                std::cout << "  in-place :";
                for (int t : inp)
                    std::cout << " " << t;
                std::cout << "\n";
            }
        }

        std::cout << "\n(repro harness - DIFFER at some L means the in-place bug reproduced)\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
