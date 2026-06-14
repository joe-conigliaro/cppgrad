// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Qwen3/Qwen3.5/3.6 inference example - loads safetensors weights and runs greedy decoding
//
// Usage:
//   ./qwen3_inference --model <safetensors_path> [--prompt "Hello"] [--max-tokens 50] [--config 0.6b|1.5b|3b|4b|7b|27b_qwen3_6]
//   ./qwen3_inference --model-dir <directory_with_sharded_files> [--config 27b_qwen3_6]
//
// Without --model: runs a dry run (creates model with random weights, runs one forward pass)

#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <fstream>
#include <glob.h>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/io/tokenizer.h"

// Return the list of safetensors files for a model path: the file itself if it ends in
// .safetensors, otherwise all sharded files in the directory (model-*-of-*.safetensors,
// falling back to *.safetensors).
static std::vector<std::string> gather_shards(const std::string& path) {
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
    if (out.empty()) out.push_back(path);  // let the loader report a clear error
    return out;
}

// Derive <dir>/tokenizer.json from a model file/dir path.
static std::string derive_tokenizer_path(const std::string& model_path) {
    if (model_path.empty()) return "";
    std::string dir = model_path;
    auto slash = dir.find_last_of('/');
    // If it points at a .safetensors file, take its directory.
    if (dir.size() > 12 && dir.substr(dir.size() - 12) == ".safetensors" && slash != std::string::npos)
        dir = dir.substr(0, slash);
    return dir + "/tokenizer.json";
}

struct Args {
    std::string model_path;
    std::string prompt = "Hello";
    int32_t max_tokens = 20;
    std::string config_name = "0.6b";
    std::string tokenizer_path;
    bool dry_run = false;
    bool quant = false;   // keep MLX weights 8-bit (quantized_matmul) instead of dequantizing to bf16
};

static Args parse_args(int argc, char** argv) {
    Args args;
    static struct option long_opts[] = {
        {"model",       required_argument, 0, 'm'},
        {"prompt",      required_argument, 0, 'p'},
        {"max-tokens",  required_argument, 0, 'n'},
        {"config",      required_argument, 0, 'c'},
        {"tokenizer",   required_argument, 0, 't'},
        {"quant",       no_argument,       0, 'q'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "m:p:n:c:t:qh", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'm': args.model_path = optarg; break;
            case 'p': args.prompt = optarg; break;
            case 'n': args.max_tokens = std::stoi(optarg); break;
            case 'c': args.config_name = optarg; break;
            case 'q': args.quant = true; break;
            case 't': args.tokenizer_path = optarg; break;
            case 'h':
                std::cout << "Usage: " << argv[0]
                          << " --model <path> [--prompt <text>] [--max-tokens N] [--config <name>]\n"
                          << "Without --model, runs a dry run with random weights.\n";
                exit(0);
        }
    }
    args.dry_run = args.model_path.empty();
    return args;
}

// Simple tokenization for testing: map ASCII chars to token ids.
// This is NOT a real tokenizer - just for verifying the forward pass works.
// For real inference, use the HuggingFace tokenizer or a cpp implementation.
static std::vector<int32_t> simple_tokenize(const std::string& text, int32_t vocab_size, int32_t bos_token) {
    std::vector<int32_t> ids;
    ids.push_back(bos_token);
    for (char c : text) {
        // Map printable ASCII to token ids starting from 1
        int32_t id = static_cast<int32_t>(c) + 1;
        if (id < vocab_size) ids.push_back(id);
    }
    return ids;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    // Select config
    cppgrad::nn::llm::qwen::Qwen3Config config;
    if (args.config_name == "0.6b")         config = cppgrad::nn::llm::qwen::Qwen3Config::get_0_6b();
    else if (args.config_name == "1.5b")    config = cppgrad::nn::llm::qwen::Qwen3Config::get_1_5b();
    else if (args.config_name == "3b")      config = cppgrad::nn::llm::qwen::Qwen3Config::get_3b();
    else if (args.config_name == "4b")      config = cppgrad::nn::llm::qwen::Qwen3Config::get_4b();
    else if (args.config_name == "7b")      config = cppgrad::nn::llm::qwen::Qwen3Config::get_7b();
    else if (args.config_name == "27b_qwen3_6") config = cppgrad::nn::llm::qwen::Qwen3Config::get_27b_qwen3_6();
    else {
        std::cerr << "Unknown config: " << args.config_name << "\n";
        return 1;
    }

    bool is_qwen3_5 = config.is_qwen3_5();
    int32_t bos_token = is_qwen3_5 ? 248044 : 151644;

    std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Config: " << args.config_name
              << " hidden=" << config.hidden_size
              << " layers=" << config.num_hidden_layers
              << " heads=" << config.num_attention_heads
              << " kv_heads=" << config.num_key_value_heads
              << " head_dim=" << config.head_dim
              << "\n";

    cppgrad::backend::DeviceManager::instance().init();
    auto device_type = cppgrad::backend::DeviceManager::default_device_type();
    std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Device: " << cppgrad::backend::to_string(device_type) << "\n";

    // Create model
    std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Creating model...\n";
    // Lazy weights when loading a checkpoint (avoids pre-allocating the full fp32 weight set).
    auto model = cppgrad::nn::llm::qwen::Qwen3Model(config, device_type, /*lazy_weights=*/!args.dry_run);

    // Load weights (if model path provided). Accepts a single .safetensors file or a directory
    // of sharded files (model-*-of-*.safetensors).
    if (!args.dry_run) {
        std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Loading weights from: " << args.model_path << "\n";
        std::vector<std::string> shards = gather_shards(args.model_path);
        auto t0 = std::chrono::steady_clock::now();
        model.load_from_safetensors(shards, args.quant);
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Weights loaded in " << ms << "ms\n";
    } else {
        std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] DRY RUN - using random weights\n";
    }

    // Tokenize the prompt with the real byte-level BPE tokenizer when a tokenizer.json is
    // available (from --tokenizer or derived from the model dir); else fall back to the stub.
    std::string tok_path = !args.tokenizer_path.empty() ? args.tokenizer_path
                                                        : derive_tokenizer_path(args.model_path);
    std::unique_ptr<cppgrad::io::BPETokenizer> tok;
    {
        std::ifstream tf(tok_path);
        if (tf.good()) {
            try { tok.reset(new cppgrad::io::BPETokenizer(tok_path)); }
            catch (const std::exception& e) { std::cerr << "[Qwen3] tokenizer load failed: " << e.what() << "\n"; }
        }
    }

    std::vector<int32_t> input_ids = tok ? tok->encode(args.prompt)
                                         : simple_tokenize(args.prompt, config.vocab_size, bos_token);
    std::cout << "[Qwen3] Prompt: \"" << args.prompt << "\" -> " << input_ids.size()
              << (tok ? " tokens (BPE)\n" : " tokens (ascii stub)\n");

    // Generate
    std::cout << "[" << (is_qwen3_5 ? "Qwen3.5/3.6" : "Qwen3") << "] Generating " << args.max_tokens << " tokens...\n";
    auto t0 = std::chrono::steady_clock::now();
    auto generated = model.generate(input_ids, args.max_tokens);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "[Qwen3] Generated " << generated.size() << " tokens in " << ms << "ms\n";
    if (tok) {
        std::cout << "[Qwen3] Output: " << tok->decode(generated) << "\n";
    } else {
        std::cout << "[Qwen3] Token ids:";
        for (auto id : generated) std::cout << " " << id;
        std::cout << "\n[Qwen3] (no tokenizer.json found - pass --tokenizer to decode to text)\n";
    }

    return 0;
}
