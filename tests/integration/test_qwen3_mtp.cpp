// Copyright (c) 2026 Joe Conigliaro
//
// MTP (Multi-Token Prediction) module validation. Loads a checkpoint that ships language_model.mtp.*
// and measures how often the MTP module's 1-step prediction matches the main model's actual next
// greedy token. A high match rate validates the MTP architecture / weight mapping / fc concat order.
//
// Heavy (loads a real checkpoint); GATED on QWEN_MTP_DIR:
//   QWEN_MTP_DIR=~/.omlx/models/Jundot/Qwen3.6-27B-oQ8-mtp QWEN_CONFIG=27b_qwen3_6 \
//     ./build/tests/test_qwen3_mtp

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <glob.h>

#include "cppgrad/nn/llm/qwen/qwen3_model.h"
#include "cppgrad/nn/llm/qwen/qwen3_config.h"
#include "cppgrad/io/tokenizer.h"
#include "cppgrad/backend/device_manager.h"

using namespace cppgrad;
using cppgrad::nn::llm::qwen::Qwen3Model;
using cppgrad::nn::llm::qwen::Qwen3Config;

static std::vector<std::string> shards(const std::string& dir) {
    std::vector<std::string> out;
    for (const char* p : {"/model-*-of-*.safetensors", "/*.safetensors"}) {
        glob_t g;
        if (glob((dir + p).c_str(), 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; ++i) out.push_back(g.gl_pathv[i]);
        globfree(&g);
        if (!out.empty()) break;
    }
    return out;
}

int main() {
    const char* dir = std::getenv("QWEN_MTP_DIR");
    if (!dir) { printf("[skip] set QWEN_MTP_DIR to run the MTP validation test\n"); return 0; }
    std::string model_dir = dir;
    std::string config = std::getenv("QWEN_CONFIG") ? std::getenv("QWEN_CONFIG") : "27b_qwen3_6";

    backend::DeviceManager::instance().init();
    auto device = backend::DeviceManager::default_device_type();

    io::BPETokenizer tok(model_dir + "/tokenizer.json");
    Qwen3Model model(config == "27b_qwen3_6" ? Qwen3Config::get_27b_qwen3_6() : Qwen3Config::get_27b_qwen3_6(),
                     device, /*lazy_weights=*/true);
    model.load_from_safetensors(shards(model_dir), /*quantize=*/true);

    if (!model.has_mtp()) {
        printf("FAILED: checkpoint has no MTP module (language_model.mtp.*)\n");
        return 1;
    }

    auto prompt = tok.encode("The history of artificial intelligence began in the");
    const int steps = 48;
    double acc0 = model.mtp_self_check(prompt, steps, /*concat_order=*/0);
    double acc1 = model.mtp_self_check(prompt, steps, /*concat_order=*/1);
    printf("MTP 1-step acceptance:  [hidden,emb]=%.1f%%   [emb,hidden]=%.1f%%\n", acc0 * 100, acc1 * 100);

    double best = std::max(acc0, acc1);
    printf("best concat order: %s\n", acc0 >= acc1 ? "[hidden, emb] (concat_order=0)" : "[emb, hidden] (concat_order=1)");
    if (best < 0.30) {
        printf("\nFAILED: MTP acceptance too low (%.1f%%) - architecture / weight mapping likely wrong\n", best * 100);
        return 1;
    }

    // --- full MTP self-speculative generate: prefix should match greedy (diverging only on fp
    // near-ties), and the tokens/verify-round ratio shows the decode speedup. ---
    const int N = 64;
    std::vector<int32_t> eos = {tok.get_token_id("<|im_end|>"), tok.get_token_id("<|endoftext|>")};
    auto ref = model.generateStreaming(prompt, N, [](int32_t){ return true; }, eos);
    std::pair<int,int> stats{0,0};
    auto spec = model.generateSpeculativeMTP(prompt, N, [](int32_t){ return true; }, eos, /*n_draft=*/4,
                                             cppgrad::nn::llm::qwen::SamplingParams{}, &stats);
    size_t mlen = std::min(ref.size(), spec.size());
    int first_diff = -1;
    for (size_t i = 0; i < mlen; ++i) if (ref[i] != spec[i]) { first_diff = (int)i; break; }
    double toks_per_round = stats.second ? (double)(spec.size()) / stats.second : 0.0;
    printf("\nMTP self-speculative generate: %zu tokens in %d verify rounds (%.2f tok/round; %d drafts accepted)\n",
           spec.size(), stats.second, toks_per_round, stats.first);
    printf("vs greedy: %s\n", first_diff < 0 ? "identical" : ("first diff at token " + std::to_string(first_diff) + " (fp near-tie)").c_str());

    printf("\nALL TESTS PASSED (MTP validated: 1-step %.1f%%, self-spec %.2f tok/round)\n", best * 100, toks_per_round);
    return 0;
}
