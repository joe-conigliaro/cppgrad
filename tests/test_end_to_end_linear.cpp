// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <iomanip>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/mlp.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/nn/activations.h"
#include "cppgrad/optim/adam.h"
#include "cppgrad/utils/rng.h"
#include "tests/helpers.h"

using namespace cppgrad;

static float compute_accuracy_pm1(const utils::Ref<ir::Tensor>& logits,
                                  const utils::Ref<ir::Tensor>& y_true_raw) {
    // Threshold raw logits at 0, labels in {-1, +1}
    auto pred_vec = logits->to_vector<float>();
    auto y_vec = y_true_raw->to_vector<float>();
    size_t n = y_vec.size();
    size_t correct = 0;
    for (size_t i = 0; i < n; ++i) {
        float pred_label = pred_vec[i] >= 0.0f ? 1.0f : -1.0f;
        if (pred_label == y_vec[i]) correct++;
    }
    return (float)correct / (float)n;
}

// Compute accuracy for BCE-with-logits (targets in {0,1})
static float compute_accuracy_01(const utils::Ref<ir::Tensor>& logits,
                                 const utils::Ref<ir::Tensor>& y_true_raw) {
    auto pred_vec = logits->to_vector<float>();
    auto y_vec = y_true_raw->to_vector<float>();
    size_t n = y_vec.size();
    size_t correct = 0;
    for (size_t i = 0; i < n; ++i) {
        float pred_label = pred_vec[i] >= 0.0f ? 1.0f : 0.0f; // threshold at 0 on logits
        if (pred_label == y_vec[i]) correct++;
    }
    return (float)correct / (float)n;
}

// Data generation: two blobs (separable)
static void make_blobs_pm1(int n_samples, float spread,
                           std::vector<float>& X, std::vector<float>& y_pm1) {
    // Class +1 centered at (1,1), Class -1 centered at (-1,-1)
    X.clear(); y_pm1.clear();
    X.reserve(n_samples * 2); y_pm1.reserve(n_samples);

    std::mt19937 gen = utils::global_rng();
    std::normal_distribution<float> noise(0.0f, spread);

    int half = n_samples / 2;
    for (int i = 0; i < half; ++i) {
        float x = 1.0f + noise(gen);
        float yy = 1.0f + noise(gen);
        X.push_back(x); X.push_back(yy);
        y_pm1.push_back(1.0f);
    }
    for (int i = 0; i < n_samples - half; ++i) {
        float x = -1.0f + noise(gen);
        float yy = -1.0f + noise(gen);
        X.push_back(x); X.push_back(yy);
        y_pm1.push_back(-1.0f);
    }
}

static void make_blobs_01(int n_samples, float spread,
                          std::vector<float>& X, std::vector<float>& y01) {
    // Same geometry, labels in {0,1}
    std::vector<float> y_pm1;
    make_blobs_pm1(n_samples, spread, X, y_pm1);
    y01.clear();
    y01.reserve(y_pm1.size());
    for (float v : y_pm1) y01.push_back(v > 0.0f ? 1.0f : 0.0f);
}

// Linear + Logistic loss (logistic_loss_pm1)
static void test_linear_logistic_end_to_end() {
    TEST_HEADER("End-to-end: Linear + Logistic (softplus pm1)");
    using nn::functional::logistic_loss_pm1;

    // Data
    int n_samples = 500;
    float spread = 0.25f;
    std::vector<float> Xv, yv;
    make_blobs_pm1(n_samples, spread, Xv, yv);
    auto xs = ir::from_vector(Xv, {(size_t)n_samples, 2});
    auto ys = ir::from_vector(yv, {(size_t)n_samples, 1});

    // Model: single linear layer
    cppgrad::nn::Linear linear(2, 1, true, cppgrad::nn::Init::XavierUniform);
    // Collect parameters for optimizer
    auto params = linear.direct_parameters();

    // Optimizer
    cppgrad::optim::Adam opt(params, 0.003f);

    // Training
    int epochs = 200;
    float loss0 = 0.0f, lossN = 0.0f;

    for (int e = 0; e < epochs; ++e) {
        auto logits = linear(xs);
        auto loss = logistic_loss_pm1(logits, ys); // mean(softplus(-y*z))
        if (e == 0) loss0 = loss->item<float>();

        opt.zero_grad();
        loss->backward();
        opt.step();

        if ((e+1) % 50 == 0) {
            float l = loss->item<float>();
            std::cout << "Epoch " << std::setw(3) << (e+1) << " loss=" << std::fixed << std::setprecision(6) << l << "\n";
        }
        if (e == epochs - 1) lossN = loss->item<float>();
    }

    // Metrics
    auto final_logits = linear(xs);
    float acc = compute_accuracy_pm1(final_logits, ys);
    std::cout << "Final loss=" << lossN << " (initial " << loss0 << "), accuracy=" << acc << "\n";

    // Assertions
    EXPECT_TRUE(lossN < loss0 * 0.5f, "Loss should drop >50%");
    EXPECT_TRUE(acc >= 0.95f, "Accuracy should be >= 95% on separable data");
    if (lossN < loss0 * 0.5f && acc >= 0.95f) {
        std::cout << "[PASS] End-to-end Linear + Logistic\n";
    } else {
        std::cout << "[WARN] End-to-end Linear + Logistic didn't meet thresholds\n";
    }
}

// MLP (Tanh) + Logistic loss (now using logistic_loss_pm1)
static void test_mlp_logistic_end_to_end() {
    TEST_HEADER("End-to-end: MLP (Tanh) + Logistic (softplus pm1)");
    using nn::functional::logistic_loss_pm1;

    // Data
    int n_samples = 500;
    float spread = 0.25f;
    std::vector<float> Xv, yv;
    make_blobs_pm1(n_samples, spread, Xv, yv);
    auto xs = ir::from_vector(Xv, {(size_t)n_samples, 2});
    auto ys = ir::from_vector(yv, {(size_t)n_samples, 1});

    // Model: MLP 2-16-1 with Tanh
    auto l1 = std::make_shared<cppgrad::nn::Linear>(2, 16, true, cppgrad::nn::Init::XavierUniform);
    auto a1 = std::make_shared<cppgrad::nn::Tanh>();
    auto l2 = std::make_shared<cppgrad::nn::Linear>(16, 1, true, cppgrad::nn::Init::XavierUniform);

    std::vector<std::shared_ptr<cppgrad::nn::Module>> modules = {l1, a1, l2};

    // Collect parameters
    std::vector<utils::Ref<ir::Tensor>> params;
    for (auto& m : modules) {
        auto ps = m->parameters();
        params.insert(params.end(), ps.begin(), ps.end());
    }

    // Optimizer
    cppgrad::optim::Adam opt(params, 0.003f);

    // Training
    int epochs = 200;
    float loss0 = 0.0f, lossN = 0.0f;

    auto forward = [&](const utils::Ref<ir::Tensor>& x) {
        auto cur = x;
        for (auto& m : modules) cur = (*m)(cur);
        return cur;
    };

    for (int e = 0; e < epochs; ++e) {
        auto logits = forward(xs);
        auto loss = logistic_loss_pm1(logits, ys);
        if (e == 0) loss0 = loss->item<float>();

        opt.zero_grad();
        loss->backward();
        opt.step();

        if ((e+1) % 50 == 0) {
            float l = loss->item<float>();
            std::cout << "Epoch " << std::setw(3) << (e+1) << " loss=" << std::fixed << std::setprecision(6) << l << "\n";
        }
        if (e == epochs - 1) lossN = loss->item<float>();
    }

    // Metrics
    auto final_logits = forward(xs);
    float acc = compute_accuracy_pm1(final_logits, ys);
    std::cout << "Final loss=" << lossN << " (initial " << loss0 << "), accuracy=" << acc << "\n";

    // Assertions
    EXPECT_TRUE(lossN < loss0 * 0.5f, "Loss should drop >50% (MLP)");
    EXPECT_TRUE(acc >= 0.97f, "Accuracy should be >= 97% (MLP)");
    if (lossN < loss0 * 0.5f && acc >= 0.97f) {
        std::cout << "[PASS] End-to-end MLP + Logistic\n";
    } else {
        std::cout << "[WARN] End-to-end MLP + Logistic didn't meet thresholds\n";
    }
}

// Linear + BCE-with-logits on {0,1} labels
static void test_linear_bce_with_logits_end_to_end() {
    TEST_HEADER("End-to-end: Linear + BCE-with-logits (0/1 targets)");
    using nn::functional::bce_with_logits;

    // Data with {0,1} labels
    int n_samples = 500;
    float spread = 0.25f;
    std::vector<float> Xv, y01;
    make_blobs_01(n_samples, spread, Xv, y01);
    auto xs = ir::from_vector(Xv, {(size_t)n_samples, 2});
    auto ys = ir::from_vector(y01, {(size_t)n_samples, 1});

    // Model
    cppgrad::nn::Linear linear(2, 1, true, cppgrad::nn::Init::XavierUniform);
    auto params = linear.direct_parameters();

    // Optimizer
    // eager / no batching
    // cppgrad::optim::Adam opt(params, 0.03f);
    // lazy / batched
    cppgrad::optim::Adam opt(params, 0.003f);

    // Training
    int epochs = 200;
    float loss0 = 0.0f, lossN = 0.0f;

    for (int e = 0; e < epochs; ++e) {
        auto logits = linear(xs);
        auto loss = bce_with_logits(logits, ys); // mean over samples
        if (e == 0) loss0 = loss->item<float>();

        opt.zero_grad();
        loss->backward();
        opt.step();

        if ((e+1) % 50 == 0) {
            float l = loss->item<float>();
            std::cout << "Epoch " << std::setw(3) << (e+1) << " loss=" << std::fixed << std::setprecision(6) << l << "\n";
        }
        if (e == epochs - 1) lossN = loss->item<float>();
    }

    // Metrics (threshold at 0 on logits)
    auto final_logits = linear(xs);
    float acc = compute_accuracy_01(final_logits, ys);
    std::cout << "Final loss=" << lossN << " (initial " << loss0 << "), accuracy=" << acc << "\n";

    // Assertions (slightly looser than pm1)
    EXPECT_TRUE(lossN < loss0 * 0.6f, "Loss should drop >40% for BCE 0/1");
    EXPECT_TRUE(acc >= 0.94f, "Accuracy should be >= 94% on separable data");
    if (lossN < loss0 * 0.6f && acc >= 0.94f) {
        std::cout << "[PASS] End-to-end Linear + BCE-with-logits\n";
    } else {
        std::cout << "[WARN] End-to-end Linear + BCE-with-logits didn't meet thresholds\n";
    }
}

int main() {
    try {
        backend::DeviceManager::instance().init();

        test_linear_logistic_end_to_end();
        test_mlp_logistic_end_to_end();
        test_linear_bce_with_logits_end_to_end();

        if (g_fail_count == 0) {
            std::cout << "\nALL END-TO-END TESTS PASSED (or met thresholds)\n";
            return 0;
        } else {
            std::cerr << "\nEND-TO-END TESTS FAILED: " << g_fail_count << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
