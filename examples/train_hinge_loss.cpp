// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/activations.h"
#include "cppgrad/optim/adam.h"
#include "cppgrad/nn/functional.h"
#include "examples/moons/moons_data.h"

using namespace cppgrad;

static float compute_accuracy(const utils::Ref<ir::Tensor>& logits, const utils::Ref<ir::Tensor>& y_true) {
    auto pv = logits->to_vector<float>();
    auto yv = y_true->to_vector<float>();
    size_t n = yv.size(), correct = 0;
    for (size_t i = 0; i < n; ++i)
        correct += (pv[i] >= 0.0f ? 1.0f : -1.0f) == yv[i];
    return (float)correct / (float)n;
}

int main() {
    backend::DeviceManager::instance().init();

    MoonsParams P; P.n_samples=1000; P.noise=0.2f; P.seed=123;
    std::vector<float> Xv, yv; make_moons(P, Xv, yv);
    // standardize_xy(Xv);

    auto xs = ir::from_vector(Xv, {(size_t)P.n_samples, 2});
    auto ys = ir::from_vector(yv, {(size_t)P.n_samples, 1});

    auto l1 = std::make_shared<nn::Linear>(2, 32, true, nn::Init::KaimingUniform);
    auto a1 = std::make_shared<nn::ReLU>();
    auto l2 = std::make_shared<nn::Linear>(32, 32, true, nn::Init::KaimingUniform);
    auto a2 = std::make_shared<nn::ReLU>();
    auto l3 = std::make_shared<nn::Linear>(32, 1, true, nn::Init::KaimingUniform);

    std::vector<std::shared_ptr<nn::Module>> modules = {l1, a1, l2, a2, l3};
    auto forward = [&](const utils::Ref<ir::Tensor>& x) {
        auto cur = x;
        for (auto& m : modules) cur = (*m)(cur);
        return cur;
    };

    std::vector<utils::Ref<ir::Tensor>> params;
    for (auto& m : modules) {
        auto ps = m->parameters();
        params.insert(params.end(), ps.begin(), ps.end());
    }
    optim::Adam opt(params, 0.003f);

    auto tic = [](){ return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto t0, auto t1){ return std::chrono::duration<double, std::milli>(t1 - t0).count(); };
    for (int e = 1; e <= 300; ++e) {
        ir::GraphScope scope;
        auto t0 = tic();
        auto logits = forward(xs);
        auto loss = nn::functional::hinge_loss(logits, ys, 1.0f);
        auto t_fw = tic();
        opt.zero_grad();
        loss->backward();
        auto t_bw = tic();
        opt.step();
        auto t_step = tic();
        if (e % 10 == 0) {
            float l = loss->item<float>();
            float acc = compute_accuracy(logits, ys);
            auto t_log = tic();
            std::cout << "Epoch " << std::setw(3) << (e)
            << " loss=" << std::fixed << std::setprecision(6) << l
            << " acc="  << std::setprecision(4) << acc
            << "  [fw=" << ms(t0,t_fw) << "ms, bw=" << ms(t_fw,t_bw)
            << "ms, step=" << ms(t_bw,t_step) << "ms, log=" << ms(t_step,t_log) << "ms]" << std::endl;
        }
    }
}
