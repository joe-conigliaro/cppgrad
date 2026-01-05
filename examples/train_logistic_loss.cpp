// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iomanip>
#include <iostream>
#include <iomanip>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/activations.h"
#include "cppgrad/optim/adam.h"
#include "examples/moons/moons_data.h"

using namespace cppgrad;

// Logistic loss on raw logits: mean(log(1 + exp(-y * logits)))
static utils::Ref<ir::Tensor> logistic_loss(const utils::Ref<ir::Tensor>& logits,
                                                  const utils::Ref<ir::Tensor>& y_true) {
  auto product = ir::mul(y_true, logits);
  auto neg_product = ir::neg(product);
  auto one = ir::full(neg_product->shape(), 1.0f);
  return ir::mean(ir::log(ir::add(one, ir::exp(neg_product))));
}

static float compute_accuracy(const utils::Ref<ir::Tensor>& logits,
                              const utils::Ref<ir::Tensor>& y_true) {
  auto pv = logits->to_vector<float>();
  auto yv = y_true->to_vector<float>();
  size_t n = yv.size();
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    float pred_label = pv[i] >= 0.0f ? 1.0f : -1.0f;
    if (pred_label == yv[i]) correct++;
  }
  return (float)correct / (float)n;
}

int main() {
  backend::DeviceManager::instance().init();

  // Data
  MoonsParams P;
  P.n_samples = 1000; P.noise = 0.2f; P.dx = 1.0f; P.dy = 0.0f; P.seed = 123;
  std::vector<float> Xv, yv;
  make_moons(P, Xv, yv);
  // standardize_xy(Xv); // optional but helps training stability

  auto xs = ir::from_vector(Xv, {(size_t)P.n_samples, 2});
  auto ys = ir::from_vector(yv, {(size_t)P.n_samples, 1});

  // Model: small MLP with Tanh + Xavier
  auto l1 = std::make_shared<nn::Linear>(2, 32, true, nn::Init::XavierUniform);
  auto a1 = std::make_shared<nn::Tanh>();
  auto l2 = std::make_shared<nn::Linear>(32, 32, true, nn::Init::XavierUniform);
  auto a2 = std::make_shared<nn::Tanh>();
  auto l3 = std::make_shared<nn::Linear>(32, 1, true, nn::Init::XavierUniform);

  std::vector<std::shared_ptr<nn::Module>> modules = {l1, a1, l2, a2, l3};
  auto forward = [&](const utils::Ref<ir::Tensor>& x) {
    auto cur = x;
    for (auto& m : modules) cur = (*m)(cur);
    return cur;
  };

  // Optimizer: Adam LR 0.003
  std::vector<utils::Ref<ir::Tensor>> params;
  for (auto& m : modules) {
    auto ps = m->parameters();
    params.insert(params.end(), ps.begin(), ps.end());
  }
  optim::Adam opt(params, 0.003f);

  // Train
  int epochs = 300;
  for (int e = 1; e <= epochs; ++e) {
    auto logits = forward(xs); // raw logits
    auto loss = logistic_loss(logits, ys);

    opt.zero_grad();
    loss->backward();
    opt.step();

    if (e % 50 == 0) {
      float l = loss->item<float>();
      float acc = compute_accuracy(logits, ys);
      std::cout << "Epoch " << std::setw(3) << (e+1)
                << " loss=" << std::fixed << std::setprecision(6) << l
                << " acc="  << std::setprecision(4) << acc << "\n";
    }
  }

  // Final metrics
  auto logits = forward(xs);
  float final_acc = compute_accuracy(logits, ys);
  std::cout << "Final training accuracy: " << final_acc << "\n";
}
