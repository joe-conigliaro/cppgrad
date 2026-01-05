// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iomanip>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/activations.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/optim/adam.h"
#include "examples/moons/moons_data.h"

using namespace cppgrad;

static float compute_accuracy_tanh(const utils::Ref<ir::Tensor>& tanh_out,
                                   const utils::Ref<ir::Tensor>& y_true) {
  auto pv = tanh_out->to_vector<float>();
  auto yv = y_true->to_vector<float>();
  size_t n = yv.size(), correct = 0;
  for (size_t i = 0; i < n; ++i) {
    float pred = pv[i] >= 0.0f ? 1.0f : -1.0f;
    if (pred == yv[i]) correct++;
  }
  return (float)correct / (float)n;
}

int main() {
  backend::DeviceManager::instance().init();

  MoonsParams P; P.n_samples=1000; P.noise=0.2f; P.seed=123;
  std::vector<float> Xv, yv; make_moons(P, Xv, yv);
  // standardize_xy(Xv);

  auto xs = ir::from_vector(Xv, {(size_t)P.n_samples, 2});
  auto ys = ir::from_vector(yv, {(size_t)P.n_samples, 1});

  auto l1 = std::make_shared<nn::Linear>(2, 32, true, nn::Init::XavierUniform);
  auto a1 = std::make_shared<nn::Tanh>();
  auto l2 = std::make_shared<nn::Linear>(32, 32, true, nn::Init::XavierUniform);
  auto a2 = std::make_shared<nn::Tanh>();
  auto l3 = std::make_shared<nn::Linear>(32, 1, true, nn::Init::XavierUniform);

  std::vector<std::shared_ptr<nn::Module>> modules = {l1, a1, l2, a2, l3};
  auto forward_logits = [&](const utils::Ref<ir::Tensor>& x) {
    auto cur = x;
    for (auto& m : modules) cur = (*m)(cur);
    return cur;
  };
  auto forward_tanh = [&](const utils::Ref<ir::Tensor>& x) {
    return ir::tanh(forward_logits(x));
  };

  std::vector<utils::Ref<ir::Tensor>> params;
  for (auto& m : modules) {
    auto ps = m->parameters();
    params.insert(params.end(), ps.begin(), ps.end());
  }
  optim::Adam opt(params, 0.003f); // lower LR helps with tanh saturation

  for (int e = 1; e <= 300; ++e) {
    auto out = forward_tanh(xs); // tanh to [-1, 1]
    auto loss = nn::functional::mse_loss(out, ys);

    opt.zero_grad();
    loss->backward();
    opt.step();

    if (e % 50 == 0) {
      float l = loss->item<float>();
      float acc = compute_accuracy_tanh(out, ys);
      std::cout << "Epoch " << std::setw(3) << (e+1)
                << " loss=" << std::fixed << std::setprecision(6) << l
                << " acc="  << std::setprecision(4) << acc << "\n";
    }
  }
}
