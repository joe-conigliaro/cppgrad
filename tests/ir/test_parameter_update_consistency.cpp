// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cmath>
#include <vector>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/optim/adam.h"
#include "tests/helpers.h"

using namespace cppgrad;

static float train_step(nn::Linear& lin, const utils::Ref<ir::Tensor>& xs, const utils::Ref<ir::Tensor>& ys) {
    using nn::functional::logistic_loss_pm1;
    cppgrad::optim::Adam opt(lin.direct_parameters(), 0.003f);
    auto logits = lin(xs);
    auto loss = logistic_loss_pm1(logits, ys);
    opt.zero_grad();
    loss->backward();
    opt.step();
    return loss->item<float>();
}

int main() {
    backend::DeviceManager::instance().init();

    // Small synthetic data
    std::vector<float> Xv{1,0, -1,0, 0,1, 0,-1};
    std::vector<float> yv{1,-1, 1,-1};
    auto xs = ir::from_vector(Xv, {4,2});
    auto ys = ir::from_vector(yv, {4,1});

    // Reference model A (used only to source an initial snapshot)
    nn::Linear A(2,1,true, nn::Init::XavierUniform);

    // Take a single immutable snapshot of A's parameters as host vectors
    // (no aliasing, independent of eval scope timing)
    auto w_vec = A.weight->to_vector<float>();
    std::vector<float> b_vec;
    if (A.bias) b_vec = A.bias->to_vector<float>();

    // Recreate snapshot as IR tensors and realize to buffers (no backend::copy)
    // These become our canonical "frozen" buffers for initializing both paths.
    auto w_t = ir::from_vector(w_vec, A.weight->shape());
    auto w_buf = w_t->schedule(); // std::shared_ptr<backend::Buffer>

    std::shared_ptr<backend::Buffer> b_buf;
    if (A.bias) {
        auto b_t = ir::from_vector(b_vec, A.bias->shape());
        b_buf = b_t->schedule();
    }

    // Two identical models to compare
    nn::Linear B(2,1,true, nn::Init::XavierUniform);
    nn::Linear C(2,1,true, nn::Init::XavierUniform);

    // Initialize via the two paths from the SAME realized snapshot buffers
    B.weight->set_parameter_data(w_buf);
    if (B.bias && b_buf) B.bias->set_parameter_data(b_buf);

    C.weight->copy_into_parameter(w_buf);
    if (C.bias && b_buf) C.bias->copy_into_parameter(b_buf);

    // Both must be leaf and canonical
    EXPECT_TRUE(B.weight->is_leaf(), "B.weight leaf");
    if (B.bias) EXPECT_TRUE(B.bias->is_leaf(), "B.bias leaf");
    EXPECT_TRUE(C.weight->is_leaf(), "C.weight leaf");
    if (C.bias) EXPECT_TRUE(C.bias->is_leaf(), "C.bias leaf");

    // Compare weights immediately (before any training step)
    auto Bw0 = B.weight->to_vector<float>();
    auto Cw0 = C.weight->to_vector<float>();
    EXPECT_TRUE(Bw0.size() == Cw0.size(), "weight size equal at init");
    for (size_t i = 0; i < Bw0.size(); ++i)
        EXPECT_CLOSE(Bw0[i], Cw0[i], 1e-7f, "weight equal at init");

    if (B.bias && C.bias) {
        auto Bb0 = B.bias->to_vector<float>();
        auto Cb0 = C.bias->to_vector<float>();
        EXPECT_TRUE(Bb0.size() == Cb0.size(), "bias size equal at init");
        for (size_t i = 0; i < Bb0.size(); ++i)
            EXPECT_CLOSE(Bb0[i], Cb0[i], 1e-7f, "bias equal at init");
    }

    // Now train exactly one step on each model and compare losses
    float loss1 = train_step(B, xs, ys);
    float loss2 = train_step(C, xs, ys);
    EXPECT_CLOSE(loss1, loss2, 1e-6f, "loss equal after one step");

    if (g_fail_count == 0) {
        std::cout << "Parameter update consistency OK\n";
        return 0;
    }
    return 1;
}
