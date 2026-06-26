// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/utils/shape.h"
#include "cppgrad/utils/vector.h"
#include "tests/helpers.h"

using namespace cppgrad;

// Backend numeric sanity
static void test_full_and_zeros_leaf() {
    TEST_HEADER("full() and zeros() produce correct values (may be non-leaf)");

    auto t1 = ir::zeros({3});
    auto t2 = ir::full({3}, 0.0f);

    // Realize and check values
    auto v1 = t1->to_vector<float>();
    auto v2 = t2->to_vector<float>();
    expect_allclose(v1, std::vector<float>{0,0,0}, 1e-7f, "zeros values");
    expect_allclose(v2, std::vector<float>{0,0,0}, 1e-7f, "full(0) values");

    // RandomOp/ConstantOp nodes are non-leaf by design.
    EXPECT_TRUE(!t1->is_leaf(), "zeros are created as non-leaf op (lazy)");
    EXPECT_TRUE(!t2->is_leaf(), "full is created as non-leaf op (lazy)");

    std::cout << "[PASS] full/zeros values (non-leaf as expected)\n";
}

static void test_uniform_basic_stats() {
    TEST_HEADER("uniform() basic stats (mean approx 0 when symmetric)");
    auto t = ir::uniform({10000}, -0.5f, 0.5f);
    auto v = t->to_vector<float>();
    double mean = 0.0;
    for (float x : v) mean += x;
    mean /= (double)v.size();
    EXPECT_TRUE(std::fabs(mean) < 0.02, "uniform mean close to 0");

    // Uniform is a RandomOp node (non-leaf).
    EXPECT_TRUE(!t->is_leaf(), "uniform is created as non-leaf op (lazy)");
    std::cout << "[PASS] uniform stats\n";
}


static void test_matmul_numeric() {
    TEST_HEADER("matmul numeric correctness");
    auto A = ir::from_vector(std::vector<float>{1,2,3,4}, {2,2});
    auto B = ir::from_vector(std::vector<float>{5,6,7,8}, {2,2});
    auto C = ir::matmul(A, B);
    auto vc = C->to_vector<float>();
    // Expected [[19,22],[43,50]]
    expect_allclose(vc, std::vector<float>{19,22,43,50}, 1e-6f, "matmul 2x2");
    std::cout << "[PASS] matmul numeric\n";
}

static void test_broadcast_add_numeric() {
    TEST_HEADER("broadcast add numeric correctness");
    auto X = ir::from_vector(std::vector<float>{1,2,3,4,5,6}, {2,3});
    auto b = ir::from_vector(std::vector<float>{10,20,30}, {1,3});
    auto Y = ir::add(X, b);
    auto vy = Y->to_vector<float>();
    // Row-wise: [1+10,2+20,3+30, 4+10,5+20,6+30]
    expect_allclose(vy, std::vector<float>{11,22,33,14,25,36}, 1e-6f, "broadcast add");
    std::cout << "[PASS] broadcast add\n";
}

static void test_sum_mean_consistency() {
    TEST_HEADER("mean equals sum/numel");
    auto T = ir::from_vector(std::vector<float>{1,2,3,4}, {2,2});
    auto S = ir::sum(T);   // scalar
    auto M = ir::mean(T);  // scalar
    float s = S->item<float>();
    float m = M->item<float>();
    EXPECT_CLOSE(m, s / 4.0f, 1e-7f, "mean=sum/numel");
    std::cout << "[PASS] sum/mean consistency\n";
}

static void test_sum_is_sum() {
    TEST_HEADER("sum is sum");
    auto t = cppgrad::ir::full({4,3}, 1.0f);
    auto s = cppgrad::ir::sum(t)->item<float>();
    EXPECT_CLOSE(s, 12.0f, 1e-6f, "sum should be 12");

}

static void test_mean_backprop_micro() {
    TEST_HEADER("mean backprop micro");
    auto t = cppgrad::ir::parameterize(cppgrad::ir::full({2,2}, 1.0f));
    auto L = cppgrad::ir::mean(t);
    L->backward();
    auto g = t->grad();
    EXPECT_TRUE(g != nullptr, "grad exists");
    EXPECT_CLOSE(cppgrad::ir::sum(g)->item<float>(), 1.0f, 1e-6f, "mean total grad");
    for (float v : g->to_vector<float>()) {
        EXPECT_CLOSE(v, 0.25f, 1e-6f, "each grad 1/4");
    }
    std::cout << "[PASS] mean backprop micro\n";
}

static void test_unbroadcast_accumulation() {
     TEST_HEADER("unbroadcast accumulation");
    // simulate incoming grad after mean: 1/12 everywhere in [4,3]
    auto incoming = cppgrad::ir::full({4,3}, 1.0f/12.0f);
    auto reduced  = incoming;
    // emulate reduce_to_shape to [1,3]
    auto axes = utils::shape::get_reduce_axes(reduced->shape(), std::vector<size_t>{1,3});
    reduced = cppgrad::ir::sum(reduced, axes, /*keep_dims=*/true);
    if (reduced->shape() != std::vector<size_t>{1,3})
        reduced = cppgrad::ir::reshape(reduced, {1,3});
    // expect 1/3 per entry
    for (float v : reduced->to_vector<float>()) {
        EXPECT_CLOSE(v, 1.0f/3.0f, 1e-6f, "unbroadcast SUM");
    }
    EXPECT_CLOSE(cppgrad::ir::sum(reduced)->item<float>(), 1.0f, 1e-6f, "unbroadcast total");
    std::cout << "[PASS] unbroadcast accumulation\n";
}

// Gradients: broadcasting
static void test_broadcast_backward_add_mean() {
    TEST_HEADER("backward: y = mean(X + b), shapes X[4,3], b[1,3]");
    // New (leaf-only):
    size_t A=4, B=3, C=1;
    auto X_init = ir::uniform({A,B}, -1.0f, 1.0f);
    auto X = ir::parameterize(X_init);
    auto b_init = ir::uniform({C,B}, -1.0f, 1.0f);
    auto b = ir::parameterize(b_init);
    auto y = ir::add(X, b);
    auto L = ir::mean(y);
    L->backward();

    // EXPECT_TRUE(X->grad() != nullptr, "grad(X) must be non-null after backward");
    // EXPECT_TRUE(b->grad() != nullptr, "grad(b) must be non-null after backward");
    // if (!X->grad() || !b->grad()) return;

    EXPECT_TRUE(X->grad()->shape() == std::vector<size_t>({A,B}), "grad X shape");
    EXPECT_TRUE(b->grad()->shape() == std::vector<size_t>({C,B}), "grad b shape");
    EXPECT_CLOSE(ir::sum(X->grad())->item<float>(), 1.0f, 1e-6f, "sum grad X");
    EXPECT_CLOSE(ir::sum(b->grad())->item<float>(), 1.0f, 1e-6f, "sum grad b");

    auto gX = X->grad()->to_vector<float>();
    auto gb = b->grad()->to_vector<float>();

    auto GX = X->grad();
    std::cout << "grad(X) device=" << (int)GX->device_type() << " dtype=" << (int)GX->dtype()
              << " shape=" << utils::vector::to_string(GX->shape()) << "\n";
    auto sGX = ir::sum(GX)->item<float>();
    std::cout << "sum grad X = " << sGX << "\n";
    auto vec = GX->to_vector<float>();
    float mn=*std::min_element(vec.begin(), vec.end()), mx=*std::max_element(vec.begin(), vec.end());
    std::cout << "grad X min=" << mn << " max=" << mx << "\n";

    for (float v : gX) EXPECT_CLOSE(v, 1.0f/12.0f, 1e-7f, "grad X val");
    for (float v : gb) EXPECT_CLOSE(v, 1.0f/3.0f, 1e-7f, "grad b val");
    std::cout << "[PASS] backward broadcast add/mean\n";
}

// Gradients: matmul + bias
static void test_linear_bias_grad_mean() {
    TEST_HEADER("backward: L = mean(XW + b), check dL/db");
    size_t N=4, I=3, O=2;
    auto X = ir::parameterize(ir::uniform({N,I}, -1, 1));
    auto W = ir::parameterize(ir::uniform({I,O}, -1, 1));
    auto b = ir::parameter({1,O});
    auto y = ir::add(ir::matmul(X, W), b);
    auto L = ir::mean(y);
    L->backward();
    auto gb = b->grad()->to_vector<float>();
    // dL/dy = 1/(N*O), and db adds across batch -> dL/db_j = N*(1/(N*O)) = 1/O
    for (float v : gb) EXPECT_CLOSE(v, 1.0f / (float)O, 1e-7f, "grad b = 1/O");
    std::cout << "[PASS] linear bias grad with mean\n";
}

static void test_matmul_backward_shapes() {
    TEST_HEADER("backward: matmul grad shapes");
    size_t N=5, I=7, O=3;
    auto X = ir::parameterize(ir::uniform({N,I}, -1, 1));
    auto W = ir::parameterize(ir::uniform({I,O}, -1, 1));
    auto y = ir::matmul(X, W);
    auto L = ir::mean(y);
    L->backward();
    // Check grad shapes
    EXPECT_TRUE(X->grad()->shape() == std::vector<size_t>({N,I}), "grad X shape");
    EXPECT_TRUE(W->grad()->shape() == std::vector<size_t>({I,O}), "grad W shape");
    std::cout << "[PASS] matmul backward shapes\n";
}

// Unary gradients: tanh, relu, exp/log stability
static void test_tanh_backward() {
    TEST_HEADER("backward: tanh");
    auto x = ir::from_vector(std::vector<float>{-1.0f, 0.0f, 1.0f}, {3}); x->set_requires_grad(true);
    auto y = ir::tanh(x);
    auto L = ir::mean(y);
    L->backward();
    auto gx = x->grad()->to_vector<float>();
    auto yv = y->to_vector<float>();
    for (int i=0;i<3;i++) {
        float expected = (1.0f/3.0f)*(1.0f - yv[i]*yv[i]);
        EXPECT_CLOSE(gx[i], expected, 1e-6f, "tanh backward component");
    }
    std::cout << "[PASS] tanh backward\n";
}

static void test_relu_backward() {
    TEST_HEADER("backward: relu");
    auto x = ir::from_vector(std::vector<float>{-1.0f, 0.0f, 2.0f}, {3}); x->set_requires_grad(true);
    auto y = ir::relu(x);
    auto L = ir::mean(y);
    L->backward();
    auto gx = x->grad()->to_vector<float>();
    EXPECT_CLOSE(gx[0], 0.0f, 1e-7f, "relu grad x<0");
    EXPECT_CLOSE(gx[1], 0.0f, 1e-7f, "relu grad x==0");
    EXPECT_CLOSE(gx[2], 1.0f/3.0f, 1e-7f, "relu grad x>0");
    std::cout << "[PASS] relu backward\n";
}

static void test_exp_log_numeric() {
    TEST_HEADER("exp/log numeric sanity");
    auto x = ir::from_vector(std::vector<float>{-2.0f, 0.0f, 2.0f}, {3});
    auto e = ir::exp(x);
    auto l = ir::log(ir::exp(x));
    auto ve = e->to_vector<float>();
    auto vl = l->to_vector<float>();
    expect_allclose(ve, std::vector<float>{std::exp(-2.0f), 1.0f, std::exp(2.0f)}, 1e-6f, "exp values");
    expect_allclose(vl, std::vector<float>{-2.0f, 0.0f, 2.0f}, 1e-6f, "log(exp(x)) values");
    std::cout << "[PASS] exp/log numeric\n";
}

static void test_neg_numeric() {
    TEST_HEADER("neg numeric sanity");
    auto x = ir::from_vector(std::vector<float>{-1.0f, 0.5f}, {2});
    auto y = ir::neg(x);
    auto vy = y->to_vector<float>();
    expect_allclose(vy, std::vector<float>{1.0f, -0.5f}, 1e-7f, "neg values");
    std::cout << "[PASS] neg numeric\n";
}

// Losses: MSE, hinge, logistic_pm1, BCE-with-logits, softplus
static void test_mse_loss_perfect_fit() {
    TEST_HEADER("mse_loss: perfect fit -> zero grad");
    auto y = ir::from_vector(std::vector<float>{1,2,3}, {3});
    auto yhat = ir::from_vector(std::vector<float>{1,2,3}, {3});
    yhat->set_requires_grad(true);
    auto loss = cppgrad::nn::functional::mse_loss(yhat, y);
    loss->backward();
    auto gyhat = yhat->grad()->to_vector<float>();
    for (float v : gyhat) EXPECT_CLOSE(v, 0.0f, 1e-7f, "mse grad at perfect fit");
    std::cout << "[PASS] mse perfect fit\n";
}

static void test_mse_loss_known_values() {
    TEST_HEADER("mse_loss: known values");
    auto y = ir::from_vector(std::vector<float>{1,2,3}, {3});
    auto yhat = ir::from_vector(std::vector<float>{0,0,0}, {3});
    yhat->set_requires_grad(true);
    auto loss = cppgrad::nn::functional::mse_loss(yhat, y);
    float l = loss->item<float>();
    EXPECT_CLOSE(l, (1*1 + 2*2 + 3*3)/3.0f, 1e-6f, "mse numeric");
    loss->backward();
    auto gyhat = yhat->grad()->to_vector<float>();
    std::vector<float> expected = {-2.0f/3.0f, -4.0f/3.0f, -2.0f};
    expect_allclose(gyhat, expected, 1e-7f, "mse grad values");
    std::cout << "[PASS] mse known values\n";
}

static void test_hinge_loss_simple() {
    TEST_HEADER("hinge_loss: simple cases");
    using nn::functional::hinge_loss;
    // y in {+1,-1}
    auto y = ir::from_vector(std::vector<float>{1.0f, -1.0f, 1.0f, -1.0f}, {4,1});
    auto logits = ir::from_vector(std::vector<float>{2.0f, -2.0f, 0.2f, -0.2f}, {4,1});
    logits->set_requires_grad(true);
    auto loss = hinge_loss(logits, y, 1.0f);
    float L = loss->item<float>();
    EXPECT_CLOSE(L, 0.4f, 1e-6f, "hinge numeric");

    loss->backward();
    auto g = logits->grad()->to_vector<float>();
    std::vector<float> exp = {0.0f, 0.0f, -1.0f/4.0f, +1.0f/4.0f};
    expect_allclose(g, exp, 1e-7f, "hinge grad");
    std::cout << "[PASS] hinge loss\n";
}

static void test_logistic_loss_pm1() {
    TEST_HEADER("logistic_loss_pm1: numeric and gradient");
    using nn::functional::logistic_loss_pm1;

    auto y = ir::from_vector(std::vector<float>{1.0f, -1.0f}, {2,1});
    auto logits = ir::from_vector(std::vector<float>{2.0f, -2.0f}, {2,1});
    logits->set_requires_grad(true);

    auto loss = logistic_loss_pm1(logits, y); // mean(softplus(-y*z))
    float L = loss->item<float>();
    EXPECT_CLOSE(L, 0.126928f, 1e-5f, "logistic_pm1 numeric ~0.1269");

    loss->backward();
    auto g = logits->grad()->to_vector<float>();
    float s = 1.0f / (1.0f + std::exp(2.0f)); // sigmoid(-2)
    std::vector<float> exp = { -s / 2.0f, +s / 2.0f };
    expect_allclose(g, exp, 1e-5f, "logistic_pm1 grad");
    std::cout << "[PASS] logistic_loss_pm1\n";
}

static void test_bce_with_logits_basic() {
    TEST_HEADER("bce_with_logits: numeric and gradient");
    using nn::functional::bce_with_logits;

    auto targets = ir::from_vector(std::vector<float>{1.0f, 0.0f}, {2,1}); // y ∈ {0,1}
    auto logits = ir::from_vector(std::vector<float>{2.0f, -2.0f}, {2,1});
    logits->set_requires_grad(true);

    auto loss = bce_with_logits(logits, targets);
    float L = loss->item<float>();
    // Canonical: BCEWithLogits(z,y) = softplus(z) − z*y
    // Sample1: softplus(2) - 2 ≈ 0.126928
    // Sample2: softplus(-2) - 0 ≈ 0.126928
    float expected_mean = 0.126928f; // (0.126928 + 0.126928)/2
    EXPECT_CLOSE(L, expected_mean, 1e-5f, "bce_with_logits numeric");

    // loss->set_requires_grad(true);
    loss->backward();
    auto g = logits->grad()->to_vector<float>();
    auto sig = [](float z){ return 1.0f / (1.0f + std::exp(-z)); };
    // d/dz mean = (sigmoid(z) - y) / N
    std::vector<float> exp = { (sig(2.0f) - 1.0f) / 2.0f, (sig(-2.0f) - 0.0f) / 2.0f };
    expect_allclose(g, exp, 1e-5f, "bce_with_logits grad");
    std::cout << "[PASS] bce_with_logits\n";
}

static void test_softplus_stability() {
    TEST_HEADER("softplus stability for large |x|");
    using nn::functional::softplus;

    auto x = ir::from_vector(std::vector<float>{-50.0f, 0.0f, 50.0f}, {3});
    auto y = softplus(x);
    auto v = y->to_vector<float>();
    EXPECT_TRUE(v[0] >= 0.0f && v[0] < 1e-21f, "softplus(-50) near 0");
    EXPECT_CLOSE(v[1], std::log(2.0f), 1e-6f, "softplus(0) = log(2)");
    EXPECT_CLOSE(v[2], 50.0f, 1e-5f, "softplus(50) ~ 50");
    std::cout << "[PASS] softplus stability\n";
}

// Softmax and cross-entropy
static void test_softmax_log_softmax() {
    TEST_HEADER("softmax/log_softmax numeric stability");
    using nn::functional::softmax;
    using nn::functional::log_softmax;

    auto logits = ir::from_vector(std::vector<float>{1000.0f, 0.0f, -1000.0f}, {3});
    auto sm = softmax(logits, -1);
    auto lsm = log_softmax(logits, -1);

    auto v_sm = sm->to_vector<float>();
    auto v_lsm = lsm->to_vector<float>();

    EXPECT_CLOSE(v_sm[0], 1.0f, 1e-6f, "softmax large pos");
    EXPECT_TRUE(v_sm[1] < 1e-6f && v_sm[2] < 1e-6f, "softmax small comps");

    EXPECT_CLOSE(v_lsm[0], 0.0f, 1e-6f, "log_softmax of the max ~ 0");
    EXPECT_TRUE(v_lsm[1] < -20.0f && v_lsm[2] < -20.0f, "log_softmax large negs");
    std::cout << "[PASS] softmax/log_softmax stability\n";
}

static void test_softmax_cross_entropy_with_logits_basic() {
    TEST_HEADER("softmax_cross_entropy_with_logits: numeric and grad shape");
    using nn::functional::softmax_cross_entropy_with_logits;

    auto logits = ir::from_vector(std::vector<float>{
        2.0f, 0.0f, -1.0f,
        -1.0f, 3.0f, 0.0f
    }, {2,3});
    logits->set_requires_grad(true);

    auto targets = ir::from_vector(std::vector<float>{
        1.0f, 0.0f, 0.0f,  // class 0
        0.0f, 1.0f, 0.0f   // class 1
    }, {2,3});

    auto loss = softmax_cross_entropy_with_logits(logits, targets);
    float L = loss->item<float>();
    EXPECT_TRUE(L > 0.0f, "softmax CE positive");

    loss->backward();

    auto g = logits->grad()->to_vector<float>();
    EXPECT_TRUE(g.size() == 6, "grad shape 2x3");
    std::cout << "[PASS] softmax_cross_entropy_with_logits\n";
}

// Scalar ops: tensor ⊕ scalar and scalar ⊕ tensor
static void test_scalar_ops() {
    TEST_HEADER("scalar ops: tensor ±×÷ scalar and reverse order");

    auto a = ir::from_vector(std::vector<float>{1.0f, -2.0f, 3.0f}, {3});
    // tensor ±×÷ scalar
    auto ap = ir::add(a, 2.0f)->to_vector<float>();
    auto am = ir::sub(a, 2.0f)->to_vector<float>();
    auto ax = ir::mul(a, 2.0f)->to_vector<float>();
    auto ad = ir::div(a, 2.0f)->to_vector<float>();
    expect_allclose(ap, std::vector<float>{3.0f, 0.0f, 5.0f}, 1e-7f, "a + 2");
    expect_allclose(am, std::vector<float>{-1.0f, -4.0f, 1.0f}, 1e-7f, "a - 2");
    expect_allclose(ax, std::vector<float>{2.0f, -4.0f, 6.0f}, 1e-7f, "a * 2");
    expect_allclose(ad, std::vector<float>{0.5f, -1.0f, 1.5f}, 1e-7f, "a / 2");

    // scalar ±×÷ tensor (reverse overloads)
    auto pa = ir::add(2.0f, a)->to_vector<float>();
    auto ma = ir::sub(2.0f, a)->to_vector<float>();
    auto xa = ir::mul(2.0f, a)->to_vector<float>();
    auto da = ir::div(2.0f, a)->to_vector<float>();
    expect_allclose(pa, std::vector<float>{3.0f, 0.0f, 5.0f}, 1e-7f, "2 + a");
    expect_allclose(ma, std::vector<float>{1.0f, 4.0f, -1.0f}, 1e-7f, "2 - a");
    expect_allclose(xa, std::vector<float>{2.0f, -4.0f, 6.0f}, 1e-7f, "2 * a");
    // For 2 / a: [2/1, 2/(-2), 2/3]
    expect_allclose(da, std::vector<float>{2.0f, -1.0f, 2.0f/3.0f}, 1e-7f, "2 / a");

    std::cout << "[PASS] scalar ops\n";
}

// Comparisons: tensor ⊕ tensor
static void test_comparisons() {
    TEST_HEADER("comparisons: cmp_eq, cmp_gt");
    auto a = ir::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});
    auto b = ir::from_vector(std::vector<float>{1.0f, 1.5f, 3.5f}, {3});
    auto eq = ir::cmp_eq(a, b)->to_vector<float>();
    auto gt = ir::cmp_gt(a, b)->to_vector<float>();
    // Assuming comparison outputs numeric masks {0,1}
    expect_allclose(eq, std::vector<float>{1.0f, 0.0f, 0.0f}, 1e-7f, "cmp_eq");
    expect_allclose(gt, std::vector<float>{0.0f, 1.0f, 0.0f}, 1e-7f, "cmp_gt");
    std::cout << "[PASS] comparisons\n";
}

// Reductions with axes/keep_dims
static void test_reductions_axes() {
    TEST_HEADER("reductions with axes and keep_dims");
    auto X = ir::from_vector(std::vector<float>{1,2,3,4,5,6}, {2,3});
    // Sum over axis 0, keep dims
    auto s0 = ir::sum(X, std::vector<int>{0}, true)->to_vector<float>();
    expect_allclose(s0, std::vector<float>{1+4, 2+5, 3+6}, 1e-7f, "sum axis 0 keep");
    // Mean over axis 1, no keep dims
    auto m1 = ir::mean(X, std::vector<int>{1}, false)->to_vector<float>();
    expect_allclose(m1, std::vector<float>{(1+2+3)/3.0f, (4+5+6)/3.0f}, 1e-7f, "mean axis 1 no keep");
    std::cout << "[PASS] reductions axes\n";
}

// Movement ops: reshape, broadcast, permute, transpose
static void test_movement_ops() {
    TEST_HEADER("movement ops: reshape, broadcast, permute, transpose");
    auto x = ir::from_vector(std::vector<float>{1,2,3,4}, {2,2});
    // auto y = ir::reshape(x, std::vector<size_t>{4});
    // EXPECT_TRUE(y->shape() == std::vector<size_t>({4}), "reshape to 1D");
    auto y = ir::reshape(x, std::vector<size_t>{4});
    EXPECT_TRUE(y->shape() == std::vector<size_t>({4}), "reshape to 1D");

    // auto b = ir::broadcast(y, std::vector<size_t>{4,3});
    // EXPECT_TRUE(b->shape() == std::vector<size_t>({4,3}), "broadcast to (4,3)");
    // Make y column-vector [4,1], then broadcast to [4,3]
    auto y2 = ir::reshape(y, std::vector<size_t>{4, 1});
    auto b = ir::broadcast(y2, std::vector<size_t>{4, 3});
    EXPECT_TRUE(b->shape() == std::vector<size_t>({4,3}), "broadcast to (4,3)");

    auto p = ir::permute(x, std::vector<size_t>{1,0});
    auto tp = ir::transpose(x, 0, 1);
    EXPECT_TRUE(p->shape() == std::vector<size_t>({2,2}), "permute shape");
    EXPECT_TRUE(tp->shape() == std::vector<size_t>({2,2}), "transpose shape");
    std::cout << "[PASS] movement ops\n";
}

int main() {
    try {
        backend::DeviceManager::instance().init();

        // Backend numeric tests
        test_full_and_zeros_leaf();
        test_uniform_basic_stats();
        test_matmul_numeric();
        test_broadcast_add_numeric();
        test_sum_mean_consistency();

        // Gradients with broadcasting
        test_broadcast_backward_add_mean();

        // Matmul + bias gradients and shapes
        test_linear_bias_grad_mean();
        test_matmul_backward_shapes();

        // Unary gradients and numeric
        test_tanh_backward();
        test_relu_backward();
        test_exp_log_numeric();
        test_neg_numeric();

        // Losses and stability
        test_mse_loss_perfect_fit();
        test_mse_loss_known_values();
        test_hinge_loss_simple();
        test_logistic_loss_pm1();
        test_bce_with_logits_basic();
        test_softplus_stability();

        // Softmax family
        test_softmax_log_softmax();
        test_softmax_cross_entropy_with_logits_basic();

        // Scalar ops
        test_scalar_ops();

        // Comparisons
        test_comparisons();

        // Reductions axes
        test_reductions_axes();

        // Movement ops
        test_movement_ops();

        if (g_fail_count == 0) {
            std::cout << "\nALL TESTS PASSED\n";
            return 0;
        } else {
            std::cerr << "\nTESTS FAILED: " << g_fail_count << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
