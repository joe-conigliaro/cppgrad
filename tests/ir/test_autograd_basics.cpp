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
#include "tests/helpers.h"

using namespace cppgrad;

static void test_mean_scalar_and_grad() {
    TEST_HEADER("mean on ones: scalar value and grad");
    auto X = ir::ones({4,3});
    auto L = ir::mean(X);

    float lv = L->item<float>();
    std::cout << "mean value on ones: " << lv << "\n"; // expect 1.0

    auto Xp = ir::parameterize(X);
    auto Lp = ir::mean(Xp);
    Lp->backward();
    auto gx = Xp->grad()->to_vector<float>();
    float mn = *std::min_element(gx.begin(), gx.end());
    float mx = *std::max_element(gx.begin(), gx.end());
    float s = ir::sum(Xp->grad())->item<float>();
    std::cout << "grad mean(ones): sum=" << s << " min=" << mn << " max=" << mx << "\n";
}

static void test_sum_all_then_div() {
    TEST_HEADER("sum-all then div sanity");
    auto X = ir::ones({4,3});
    // S = sum(X) -> shape {1}, expect value 12
    auto S = ir::sum(X);
    float sv = S->item<float>();
    EXPECT_CLOSE(sv, 12.0f, 1e-6f, "sum-all scalar");

    // L = S / 12 => scalar 1.0
    auto L = ir::div(S, 12.0f);
    float lv = L->item<float>();
    EXPECT_CLOSE(lv, 1.0f, 1e-6f, "sum/12 scalar");
}

static void test_broadcast_full_vs_direct_full() {
    TEST_HEADER("broadcast({1,1}) vs full({4,3}) as upstream grad");

    size_t A=4, B=3;
    auto X1 = ir::parameterize(ir::uniform({A,B}, -1, 1));

    auto dY_direct = ir::full({A,B}, 1.0f/float(A*B));
    auto L1 = ir::sum(ir::mul(X1, dY_direct));
    L1->backward();
    auto v1 = X1->grad()->to_vector<float>();
    float mn1 = *std::min_element(v1.begin(), v1.end());
    float mx1 = *std::max_element(v1.begin(), v1.end());
    std::cout << "direct full: min=" << mn1 << " max=" << mx1 << "\n";

    auto X2 = ir::parameterize(ir::uniform({A,B}, -1, 1));
    auto kd = ir::full(std::vector<size_t>{1,1}, 1.0f/float(A*B));
    auto dY_bcast = ir::broadcast(kd, std::vector<size_t>{A,B});
    auto L2 = ir::sum(ir::mul(X2, dY_bcast));
    L2->backward();
    auto v2 = X2->grad()->to_vector<float>();
    float mn2 = *std::min_element(v2.begin(), v2.end());
    float mx2 = *std::max_element(v2.begin(), v2.end());
    std::cout << "broadcast full: min=" << mn2 << " max=" << mx2 << "\n";
}

static void test_backward_add_mean_explicit_vs_builtin() {
    TEST_HEADER("y = mean(X + b): builtin mean vs explicit sum*invN");

    size_t A=4, B=3, C=1;
    auto X_init = ir::uniform({A,B}, -1.0f, 1.0f);
    auto b_init = ir::uniform({C,B}, -1.0f, 1.0f);

    // Case 1: builtin mean
    auto X1 = ir::parameterize(X_init);
    auto b1 = ir::parameterize(b_init);
    auto y1 = ir::add(X1, b1);
    auto L1 = ir::mean(y1);
    L1->backward();
    auto gX1 = X1->grad()->to_vector<float>();
    float s1 = ir::sum(X1->grad())->item<float>();
    float mn1 = *std::min_element(gX1.begin(), gX1.end());
    float mx1 = *std::max_element(gX1.begin(), gX1.end());
    std::cout << "builtin mean: sum=" << s1 << " min=" << mn1 << " max=" << mx1 << "\n";

    // Reset grads (new params to avoid aliasing)
    auto X2 = ir::parameterize(X_init);
    auto b2 = ir::parameterize(b_init);

    // Case 2: explicit sum then multiply by 1/N
    auto y2 = ir::add(X2, b2);
    auto S  = ir::sum(y2); // scalar shape {1}
    auto invN = ir::scalar_like(1.0f / float(A*B), S);
    auto L2 = ir::mul(S, invN);
    L2->backward();
    auto gX2 = X2->grad()->to_vector<float>();
    float s2 = ir::sum(X2->grad())->item<float>();
    float mn2 = *std::min_element(gX2.begin(), gX2.end());
    float mx2 = *std::max_element(gX2.begin(), gX2.end());
    std::cout << "explicit mean: sum=" << s2 << " min=" << mn2 << " max=" << mx2 << "\n";

    EXPECT_CLOSE(s1, 1.0f, 1e-6f, "builtin mean sum dX");
    EXPECT_CLOSE(s2, 1.0f, 1e-6f, "explicit mean sum dX");
    // print the first row
    std::cout << "gX1[0..5]: ";
    for (int i=0;i<std::min<int>(6, gX1.size());++i) std::cout << gX1[i] << (i+1<6?", ":"\n");
    std::cout << "gX2[0..5]: ";
    for (int i=0;i<std::min<int>(6, gX2.size());++i) std::cout << gX2[i] << (i+1<6?", ":"\n");
}

static void test_add_backward_with_manual_upstream() {
    TEST_HEADER("ADD backward with manual upstream grad (dY = 1/12)");

    size_t A=4, B=3, C=1;
    auto X = ir::parameterize(ir::zeros({A,B}));
    auto b = ir::parameterize(ir::zeros({C,B}));

    // Build y = X + b (to have a graph)
    auto y = ir::add(X, b);

    // Manual upstream grad node G same shape as y
    auto G = ir::full(y->shape(), 1.0f / float(A*B));

    // Hack: create a scalar loss as sum(y .* mask) so that dL/dy = G
    // i.e., L = sum(y * 1) / (A*B) == mean(y); but we will replace y with a proxy
    // A more direct way: just use L = sum( y * constant(G_values / 1) )
    auto L = ir::sum(ir::mul(y, G)); // dL/dy = G

    L->backward();

    auto vX = X->grad()->to_vector<float>();
    auto vb = b->grad()->to_vector<float>();

    float sX = ir::sum(X->grad())->item<float>();
    float sb = ir::sum(b->grad())->item<float>();

    float mnX = *std::min_element(vX.begin(), vX.end());
    float mxX = *std::max_element(vX.begin(), vX.end());

    std::cout << "ADD bw (manual upstream): sum dX=" << sX << " min=" << mnX << " max=" << mxX << "\n";
    EXPECT_CLOSE(sX, 1.0f, 1e-6f, "sum dX == 1");
    EXPECT_CLOSE(sb, 1.0f, 1e-6f, "sum db == 1");
    for (float v : vX) EXPECT_CLOSE(v, 1.0f/12.0f, 1e-6f, "dX element");
    for (float v : vb) EXPECT_CLOSE(v, 1.0f/3.0f, 1e-6f, "db element");
}

static void test_mean_backward_no_broadcast() {
    TEST_HEADER("mean backward without broadcast (L = mean(X))");
    size_t A=4, B=3;
    auto X = ir::parameterize(ir::uniform({A,B}, -1, 1));
    auto L = ir::mean(X);
    L->backward();

    auto v = X->grad()->to_vector<float>();
    float s = ir::sum(X->grad())->item<float>();
    float mn = *std::min_element(v.begin(), v.end());
    float mx = *std::max_element(v.begin(), v.end());

    std::cout << "mean(X): sum=" << s << " min=" << mn << " max=" << mx << "\n";
    EXPECT_CLOSE(s, 1.0f, 1e-6f, "sum dX == 1");
    for (float e : v) EXPECT_CLOSE(e, 1.0f/12.0f, 1e-6f, "each grad 1/12");
}

static void test_realization_order_effect() {
    TEST_HEADER("realization order effect (sum vs to_vector)");

    size_t A=4, B=3, C=1;
    auto X = ir::parameterize(ir::uniform({A,B}, -1, 1));
    auto b = ir::parameterize(ir::uniform({C,B}, -1, 1));

    auto L = ir::mean(ir::add(X, b));
    L->backward();

    // Path 1: sum first
    float s1 = ir::sum(X->grad())->item<float>();
    auto v1 = X->grad()->to_vector<float>();
    float mn1 = *std::min_element(v1.begin(), v1.end());
    float mx1 = *std::max_element(v1.begin(), v1.end());
    std::cout << "sum-first: sum=" << s1 << " min=" << mn1 << " max=" << mx1 << "\n";

    // Reset grads
    X->zero_grad(); b->zero_grad();
    L = ir::mean(ir::add(X, b));
    L->backward();

    // Path 2: to_vector first
    auto v2 = X->grad()->to_vector<float>();
    float s2 = ir::sum(X->grad())->item<float>();
    float mn2 = *std::min_element(v2.begin(), v2.end());
    float mx2 = *std::max_element(v2.begin(), v2.end());
    std::cout << "vec-first: sum=" << s2 << " min=" << mn2 << " max=" << mx2 << "\n";

    EXPECT_CLOSE(s1, s2, 1e-7f, "sum equal across orders");
}

int main() {
    backend::DeviceManager::instance().init();

    test_mean_scalar_and_grad();
    test_sum_all_then_div();
    test_broadcast_full_vs_direct_full();

    test_backward_add_mean_explicit_vs_builtin();
    test_add_backward_with_manual_upstream();
    test_mean_backward_no_broadcast();
    test_realization_order_effect();

    return 0;
}
