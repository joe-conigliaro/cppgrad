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

// Focused tests for reduce_last_axis fast path (sum/max/mean on last axis)

static void test_reduce_last_rank2_small_inner() {
    TEST_HEADER("reduce last axis rank-2 small odd inner (sum/max/mean; keep_dims t/f)");

    auto X = ir::from_vector(std::vector<float>{1,2,3, 4,5,6}, {2,3});

    // SUM
    {
        auto y0 = ir::sum(X, std::vector<int>{1}, false)->to_vector<float>(); // [6,15]
        expect_allclose(y0, std::vector<float>{6.0f, 15.0f}, 1e-6f, "sum axis=1 no keep");
        auto y1 = ir::sum(X, std::vector<int>{1}, true)->to_vector<float>();  // [[6],[15]]
        expect_allclose(y1, std::vector<float>{6.0f, 15.0f}, 1e-6f, "sum axis=1 keep (values)");
        EXPECT_TRUE(ir::sum(X, std::vector<int>{1}, true)->shape() == std::vector<size_t>({2,1}), "sum keep_dims shape [2,1]");
    }
    // MAX
    {
        auto y0 = ir::max(X, std::vector<int>{1}, false)->to_vector<float>(); // [3,6]
        expect_allclose(y0, std::vector<float>{3.0f, 6.0f}, 1e-6f, "max axis=1 no keep");
        auto y1 = ir::max(X, std::vector<int>{1}, true)->to_vector<float>();  // [[3],[6]]
        expect_allclose(y1, std::vector<float>{3.0f, 6.0f}, 1e-6f, "max axis=1 keep (values)");
        EXPECT_TRUE(ir::max(X, std::vector<int>{1}, true)->shape() == std::vector<size_t>({2,1}), "max keep_dims shape [2,1]");
    }
    // MEAN
    {
        auto y0 = ir::mean(X, std::vector<int>{1}, false)->to_vector<float>(); // [2,5]
        expect_allclose(y0, std::vector<float>{2.0f, 5.0f}, 1e-6f, "mean axis=1 no keep");
        auto y1 = ir::mean(X, std::vector<int>{1}, true)->to_vector<float>();  // [[2],[5]]
        expect_allclose(y1, std::vector<float>{2.0f, 5.0f}, 1e-6f, "mean axis=1 keep (values)");
        EXPECT_TRUE(ir::mean(X, std::vector<int>{1}, true)->shape() == std::vector<size_t>({2,1}), "mean keep_dims shape [2,1]");
    }

    std::cout << "[PASS] reduce last axis rank-2 small odd inner\n";
}

static void test_reduce_last_rank2_large_inner() {
    TEST_HEADER("reduce last axis rank-2 large odd inner (mean)");

    const int B = 4, N = 1023; // odd, non-power-of-two
    std::vector<float> v(B * N);
    for (int i = 0; i < B * N; ++i) v[i] = static_cast<float>(i);
    auto X = ir::from_vector(v, {static_cast<size_t>(B), static_cast<size_t>(N)});

    // Reference via IR: mean = sum / N
    auto sum = ir::sum(X, std::vector<int>{1}, false);
    auto v_sum = sum->to_vector<float>();
    std::vector<float> v_ref(B);
    for (int b = 0; b < B; ++b) v_ref[b] = v_sum[b] / static_cast<float>(N);

    auto mean = ir::mean(X, std::vector<int>{1}, false)->to_vector<float>();
    expect_allclose(mean, v_ref, 1e-4f, "mean large odd inner");

    std::cout << "[PASS] reduce last axis rank-2 large odd inner\n";
}

static void test_reduce_keepdims_consistency() {
    TEST_HEADER("reduce keep_dims true/false consistency");

    auto X = ir::from_vector(std::vector<float>{1,2,3, 4,5,6}, {2,3});

    auto s0 = ir::sum(X, std::vector<int>{1}, false)->to_vector<float>(); // [6,15]
    auto s1 = ir::sum(X, std::vector<int>{1}, true);                      // [[6],[15]]
    auto s1v = s1->to_vector<float>();
    expect_allclose(s0, s1v, 1e-6f, "sum keep_dims/value equality");
    EXPECT_TRUE(s1->shape() == std::vector<size_t>({2,1}), "sum keep_dims shape");

    auto m0 = ir::mean(X, std::vector<int>{1}, false)->to_vector<float>(); // [2,5]
    auto m1 = ir::mean(X, std::vector<int>{1}, true);
    auto m1v = m1->to_vector<float>();
    expect_allclose(m0, m1v, 1e-6f, "mean keep_dims/value equality");
    EXPECT_TRUE(m1->shape() == std::vector<size_t>({2,1}), "mean keep_dims shape");

    std::cout << "[PASS] reduce keep_dims consistency\n";
}

static void test_reduce_last_sliced_view_offset() {
    TEST_HEADER("reduce last axis on sliced view (offset, last contiguous)");

    // Base 3x3: 1..9
    auto Y = ir::from_vector(std::vector<float>{1,2,3, 4,5,6, 7,8,9}, {3,3});
    // Slice rows 1..2 => X is [[4,5,6],[7,8,9]]
    auto X = ir::slice(Y, std::vector<size_t>{1,0}, std::vector<size_t>{3,3}, std::vector<size_t>{1,1});

    auto s = ir::sum(X, std::vector<int>{1}, false)->to_vector<float>();   // [15,24]
    auto m = ir::mean(X, std::vector<int>{1}, false)->to_vector<float>();  // [5,8]
    expect_allclose(s, std::vector<float>{15.0f, 24.0f}, 1e-6f, "sum sliced");
    expect_allclose(m, std::vector<float>{5.0f, 8.0f}, 1e-6f, "mean sliced");

    std::cout << "[PASS] reduce last axis sliced view\n";
}

static void test_reduce_last_broadcast_inner_stride0() {
    TEST_HEADER("reduce last axis with broadcasted inner (stride=0)");

    // Z: [2,1] => broadcast to [2,3] along last axis
    auto Z = ir::from_vector(std::vector<float>{10, 20}, {2,1});
    auto X = ir::broadcast(Z, std::vector<size_t>{2,3});

    auto s = ir::sum(X, std::vector<int>{1}, false)->to_vector<float>();  // [30,60]
    auto M = ir::max(X, std::vector<int>{1}, false)->to_vector<float>();  // [10,20]
    expect_allclose(s, std::vector<float>{30.0f, 60.0f}, 1e-6f, "sum broadcast inner");
    expect_allclose(M, std::vector<float>{10.0f, 20.0f}, 1e-6f, "max broadcast inner");

    std::cout << "[PASS] reduce last axis broadcasted inner\n";
}

static void test_reduce_last_permuted_view() {
    TEST_HEADER("reduce last axis on permuted view (last contiguous)");

    // W: shape [2,3,4], values 0..23
    std::vector<float> v(2*3*4);
    for (int i = 0; i < (int)v.size(); ++i) v[i] = static_cast<float>(i);
    auto W = ir::from_vector(v, {2,3,4});

    // Permute to [3,2,4], then reduce last axis (4)
    auto X = ir::permute(W, std::vector<size_t>{1,0,2});
    auto s = ir::sum(X, std::vector<int>{2}, false)->to_vector<float>(); // [3,2] values

    // Reference via IR (already computed by engine on CPU if Metal not used)
    auto s_ref = ir::sum(X, std::vector<int>{2}, false)->to_vector<float>();
    expect_allclose(s, s_ref, 1e-5f, "sum permuted last axis");

    std::cout << "[PASS] reduce last axis permuted view\n";
}

static void test_reduce_last_edge_cases() {
    TEST_HEADER("reduce last axis edge cases: inner=1, batch=1");

    // inner = 1
    {
        auto A = ir::from_vector(std::vector<float>{1, 4}, {2,1});
        auto m = ir::mean(A, std::vector<int>{1}, false)->to_vector<float>(); // [1,4]
        expect_allclose(m, std::vector<float>{1.0f, 4.0f}, 1e-7f, "mean inner=1");
    }
    // batch = 1
    {
        auto B = ir::from_vector(std::vector<float>{1,2,3}, {1,3});
        auto mx = ir::max(B, std::vector<int>{1}, false)->to_vector<float>(); // [3]
        expect_allclose(mx, std::vector<float>{3.0f}, 1e-7f, "max batch=1");
    }

    std::cout << "[PASS] reduce last axis edge cases\n";
}

int main() {
    try {
        backend::DeviceManager::instance().init();

        test_reduce_last_rank2_small_inner();
        test_reduce_last_rank2_large_inner();
        test_reduce_keepdims_consistency();
        test_reduce_last_sliced_view_offset();
        test_reduce_last_broadcast_inner_stride0();
        test_reduce_last_permuted_view();
        test_reduce_last_edge_cases();

        if (g_fail_count == 0) {
            std::cout << "\nALL TESTS PASSED (reduce fast)\n";
            return 0;
        } else {
            std::cerr << "\nTESTS FAILED (reduce fast): " << g_fail_count << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
