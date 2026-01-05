// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <vector>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/dtype.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;

static void test_deep_copy() {
    TEST_HEADER("CPU allocator: deep copy");

    std::vector<float> v{1.f, 2.f, 3.f};
    auto t = ir::from_vector<float>(v, {3});

    // Mutate source to ensure deep copy
    v[0] = 999.f;

    auto out = t->to_vector<float>();
    EXPECT_TRUE(out.size() == 3, "size should be 3");
    EXPECT_TRUE(out[0] == 1.f && out[1] == 2.f && out[2] == 3.f, "buffer must be a deep copy");
    if (g_fail_count == 0) std::cout << "[PASS] deep copy behavior\n";
}

static void test_zero_length() {
    TEST_HEADER("CPU allocator: zero-length");

    std::vector<float> v;
    auto t = ir::from_vector<float>(v, {0});

    EXPECT_TRUE(t->numel() == 0, "numel must be 0 for empty tensor");
    auto out = t->to_vector<float>();
    EXPECT_TRUE(out.empty(), "to_vector must return empty vector");
    if (g_fail_count == 0) std::cout << "[PASS] zero-length behavior\n";
}

int main() {
    try {
        backend::DeviceManager::instance().init();

        test_deep_copy();
        test_zero_length();

        if (g_fail_count == 0) {
            std::cout << "\nALL CPU ALLOCATOR TESTS PASSED\n";
            return 0;
        } else {
            std::cerr << "\nCPU ALLOCATOR TESTS FAILED: " << g_fail_count << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
