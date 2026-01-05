// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <vector>
#include <cassert>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor.h"
#include "tests/helpers.h"

using namespace cppgrad;

int main() {
    try {
        backend::DeviceManager::instance().init();

        TEST_HEADER("Tensor IO tests");

        // Round-trip
        {
            std::vector<float> v{1.f, 2.f, 3.5f};
            auto t = ir::from_vector<float>(v, {3});
            auto out = t->to_vector<float>();
            EXPECT_TRUE(out == v, "round-trip from_vector/to_vector should be equal");
            if (g_fail_count == 0) std::cout << "[PASS] round-trip\n";
        }

        // data_span
        {
            std::vector<float> v{1.f, 2.f, 3.5f};
            auto t = ir::from_vector<float>(v, {3});
            // auto s = t->data_span<float>();
            auto s = t->device() != backend::DeviceType::CPU ? t->to(backend::DeviceType::CPU)->data_span<float>() : t->data_span<float>();
            EXPECT_TRUE(s.size() == 3, "data_span size must match numel");
            EXPECT_TRUE(s[0] == 1.f, "data_span content check");
            if (g_fail_count == 0) std::cout << "[PASS] data_span\n";
        }

        // Zero-length
        {
            std::vector<float> z;
            auto tz = ir::from_vector<float>(z, {0});
            auto oz = tz->to_vector<float>();
            // auto sz = tz->data_span<float>();
            auto sz = tz->device() != backend::DeviceType::CPU ? tz->to(backend::DeviceType::CPU)->data_span<float>() : tz->data_span<float>();
            EXPECT_TRUE(oz.empty(), "to_vector on zero-length should be empty");
            EXPECT_TRUE(sz.size() == 0, "data_span size should be 0 for zero-length");
            if (g_fail_count == 0) std::cout << "[PASS] zero-length\n";
        }

        if (g_fail_count == 0) {
            std::cout << "\nALL TENSOR IO TESTS PASSED\n";
            return 0;
        } else {
            std::cerr << "\nTENSOR IO TESTS FAILED: " << g_fail_count << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }
}
