// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <vector>
#include <iostream>
#include <functional>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/utils/vector.h"
#include "tests/helpers.h"

using namespace cppgrad;

int main() {
    try {
        backend::DeviceManager::instance().init();
        Tolerances tol{1e-6, 1e-5};

        // Elementwise binary ops with broadcasting
        {
            TEST_HEADER("ew add/mul/div/pow/min/max (broadcast)");

            std::vector<size_t> sa = {2, 3, 1};
            std::vector<size_t> sb = {1, 3, 4};

            size_t na = utils::vector::numel(sa);
            size_t nb = utils::vector::numel(sb);

            std::vector<float> ha(na), hb(nb);
            fill_random(ha, -1, 1, 42);
            fill_random(hb, 0.1f, 1, 43); // Use positive values for b to avoid div/pow issues

            // Create Tensors on both devices
            auto a_cpu = ir::from_vector(ha, sa, backend::DeviceType::CPU);
            auto b_cpu = ir::from_vector(hb, sb, backend::DeviceType::CPU);

            auto a_gpu = ir::from_vector(ha, sa, backend::DeviceType::METAL);
            auto b_gpu = ir::from_vector(hb, sb, backend::DeviceType::METAL);

            using BinaryOpFn = utils::Ref<ir::Tensor>(*)(const utils::Ref<const ir::Tensor>&, const utils::Ref<const ir::Tensor>&);
            struct Op {
                const char* name;
                BinaryOpFn fn;
            };

            Op ops[] = {
                {"add", &ir::add},
                {"mul", &ir::mul},
                {"div", &ir::div},
                {"pow", &ir::pow},
                {"min", &ir::min},
                {"max", &ir::max}
            };

            for (const auto& op : ops) {
                // Perform the operation on both CPU and GPU tensors
                auto out_cpu_tensor = op.fn(a_cpu, b_cpu);
                auto out_gpu_tensor = op.fn(a_gpu, b_gpu);

                // Download results and compare
                auto v_cpu = out_cpu_tensor->to_vector<float>();
                auto v_gpu = out_gpu_tensor->to_vector<float>();

                EXPECT_TRUE(v_cpu.size() == v_gpu.size(), std::string(op.name) + ": size mismatch");
                if (!v_cpu.empty()) {
                    EXPECT_ALLCLOSE(v_gpu.data(), v_cpu.data(), v_cpu.size(), tol, std::string(op.name));
                }
            }
        }

        // Permute and Broadcast
        {
            TEST_HEADER("permute & broadcast");

            std::vector<size_t> s = {2, 3, 4};
            std::vector<float> h(utils::vector::numel(s));
            fill_random(h, -2, 2, 123);

            // Permute
            std::vector<size_t> axes = {0, 2, 1};
            auto cpu_permuted = ir::permute(ir::from_vector(h, s, backend::DeviceType::CPU), axes);
            auto gpu_permuted = ir::permute(ir::from_vector(h, s, backend::DeviceType::METAL), axes);

            auto v_cpu_permute = cpu_permuted->to_vector<float>();
            auto v_gpu_permute = gpu_permuted->to_vector<float>();
            EXPECT_TRUE(v_cpu_permute.size() == v_gpu_permute.size(), "permute: size mismatch");
            if (!v_cpu_permute.empty()) {
                EXPECT_ALLCLOSE(v_gpu_permute.data(), v_cpu_permute.data(), v_cpu_permute.size(), tol, "permute");
            }

            // Broadcast
            std::vector<size_t> in_shape = {2, 1, 4};
            std::vector<size_t> out_shape = {2, 3, 4};
            std::vector<float> hb(utils::vector::numel(in_shape));
            fill_random(hb, -1, 1, 777);

            auto cpu_broadcasted = ir::broadcast(ir::from_vector(hb, in_shape, backend::DeviceType::CPU), out_shape);
            auto gpu_broadcasted = ir::broadcast(ir::from_vector(hb, in_shape, backend::DeviceType::METAL), out_shape);

            auto v_cpu_broadcast = cpu_broadcasted->to_vector<float>();
            auto v_gpu_broadcast = gpu_broadcasted->to_vector<float>();
            EXPECT_TRUE(v_cpu_broadcast.size() == v_gpu_broadcast.size(), "broadcast: size mismatch");
            if (!v_cpu_broadcast.empty()) {
                EXPECT_ALLCLOSE(v_gpu_broadcast.data(), v_cpu_broadcast.data(), v_cpu_broadcast.size(), tol, "broadcast");
            }
        }


        // Diagnostic: read first few elements of the input that is to be broadcast
        {
            std::vector<size_t> in_shape = {2, 1, 4};
            std::vector<size_t> out_shape = {2, 3, 4};
            std::vector<float> hb(utils::vector::numel(in_shape));
            fill_random(hb, -1, 1, 777);
            auto cpu_in  = ir::from_vector(hb, in_shape, backend::DeviceType::CPU);
            auto gpu_in  = ir::from_vector(hb, in_shape, backend::DeviceType::METAL);
            auto cpu_in_v = cpu_in->to_vector<float>();
            auto gpu_in_v = gpu_in->to_vector<float>();
            for (int i = 0; i < std::min<int>(hb.size(), 8); ++i) {
                EXPECT_CLOSE(cpu_in_v[i], hb[i], 1e-7f, "CPU in matches host");
                EXPECT_CLOSE(gpu_in_v[i], hb[i], 1e-7f, "GPU in matches host");
            }
            // Now explicit broadcast and read index 8 (coords [0,2,0])
            auto cpu_b = ir::broadcast(cpu_in, out_shape);
            auto gpu_b = ir::broadcast(gpu_in, out_shape);
            auto v_cpu = cpu_b->to_vector<float>();
            auto v_gpu = gpu_b->to_vector<float>();
            size_t idx = 8; // coords (0,2,0) for out_shape {2,3,4}
            EXPECT_CLOSE(v_cpu[idx], hb[0], 1e-6f, "CPU broadcast idx=8");
            EXPECT_CLOSE(v_gpu[idx], hb[0], 1e-6f, "GPU broadcast idx=8");
        }


        // Reductions
        {
            TEST_HEADER("reductions (sum/mean/max) axis=1");

            std::vector<size_t> s = {3, 4, 5};
            std::vector<float> h(utils::vector::numel(s));
            fill_random(h, -1, 1, 888);

            auto a_cpu = ir::from_vector(h, s, backend::DeviceType::CPU);
            auto a_gpu = ir::from_vector(h, s, backend::DeviceType::METAL);

            std::vector<int> axes = {1};

            // Test sum
            auto sum_cpu = ir::sum(a_cpu, axes, false)->to_vector<float>();
            auto sum_gpu = ir::sum(a_gpu, axes, false)->to_vector<float>();
            EXPECT_ALLCLOSE(sum_gpu.data(), sum_cpu.data(), sum_cpu.size(), tol, "sum");

            // Test mean
            auto mean_cpu = ir::mean(a_cpu, axes, false)->to_vector<float>();
            auto mean_gpu = ir::mean(a_gpu, axes, false)->to_vector<float>();
            EXPECT_ALLCLOSE(mean_gpu.data(), mean_cpu.data(), mean_cpu.size(), tol, "mean");

            // Test max
            auto max_cpu = ir::max(a_cpu, axes, false)->to_vector<float>();
            auto max_gpu = ir::max(a_gpu, axes, false)->to_vector<float>();
            EXPECT_ALLCLOSE(max_gpu.data(), max_cpu.data(), max_cpu.size(), tol, "max");
        }

        if (g_fail_count == 0) {
            std::cout << "\nALL CPU vs Metal tests passed\n";
            return 0;
        } else {
            std::cout << "\nCPU vs Metal tests had " << g_fail_count << " failures\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
