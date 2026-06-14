// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// A linear projection that is either dense or quantized, behind one forward(x). Lets the Qwen
// blocks keep weights 8-bit for inference (call ir::quantized_matmul) while the dense path is still
// used for random-init / dry-run / training.
//
//   dense    : weight [in, out]                       -> matmul(x, weight)
//   quantized: qweight [out, in/pack], scales/biases  -> quantized_matmul(x, qweight, scales, biases)
//              [out, in/group_size]
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/ops.h"      // ir::QuantParams
#include "cppgrad/utils/ref.h"

namespace cppgrad {
namespace nn {
namespace llm {
namespace qwen {

struct QLinear {
    // Dense weight [in, out] (matmul convention). Used when !quantized.
    utils::Ref<ir::Tensor> weight;
    // Quantized weights (MLX affine etc.): qweight [out, in/pack_factor], scales/biases [out, groups].
    utils::Ref<ir::Tensor> qweight, scales, biases;
    ir::QuantParams params;
    bool quantized = false;

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& x) const {
        return quantized ? ir::quantized_matmul(x, qweight, scales, biases, params)
                         : ir::matmul(x, weight);
    }
};

}  // namespace qwen
}  // namespace llm
}  // namespace nn
}  // namespace cppgrad
