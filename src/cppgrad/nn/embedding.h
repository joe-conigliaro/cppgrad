// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <cstddef>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/nn/module.h"

namespace cppgrad::nn {

class Embedding : public Module {
  public:
    utils::Ref<ir::Tensor> weight;

    Embedding(size_t vocab_size, size_t embed_dim,
              backend::DeviceType device_type = backend::DeviceManager::default_device_type()) {
        float limit = std::sqrt(1.0f / static_cast<float>(embed_dim));
        auto w_init = ir::uniform({vocab_size, embed_dim}, -limit, limit, device_type);
        weight = ir::parameterize(w_init);
        register_parameter("weight", weight);
    }

    utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor> &ids) override { return ir::gather(weight, ids); }
};

} // namespace cppgrad::nn
