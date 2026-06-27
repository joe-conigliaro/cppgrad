// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <stdexcept>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/device.h"
#include "cppgrad/common/dtype.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/utils/ref.h"

namespace cppgrad::ir {

// Create a leaf parameter. Storage can be allocated now or deferred until first assign or eval.
inline utils::Ref<Tensor> parameter(const std::vector<size_t>& shape,
    cppgrad::backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
    cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32, bool allocate_now = true) {
    std::shared_ptr<cppgrad::backend::Buffer> storage = nullptr;
    if (allocate_now) {
        auto* device_obj = cppgrad::backend::DeviceManager::device(device_type);
        if (!device_obj) throw std::runtime_error("parameter: device not found");
        storage = device_obj->allocator()->allocate(cppgrad::utils::vector::numel(shape), dtype);
    } else {
        // deferred allocation
        storage = nullptr;
    }
    auto param = Tensor::make_leaf(storage, shape, device_type, dtype);
    // Leaf parameter must be leaf op; set requires_grad true
    param->set_requires_grad(true);

    if (!param->is_canonical_leaf()) throw std::runtime_error("parameter: non-canonical leaf");

    return param;
}

// Convert any tensor (graph or leaf) into a leaf Parameter.
// Ensures: becomes LeafOp, same shape/device/dtype, storage filled, requires_grad=true.
inline utils::Ref<Tensor> parameterize(const utils::Ref<Tensor>& t) {
    if (!t) throw std::runtime_error("parameterize: null tensor");

    if (t->is_canonical_leaf()) {
        auto buf = t->eval();
        if (!buf) throw std::runtime_error("parameterize: canonical leaf without buffer");
        t->set_requires_grad(true);
        return t;
    }

    // Realize once and attach to a new leaf
    auto buf = t->eval();
    if (!buf) throw std::runtime_error("parameterize: realization failed (null buffer)");

    auto param = Tensor::make_leaf(buf, t->shape(), t->device_type(), t->dtype());
    param->set_requires_grad(true);

    if (!param->is_canonical_leaf()) throw std::runtime_error("parameterize: non-canonical leaf");

    return param;
}

// Commit a (possibly in-place / effectful) graph node to a detached leaf.
//
// Realizes `t` and returns a fresh canonical leaf wrapping the realized buffer, dropping the
// producing graph. Unlike parameterize(), this is the decode-runtime commit primitive (see
// docs/decode-runtime.md): it is correct in the presence of committed in-place effects because it
// relies on the executor's "realized = immutable boundary" memoization -- a node that already holds
// its buffer is never recomputed, so committed in-place writes (cache_update / assign) are never
// re-applied or re-ordered. Use it to detach the linear-attention recurrent/conv state (and any
// per-chunk hidden) between prefill chunks / decode steps so the upstream graph -- and its buffers --
// can be freed, bounding memory on long prefills.
//
// If `t` is a non-canonical view (offset / strided), its prefix is materialized into a dense buffer
// so the returned leaf is canonical (contiguous, offset 0). requires_grad is false: commit is an
// inference-only detach (it severs autograd history by construction).
inline utils::Ref<Tensor> commit(const utils::Ref<Tensor>& t) {
    if (!t) throw std::runtime_error("commit: null tensor");

    std::shared_ptr<cppgrad::backend::Buffer> buf;
    const auto& acc = t->access_meta();
    const bool canonical = acc.contiguous && acc.offset == 0;
    if (canonical) {
        buf = t->eval();   // realizes; realized nodes are memoized (committed writes never re-run)
    } else {
        // View: copy the prefix into its own dense buffer so the leaf doesn't alias a mutating cache.
        buf = t->materialize_buffer();
    }
    if (!buf) throw std::runtime_error("commit: realization failed (null buffer)");

    auto leaf = Tensor::make_leaf(buf, t->shape(), t->device_type(), t->dtype());
    leaf->set_requires_grad(false);
    if (!leaf->is_canonical_leaf()) throw std::runtime_error("commit: non-canonical leaf");
    return leaf;
}

// Convenience alias
inline utils::Ref<ir::Tensor> trainable_from(const utils::Ref<ir::Tensor>& init) {
    return ir::parameterize(init);
}

} // namespace cppgrad::ir
