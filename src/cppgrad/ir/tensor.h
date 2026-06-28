// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/common/access_meta.h"
#include "cppgrad/common/dtype.h"
#include "cppgrad/compat/span.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/ir/ops.h"
#include "cppgrad/ir/storage_view.h"
#include "cppgrad/utils/ref.h"

namespace cppgrad::utils {
class Arena;
}

namespace cppgrad::ir {

class GraphContext;

// Tensor
class Tensor : public utils::RefCounted {
  public:
    // Factory methods
    static utils::Ref<Tensor>
    make(Op op, std::vector<utils::Ref<const Tensor>> children, const std::vector<size_t> &shape,
         backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
         cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);
    static utils::Ref<Tensor>
    make(Op op, std::vector<utils::Ref<const Tensor>> children, const common::AccessMeta &access,
         backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
         cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);
    static utils::Ref<Tensor>
    make_leaf(std::shared_ptr<backend::Buffer> data, const std::vector<size_t> &shape,
              backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
              cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);

    // Properties
    const std::vector<size_t> &shape() const noexcept;
    size_t numel() const noexcept;
    backend::DeviceType device_type() const noexcept { return _device_type; }
    common::DType dtype() const noexcept { return _dtype; }

    // Access meta
    const common::AccessMeta &access_meta() const noexcept;
    void set_access_meta(common::AccessMeta m);

    // Eval / IO
    std::shared_ptr<backend::Buffer> schedule() const;
    std::shared_ptr<backend::Buffer> eval() const;
    std::shared_ptr<backend::Buffer> realized_buffer() const;
    utils::Ref<Tensor> to(backend::DeviceType device) const;

    std::shared_ptr<backend::Buffer> materialize_buffer() const;

    template <typename T> T item() const;
    template <typename T> std::vector<T> to_vector() const;
    template <typename T> cppgrad::compat::Span<const T> data_span() const;

    // Grad
    bool requires_grad() const noexcept { return _requires_grad; }
    void set_requires_grad(bool rg);
    utils::Ref<Tensor> grad() const noexcept { return _grad; }
    // TODO:
    // void set_grad(utils::Ref<Tensor> g) const { _grad = std::move(g); }
    // OR:
    void set_grad(const utils::Ref<Tensor> &g) const { _grad = g; }
    void set_grad(utils::Ref<Tensor> &&g) const { _grad = std::move(g); }
    void zero_grad() noexcept { _grad = nullptr; }

    // Backward
    void backward();

    // Mutations for optimizers and compiler
    // void attach_buffer(const std::shared_ptr<backend::Buffer>& src);
    void attach_buffer(std::shared_ptr<backend::Buffer> b) const;
    void set_parameter_data(const std::shared_ptr<backend::Buffer> &src);
    void copy_into_parameter(const std::shared_ptr<backend::Buffer> &src);

    utils::Ref<Tensor> assign(const utils::Ref<const Tensor> &src) const;

    // Graph access
    const Op &op() const noexcept { return _op; }
    const std::vector<utils::Ref<const Tensor>> &children() const noexcept { return _children; }

    // Drop references to child nodes. Only valid once this node is realized (its buffer holds the
    // value) and outside autograd (backward needs the children). Used by the executor in no-grad to
    // free the upstream graph as soon as a node is realized, so long autoregressive chains (e.g. the
    // KV concat chain) stay O(n) memory instead of O(n^2). Like attach_buffer(), mutates a const
    // node (the graph node is logically frozen once realized).
    void detach_children() const { const_cast<Tensor *>(this)->_children.clear(); }

    // Convenience
    bool is_leaf() const noexcept { return std::holds_alternative<LeafOp>(_op); }
    bool is_canonical_leaf() const {
        if (!is_leaf())
            return false;
        const auto &acc = access_meta();
        return acc.contiguous && acc.offset == 0;
    }

    const StorageView &storage_view() const noexcept { return _sv; }

    // Memory
    void check_liveness(const char *caller_name) const;

    void destroy() const override {
        if (_generation_id == 0) {
            // Heap Tensor (Deallocate now).
            delete this;
        } else {
            // Arena Tensor (Arena destructor will reclaim memory later).
        }
    }

  private:
    Tensor(Op op, std::vector<utils::Ref<const Tensor>> children, const std::vector<size_t> &shape,
           backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
           cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);
    Tensor(Op op, std::vector<utils::Ref<const Tensor>> children, const common::AccessMeta &access,
           backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
           cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);
    Tensor(std::shared_ptr<backend::Buffer> data, const std::vector<size_t> &shape,
           backend::DeviceType device_type = cppgrad::backend::DeviceManager::default_device_type(),
           cppgrad::common::DType dtype = cppgrad::common::DType::FLOAT32);

    utils::Ref<Tensor> self() { return utils::Ref<Tensor>(this); }
    utils::Ref<const Tensor> self() const { return utils::Ref<const Tensor>(this); }
    utils::Ref<Tensor> self_mut() const { return utils::Ref<Tensor>(const_cast<Tensor *>(this)); }

    void compute_requires_grad() {
        if (!GradMode::enabled || !ir::is_differentiable(_op)) {
            _requires_grad = false;
            return;
        }
        bool req_grad = false;
        for (const auto &c : _children)
            if (c && c->requires_grad()) {
                req_grad = true;
                break;
            }
        _requires_grad = req_grad;
    }

    StorageView _sv;
    Op _op;
    std::vector<utils::Ref<const Tensor>> _children;
    backend::DeviceType _device_type;
    common::DType _dtype;
    bool _requires_grad = false;
    mutable utils::Ref<Tensor> _grad = nullptr;
    //  0 = Heap  (always alive)
    // >0 = Arena (scoped)
    size_t _generation_id = 0;
    mutable size_t _last_enqueued_token = 0;

    // Allow utils::Arena::alloc<Tensor>(...) to invoke private constructors
    friend class cppgrad::utils::Arena;
    friend class GraphContext;
};

// TODO: make these view aware
// template<typename T>
// T Tensor::item() const {
//     if (numel() != 1) {
//         throw std::runtime_error("item(): tensor must be scalar (numel==1)");
//     }
//     if (common::dtype_v<T> != this->dtype()) {
//         throw std::runtime_error("item(): dtype mismatch");
//     }
//     // const auto& buf = schedule();
//     const auto& buf = eval();
//     T result{};
//     auto dev = backend::DeviceManager::device(this->device_type());
//     if (!dev) throw std::runtime_error("item(): device not found");
//     dev->allocator()->copy_device_to_host(&result, *buf);
//     return result;
// }
template <typename T> T Tensor::item() const {
    if (numel() != 1)
        throw std::runtime_error("item(): tensor must be scalar (numel==1)");
    if (common::dtype_v<T> != this->dtype())
        throw std::runtime_error("item(): dtype mismatch");

    auto dev = backend::DeviceManager::device(this->device_type());
    if (!dev)
        throw std::runtime_error("item(): device not found");

    // Materialize view to a 1-element dense buffer, then copy.
    auto dense = materialize_buffer();

    T result{};
    dev->allocator()->copy_device_to_host(&result, *dense);
    return result;
}

// template<typename T>
// std::vector<T> Tensor::to_vector() const {
//     if (common::dtype_v<T> != this->dtype()) {
//         throw std::runtime_error("to_vector: dtype mismatch");
//     }
//     // const auto& buf = schedule();
//     const auto& buf = eval();
//     std::vector<T> host(this->numel());
//     if (this->numel() > 0) {
//         auto dev = backend::DeviceManager::device(this->device_type());
//         if (!dev) throw std::runtime_error("to_vector: device not found");
//         dev->allocator()->copy_device_to_host(host.data(), *buf);
//     }
//     return host;
// }
template <typename T> std::vector<T> Tensor::to_vector() const {
    if (common::dtype_v<T> != this->dtype())
        throw std::runtime_error("to_vector: dtype mismatch");

    std::vector<T> host(this->numel());
    if (this->numel() == 0)
        return host;

    auto dev = backend::DeviceManager::device(this->device_type());
    if (!dev)
        throw std::runtime_error("to_vector: device not found");

    auto dense = materialize_buffer();
    dev->allocator()->copy_device_to_host(host.data(), *dense);
    return host;
}

template <typename T> cppgrad::compat::Span<const T> Tensor::data_span() const {
    if (common::dtype_v<T> != this->dtype())
        throw std::runtime_error("data_span: dtype mismatch");
    // const auto& buf = schedule();
    const auto &buf = eval();

    if (buf->device_type() != backend::DeviceType::CPU) {
        throw std::runtime_error("data_span: only valid for CPU tensors. Use to_vector()/item() for device tensors.");
    }

    if (!buf)
        throw std::runtime_error("data_span: null buffer");
    const T *p = static_cast<const T *>(buf->data());
    if (!p && numel() > 0)
        throw std::runtime_error("data_span: null ptr on non-empty tensor");
    return cppgrad::compat::Span<const T>(p, numel());
}

} // namespace cppgrad::ir
