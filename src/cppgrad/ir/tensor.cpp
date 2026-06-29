// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/ir/tensor.h"

#include <iostream>

#include "cppgrad/backend/backend.h"
#include "cppgrad/backend/copy.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/common/access_meta.h"
#include "cppgrad/executor/interpreter/interpreter_executor.h"
#include "cppgrad/ir/autograd.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/ops.h"
#include "cppgrad/ir/storage_view.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/utils/vector.h"

namespace cppgrad::ir {

static size_t generation_id() { return GraphContext::active() ? GraphContext::instance().generation() : 0; }

static void warn_no_auto_graph_scope() {
    static bool warned = false;
    if (!warned) {
        std::cerr << "[WARNING] cppgrad: Creating graph nodes outside of an GraphScope.\n"
                  << "          - Tensors are being allocated on the Heap.\n"
                  << "          - Memory will be managed via Ref-counting (RAII) instead of Arena Allocation.\n"
                  << "          - Ensure you do not mix dead Arena tensors with Heap tensors.\n";
        warned = true;
    }
}

std::shared_ptr<backend::Buffer> Tensor::materialize_buffer() const {
    auto dev = backend::DeviceManager::device(device_type());
    if (!dev)
        throw std::runtime_error("materialize: device not found");

    auto src = eval();
    if (!src)
        throw std::runtime_error("materialize: null src buffer");

    auto dst = dev->allocator()->allocate(numel(), dtype());

    auto vs = backend::View::from(access_meta());
    auto vd = backend::View::from(common::AccessMeta::contiguous_from(shape(), 0));

    dev->backend()->copy_view(*src, vs, *dst, vd);
    return dst;
}

// Overload 1: Shape-based
// Used by: Compute Ops (Add, Mul, MatMul) and RandomOp
utils::Ref<Tensor> Tensor::make(Op op, std::vector<utils::Ref<const Tensor>> children, const std::vector<size_t> &shape,
                                backend::DeviceType device_type, common::DType dtype) {
    // Fast Path: Inside GraphScope -> Arena Allocation
    if (GraphContext::active()) {
        return GraphContext::instance().make_node(std::move(op), std::move(children), shape, device_type, dtype);
    }

// Slow Path: Heap Allocation (Fallback)
// If this is a compute node (like Add/Mul) happening outside a scope, warn the user.
#ifdef CPPGRAD_DEBUG
    if (!is_start_node(op))
        warn_no_auto_graph_scope();
#endif

    // Allocate on Heap (Generation 0)
    // Ref<T> will handle 'delete' automatically when ref_count hits 0.
    return utils::Ref<Tensor>(new Tensor(std::move(op), std::move(children), shape, device_type, dtype));
}

// Overload 2: common::AccessMeta-based
// Used by: View Ops (Reshape, Slice, Permute, Broadcast)
utils::Ref<Tensor> Tensor::make(Op op, std::vector<utils::Ref<const Tensor>> children, const common::AccessMeta &access,
                                backend::DeviceType device_type, common::DType dtype) {
    // Fast Path: Inside GraphScope -> Arena Allocation
    if (GraphContext::active()) {
        return GraphContext::instance().make_node(std::move(op), std::move(children), access, device_type, dtype);
    }

    // Slow Path: Heap Allocation (Fallback)

#ifdef CPPGRAD_DEBUG
    if (!is_start_node(op))
        warn_no_auto_graph_scope();
#endif

    // Allocate on Heap (Generation 0)
    return utils::Ref<Tensor>(new Tensor(std::move(op), std::move(children), access, device_type, dtype));
}

utils::Ref<Tensor> Tensor::make_leaf(std::shared_ptr<backend::Buffer> data, const std::vector<size_t> &shape,
                                     backend::DeviceType device_type, common::DType dtype) {
    return utils::Ref<Tensor>(new Tensor(std::move(data), shape, device_type, dtype));
}

void Tensor::check_liveness(const char *caller_name) const {
    // If gen is 0, it's a parameter/heap tensor, always valid.
    if (_generation_id == 0)
        return;
    // If gen > 0, it must match the current context generation.
    size_t current_gen = GraphContext::instance().generation();
    if (_generation_id != current_gen) {
        std::string msg = "Tensor Error: Accessing a tensor from a closed/reset GraphScope (Generation mismatch). ";
        msg += "Tensor Gen: " + std::to_string(_generation_id) + ", Current Gen: " + std::to_string(current_gen);
        msg += ". Operation: " + std::string(caller_name);
        throw std::runtime_error(msg);
    }
}

Tensor::Tensor(Op op, std::vector<utils::Ref<const Tensor>> children, const std::vector<size_t> &shape,
               backend::DeviceType device_type, common::DType dtype)
    : _op(std::move(op)), _children(std::move(children)), _device_type(device_type), _dtype(dtype) {
    _sv.buffer = nullptr;
    _sv.access_meta = common::AccessMeta::contiguous_from(shape, 0);
    _generation_id = generation_id();
    compute_requires_grad();
}

Tensor::Tensor(Op op, std::vector<utils::Ref<const Tensor>> children, const common::AccessMeta &access,
               backend::DeviceType device_type, common::DType dtype)
    : _op(std::move(op)), _children(std::move(children)), _device_type(device_type), _dtype(dtype) {
    _sv.buffer = nullptr;
    _sv.access_meta = access;
    _sv.access_meta.recompute_contiguity();
    _generation_id = generation_id();
    compute_requires_grad();
}

Tensor::Tensor(std::shared_ptr<backend::Buffer> data, const std::vector<size_t> &shape, backend::DeviceType device_type,
               common::DType dtype)
    : _op(LeafOp{}), _children(), _device_type(device_type), _dtype(dtype) {
    _sv = StorageView::contiguous_from(std::move(data), shape, 0);
    _generation_id = generation_id();
}

// Basic methods

const std::vector<size_t> &Tensor::shape() const noexcept { return _sv.access_meta.shape; }

size_t Tensor::numel() const noexcept { return cppgrad::utils::vector::numel(shape()); }

const common::AccessMeta &Tensor::access_meta() const noexcept { return _sv.access_meta; }

void Tensor::set_access_meta(common::AccessMeta m) {
    _sv.access_meta = std::move(m);
    _sv.access_meta.recompute_contiguity();
}

// If `GraphContext` (`GraphScope`) is active then schedule batched realization.
// Otherwise realize now through `eval()`.
std::shared_ptr<backend::Buffer> Tensor::schedule() const {
    check_liveness("Tensor::schedule");
    if (_sv.buffer)
        return _sv.buffer;
    if (!GraphContext::active()) {
        return eval();
    }
    GraphContext::instance().schedule_realization(self());
    return _sv.buffer;
}

// Synchronously realize this node now using the interpreter executor.
std::shared_ptr<backend::Buffer> Tensor::eval() const {
    check_liveness("Tensor::eval");
    if (_sv.buffer)
        return _sv.buffer;
    executor::interpreter::InterpreterExecutor compiler;
    compiler.realize(self());
    return _sv.buffer;
}

std::shared_ptr<backend::Buffer> Tensor::realized_buffer() const { return _sv.buffer; }

utils::Ref<Tensor> Tensor::to(backend::DeviceType device) const {
    if (this->device_type() == device)
        return self_mut();
    return Tensor::make(CopyOp{}, {self()}, this->shape(), device, this->dtype());
}

void Tensor::set_requires_grad(bool rg) {
    if (rg && !std::holds_alternative<LeafOp>(_op)) {
        throw std::runtime_error(
            "set_requires_grad(true): only allowed on leaf tensors. Use parameter()/parameterize().");
    }
    _requires_grad = rg;
}

void Tensor::attach_buffer(std::shared_ptr<backend::Buffer> buf) const {
    if (!buf && this->numel() > 0)
        throw std::runtime_error("attach_buffer: null buffer for non-empty tensor");

#ifdef CPPGRAD_DEBUG
    const size_t cap_elems = buf ? (buf->size_bytes() / common::size(this->dtype())) : 0;

    const auto &am = this->access_meta();

    if (this->numel() == 0) {
        // ok
    } else if (am.offset == 0 && am.contiguous) {
        // materialized/owned tensor: buffer must hold at least numel elements. Usually exact,
        // but a contiguous prefix view into a larger pool buffer (e.g. CacheUpdateOp returning
        // cache[0:end] of a preallocated [0:max_len]) is legitimately backed by a bigger buffer.
        const size_t expect = this->numel();
        if (cap_elems < expect) {
            throw std::runtime_error("attach_buffer: backing buffer too small (identity tensor)");
        }
    } else {
        // view tensor: buffer may be larger, must be large enough
        size_t max_idx = am.offset;
        for (size_t d = 0; d < am.shape.size(); ++d) {
            const size_t dim = am.shape[d];
            const size_t st = am.strides[d];
            if (dim)
                max_idx += (dim - 1) * st;
        }
        const size_t required = max_idx + 1;
        if (cap_elems < required) {
            throw std::runtime_error("attach_buffer: backing buffer too small for view");
        }
    }
#endif

    _sv.buffer = std::move(buf);
}

// Canonical parameter update: replace buffer, retag leaf, normalize view
void Tensor::set_parameter_data(const std::shared_ptr<backend::Buffer> &src) {
    const size_t n = this->numel();
    if (!src && n > 0) {
        throw std::runtime_error("set_parameter_data: null buffer for non-empty param");
    }

    const size_t elems_src = src ? (src->size_bytes() / common::size(this->dtype())) : 0;
    if (src && elems_src != n) {
        throw std::runtime_error("set_parameter_data: size mismatch");
    }
    if (src && src->dtype() != this->dtype()) {
        throw std::runtime_error("set_parameter_data: dtype mismatch");
    }

    _sv.buffer = src;
    _op = LeafOp{};
    _sv.access_meta = common::AccessMeta::contiguous_from(this->shape(), 0);
    // Clear children to detach from the computation graph - a re-tagged leaf must not
    // keep the upstream graph alive through refcounted children references.
    _children.clear();

#ifdef CPPGRAD_DEBUG
    if (!is_canonical_leaf())
        throw std::runtime_error("copy_into_parameter: non-canonical param view");
#endif
}

void Tensor::copy_into_parameter(const std::shared_ptr<backend::Buffer> &src) {
    if (!src)
        throw std::runtime_error("copy_into_parameter: src null");
    if (src->dtype() != this->dtype())
        throw std::runtime_error("copy_into_parameter: dtype mismatch");

    if (!_sv.buffer) {
        auto *dev = backend::DeviceManager::device(this->device_type());
        if (!dev)
            throw std::runtime_error("copy_into_parameter: device not found");
        _sv.buffer = dev->allocator()->allocate(this->numel(), this->dtype());
    }

#ifdef CPPGRAD_DEBUG
    const size_t expect_bytes = this->numel() * common::size(this->dtype());
    if (_sv.buffer->size_bytes() != expect_bytes) {
        throw std::runtime_error("copy_into_parameter: dst buffer inconsistent with tensor metadata (debug)");
    }
#endif

    if (src->size_bytes() != _sv.buffer->size_bytes()) {
        throw std::runtime_error("copy_into_parameter: size mismatch");
    }

    backend::copy(*_sv.buffer, *src);

    _op = LeafOp{};
    _sv.access_meta = common::AccessMeta::contiguous_from(this->shape(), 0);
    // Clear children to detach from the computation graph.
    _children.clear();

#ifdef CPPGRAD_DEBUG
    if (!is_canonical_leaf())
        throw std::runtime_error("copy_into_parameter: non-canonical param view");
#endif
}

utils::Ref<Tensor> Tensor::assign(const utils::Ref<const Tensor> &src) const { return ir::assign(self(), src); }

//  Backward


void Tensor::backward() {
    check_liveness("Tensor::backward");

    if (!_requires_grad) {
        throw std::runtime_error("backward(): tensor does not require grad");
    }
    if (numel() != 1) {
        throw std::runtime_error("backward(): only scalar tensors supported");
    }

    // Build tape (topological post-order)
    std::vector<const Tensor *> tape;
    tape.reserve(128);
    std::unordered_set<const Tensor *> visited;
    visited.reserve(256);
    std::function<void(const Tensor *)> dfs = [&](const Tensor *t) {
        if (!t || visited.count(t))
            return;
        if (!t->requires_grad())
            return;
        visited.insert(t);
        for (const auto &c : t->children())
            dfs(c.get());
        tape.push_back(t);
    };
    dfs(this);

    // Incoming grads map
    using GradMap = std::unordered_map<const Tensor *, utils::Ref<Tensor>>;
    GradMap incoming;
    incoming.reserve(tape.size());

    // Seed grad (scalar one)
    incoming[this] =
        Tensor::make(ConstantOp{ConstantOpType::SCALAR, 1.0}, {}, std::vector<size_t>{}, device_type(), dtype());

    // Reverse sweep
    for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
        const Tensor *node = *it;

        auto itg = incoming.find(node);
        if (itg == incoming.end())
            continue;

        utils::Ref<Tensor> grad_this = itg->second;
        if (!grad_this)
            continue;

        // Leaf: do not propagate further. The leaf's final grad is incoming[node].
        if (std::holds_alternative<LeafOp>(node->_op)) {
            continue;
        }

        const auto &children = node->children();

        // Dispatch the node's gradient rule (one free function per op; see ir/autograd.{h,cpp}).
        BackwardCtx ctx{node, children, grad_this};
        Grads child_grads = std::visit([&](auto &&op) { return backward_op(op, ctx); }, node->_op);

        // Align and accumulate each child's gradient into incoming[child]
        for (size_t i = 0; i < children.size(); ++i) {
            auto child = children[i];
            if (!child)
                continue;
            if (i >= child_grads.size())
                continue;

            auto cgrad = child_grads[i];
            if (!cgrad)
                continue;

            cgrad = unify_to_shape(cgrad, child->shape());

            // accumulate
            auto &slot = incoming[child.get()];
            slot = slot ? Tensor::make(BinaryOp{BinaryOpType::ADD}, {slot, cgrad}, slot->shape(), slot->device_type(),
                                       slot->dtype())
                        : cgrad;
        }
    }

    // Materialize grads onto leaf tensors
    for (auto &[n, g] : incoming) {
        if (!n || !g)
            continue;
        if (!std::holds_alternative<LeafOp>(n->_op))
            continue;
        if (!n->requires_grad())
            continue;

        if (auto old = n->grad()) {
            n->set_grad(
                Tensor::make(BinaryOp{BinaryOpType::ADD}, {old, g}, old->shape(), old->device_type(), old->dtype()));
        } else {
            n->set_grad(g);
        }
    }
}

} // namespace cppgrad::ir
