// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iostream>
#include "cppgrad/ir/ops.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/storage_view.h"
#include "cppgrad/ir/access_meta.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/executor/interpreter/interpreter_executor.h"
#include "cppgrad/backend/backend.h"
#include "cppgrad/backend/copy.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/utils/vector.h"

namespace cppgrad {
namespace ir {

static size_t generation_id() {
    return GraphContext::active() ? GraphContext::instance().generation() : 0;
}

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
    if (!dev) throw std::runtime_error("materialize: device not found");

    auto src = eval();
    if (!src) throw std::runtime_error("materialize: null src buffer");

    auto dst = dev->allocator()->allocate(numel(), dtype());

    auto vs = backend::View::from(access_meta());
    auto vd = backend::View::from(ir::AccessMeta::contiguous_from(shape(), 0));

    dev->backend()->copy_view(*src, vs, *dst, vd);
    return dst;
}

// Overload 1: Shape-based
// Used by: Compute Ops (Add, Mul, MatMul) and RandomOp
utils::Ref<Tensor> Tensor::make(Op op, std::vector<utils::Ref<const Tensor>> children,
    const std::vector<size_t>& shape, backend::DeviceType device_type, backend::DType dtype) {
    // Fast Path: Inside GraphScope -> Arena Allocation
    if (GraphContext::active()) {
        return GraphContext::instance().make_node(
            std::move(op),
            std::move(children),
            shape,
            device_type,
            dtype);
    }

    // Slow Path: Heap Allocation (Fallback)
    // If this is a compute node (like Add/Mul) happening outside a scope, warn the user.
    #ifdef CPPGRAD_DEBUG
        if (!is_start_node(op)) warn_no_auto_graph_scope();
    #endif

    // Allocate on Heap (Generation 0)
    // Ref<T> will handle 'delete' automatically when ref_count hits 0.
    return utils::Ref<Tensor>(new Tensor(std::move(op), std::move(children), shape, device_type, dtype));
}

// Overload 2: AccessMeta-based
// Used by: View Ops (Reshape, Slice, Permute, Broadcast)
utils::Ref<Tensor> Tensor::make(Op op, std::vector<utils::Ref<const Tensor>> children,
    const AccessMeta& access, backend::DeviceType device_type, backend::DType dtype) {
    // Fast Path: Inside GraphScope -> Arena Allocation
    if (GraphContext::active()) {
        return GraphContext::instance().make_node(std::move(op), std::move(children), access, device_type, dtype);
    }

    // Slow Path: Heap Allocation (Fallback)

    #ifdef CPPGRAD_DEBUG
        if (!is_start_node(op)) warn_no_auto_graph_scope();
    #endif

    // Allocate on Heap (Generation 0)
    return utils::Ref<Tensor>(new Tensor(std::move(op), std::move(children), access, device_type, dtype));
}

utils::Ref<Tensor> Tensor::make_leaf(std::shared_ptr<backend::Buffer> data,
    const std::vector<size_t>& shape, backend::DeviceType device_type, backend::DType dtype) {
    return utils::Ref<Tensor>(new Tensor(std::move(data), shape, device_type, dtype));
}

void Tensor::check_liveness(const char* caller_name) const {
    // If gen is 0, it's a parameter/heap tensor, always valid.
    if (_generation_id == 0) return;
    // If gen > 0, it must match the current context generation.
    size_t current_gen = GraphContext::instance().generation();
    if (_generation_id != current_gen) {
        std::string msg = "Tensor Error: Accessing a tensor from a closed/reset GraphScope (Generation mismatch). ";
        msg += "Tensor Gen: " + std::to_string(_generation_id) + ", Current Gen: " + std::to_string(current_gen);
        msg += ". Operation: " + std::string(caller_name);
        throw std::runtime_error(msg);
    }
}

Tensor::Tensor(Op op, std::vector<utils::Ref<const Tensor>> children,
    const std::vector<size_t>& shape, backend::DeviceType device_type, backend::DType dtype)
    : _op(std::move(op)), _children(std::move(children)), _device_type(device_type), _dtype(dtype) {
    _sv.buffer = nullptr;
    _sv.access_meta = AccessMeta::contiguous_from(shape, 0);
    _generation_id = generation_id();
    compute_requires_grad();
}

Tensor::Tensor(Op op, std::vector<utils::Ref<const Tensor>> children, const AccessMeta& access,
    backend::DeviceType device_type, backend::DType dtype)
    : _op(std::move(op)), _children(std::move(children)), _device_type(device_type), _dtype(dtype) {
    _sv.buffer = nullptr;
    _sv.access_meta = access;
    _sv.access_meta.recompute_contiguity();
    _generation_id = generation_id();
    compute_requires_grad();
}

Tensor::Tensor(std::shared_ptr<backend::Buffer> data, const std::vector<size_t>& shape,
    backend::DeviceType device_type, backend::DType dtype)
    : _op(LeafOp{}), _children(), _device_type(device_type), _dtype(dtype) {
    _sv = StorageView::contiguous_from(std::move(data), shape, 0);
    _generation_id = generation_id();
}

// Basic methods

const std::vector<size_t>& Tensor::shape() const noexcept { return _sv.access_meta.shape; }

size_t Tensor::numel() const noexcept {
    return cppgrad::utils::vector::numel(shape());
}

const AccessMeta& Tensor::access_meta() const noexcept { return _sv.access_meta; }

void Tensor::set_access_meta(AccessMeta m) {
    _sv.access_meta = std::move(m);
    _sv.access_meta.recompute_contiguity();
}

// If `GraphContext` (`GraphScope`) is active then schedule batched realization.
// Otherwise realize now through `eval()`.
std::shared_ptr<backend::Buffer> Tensor::schedule() const {
    check_liveness("Tensor::schedule");
    if (_sv.buffer) return _sv.buffer;
    if(!GraphContext::active()) {
        return eval();
    }
    GraphContext::instance().schedule_realization(self());
    return _sv.buffer;
}

// Synchronously realize this node now using the interpreter executor.
std::shared_ptr<backend::Buffer> Tensor::eval() const {
    check_liveness("Tensor::eval");
    if (_sv.buffer) return _sv.buffer;
    executor::interpreter::InterpreterExecutor compiler;
    compiler.realize(self());
    return _sv.buffer;
}

std::shared_ptr<backend::Buffer> Tensor::realized_buffer() const {
    return _sv.buffer;
}

utils::Ref<Tensor> Tensor::to(backend::DeviceType device) const {
    if (this->device_type() == device) return self_mut();
    return Tensor::make(CopyOp{}, {self()}, this->shape(), device, this->dtype());
}

void Tensor::set_requires_grad(bool rg) {
    if (rg && !std::holds_alternative<LeafOp>(_op)) {
        throw std::runtime_error("set_requires_grad(true): only allowed on leaf tensors. Use parameter()/parameterize().");
    }
    _requires_grad = rg;
}

void Tensor::attach_buffer(std::shared_ptr<backend::Buffer> buf) const {
    if (!buf && this->numel() > 0) throw std::runtime_error("attach_buffer: null buffer for non-empty tensor");

    #ifdef CPPGRAD_DEBUG
        const size_t cap_elems = buf ? (buf->size_bytes() / backend::size(this->dtype())) : 0;

        const auto& am = this->access_meta();

        if (this->numel() == 0) {
            // ok
        } else if (am.offset == 0 && am.contiguous) {
            // materialized/owned tensor: exact match is a useful invariant
            const size_t expect = this->numel();
            if (cap_elems != expect) {
                throw std::runtime_error("attach_buffer: buffer elements mismatch (identity tensor)");
            }
        } else {
            // view tensor: buffer may be larger, must be large enough
            size_t max_idx = am.offset;
            for (size_t d = 0; d < am.shape.size(); ++d) {
                const size_t dim = am.shape[d];
                const size_t st  = am.strides[d];
                if (dim) max_idx += (dim - 1) * st;
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
void Tensor::set_parameter_data(const std::shared_ptr<backend::Buffer>& src) {
    const size_t n = this->numel();
    if (!src && n > 0) {
        throw std::runtime_error("set_parameter_data: null buffer for non-empty param");
    }

    const size_t elems_src = src ? (src->size_bytes() / backend::size(this->dtype())) : 0;
    if (src && elems_src != n) {
        throw std::runtime_error("set_parameter_data: size mismatch");
    }
    if (src && src->dtype() != this->dtype()) {
        throw std::runtime_error("set_parameter_data: dtype mismatch");
    }

    _sv.buffer = src;
    _op = LeafOp{};
    _sv.access_meta = AccessMeta::contiguous_from(this->shape(), 0);

    #ifdef CPPGRAD_DEBUG
        if (!is_canonical_leaf()) throw std::runtime_error("copy_into_parameter: non-canonical param view");
    #endif
}

void Tensor::copy_into_parameter(const std::shared_ptr<backend::Buffer>& src) {
    if (!src) throw std::runtime_error("copy_into_parameter: src null");
    if (src->dtype() != this->dtype()) throw std::runtime_error("copy_into_parameter: dtype mismatch");

    if (!_sv.buffer) {
        auto* dev = backend::DeviceManager::device(this->device_type());
        if (!dev) throw std::runtime_error("copy_into_parameter: device not found");
        _sv.buffer = dev->allocator()->allocate(this->numel(), this->dtype());
    }

    #ifdef CPPGRAD_DEBUG
        const size_t expect_bytes = this->numel() * backend::size(this->dtype());
        if (_sv.buffer->size_bytes() != expect_bytes) {
            throw std::runtime_error("copy_into_parameter: dst buffer inconsistent with tensor metadata (debug)");
        }
    #endif

    if (src->size_bytes() != _sv.buffer->size_bytes()) {
        throw std::runtime_error("copy_into_parameter: size mismatch");
    }

    backend::copy(*_sv.buffer, *src);

    _op = LeafOp{};
    _sv.access_meta = AccessMeta::contiguous_from(this->shape(), 0);

    #ifdef CPPGRAD_DEBUG
        if (!is_canonical_leaf()) throw std::runtime_error("copy_into_parameter: non-canonical param view");
    #endif
}

utils::Ref<Tensor> Tensor::assign(const utils::Ref<const Tensor>& src) const {
    return ir::assign(self(), src);
}

//  Backward

static utils::Ref<Tensor>
broadcast_to_shape_dbg(utils::Ref<Tensor> g,
                       const std::vector<size_t>& target,
                       const char* tag) {
    if (!g) return g;
    const auto& gs = g->shape();
    if (gs == target) return g;
    std::vector<size_t> g_aligned = gs;
    if (g_aligned.size() < target.size()) {
        g_aligned.insert(g_aligned.begin(), target.size() - g_aligned.size(), 1);
    }
    #ifdef CPPGRAD_DEBUG
        for (size_t d = 0; d < target.size(); ++d) {
            size_t gd = g_aligned[d];
            size_t td = target[d];
            if (!(gd == td || gd == 1)) {
                std::ostringstream os;
                os << "broadcast_to_shape[" << tag << "]: incompatible axis " << d
                << " (grad=" << gd << ", target=" << td << ")"
                << " grad_aligned=" << cppgrad::utils::vector::to_string(g_aligned)
                << " target=" << cppgrad::utils::vector::to_string(target)
                << " original_grad=" << cppgrad::utils::vector::to_string(gs);
                throw std::runtime_error(os.str());
            }
        }
    #endif
    utils::Ref<Tensor> src = g;
    if (g->shape() != g_aligned) {
        src = Tensor::make(
            MovementOp{MovementOpType::RESHAPE, g_aligned},
            {g},
            g_aligned,
            g->device_type(),
            g->dtype());
    }
    return Tensor::make(
        MovementOp{MovementOpType::BROADCAST, target},
        {src},
        target,
        g->device_type(),
        g->dtype());
}

static utils::Ref<Tensor>
broadcast_to_shape(utils::Ref<Tensor> g, const std::vector<size_t>& target) {
    return broadcast_to_shape_dbg(g, target, "default");
}

static utils::Ref<Tensor>
broadcast_grad_for_sum_backward(const utils::Ref<Tensor>& grad_this,
                                const std::vector<size_t>& in_shape,
                                const std::vector<int>& axes,
                                bool /*keep_dims_fwd*/)
{
    if (!grad_this) return grad_this;

    const size_t rank = in_shape.size();
    std::vector<int> axes_norm;
    axes_norm.reserve(axes.size());
    for (int ax : axes) {
        int a = ax < 0 ? ax + static_cast<int>(rank) : ax;
        if (a < 0 || a >= static_cast<int>(rank)) {
            throw std::runtime_error("broadcast_grad_for_sum_backward: axis out of bounds");
        }
        axes_norm.push_back(a);
    }

    std::vector<size_t> kd_shape = in_shape;
    for (int a : axes_norm) kd_shape[(size_t)a] = 1;

    utils::Ref<Tensor> g = grad_this;
    const auto& gs = g->shape();
    if (gs != kd_shape) {
        if (gs.size() > kd_shape.size()) {
            throw std::runtime_error("broadcast_grad_for_sum_backward: grad rank > input rank");
        }
        // If rank differs, insert leading 1s (view reshape)
        if (gs.size() < kd_shape.size()) {
            std::vector<size_t> padded = gs;
            padded.insert(padded.begin(), kd_shape.size() - padded.size(), 1);
            g = reshape_view(g, padded);
        }
        // Reshape to kd_shape (safe reshape; may materialize)
        if (g->shape() != kd_shape) g = reshape(g, kd_shape);
    }
    // Broadcast singleton dims back to input shape
    g = Tensor::make(
        MovementOp{MovementOpType::BROADCAST, in_shape},
        {g},
        AccessMeta::broadcast_from(g->access_meta(), in_shape),
        g->device_type(),
        g->dtype());

    return g;
}

static utils::Ref<Tensor> reduce_to_shape_sum(utils::Ref<Tensor> g, const std::vector<size_t>& target) {
    if (!g) return g;
    const auto& grad_shape = g->shape();
    if (grad_shape == target) return g;

    const size_t gr = grad_shape.size();
    const size_t tr = target.size();

    auto pad_left = [](const std::vector<size_t>& v, size_t R) {
        if (v.size() >= R) return v;
        std::vector<size_t> out;
        out.reserve(R);
        out.insert(out.end(), R - v.size(), 1);
        out.insert(out.end(), v.begin(), v.end());
        return out;
    };
    const size_t R = std::max(gr, tr);
    auto ga = pad_left(grad_shape, R);
    auto ta = pad_left(target, R);

    std::vector<int> axes;
    axes.reserve(R);
    for (size_t d = 0; d < R; ++d) {
        const size_t gd = ga[d], td = ta[d];
        if (td == 1 && gd > 1) {
            axes.push_back(static_cast<int>(d));
        } else if (td != 1 && gd != td) {
            if (!(gd == 1 && td > 1)) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                        "reduce_to_shape_sum: incompatible dims at axis %zu (grad=%zu, target=%zu)",
                        d, gd, td);
                throw std::runtime_error(buf);
            }
        }
    }

    if (!axes.empty()) {
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
        auto target_aligned = ta;
        for (int ax : axes) target_aligned[(size_t)ax] = 1;
        g = Tensor::make(
            ReduceOp{ReduceOpType::SUM, axes, /*keep_dims=*/true},
            {g},
            target_aligned,
            g->device_type(),
            g->dtype());
    }

    return g;
}

static utils::Ref<Tensor>
unify_to_shape(utils::Ref<Tensor> g, const std::vector<size_t>& parent_shape) {
    if (!g) return g;
    const auto& gs = g->shape();
    if (gs == parent_shape) return g;

    #ifdef CPPGRAD_DEBUG
        const size_t gr = gs.size(), pr = parent_shape.size();
        const size_t R = std::max(gr, pr);

        auto pad_left = [&](const std::vector<size_t>& v, size_t Rq) {
            if (v.size() >= Rq) return v;
            std::vector<size_t> out;
            out.reserve(Rq);
            out.insert(out.end(), Rq - v.size(), 1);
            out.insert(out.end(), v.begin(), v.end());
            return out;
        };

        std::vector<size_t> ga = pad_left(gs, R);
        std::vector<size_t> pa = pad_left(parent_shape, R);

        bool need_reduce = false;
        for (size_t d = 0; d < R; ++d) {
            const size_t gd = ga[d], pd = pa[d];
            if (gd == pd) continue;
            if (pd == 1 && gd > 1) need_reduce = true;
            else if (gd == 1 && pd > 1) {
                // ok, broadcast needed (will be handled by broadcast_to_shape)
            } else {
                char buf[512];
                snprintf(buf, sizeof(buf),
                        "unify_to_shape: incompatible dims at axis %zu (grad=%zu, parent=%zu)"
                        " grad_shape=%s parent_shape=%s",
                        d, gd, pd,
                        cppgrad::utils::vector::to_string(ga).c_str(),
                        cppgrad::utils::vector::to_string(pa).c_str());
                throw std::runtime_error(buf);
            }
        }

        if (need_reduce) {
            std::ostringstream os;
            os << "unify_to_shape: upstream grad requires reduction\n";
            os << "  grad shape=" << cppgrad::utils::vector::to_string(gs)
            << " parent shape=" << cppgrad::utils::vector::to_string(parent_shape) << "\n";
            os << "  aligned grad=" << cppgrad::utils::vector::to_string(ga)
            << " aligned parent=" << cppgrad::utils::vector::to_string(pa) << "\n";
            throw std::runtime_error(os.str());
        }
    #endif

    return broadcast_to_shape_dbg(g, parent_shape, "unify_to_shape");
}

static utils::Ref<Tensor>
reduce_grad_to_parent(const utils::Ref<Tensor>& gz, const std::vector<size_t>& parent_shape) {
    auto g_red = reduce_to_shape_sum(gz, parent_shape);
    return unify_to_shape(g_red, parent_shape);
}

void Tensor::backward() {
    check_liveness("Tensor::backward");

    if (!_requires_grad) {
        throw std::runtime_error("backward(): tensor does not require grad");
    }
    if (numel() != 1) {
        throw std::runtime_error("backward(): only scalar tensors supported");
    }

    // Build tape (topological post-order)
    std::vector<const Tensor*> tape;
    tape.reserve(128);
    std::unordered_set<const Tensor*> visited;
    visited.reserve(256);
    std::function<void(const Tensor*)> dfs = [&](const Tensor* t) {
        if (!t || visited.count(t)) return;
        if (!t->requires_grad()) return;
        visited.insert(t);
        for (const auto& c : t->children()) dfs(c.get());
        tape.push_back(t);
    };
    dfs(this);

    // Incoming grads map
    using GradMap = std::unordered_map<const Tensor*, utils::Ref<Tensor>>;
    GradMap incoming;
    incoming.reserve(tape.size());

    // Seed grad (scalar one)
    incoming[this] = Tensor::make(
        ConstantOp{ConstantOpType::SCALAR, 1.0},
        {},
        std::vector<size_t>{},
        device_type(),
        dtype());

    // Reverse sweep
    for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
        const Tensor* node = *it;

        auto itg = incoming.find(node);
        if (itg == incoming.end()) continue;

        utils::Ref<Tensor> grad_this = itg->second;
        if (!grad_this) continue;

        // Leaf: do not propagate further. The leaf's final grad is incoming[node].
        if (std::holds_alternative<LeafOp>(node->_op)) {
            continue;
        }

        const auto& children = node->children();

        std::vector<utils::Ref<Tensor>> parent_grads;

        std::visit([&](auto&& op) {
            using T = std::decay_t<decltype(op)>;

            if constexpr (std::is_same_v<T, UnaryOp>) {
                auto x = children[0];

                switch (op.type) {
                    case UnaryOpType::RELU: {
                        auto zero = Tensor::make(
                            ConstantOp{ConstantOpType::FULL, 0.0},
                            {},
                            x->shape(),
                            x->device_type(),
                            x->dtype());
                        auto mask = Tensor::make(
                            BinaryOp{BinaryOpType::CMP_GT},
                            {utils::Ref<const Tensor>(node), zero},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        auto g = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, mask},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        parent_grads = {g};
                        break;
                    }
                    case UnaryOpType::EXP: {
                        auto g = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, utils::Ref<const Tensor>(node)},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        parent_grads = {g};
                        break;
                    }
                    case UnaryOpType::LOG: {
                        auto g = Tensor::make(
                            BinaryOp{BinaryOpType::DIV},
                            {grad_this, x},
                            x->shape(),
                            x->device_type(),
                            x->dtype());
                        parent_grads = {g};
                        break;
                    }
                    case UnaryOpType::NEG: {
                        auto g = Tensor::make(
                            UnaryOp{UnaryOpType::NEG},
                            {grad_this},
                            grad_this->shape(),
                            grad_this->device_type(),
                            grad_this->dtype());
                        parent_grads = {g};
                        break;
                    }
                    case UnaryOpType::TANH: {
                        auto one = Tensor::make(
                            ConstantOp{ConstantOpType::FULL, 1.0},
                            {},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        auto out2 = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {utils::Ref<const Tensor>(node), utils::Ref<const Tensor>(node)},
                            node->shape(),
                            node->device_type(),
                            node->dtype()
                        );
                        auto local = Tensor::make(
                            BinaryOp{BinaryOpType::SUB},
                            {one, out2},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        auto g = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, local},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        parent_grads = {g};
                        break;
                    }
                }
            }
            else if constexpr (std::is_same_v<T, BinaryOp>) {
                auto a = children[0];
                auto b = children[1];
                utils::Ref<Tensor> ga, gb;
                switch (op.type) {
                    case BinaryOpType::ADD: {
                        ga = reduce_grad_to_parent(grad_this, a->shape());
                        gb = reduce_grad_to_parent(grad_this, b->shape());
                        break;
                    }
                    case BinaryOpType::SUB: {
                        ga = reduce_grad_to_parent(grad_this, a->shape());
                        auto neg_g = Tensor::make(
                            UnaryOp{UnaryOpType::NEG},
                            {grad_this},
                            grad_this->shape(),
                            grad_this->device_type(),
                            grad_this->dtype());
                        gb = reduce_grad_to_parent(neg_g, b->shape());
                        break;
                    }
                    case BinaryOpType::MUL: {
                        auto ga_raw = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, b}, node->shape(),
                            node->device_type(),
                            node->dtype());
                        ga = reduce_grad_to_parent(ga_raw, a->shape());
                        auto gb_raw = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, a},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        gb = reduce_grad_to_parent(gb_raw, b->shape());
                        break;
                    }
                    case BinaryOpType::DIV: {
                        auto ga_raw = Tensor::make(
                            BinaryOp{BinaryOpType::DIV},
                            {grad_this, b},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        ga = reduce_grad_to_parent(ga_raw, a->shape());
                        auto b2 = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {b, b},
                            b->shape(),
                            b->device_type(),
                            b->dtype());
                        auto neg_g = Tensor::make(
                            UnaryOp{UnaryOpType::NEG},
                            {grad_this},
                            grad_this->shape(),
                            grad_this->device_type(),
                            grad_this->dtype());
                        auto num = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {neg_g, a},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        auto gb_raw = Tensor::make(
                            BinaryOp{BinaryOpType::DIV},
                            {num, b2},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        gb = reduce_grad_to_parent(gb_raw, b->shape());
                        break;
                    }
                    case BinaryOpType::POW: {
                        auto one = Tensor::make(
                            ConstantOp{ConstantOpType::FULL, 1.0},
                            {},
                            b->shape(),
                            b->device_type(),
                            b->dtype());
                        auto b_minus_1 = Tensor::make(
                            BinaryOp{BinaryOpType::SUB},
                            {b, one},
                            b->shape(),
                            b->device_type(),
                            b->dtype());
                        auto a_pow_bm1 = Tensor::make(
                            BinaryOp{BinaryOpType::POW},
                            {a, b_minus_1},
                            node->shape(),
                            a->device_type(),
                            a->dtype());
                        auto term_a = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {b, a_pow_bm1},
                            node->shape(),
                            a->device_type(),
                            a->dtype());
                        auto ga_raw = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, term_a},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        ga = reduce_grad_to_parent(ga_raw, a->shape());

                        auto ln_a = Tensor::make(
                            UnaryOp{UnaryOpType::LOG},
                            {a},
                            a->shape(),
                            a->device_type(),
                            a->dtype());
                        auto out_times_ln_a = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {utils::Ref<const Tensor>(node), ln_a},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        auto gb_raw = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {grad_this, out_times_ln_a},
                            node->shape(),
                            node->device_type(),
                            node->dtype());
                        gb = reduce_grad_to_parent(gb_raw, b->shape());
                        break;
                    }
                    case BinaryOpType::CMP_EQ:
                    case BinaryOpType::CMP_GT:
                    case BinaryOpType::MIN:
                    case BinaryOpType::MAX: {
                        auto zero = Tensor::make(
                            ConstantOp{ConstantOpType::FULL, 0.0},
                            {},
                            grad_this->shape(),
                            grad_this->device_type(),
                            grad_this->dtype());
                        ga = zero; gb = zero;
                        break;
                    }
                }
                parent_grads = {ga, gb};
            }
            else if constexpr (std::is_same_v<T, ReduceOp>) {
                auto x = children[0];
                switch (op.type) {
                    case ReduceOpType::SUM: {
                        auto g = broadcast_grad_for_sum_backward(grad_this, x->shape(), op.axes, op.keep_dims);
                        parent_grads = {g};
                        break;
                    }
                    case ReduceOpType::MAX: {
                        auto b_out = Tensor::make(
                            MovementOp{MovementOpType::BROADCAST, x->shape()},
                            {utils::Ref<const Tensor>(node)},
                            AccessMeta::broadcast_from(node->access_meta(),
                            x->shape()),
                            x->device_type(),
                            x->dtype());
                        auto mask = Tensor::make(
                            BinaryOp{BinaryOpType::CMP_EQ},
                            {x, b_out},
                            x->shape(),
                            x->device_type(),
                            x->dtype());
                        auto b_g = Tensor::make(
                            MovementOp{MovementOpType::BROADCAST, x->shape()},
                            {grad_this},
                            AccessMeta::broadcast_from(grad_this->access_meta(), x->shape()),
                            grad_this->device_type(),
                            grad_this->dtype());
                        auto g = Tensor::make(
                            BinaryOp{BinaryOpType::MUL},
                            {b_g, mask},
                            x->shape(),
                            x->device_type(),
                            x->dtype());
                        parent_grads = {g};
                        break;
                    }
                }
            }
            else if constexpr (std::is_same_v<T, MovementOp>) {
                auto x = children[0];
                switch (op.type) {
                    case MovementOpType::RESHAPE: {
                        parent_grads = {grad_this};
                        break;
                    }
                    case MovementOpType::PERMUTE: {
                        const auto& axes = op.arg;
                        std::vector<size_t> undo(axes.size());
                        for (size_t i = 0; i < axes.size(); ++i) undo[axes[i]] = i;
                        auto gperm = Tensor::make(
                            MovementOp{MovementOpType::PERMUTE, undo},
                            {grad_this},
                            AccessMeta::permute_from(grad_this->access_meta(), undo),
                            x->device_type(),
                            x->dtype());
                        parent_grads = {gperm};
                        break;
                    }
                    case MovementOpType::BROADCAST: {
                        auto gred = reduce_to_shape_sum(grad_this, x->shape());
                        gred = unify_to_shape(gred, x->shape());
                        parent_grads = {gred};
                        break;
                    }
                    case MovementOpType::SLICE: {
                        MovementOp slice_op = op;
                        // TODO: proper scatter?
                        auto scatter = Tensor::make(
                            slice_op,
                            {grad_this},
                            AccessMeta::slice_from(x->access_meta(), op.slice_begin, op.slice_end, op.arg),
                            x->device_type(),
                            x->dtype());
                        parent_grads = {scatter};
                        break;
                    }
                }
            }
            else if constexpr (std::is_same_v<T, MatMulOp>) {
                auto A = children[0];
                auto B = children[1];
                auto Xt_axes = std::vector<size_t>{1,0};
                auto Bt = Tensor::make(
                    MovementOp{MovementOpType::PERMUTE, Xt_axes},
                    {B},
                    AccessMeta::permute_from(B->access_meta(), Xt_axes),
                    B->device_type(),
                    B->dtype());
                auto dA = Tensor::make(
                    MatMulOp{},
                    std::vector<utils::Ref<const Tensor>>{grad_this, Bt},
                    {A->shape()[0], A->shape()[1]},
                    A->device_type(),
                    A->dtype());
                auto At = Tensor::make(
                    MovementOp{MovementOpType::PERMUTE, Xt_axes},
                    {A},
                    AccessMeta::permute_from(A->access_meta(), Xt_axes),
                    A->device_type(),
                    A->dtype());
                auto dB = Tensor::make(
                    MatMulOp{},
                    {At, grad_this},
                    {B->shape()[0], B->shape()[1]},
                    B->device_type(),
                    B->dtype());
                parent_grads = {dA, dB};
            }
            else if constexpr (std::is_same_v<T, CopyOp>) {
                auto src = children[0];
                auto back = Tensor::make(
                    CopyOp{},
                    {grad_this},
                    src->shape(),
                    src->device_type(),
                    src->dtype());
                parent_grads = {back};
            }
            else if constexpr (!ir::is_differentiable_v<T>) {
                parent_grads = {};
            }
            else {
                static_assert(!sizeof(T), "Unhandled op type in backward()");
            }
        }, node->_op);

        // Align and accumulate into incoming[parent]
        for (size_t i = 0; i < children.size(); ++i) {
            auto parent = children[i];
            if (!parent) continue;
            if (i >= parent_grads.size()) continue;

            auto pgrad = parent_grads[i];
            if (!pgrad) continue;

            pgrad = unify_to_shape(pgrad, parent->shape());

            // accumulate
            auto& slot = incoming[parent.get()];
            slot = slot ? Tensor::make(
                BinaryOp{BinaryOpType::ADD},
                {slot, pgrad},
                slot->shape(),
                slot->device_type(),
                slot->dtype()) : pgrad;
        }
    }

    // Materialize grads onto leaf tensors
    for (auto& [n, g] : incoming) {
        if (!n || !g) continue;
        if (!std::holds_alternative<LeafOp>(n->_op)) continue;
        if (!n->requires_grad()) continue;

        if (auto old = n->grad()) {
            n->set_grad(Tensor::make(
                BinaryOp{BinaryOpType::ADD},
                {old, g},
                old->shape(),
                old->device_type(),
                old->dtype()));
        } else {
            n->set_grad(g);
        }
    }
}

} // namespace ir
} // namespace cppgrad
