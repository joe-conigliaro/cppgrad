// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/ir/autograd.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

#include "cppgrad/common/access_meta.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/utils/vector.h"

namespace cppgrad::ir {

// ---- shape-alignment helpers (grad broadcast / reduction) ----

static utils::Ref<Tensor> broadcast_to_shape_dbg(utils::Ref<Tensor> g, const std::vector<size_t> &target,
                                                 const char *tag) {
    if (!g)
        return g;
    const auto &gs = g->shape();
    if (gs == target)
        return g;
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
            os << "broadcast_to_shape[" << tag << "]: incompatible axis " << d << " (grad=" << gd << ", target=" << td
               << ")"
               << " grad_aligned=" << cppgrad::utils::vector::to_string(g_aligned)
               << " target=" << cppgrad::utils::vector::to_string(target)
               << " original_grad=" << cppgrad::utils::vector::to_string(gs);
            throw std::runtime_error(os.str());
        }
    }
#else
    (void)tag;
#endif
    utils::Ref<Tensor> src = g;
    if (g->shape() != g_aligned) {
        src =
            Tensor::make(MovementOp{MovementOpType::RESHAPE, g_aligned}, {g}, g_aligned, g->device_type(), g->dtype());
    }
    return Tensor::make(MovementOp{MovementOpType::BROADCAST, target}, {src}, target, g->device_type(), g->dtype());
}

static utils::Ref<Tensor> broadcast_grad_for_sum_backward(const utils::Ref<Tensor> &grad_this,
                                                          const std::vector<size_t> &in_shape,
                                                          const std::vector<int> &axes, bool /*keep_dims_fwd*/) {
    if (!grad_this)
        return grad_this;

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
    for (int a : axes_norm)
        kd_shape[(size_t)a] = 1;

    utils::Ref<Tensor> g = grad_this;
    const auto &gs = g->shape();
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
        if (g->shape() != kd_shape)
            g = reshape(g, kd_shape);
    }
    // Broadcast singleton dims back to input shape
    g = Tensor::make(MovementOp{MovementOpType::BROADCAST, in_shape}, {g},
                     common::AccessMeta::broadcast_from(g->access_meta(), in_shape), g->device_type(), g->dtype());

    return g;
}

static utils::Ref<Tensor> reduce_to_shape_sum(utils::Ref<Tensor> g, const std::vector<size_t> &target) {
    if (!g)
        return g;
    const auto &grad_shape = g->shape();
    if (grad_shape == target)
        return g;

    const size_t gr = grad_shape.size();
    const size_t tr = target.size();

    auto pad_left = [](const std::vector<size_t> &v, size_t R) {
        if (v.size() >= R)
            return v;
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
                snprintf(buf, sizeof(buf), "reduce_to_shape_sum: incompatible dims at axis %zu (grad=%zu, target=%zu)",
                         d, gd, td);
                throw std::runtime_error(buf);
            }
        }
    }

    if (!axes.empty()) {
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
        auto target_aligned = ta;
        for (int ax : axes)
            target_aligned[(size_t)ax] = 1;
        g = Tensor::make(ReduceOp{ReduceOpType::SUM, axes, /*keep_dims=*/true}, {g}, target_aligned, g->device_type(),
                         g->dtype());
    }

    return g;
}

utils::Ref<Tensor> unify_to_shape(utils::Ref<Tensor> g, const std::vector<size_t> &parent_shape) {
    if (!g)
        return g;
    const auto &gs = g->shape();
    if (gs == parent_shape)
        return g;

#ifdef CPPGRAD_DEBUG
    const size_t gr = gs.size(), pr = parent_shape.size();
    const size_t R = std::max(gr, pr);

    auto pad_left = [&](const std::vector<size_t> &v, size_t Rq) {
        if (v.size() >= Rq)
            return v;
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
        if (gd == pd)
            continue;
        if (pd == 1 && gd > 1)
            need_reduce = true;
        else if (gd == 1 && pd > 1) {
            // ok, broadcast needed (will be handled by broadcast_to_shape)
        } else {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "unify_to_shape: incompatible dims at axis %zu (grad=%zu, parent=%zu)"
                     " grad_shape=%s parent_shape=%s",
                     d, gd, pd, cppgrad::utils::vector::to_string(ga).c_str(),
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

static utils::Ref<Tensor> reduce_grad_to_parent(const utils::Ref<Tensor> &gz, const std::vector<size_t> &parent_shape) {
    auto g_red = reduce_to_shape_sum(gz, parent_shape);
    return unify_to_shape(g_red, parent_shape);
}

// ---- per-op backward rules ----

Grads backward_op(const UnaryOp &op, const BackwardCtx &ctx) {
    const Tensor *node = ctx.node;
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    auto x = children[0];

    switch (op.type) {
    case UnaryOpType::RELU: {
        auto zero = Tensor::make(ConstantOp{ConstantOpType::FULL, 0.0}, {}, x->shape(), x->device_type(), x->dtype());
        auto mask = Tensor::make(BinaryOp{BinaryOpType::CMP_GT}, {utils::Ref<const Tensor>(node), zero}, node->shape(),
                                 node->device_type(), node->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, mask}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    case UnaryOpType::EXP: {
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, utils::Ref<const Tensor>(node)}, node->shape(),
                              node->device_type(), node->dtype());
        return {g};
    }
    case UnaryOpType::LOG: {
        auto g = Tensor::make(BinaryOp{BinaryOpType::DIV}, {grad_this, x}, x->shape(), x->device_type(), x->dtype());
        return {g};
    }
    case UnaryOpType::NEG: {
        auto g = Tensor::make(UnaryOp{UnaryOpType::NEG}, {grad_this}, grad_this->shape(), grad_this->device_type(),
                              grad_this->dtype());
        return {g};
    }
    case UnaryOpType::TANH: {
        auto one = Tensor::make(ConstantOp{ConstantOpType::FULL, 1.0}, {}, node->shape(), node->device_type(),
                                node->dtype());
        auto out2 = Tensor::make(BinaryOp{BinaryOpType::MUL},
                                 {utils::Ref<const Tensor>(node), utils::Ref<const Tensor>(node)}, node->shape(),
                                 node->device_type(), node->dtype());
        auto local = Tensor::make(BinaryOp{BinaryOpType::SUB}, {one, out2}, node->shape(), node->device_type(),
                                  node->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, local}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    case UnaryOpType::SIN: {
        // d(sin(x))/dx = cos(x) * grad
        auto cx = Tensor::make(UnaryOp{UnaryOpType::COS}, {utils::Ref<const Tensor>(node)}, node->shape(),
                               node->device_type(), node->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, cx}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    case UnaryOpType::COS: {
        // d(cos(x))/dx = -sin(x) * grad
        auto sx = Tensor::make(UnaryOp{UnaryOpType::SIN}, {utils::Ref<const Tensor>(node)}, node->shape(),
                               node->device_type(), node->dtype());
        auto neg_sx =
            Tensor::make(UnaryOp{UnaryOpType::NEG}, {sx}, node->shape(), node->device_type(), node->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, neg_sx}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    case UnaryOpType::SIGMOID: {
        // d(sigmoid)/dx = s*(1-s), s = node (the output).
        auto one = Tensor::make(ConstantOp{ConstantOpType::FULL, 1.0}, {}, node->shape(), node->device_type(),
                                node->dtype());
        auto one_minus = Tensor::make(BinaryOp{BinaryOpType::SUB}, {one, utils::Ref<const Tensor>(node)}, node->shape(),
                                      node->device_type(), node->dtype());
        auto deriv = Tensor::make(BinaryOp{BinaryOpType::MUL}, {utils::Ref<const Tensor>(node), one_minus},
                                  node->shape(), node->device_type(), node->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, deriv}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    case UnaryOpType::SILU: {
        // silu(x) = x*sigmoid(x); d/dx = s*(1 + x*(1-s)), s = sigmoid(x).
        auto s = Tensor::make(UnaryOp{UnaryOpType::SIGMOID}, {x}, x->shape(), x->device_type(), x->dtype());
        auto one =
            Tensor::make(ConstantOp{ConstantOpType::FULL, 1.0}, {}, x->shape(), x->device_type(), x->dtype());
        auto one_minus =
            Tensor::make(BinaryOp{BinaryOpType::SUB}, {one, s}, x->shape(), x->device_type(), x->dtype());
        auto x_term =
            Tensor::make(BinaryOp{BinaryOpType::MUL}, {x, one_minus}, x->shape(), x->device_type(), x->dtype());
        auto inner =
            Tensor::make(BinaryOp{BinaryOpType::ADD}, {one, x_term}, x->shape(), x->device_type(), x->dtype());
        auto deriv =
            Tensor::make(BinaryOp{BinaryOpType::MUL}, {s, inner}, x->shape(), x->device_type(), x->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, deriv}, node->shape(), node->device_type(),
                              node->dtype());
        return {g};
    }
    }
    return {};
}

Grads backward_op(const BinaryOp &op, const BackwardCtx &ctx) {
    const Tensor *node = ctx.node;
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
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
        auto neg_g = Tensor::make(UnaryOp{UnaryOpType::NEG}, {grad_this}, grad_this->shape(), grad_this->device_type(),
                                  grad_this->dtype());
        gb = reduce_grad_to_parent(neg_g, b->shape());
        break;
    }
    case BinaryOpType::MUL: {
        auto ga_raw = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, b}, node->shape(), node->device_type(),
                                   node->dtype());
        ga = reduce_grad_to_parent(ga_raw, a->shape());
        auto gb_raw = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, a}, node->shape(), node->device_type(),
                                   node->dtype());
        gb = reduce_grad_to_parent(gb_raw, b->shape());
        break;
    }
    case BinaryOpType::DIV: {
        auto ga_raw = Tensor::make(BinaryOp{BinaryOpType::DIV}, {grad_this, b}, node->shape(), node->device_type(),
                                   node->dtype());
        ga = reduce_grad_to_parent(ga_raw, a->shape());
        auto b2 = Tensor::make(BinaryOp{BinaryOpType::MUL}, {b, b}, b->shape(), b->device_type(), b->dtype());
        auto neg_g = Tensor::make(UnaryOp{UnaryOpType::NEG}, {grad_this}, grad_this->shape(), grad_this->device_type(),
                                  grad_this->dtype());
        auto num = Tensor::make(BinaryOp{BinaryOpType::MUL}, {neg_g, a}, node->shape(), node->device_type(),
                                node->dtype());
        auto gb_raw =
            Tensor::make(BinaryOp{BinaryOpType::DIV}, {num, b2}, node->shape(), node->device_type(), node->dtype());
        gb = reduce_grad_to_parent(gb_raw, b->shape());
        break;
    }
    case BinaryOpType::POW: {
        auto one =
            Tensor::make(ConstantOp{ConstantOpType::FULL, 1.0}, {}, b->shape(), b->device_type(), b->dtype());
        auto b_minus_1 =
            Tensor::make(BinaryOp{BinaryOpType::SUB}, {b, one}, b->shape(), b->device_type(), b->dtype());
        auto a_pow_bm1 = Tensor::make(BinaryOp{BinaryOpType::POW}, {a, b_minus_1}, node->shape(), a->device_type(),
                                      a->dtype());
        auto term_a = Tensor::make(BinaryOp{BinaryOpType::MUL}, {b, a_pow_bm1}, node->shape(), a->device_type(),
                                   a->dtype());
        auto ga_raw = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, term_a}, node->shape(),
                                   node->device_type(), node->dtype());
        ga = reduce_grad_to_parent(ga_raw, a->shape());

        auto ln_a = Tensor::make(UnaryOp{UnaryOpType::LOG}, {a}, a->shape(), a->device_type(), a->dtype());
        auto out_times_ln_a = Tensor::make(BinaryOp{BinaryOpType::MUL}, {utils::Ref<const Tensor>(node), ln_a},
                                           node->shape(), node->device_type(), node->dtype());
        auto gb_raw = Tensor::make(BinaryOp{BinaryOpType::MUL}, {grad_this, out_times_ln_a}, node->shape(),
                                   node->device_type(), node->dtype());
        gb = reduce_grad_to_parent(gb_raw, b->shape());
        break;
    }
    case BinaryOpType::CMP_EQ:
    case BinaryOpType::CMP_GT:
    case BinaryOpType::MIN:
    case BinaryOpType::MAX: {
        auto zero = Tensor::make(ConstantOp{ConstantOpType::FULL, 0.0}, {}, grad_this->shape(),
                                 grad_this->device_type(), grad_this->dtype());
        ga = zero;
        gb = zero;
        break;
    }
    }
    return {ga, gb};
}

Grads backward_op(const ReduceOp &op, const BackwardCtx &ctx) {
    const Tensor *node = ctx.node;
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    auto x = children[0];
    switch (op.type) {
    case ReduceOpType::SUM: {
        auto g = broadcast_grad_for_sum_backward(grad_this, x->shape(), op.axes, op.keep_dims);
        return {g};
    }
    case ReduceOpType::MAX: {
        auto b_out = Tensor::make(MovementOp{MovementOpType::BROADCAST, x->shape()}, {utils::Ref<const Tensor>(node)},
                                  common::AccessMeta::broadcast_from(node->access_meta(), x->shape()), x->device_type(),
                                  x->dtype());
        auto mask = Tensor::make(BinaryOp{BinaryOpType::CMP_EQ}, {x, b_out}, x->shape(), x->device_type(), x->dtype());
        auto b_g = Tensor::make(MovementOp{MovementOpType::BROADCAST, x->shape()}, {grad_this},
                                common::AccessMeta::broadcast_from(grad_this->access_meta(), x->shape()),
                                grad_this->device_type(), grad_this->dtype());
        auto g = Tensor::make(BinaryOp{BinaryOpType::MUL}, {b_g, mask}, x->shape(), x->device_type(), x->dtype());
        return {g};
    }
    }
    return {};
}

Grads backward_op(const MovementOp &op, const BackwardCtx &ctx) {
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    auto x = children[0];
    switch (op.type) {
    case MovementOpType::RESHAPE: {
        return {grad_this};
    }
    case MovementOpType::PERMUTE: {
        const auto &axes = op.arg;
        std::vector<size_t> undo(axes.size());
        for (size_t i = 0; i < axes.size(); ++i)
            undo[axes[i]] = i;
        auto gperm = Tensor::make(MovementOp{MovementOpType::PERMUTE, undo}, {grad_this},
                                  common::AccessMeta::permute_from(grad_this->access_meta(), undo), x->device_type(),
                                  x->dtype());
        return {gperm};
    }
    case MovementOpType::BROADCAST: {
        auto gred = reduce_to_shape_sum(grad_this, x->shape());
        gred = unify_to_shape(gred, x->shape());
        return {gred};
    }
    case MovementOpType::SLICE: {
        MovementOp slice_op = op;
        // TODO: proper scatter?
        auto scatter = Tensor::make(
            slice_op, {grad_this},
            common::AccessMeta::slice_from(x->access_meta(), op.slice_begin, op.slice_end, op.arg), x->device_type(),
            x->dtype());
        return {scatter};
    }
    }
    return {};
}

Grads backward_op(const MatMulOp &, const BackwardCtx &ctx) {
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    auto A = children[0];
    auto B = children[1];
    auto Xt_axes = std::vector<size_t>{1, 0};
    auto Bt = Tensor::make(MovementOp{MovementOpType::PERMUTE, Xt_axes}, {B},
                           common::AccessMeta::permute_from(B->access_meta(), Xt_axes), B->device_type(), B->dtype());
    auto dA = Tensor::make(MatMulOp{}, std::vector<utils::Ref<const Tensor>>{grad_this, Bt},
                           {A->shape()[0], A->shape()[1]}, A->device_type(), A->dtype());
    auto At = Tensor::make(MovementOp{MovementOpType::PERMUTE, Xt_axes}, {A},
                           common::AccessMeta::permute_from(A->access_meta(), Xt_axes), A->device_type(), A->dtype());
    auto dB = Tensor::make(MatMulOp{}, {At, grad_this}, {B->shape()[0], B->shape()[1]}, B->device_type(), B->dtype());
    return {dA, dB};
}

Grads backward_op(const CopyOp &, const BackwardCtx &ctx) {
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    auto src = children[0];
    auto back = Tensor::make(CopyOp{}, {grad_this}, src->shape(), src->device_type(), src->dtype());
    return {back};
}

Grads backward_op(const ConcatOp &op, const BackwardCtx &ctx) {
    const auto &children = ctx.children;
    const auto &grad_this = ctx.grad_this;
    // Backward: split gradient along axis
    auto a = children[0];
    auto b = children[1];
    int axis = op.axis;
    const auto &gs = grad_this->shape();
    int rank = static_cast<int>(gs.size());
    if (axis < 0)
        axis += rank;

    // Build begin/end for slicing a's portion
    std::vector<size_t> begin_a(rank, 0), end_a(rank, 0), begin_b(rank, 0), end_b(rank, 0);
    for (int d = 0; d < rank; ++d) {
        end_a[d] = gs[d];
        end_b[d] = gs[d];
        begin_b[d] = (d == axis) ? a->shape()[static_cast<size_t>(axis)] : 0;
    }
    end_a[axis] = a->shape()[static_cast<size_t>(axis)];

    auto ga = slice(grad_this, begin_a, end_a);
    auto gb = slice(grad_this, begin_b, end_b);

    return {ga, gb};
}

} // namespace cppgrad::ir
