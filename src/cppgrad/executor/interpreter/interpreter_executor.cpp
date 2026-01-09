// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iostream>
#include <unordered_map>
#include "cppgrad/executor/interpreter/interpreter_executor.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/backend/backend.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/copy.h"
#include "cppgrad/backend/view.h"
#include "cppgrad/ir/tensor.h"

namespace cppgrad {
namespace executor {
namespace interpreter {

std::shared_ptr<backend::Buffer> InterpreterExecutor::realize(const utils::Ref<const cppgrad::ir::Tensor>& out) {
    if (!out) return nullptr;
    std::vector<utils::Ref<const cppgrad::ir::Tensor>> outs = { out };
    realize_many(outs);
    return out->schedule();
}

void InterpreterExecutor::realize_many(const std::vector<utils::Ref<const cppgrad::ir::Tensor>>& outs) {
    if (outs.empty()) return;
    auto schedules = build_device_schedules(outs);
    realize_scheduled(schedules);
}

void InterpreterExecutor::realize_scheduled(const std::vector<DeviceSchedule>& schedules) {
    std::unordered_map<utils::Ref<const ir::Tensor>, std::shared_ptr<backend::Buffer>> realized;

    auto get_buf = [&](const utils::Ref<const cppgrad::ir::Tensor>& t) -> std::shared_ptr<backend::Buffer> {
        if (!t) return nullptr;
        auto it = realized.find(t);
        if (it != realized.end()) return it->second;
        if (t->schedule()) return realized[t] = t->schedule();
        return nullptr;
    };

    for (const auto& ds : schedules) {
        backend::Device* out_device = backend::DeviceManager::device(ds.device);
        if (!out_device) throw std::runtime_error("InterpreterExecutor: device not found");

        for (const auto& t : ds.schedule) {
            if (!t) continue;
            if (realized.count(t)) continue;

            if (std::holds_alternative<cppgrad::ir::LeafOp>(t->op())) {
                if (!t->realized_buffer() && t->numel() > 0) {
                    throw std::runtime_error("InterpreterExecutor: leaf tensor has no storage. Did you create a parameter with allocate_now=false and forget to initialize it?");
                }
                if (t->schedule()) realized[t] = t->schedule();
                continue;
            }

            std::vector<std::shared_ptr<backend::Buffer>> parents;
            parents.reserve(t->children().size());

            for (const auto& p_ref : t->children()) {
                auto it = realized.find(p_ref);

                if (it == realized.end()) {
                    realize(p_ref); // Recursion
                    it = realized.find(p_ref);
                    if (it == realized.end() && p_ref->schedule()) {
                        realized[p_ref] = p_ref->schedule();
                        it = realized.find(p_ref);
                    }
                }
                if (it == realized.end() || !it->second) {
                    throw std::runtime_error("InterpreterExecutor: parent buffer missing");
                }
                parents.push_back(it->second);
            }

            std::shared_ptr<backend::Buffer> out_buf = nullptr;
            std::visit([&](auto&& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, cppgrad::ir::ConstantOp>) {
                    if (op.type == cppgrad::ir::ConstantOpType::SCALAR && (t->numel() != 1 || !t->shape().empty())) {
                        throw std::runtime_error("InterpreterExecutor: ConstantOp of type SCALAR must be rank-0 scalar");
                    }
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    out_device->backend()->fill(*out_buf, float(op.value));
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::RandomOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    if (op.type == cppgrad::ir::RandomOpType::UNIFORM) {
                        auto p = std::get<cppgrad::ir::UniformParams>(op.params);
                        out_device->backend()->rand_uniform(*out_buf, p.min, p.max);
                    } else {
                        auto p = std::get<cppgrad::ir::NormalParams>(op.params);
                        out_device->backend()->rand_normal(*out_buf, p.mean, p.stddev);
                    }
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::UnaryOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    out_device->backend()->unary_op(op.type, *parents[0], va, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::BinaryOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vb = backend::View::from(t->children()[1]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    out_device->backend()->binary_op(op.type, *parents[0], va, *parents[1], vb, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::ReduceOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    if (!vo.is_identity()) {
                        // Catch here instead of just creating an identity view, it should be identity from upstream.
                        throw std::runtime_error("InterpreterExecutor: ReduceOp output must be identity (contiguous / row-major dense, with offset=0)");
                    }
                    out_device->backend()->reduce_op(op.type, *parents[0], va, *out_buf, vo, op.axes, op.keep_dims);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::MatMulOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vb = backend::View::from(t->children()[1]->access_meta());
                    // auto vo = backend::View::from(t->access_meta());
                    // NOTE: We just allocated a fresh output buffer (dense storage starting at offset 0).
                    // Don't use t->access_meta() here as it may represent a view with non-zero offset/strides,
                    // which would make kernels write out-of-bounds into out_buf.
                    // Realized outputs must use an identity (row-major dense, offset=0) view.
                    auto vo = backend::View::from(ir::AccessMeta::contiguous_from(t->shape(), /*offset=*/0));
                    out_device->backend()->matmul(*parents[0], va, *parents[1], vb, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::CopyOp>) {
                    auto src_buf = parents[0];
                    const auto vs = backend::View::from(t->children()[0]->access_meta());
                    const auto vd = backend::View::from(t->access_meta());

                    // Same-device: view copy.
                    if (src_buf->device() == t->device()) {
                        out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                        out_device->backend()->copy_view(*src_buf, vs, *out_buf, vd);
                    }
                    // Cross-device: materialize identity on src, transfer identity, write to output on dst.
                    else {
                        // src -> tmp src (identity)
                        auto* src_device = backend::DeviceManager::device(src_buf->device());
                        auto tmp_src = src_device->allocator()->allocate(t->numel(), t->dtype());
                        const auto vid = backend::View::from(ir::AccessMeta::contiguous_from(t->shape(), 0));
                        src_device->backend()->copy_view(*src_buf, vs, *tmp_src, vid);

                        // tmp src (identity) -> tmp dst (identity)
                        auto tmp_dst = out_device->allocator()->allocate(t->numel(), t->dtype());
                        backend::copy(*tmp_dst, *tmp_src);

                        // tmp dest (identity) -> output device
                        if (vd.is_identity()) {
                            out_buf = tmp_dst;
                        } else {
                            out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                            out_device->backend()->copy_view(*tmp_dst, vid, *out_buf, vd);
                        }
                    }
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::AssignOp>) {
                    auto dst_tensor = t->children()[0];
                    auto src_tensor = t->children()[1];
                    auto src_buf = parents[1];
                    auto dst_buf = parents[0];
                    auto va = backend::View::from(src_tensor->access_meta());
                    auto vo = backend::View::from(dst_tensor->access_meta());
                    out_device->backend()->copy_view(*src_buf, va, *dst_buf, vo);
                    out_buf = dst_buf;
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::MovementOp>) {
                    if (!parents[0]) {
                         throw std::runtime_error("InterpreterExecutor: MovementOp parent buffer not realized");
                     }
                    out_buf = parents[0];
                }
            }, t->op());

            // if (!out_buf && t->numel() == 0) {
            //     out_buf = out_device->allocator()->allocate(0, t->dtype());
            // }
            if (!out_buf) throw std::runtime_error("InterpreterExecutor: failed to realize tensor");

            t->attach_buffer(out_buf);
            realized[t] = out_buf;
        }
    }
}

} // namespace interpreter
} // namespace executor
} // namespace cppgrad
