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
        backend::Device* device_obj = backend::DeviceManager::device(ds.device);
        if (!device_obj) throw std::runtime_error("InterpreterExecutor: device not found");

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
                        throw std::runtime_error("ConstantOp of type SCALAR must be rank-0 scalar");
                    }
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    device_obj->backend()->fill(*out_buf, float(op.value));
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::RandomOp>) {
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    if (op.type == cppgrad::ir::RandomOpType::UNIFORM) {
                        auto p = std::get<cppgrad::ir::UniformParams>(op.params);
                        device_obj->backend()->rand_uniform(*out_buf, p.min, p.max);
                    } else {
                        auto p = std::get<cppgrad::ir::NormalParams>(op.params);
                        device_obj->backend()->rand_normal(*out_buf, p.mean, p.stddev);
                    }
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::UnaryOp>) {
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    device_obj->backend()->unary_op(op.type, *parents[0], va, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::BinaryOp>) {
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vb = backend::View::from(t->children()[1]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    device_obj->backend()->binary_op(op.type, *parents[0], va, *parents[1], vb, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::ReduceOp>) {
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    if (!vo.is_contiguous() || !vo.is_offset_zero()) {
                        throw std::runtime_error("ReduceOp: output must be contiguous (row-major dense) with offset==0");
                    }
                    device_obj->backend()->reduce_op(op.type, *parents[0], va, *out_buf, vo, op.axes, op.keep_dims);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::MatMulOp>) {
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vb = backend::View::from(t->children()[1]->access_meta());
                    // auto vo = backend::View::from(t->access_meta());
                    // NOTE: We just allocated a fresh output buffer (dense storage starting at offset 0).
                    // Don't use t->access_meta() here as it may represent a view with non-zero offset/strides,
                    // which would make kernels write out-of-bounds into out_buf.
                    // Realized outputs must use an identity (row-major dense, offset=0) view.
                    auto vo = backend::View::from(ir::AccessMeta::contiguous_from(t->shape(), /*offset=*/0));
                    device_obj->backend()->matmul(*parents[0], va, *parents[1], vb, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::CopyOp>) {
                    // Copy via host staging.
                    // out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    // device_obj->backend()->copy(*out_buf, *parents[0]);
                    // Copy using view aware copy kernel.
                    out_buf = device_obj->allocator()->allocate(t->numel(), t->dtype());
                    auto va = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(t->access_meta());
                    device_obj->backend()->copy_view(*parents[0], va, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::AssignOp>) {
                    auto dst_tensor = t->children()[0];
                    auto src_tensor = t->children()[1];
                    auto src_buf = parents[1];
                    auto dst_buf = dst_tensor->realized_buffer();
                    if (!dst_buf) {
                        dst_buf = device_obj->allocator()->allocate(dst_tensor->numel(), dst_tensor->dtype());
                        dst_tensor->attach_buffer(dst_buf);
                    }
                    // if (!dst_buf) {
                    //     throw std::runtime_error("AssignOp: trying to assign to invalid destination buffer");
                    // }
                    auto va = backend::View::from(src_tensor->access_meta());
                    auto vo = backend::View::from(dst_tensor->access_meta());
                    device_obj->backend()->copy_view(*src_buf, va, *dst_buf, vo);
                    out_buf = dst_buf;
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::MovementOp>) {
                    if (!parents[0]) {
                         throw std::runtime_error("MovementOp: parent buffer not realized");
                     }
                    out_buf = parents[0];
                }
            }, t->op());

            if (!out_buf && t->numel() == 0) {
                out_buf = device_obj->allocator()->allocate(0, t->dtype());
            }
            if (!out_buf) throw std::runtime_error("InterpreterExecutor: failed to realize tensor");

            t->attach_buffer(out_buf);
            realized[t] = out_buf;
        }
    }
}

} // namespace interpreter
} // namespace executor
} // namespace cppgrad
