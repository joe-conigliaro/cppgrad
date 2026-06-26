// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include "cppgrad/executor/interpreter/interpreter_executor.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/backend/copy.h"
#include "cppgrad/common/dtype.h"
#include "cppgrad/utils/profiler.h"
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
    const bool profile_on = cppgrad::utils::Profiler::enabled();  // per-op memory-traffic profiling
    // Persistent (process-lifetime) cache of immutable constant buffers. See the ConstantOp branch.
    static std::map<std::tuple<int,int,size_t,double>, std::shared_ptr<backend::Buffer>> const_cache;

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

            // Already realized (this call's DFS boundary, or carried over from a
            // prior realize): reuse the buffer instead of recomputing. Realized
            // nodes are immutable values, so this is safe; it is what makes
            // overlapping re-realizes (decode KV reuse, backward reusing forward
            // activations) O(new-work) rather than O(whole-graph).
            if (auto b = t->realized_buffer()) { realized[t] = b; continue; }

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
                    // Constants are immutable and overwhelmingly repeat (the `1.0` in every sigmoid,
                    // eps, per-layer scale factors, zero-pads). Cache the filled buffer by
                    // (device, dtype, numel, value) so we dispatch one fill per distinct constant
                    // instead of one per occurrence per step (~1600 fills/decode-step otherwise).
                    // Buffers are shared read-only and kept alive by the cache's shared_ptr. Gated to
                    // no-grad regions (inference: generate() runs under NoGradScope). In grad-enabled
                    // regions (training forward/backward) we keep the plain per-node fill, so the
                    // autograd tape never shares constant storage across nodes.
                    if (!cppgrad::ir::GradMode::enabled) {
                        auto key = std::make_tuple((int)t->device_type(), (int)t->dtype(),
                                                   t->numel(), op.value);
                        auto it = const_cache.find(key);
                        if (it != const_cache.end()) {
                            out_buf = it->second;
                        } else {
                            out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                            out_device->backend()->fill(*out_buf, float(op.value));
                            const_cache.emplace(key, out_buf);
                        }
                    } else {
                        out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                        out_device->backend()->fill(*out_buf, float(op.value));
                    }
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
                    auto vo = backend::View::from(common::AccessMeta::contiguous_from(t->shape(), /*offset=*/0));
                    out_device->backend()->matmul(*parents[0], va, *parents[1], vb, *out_buf, vo);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::FlashAttentionOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto q_t = t->children()[0];   // [B,S,nH,Dh]
                    auto k_t = t->children()[1];   // [B,KV,nKV,Dh]
                    const size_t B = q_t->shape()[0], S = q_t->shape()[1], nH = q_t->shape()[2], Dh = q_t->shape()[3];
                    const size_t KV = k_t->shape()[1], nKV = k_t->shape()[2];
                    out_device->backend()->flash_attention(*parents[0], *parents[1], *parents[2], *out_buf,
                        B, S, nH, Dh, KV, nKV, op.scale, op.n_rep, op.causal, op.q_offset);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::QuantizedMatMulOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    const size_t M = t->children()[0]->shape()[0];
                    const size_t K = t->children()[0]->shape()[1];
                    const size_t N = t->children()[1]->shape()[0];   // qweight [N, K/pack_factor]
                    // parents = [a, qweight, aux...]; aux meaning is scheme-defined.
                    std::vector<const backend::Buffer*> aux;
                    for (size_t pi = 2; pi < parents.size(); ++pi) aux.push_back(parents[pi].get());
                    out_device->backend()->quantized_matmul(*parents[0], *parents[1], aux,
                                                            *out_buf, M, N, K, op.params);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::CopyOp>) {
                    auto src_buf = parents[0];
                    const auto vs = backend::View::from(t->children()[0]->access_meta());
                    const auto vd = backend::View::from(t->access_meta());

                    // Same-device: view copy.
                    if (src_buf->device_type() == t->device_type()) {
                        out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                        out_device->backend()->copy_view(*src_buf, vs, *out_buf, vd);
                    }
                    // Cross-device: materialize identity on src, transfer identity, write to output on dst.
                    else {
                        // src -> tmp src (identity)
                        auto* src_device = backend::DeviceManager::device(src_buf->device_type());
                        auto tmp_src = src_device->allocator()->allocate(t->numel(), t->dtype());
                        const auto vid = backend::View::from(common::AccessMeta::contiguous_from(t->shape(), 0));
                        src_device->backend()->copy_view(*src_buf, vs, *tmp_src, vid);

                        // tmp src (identity) -> tmp dst (identity)
                        auto tmp_dst = out_device->allocator()->allocate(t->numel(), t->dtype());
                        // TODO: do we want to avoid device lookup fron copy helper? and see src_device...
                        // out_device->allocator()->copy_device_to_device(*tmp_dst, *tmp_src);
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
                else if constexpr (std::is_same_v<T, cppgrad::ir::CacheUpdateOp>) {
                    // In-place append: write values into the preallocated cache at
                    // [.., start:start+S, ..] along op.axis, then return the cache buffer.
                    // The node's own access_meta already describes the [.., 0:start+S, ..]
                    // read view (contiguous from 0, batch dim 1).
                    auto cache_t = t->children()[0];
                    auto val_t   = t->children()[1];
                    auto cache_buf = parents[0];
                    auto val_buf   = parents[1];
                    std::vector<size_t> begin(cache_t->shape().size(), 0);
                    std::vector<size_t> end = cache_t->shape();
                    const size_t S = val_t->shape()[op.axis];
                    begin[op.axis] = op.start;
                    end[op.axis]   = op.start + S;
                    auto write_am = cppgrad::common::AccessMeta::slice_from(
                        cache_t->access_meta(), begin, end,
                        std::vector<size_t>(cache_t->shape().size(), 1));
                    auto vdst = backend::View::from(write_am);
                    auto vsrc = backend::View::from(val_t->access_meta());
                    out_device->backend()->copy_view(*val_buf, vsrc, *cache_buf, vdst);
                    out_buf = cache_buf;
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
                else if constexpr (std::is_same_v<T, cppgrad::ir::GatherOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto table_tensor = t->children()[0];
                    auto table_buf = parents[0];
                    auto indices_buf = parents[1];
                    // V = table dim 0, D = product of all remaining table dims
                    size_t V = table_tensor->shape()[0];
                    size_t D = 1;
                    for (size_t i = 1; i < table_tensor->shape().size(); ++i) D *= table_tensor->shape()[i];
                    out_device->backend()->gather_op(*table_buf, *indices_buf, *out_buf, V, D);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::ConcatOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    std::vector<const backend::Buffer*> input_bufs;
                    std::vector<backend::View> input_views;
                    for (size_t i = 0; i < t->children().size(); ++i) {
                        input_bufs.push_back(parents[i].get());
                        input_views.push_back(backend::View::from(t->children()[i]->access_meta()));
                    }
                    auto vo = backend::View::from(common::AccessMeta::contiguous_from(t->shape(), /*offset=*/0));
                    out_device->backend()->concat_op(input_bufs, input_views, *out_buf, vo, op.axis);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::GatherAxisOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto tv = backend::View::from(t->children()[0]->access_meta());
                    auto vo = backend::View::from(common::AccessMeta::contiguous_from(t->shape(), 0));
                    out_device->backend()->gather_axis_op(*parents[0], tv, *parents[1], *out_buf, vo, op.axis);
                }
                else if constexpr (std::is_same_v<T, cppgrad::ir::ScatterOp>) {
                    out_buf = out_device->allocator()->allocate(t->numel(), t->dtype());
                    auto bv = backend::View::from(t->children()[0]->access_meta());
                    auto vv = backend::View::from(t->children()[1]->access_meta());
                    auto vo = backend::View::from(common::AccessMeta::contiguous_from(t->shape(), 0));
                    out_device->backend()->scatter_axis_op(*parents[0], bv, *parents[1], vv, *parents[2], *out_buf, vo, op.axis);
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

            // No-grad only: now that this node's buffer holds its value, drop its child references
            // so the upstream graph frees as soon as it is no longer reachable. Keeps long
            // autoregressive chains (e.g. the per-step KV ConcatOp chain) at O(n) memory instead of
            // O(n^2). The `realized` map keeps any still-needed node alive for the rest of this call;
            // backward (grad mode) needs the children, so it is skipped there.
            if (!cppgrad::ir::GradMode::enabled) t->detach_children();

            if (profile_on) {
                // Memory traffic moved by this op = output + all inputs (the cost proxy in a
                // bandwidth-bound regime). Time is left to region scopes / backend GPU hooks.
                uint64_t bytes = (uint64_t)t->numel() * common::size(t->dtype());
                for (const auto& c : t->children())
                    if (c) bytes += (uint64_t)c->numel() * common::size(c->dtype());
                cppgrad::utils::Profiler::instance().record(cppgrad::ir::to_string(t->op()), 0.0, bytes);
            }
        }
    }
}

} // namespace interpreter
} // namespace executor
} // namespace cppgrad
