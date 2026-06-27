// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <string>
#include <numeric>
#include <stdexcept>
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/ops.h"
#include "cppgrad/utils/shape.h"
#include "cppgrad/utils/vector.h"

namespace cppgrad::ir {

// Helpers

static utils::Ref<Tensor> unary(UnaryOpType op, const utils::Ref<const Tensor>& t) {
  auto out = Tensor::make(UnaryOp{op}, { t }, t->shape(), t->device_type(), t->dtype());
  out->set_access_meta(common::AccessMeta::contiguous_from(out->shape()));
  return out;
}

static utils::Ref<Tensor> binary(BinaryOpType op, const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) {
  auto out_shape = utils::shape::get_broadcast_shape(a->shape(), b->shape());
  auto out = Tensor::make(BinaryOp{op}, { a, b }, out_shape, a->device_type(), a->dtype());
  out->set_access_meta(common::AccessMeta::contiguous_from(out_shape));
  return out;
}

// Public API

utils::Ref<Tensor> assign(const utils::Ref<const Tensor>& dst, const utils::Ref<const Tensor>& src) {
    if (ir::GradMode::enabled) throw std::runtime_error(
        "assign: forbidden in grad mode (AssignOp is a non-differentiable in-place update). "
        "Wrap optimizer updates in `ir::NoGradScope`.");
    if (!dst->is_canonical_leaf()) throw std::runtime_error("assign: dst tensor is not a canonical leaf");
    if (dst->shape() != src->shape()) throw std::runtime_error("assign: shape mismatch");
    if (dst->dtype() != src->dtype()) throw std::runtime_error("assign: dtype mismatch");
    if (dst->device_type() != src->device_type()) throw std::runtime_error(
        "assign: device mismatch (use `src.to(dst->device_type())` first)");
    return Tensor::make(AssignOp{}, {dst, src}, dst->shape(), dst->device_type(), dst->dtype());
}

utils::Ref<Tensor> cache_update(const utils::Ref<const Tensor>& cache,
                                const utils::Ref<const Tensor>& values,
                                int axis, size_t start) {
    if (ir::GradMode::enabled) throw std::runtime_error(
        "cache_update: forbidden in grad mode (in-place op, no backward).");
    if (!cache->is_canonical_leaf()) throw std::runtime_error("cache_update: cache must be a canonical leaf");
    if (cache->shape().size() != values->shape().size()) throw std::runtime_error("cache_update: rank mismatch");
    // The cache may be a lower-precision store (e.g. bf16 KV cache fed by fp32 activations); the
    // copy_view that performs the write converts src->dst dtype. Only same-kind float narrowing /
    // identical dtypes are allowed -- a float<->int reinterpret would be a bug, not a conversion.
    if (cache->dtype() != values->dtype()) {
        auto is_float = [](common::DType d) {
            return d == common::DType::FLOAT16 || d == common::DType::BFLOAT16 ||
                   d == common::DType::FLOAT32 || d == common::DType::FLOAT64;
        };
        if (!is_float(cache->dtype()) || !is_float(values->dtype()))
            throw std::runtime_error("cache_update: dtype mismatch (only float<->float conversion allowed)");
    }
    if (cache->device_type() != values->device_type()) throw std::runtime_error("cache_update: device mismatch");
    if (cache->shape()[0] != 1) throw std::runtime_error("cache_update: requires batch dim 1 (autoregressive decode)");
    const auto& cshape = cache->shape();
    size_t S = values->shape()[axis];
    size_t end = start + S;
    if (end > cshape[axis]) throw std::runtime_error("cache_update: write past end of preallocated cache");
    // Returned read view = cache[.., 0:end, ..]. With batch dim 1 this prefix is physically
    // contiguous from offset 0, so downstream reshape / repeat_kv see a dense tensor.
    std::vector<size_t> out_shape(cshape.begin(), cshape.end());
    out_shape[axis] = end;
    auto am = common::AccessMeta::contiguous_from(out_shape, /*offset=*/0);
    return Tensor::make(CacheUpdateOp{axis, start}, {cache, values}, am, cache->device_type(), cache->dtype());
}

// Unary Ops
utils::Ref<Tensor> relu(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::RELU, t); }
utils::Ref<Tensor> exp (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::EXP,  t); }
utils::Ref<Tensor> log (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::LOG,  t); }
utils::Ref<Tensor> neg (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::NEG,  t); }
utils::Ref<Tensor> tanh(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::TANH, t); }
utils::Ref<Tensor> silu(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::SILU, t); }     // x*sigmoid(x), fused
utils::Ref<Tensor> sigmoid(const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::SIGMOID, t); } // 1/(1+e^-x), fused
utils::Ref<Tensor> sin (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::SIN,  t); }
utils::Ref<Tensor> cos (const utils::Ref<const Tensor>& t) { return unary(UnaryOpType::COS,  t); }
utils::Ref<Tensor> sqrt(const utils::Ref<const Tensor>& t) { return pow(t, 0.5f); }

// Binary Ops
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::ADD, a, b); }
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::SUB, a, b); }
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MUL, a, b); }
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::DIV, a, b); }
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::POW, a, b); }
utils::Ref<Tensor> cmp_eq(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::CMP_EQ, a, b); }
utils::Ref<Tensor> cmp_gt(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::CMP_GT, a, b); }
utils::Ref<Tensor> min(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MIN, a, b); }
utils::Ref<Tensor> max(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) { return binary(BinaryOpType::MAX, a, b); }

// Reduction Ops
utils::Ref<Tensor> sum(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const auto& in_shape = t->shape();
    const int rank = static_cast<int>(in_shape.size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::shape::normalize_unique_sorted_axes(axes_in, rank);
    auto out_shape = utils::shape::get_reduce_shape(in_shape, axes_n, keep_dims);

    return Tensor::make(ReduceOp{ ReduceOpType::SUM, axes_n, keep_dims }, {t}, out_shape, t->device_type(), t->dtype());
}

utils::Ref<Tensor> max(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const auto& in_shape = t->shape();
    const int rank = static_cast<int>(in_shape.size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::shape::normalize_unique_sorted_axes(axes_in, rank);
    auto out_shape = utils::shape::get_reduce_shape(in_shape, axes_n, keep_dims);

    return Tensor::make(ReduceOp{ ReduceOpType::MAX, axes_n, keep_dims }, {t}, out_shape, t->device_type(), t->dtype());
}

// MatMul Op: contracts last 2 dimensions, broadcasts all leading (batch) dimensions
// A[..., M, K] @ B[..., K, N] -> C[..., M, N]
utils::Ref<Tensor> matmul(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b) {
    if (a->shape().size() < 2 || b->shape().size() < 2) {
        throw std::runtime_error("matmul: tensors must have rank >= 2");
    }
    if (a->shape().back() != b->shape()[b->shape().size() - 2]) {
        std::string msg = "matmul: inner dims mismatch. A shape={";
        for (auto d : a->shape()) msg += std::to_string(d) + " ";
        msg += "}, B shape={"  ;
        for (auto d : b->shape()) msg += std::to_string(d) + " ";
        msg += "}";
        throw std::runtime_error(msg);
    }

    size_t ra = a->shape().size();
    size_t rb = b->shape().size();

    // Batch dims: all dims except last 2
    std::vector<size_t> batch_a(a->shape().begin(), a->shape().begin() + ra - 2);
    std::vector<size_t> batch_b(b->shape().begin(), b->shape().begin() + rb - 2);

    // Matrix dims
    size_t M = a->shape()[ra - 2];
    size_t Ka = a->shape()[ra - 1];
    size_t N = b->shape()[rb - 1];

    // Broadcast batch dims RIGHT-ALIGNED (NumPy/torch rule): a missing or size-1 dim broadcasts.
    size_t nb = std::max(batch_a.size(), batch_b.size());
    std::vector<size_t> bc_batch(nb);
    const size_t pa = nb - batch_a.size(), pb = nb - batch_b.size();
    for (size_t i = 0; i < nb; ++i) {
        size_t ia = i < pa ? 1 : batch_a[i - pa];
        size_t ib = i < pb ? 1 : batch_b[i - pb];
        if (ia != 1 && ib != 1 && ia != ib)
            throw std::runtime_error("matmul: batch dims incompatible");
        bc_batch[i] = std::max(ia, ib);
    }

    // Build output shape: [bc_batch..., M, N]
    std::vector<size_t> out_shape = bc_batch;
    out_shape.push_back(M);
    out_shape.push_back(N);

    // Broadcast each operand's batch dims up to bc_batch (stride-0 views for size-1 / missing dims),
    // so the backend -- which indexes each operand by the OUTPUT batch index using that operand's own
    // strides -- reads the right slice. The builder used to compute the broadcast output shape but
    // pass the operands unmodified, so any actually-broadcast batch dim was indexed with a nonzero
    // stride -> silent wrong results. Identity (no copy) when an operand's batch already equals bc_batch.
    auto align = [&](const utils::Ref<const Tensor>& t, size_t inner0, size_t inner1) {
        std::vector<size_t> target = bc_batch;
        target.push_back(inner0);
        target.push_back(inner1);
        if (t->shape() == target) return t;
        utils::Ref<const Tensor> u = t;
        if (t->shape().size() < target.size()) {            // left-pad rank with size-1 dims
            std::vector<size_t> padded(target.size() - t->shape().size(), 1);
            padded.insert(padded.end(), t->shape().begin(), t->shape().end());
            u = ir::reshape(u, padded);
        }
        return utils::Ref<const Tensor>(ir::broadcast(u, target));
    };
    auto a_bc = align(a, M, Ka);
    auto b_bc = align(b, Ka, N);

    return Tensor::make(MatMulOp{}, {a_bc, b_bc}, out_shape, a->device_type(), a->dtype());
}

utils::Ref<Tensor> flash_attention(const utils::Ref<const Tensor>& q, const utils::Ref<const Tensor>& k,
                                   const utils::Ref<const Tensor>& v, float scale, int n_rep,
                                   bool causal, size_t q_offset) {
    if (q->shape().size() != 4 || k->shape().size() != 4 || v->shape().size() != 4)
        throw std::runtime_error("flash_attention: q,k,v must be 4-D [B,S,nH,Dh] / [B,KV,nKV,Dh]");
    const auto& qs = q->shape();   // [B, S, nH, Dh]
    const auto& ks = k->shape();   // [B, KV, nKV, Dh]
    if (qs[3] != ks[3]) throw std::runtime_error("flash_attention: head_dim mismatch");
    if (k->shape() != v->shape()) throw std::runtime_error("flash_attention: k/v shape mismatch");
    if ((int)qs[2] != n_rep * (int)ks[2]) throw std::runtime_error("flash_attention: nH != n_rep*nKV");
    // Output matches q [B,S,nH,Dh].
    return Tensor::make(FlashAttentionOp{scale, n_rep, causal, q_offset}, {q, k, v},
                        std::vector<size_t>(qs.begin(), qs.end()), q->device_type(), q->dtype());
}

// Fused RMSNorm over the last axis. Inference-only (non-differentiable); functional::rms_norm uses
// this only for a dense (contiguous, offset 0) x and falls back to the composite otherwise.
utils::Ref<Tensor> rms_norm(const utils::Ref<const Tensor>& x, const utils::Ref<const Tensor>& weight, float eps) {
    return Tensor::make(RMSNormOp{eps}, {x, weight}, x->shape(), x->device_type(), x->dtype());
}

utils::Ref<Tensor> quantized_matmul(const utils::Ref<const Tensor>& a,
                                    const utils::Ref<const Tensor>& qweight,
                                    const std::vector<utils::Ref<const Tensor>>& aux,
                                    const ir::QuantParams& params) {
    if (a->shape().size() != 2)
        throw std::runtime_error("quantized_matmul: activation must be 2D [M,K]");
    const size_t K = a->shape()[1];
    const size_t N = qweight->shape()[0];           // qweight: [N, K/pack_factor]
    if (qweight->shape().size() != 2 ||
        qweight->shape()[1] * static_cast<size_t>(params.pack_factor) != K)
        throw std::runtime_error("quantized_matmul: qweight must be [N, K/pack_factor] matching K");
    if ((int)aux.size() != ir::aux_buffer_count(params.scheme))
        throw std::runtime_error("quantized_matmul: wrong number of aux tensors for scheme");
    std::vector<utils::Ref<const Tensor>> inputs = {a, qweight};
    inputs.insert(inputs.end(), aux.begin(), aux.end());
    std::vector<size_t> out_shape = {a->shape()[0], N};
    return Tensor::make(QuantizedMatMulOp{params}, inputs,
                        out_shape, a->device_type(), a->dtype());
}

// Gather Op: table[V, D, ...] + indices[...] -> output[indices->shape(), D, ...]
// Preserves indices shape and appends table's remaining dimensions (after axis 0).
utils::Ref<Tensor> gather(const utils::Ref<const Tensor>& table, const utils::Ref<const Tensor>& indices) {
    std::vector<size_t> out_shape = indices->shape();
    for (size_t i = 1; i < table->shape().size(); ++i) {
        out_shape.push_back(table->shape()[i]);
    }
    // bf16 weight table -> fp32 output (dequantize on lookup): the embedding table is kept in
    // bf16 to save memory, but downstream activations are fp32.
    auto out_dtype = (table->dtype() == common::DType::BFLOAT16)
                         ? common::DType::FLOAT32 : table->dtype();
    return Tensor::make(GatherOp{}, {table, indices}, out_shape, table->device_type(), out_dtype);
}

// Gather along axis: select elements at integer indices along specified axis
utils::Ref<Tensor> gather_axis(const utils::Ref<const Tensor>& tensor, const utils::Ref<const Tensor>& indices, int axis) {
    const auto& shape = tensor->shape();
    int rank = static_cast<int>(shape.size());
    int ax = axis < 0 ? rank + axis : axis;
    if (ax < 0 || ax >= rank) throw std::runtime_error("gather_axis: axis out of range");
    size_t n = indices->numel();
    std::vector<size_t> out_shape = shape;
    out_shape[ax] = n;
    auto out = Tensor::make(GatherAxisOp{ax}, {tensor, indices}, out_shape, tensor->device_type(), tensor->dtype());
    out->set_access_meta(common::AccessMeta::contiguous_from(out_shape));
    return out;
}

// Scatter along axis: replace elements at indexed positions with values
utils::Ref<Tensor> scatter_axis(const utils::Ref<const Tensor>& base, const utils::Ref<const Tensor>& values, const utils::Ref<const Tensor>& indices, int axis) {
    const auto& shape = base->shape();
    int rank = static_cast<int>(shape.size());
    int ax = axis < 0 ? rank + axis : axis;
    if (ax < 0 || ax >= rank) throw std::runtime_error("scatter_axis: axis out of range");
    size_t n = indices->numel();
    if (values->shape()[ax] != n) throw std::runtime_error("scatter_axis: values shape mismatch at axis");
    // Output has same shape as base
    auto out = Tensor::make(ScatterOp{ax}, {base, values, indices}, shape, base->device_type(), base->dtype());
    out->set_access_meta(common::AccessMeta::contiguous_from(shape));
    return out;
}

// Concat: concatenate two tensors along axis
utils::Ref<Tensor> concat(const utils::Ref<const Tensor>& a, const utils::Ref<const Tensor>& b, int axis) {
    const auto& as = a->shape();
    const auto& bs = b->shape();
    if (as.size() != bs.size()) {
        throw std::runtime_error("concat: tensors must have the same rank");
    }
    int rank = static_cast<int>(as.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("concat: axis out of range");
    }
    for (int i = 0; i < rank; ++i) {
        if (i != axis && as[i] != bs[i]) {
            throw std::runtime_error("concat: non-concat dimensions must match");
        }
    }
    std::vector<size_t> out_shape = as;
    out_shape[axis] = as[axis] + bs[axis];
    auto out = Tensor::make(ConcatOp{axis}, {a, b}, out_shape, a->device_type(), a->dtype());
    out->set_access_meta(common::AccessMeta::contiguous_from(out_shape));
    return out;
}

// Materialization / Layout Ops
utils::Ref<const Tensor> contiguous(const utils::Ref<const Tensor>& t) {
    const auto& am = t->access_meta();
    if (am.contiguous && am.offset == 0) return t;
    return Tensor::make(CopyOp{}, {t}, common::AccessMeta::contiguous_from(t->shape(), 0), t->device_type(), t->dtype());
}

// Movement Ops
utils::Ref<Tensor> reshape_view(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape) {
    if (utils::vector::numel(t->shape()) != utils::vector::numel(new_shape)) throw std::runtime_error("reshape_view: numel must match");
    auto am = common::AccessMeta::reshape_from(t->access_meta(), new_shape);
    return Tensor::make(MovementOp{MovementOpType::RESHAPE, new_shape}, {t}, am, t->device_type(), t->dtype());
}

utils::Ref<Tensor> permute(const utils::Ref<const Tensor>& t, const std::vector<size_t>& axes) {
    return Tensor::make(MovementOp{MovementOpType::PERMUTE, axes}, {t}, common::AccessMeta::permute_from(t->access_meta(), axes), t->device_type(), t->dtype());
}

utils::Ref<Tensor> transpose(const utils::Ref<const Tensor>& t, int dim0, int dim1) {
    int rank = static_cast<int>(t->shape().size());
    if (dim0 < 0) dim0 += rank;
    if (dim1 < 0) dim1 += rank;
    std::vector<size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);
    std::swap(axes[(size_t)dim0], axes[(size_t)dim1]);
    return permute(t, axes);
}

utils::Ref<Tensor> broadcast(const utils::Ref<const Tensor>& t, const std::vector<size_t>& shape) {
    return Tensor::make(MovementOp{MovementOpType::BROADCAST, shape}, {t}, common::AccessMeta::broadcast_from(t->access_meta(), shape), t->device_type(), t->dtype());
}

utils::Ref<Tensor> slice(const utils::Ref<const Tensor>& t, const std::vector<size_t>& begin, const std::vector<size_t>& end, const std::vector<size_t>& step) {
    std::vector<size_t> steps = step.empty() ? std::vector<size_t>(begin.size(), 1) : step;
    return Tensor::make(MovementOp{MovementOpType::SLICE, steps, begin, end}, {t}, common::AccessMeta::slice_from(t->access_meta(), begin, end, steps), t->device_type(), t->dtype());
}

// Composite Ops
utils::Ref<Tensor> reshape(const utils::Ref<const Tensor>& t, const std::vector<size_t>& new_shape) {
    if (utils::vector::numel(t->shape()) != utils::vector::numel(new_shape)) {
        std::string msg = "reshape: numel must match. src={" ;
        for (auto d : t->shape()) msg += std::to_string(d) + " ";
        msg += "} dst={";
        for (auto d : new_shape) msg += std::to_string(d) + " ";
        msg += "}";
        throw std::runtime_error(msg);
    }
    if (t->access_meta().contiguous) return reshape_view(t, new_shape);
    return reshape_view(contiguous(t), new_shape);
}

utils::Ref<Tensor> mean(const utils::Ref<const Tensor>& t, const std::vector<int>& axes, bool keep_dims) {
    const int rank = static_cast<int>(t->shape().size());
    std::vector<int> axes_in = axes;
    if (axes_in.empty()) { axes_in.resize(rank); std::iota(axes_in.begin(), axes_in.end(), 0); }
    auto axes_n = cppgrad::utils::shape::normalize_unique_sorted_axes(axes_in, rank);
    auto summed = sum(t, axes_n, keep_dims);
    size_t reduction_size = cppgrad::utils::shape::get_reduce_count(t->shape(), axes_n);
    return div(summed, static_cast<float>(reduction_size));
}

// Scalar Ops (Tensor, float)
utils::Ref<Tensor> add(const utils::Ref<const Tensor>& a, float val) { return add(a, scalar_like(val, a)); }
utils::Ref<Tensor> sub(const utils::Ref<const Tensor>& a, float val) { return sub(a, scalar_like(val, a)); }
utils::Ref<Tensor> mul(const utils::Ref<const Tensor>& a, float val) { return mul(a, scalar_like(val, a)); }
utils::Ref<Tensor> div(const utils::Ref<const Tensor>& a, float val) { return div(a, scalar_like(val, a)); }
utils::Ref<Tensor> pow(const utils::Ref<const Tensor>& a, float val) { return pow(a, scalar_like(val, a)); }

// Scalar Ops (float, Tensor)
utils::Ref<Tensor> add(float val, const utils::Ref<const Tensor>& a) { return add(scalar_like(val, a), a); }
utils::Ref<Tensor> sub(float val, const utils::Ref<const Tensor>& a) { return sub(scalar_like(val, a), a); }
utils::Ref<Tensor> mul(float val, const utils::Ref<const Tensor>& a) { return mul(scalar_like(val, a), a); }
utils::Ref<Tensor> div(float val, const utils::Ref<const Tensor>& a) { return div(scalar_like(val, a), a); }
utils::Ref<Tensor> pow(float val, const utils::Ref<const Tensor>& a) { return pow(scalar_like(val, a), a); }

} // namespace cppgrad::ir
