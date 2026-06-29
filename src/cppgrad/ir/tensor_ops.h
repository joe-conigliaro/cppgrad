// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>

#include "cppgrad/ir/tensor.h"

namespace cppgrad::ir {

utils::Ref<Tensor> assign(const utils::Ref<const Tensor> &dst, const utils::Ref<const Tensor> &src);

// In-place autoregressive cache append. Writes `values` ([.., S, ..]) into the preallocated
// contiguous cache leaf at [.., start : start+S, ..] along `axis`, and returns a view of the
// cache covering [.., 0 : start+S, ..]. Write + read-view are one atomic node (no RAW hazard),
// making decode O(S) rather than the O(context) copy a ConcatOp incurs each step. Inference
// only (no backward). Requires batch dim (axis 0) == 1 so the returned prefix view is
// contiguous from offset 0.
utils::Ref<Tensor> cache_update(const utils::Ref<const Tensor> &cache, const utils::Ref<const Tensor> &values, int axis,
                                size_t start);

// Unary Ops
utils::Ref<Tensor> relu(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> exp(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> log(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> neg(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> tanh(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> silu(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> sigmoid(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> sin(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> cos(const utils::Ref<const Tensor> &t);
utils::Ref<Tensor> sqrt(const utils::Ref<const Tensor> &t);

// Binary Ops
utils::Ref<Tensor> add(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> sub(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> mul(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> div(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> pow(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> cmp_eq(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> cmp_gt(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> min(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);
utils::Ref<Tensor> max(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);

// Reduction Ops
utils::Ref<Tensor> sum(const utils::Ref<const Tensor> &t, const std::vector<int> &axes = {}, bool keep_dims = false);
utils::Ref<Tensor> max(const utils::Ref<const Tensor> &t, const std::vector<int> &axes = {}, bool keep_dims = false);

// MatMul Op: contracts last 2 dimensions, broadcasts all leading (batch) dimensions
// A[..., M, K] @ B[..., K, N] -> C[..., M, N]
utils::Ref<Tensor> matmul(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b);

// Fused flash attention (inference only). q [B,S,nH,Dh], k,v [B,KV,nKV,Dh] -> out [B,S,nH,Dh].
// softmax(scale * QKᵀ + causal)V, online-softmax streaming over keys (no [S,KV] materialization).
// n_rep = nH/nKV (grouped-query); causal masks query row s to keys [0, q_offset+s].
utils::Ref<Tensor> flash_attention(const utils::Ref<const Tensor> &q, const utils::Ref<const Tensor> &k,
                                   const utils::Ref<const Tensor> &v, float scale, int n_rep, bool causal,
                                   size_t q_offset);
utils::Ref<Tensor> rms_norm(const utils::Ref<const Tensor> &x, const utils::Ref<const Tensor> &weight, float eps);
utils::Ref<Tensor> pairwise_decay(const utils::Ref<const Tensor> &G);
utils::Ref<Tensor> delta_decay_mask(const utils::Ref<const Tensor> &scores, const utils::Ref<const Tensor> &Dexp,
                                    const utils::Ref<const Tensor> &beta, bool strict, bool apply_beta);
utils::Ref<Tensor> fma(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b,
                       const utils::Ref<const Tensor> &c);

// Quantized matmul (inference): out[M,N] = a[M,K] @ dequant(qweight)^T, weights kept packed and
// dequantized in-kernel. `params` selects the scheme (MLX affine 8-bit by default). For MLX affine:
// qweight [N, K/pack_factor] u32, scales/biases [N, K/group_size] fp32.
// `aux` holds the scheme's metadata tensors in scheme-defined order (MLX_AFFINE: {scales, biases}).
// Op inputs become [a, qweight, aux...].
utils::Ref<Tensor> quantized_matmul(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &qweight,
                                    const std::vector<utils::Ref<const Tensor>> &aux,
                                    const ir::QuantParams &params = {},
                                    common::DType out_dtype = common::DType::UNKNOWN);

// Gather Op: table[V, D, ...] + indices[...] -> output[indices->shape(), D, ...]
// Preserves indices shape; appends table's remaining dimensions after axis 0.
utils::Ref<Tensor> gather(const utils::Ref<const Tensor> &table, const utils::Ref<const Tensor> &indices);

// Concat: concatenate two tensors along axis
utils::Ref<Tensor> concat(const utils::Ref<const Tensor> &a, const utils::Ref<const Tensor> &b, int axis);

// Gather along axis: select elements at integer indices along specified axis
// tensor: [..., D, ...], indices: [N] int32 -> output: [..., N, ...] (D replaced by N at axis)
utils::Ref<Tensor> gather_axis(const utils::Ref<const Tensor> &tensor, const utils::Ref<const Tensor> &indices,
                               int axis);

// Scatter along axis: replace elements at indexed positions with values
// base: [..., D, ...], values: [..., N, ...], indices: [N] int32 -> output: [..., D, ...]
utils::Ref<Tensor> scatter_axis(const utils::Ref<const Tensor> &base, const utils::Ref<const Tensor> &values,
                                const utils::Ref<const Tensor> &indices, int axis);

// Materialization / Layout Ops
utils::Ref<const Tensor> contiguous(const utils::Ref<const Tensor> &t);

// Movement Ops
utils::Ref<Tensor> reshape_view(const utils::Ref<const Tensor> &t, const std::vector<size_t> &new_shape);
utils::Ref<Tensor> permute(const utils::Ref<const Tensor> &t, const std::vector<size_t> &axes);
utils::Ref<Tensor> transpose(const utils::Ref<const Tensor> &t, int dim0, int dim1);
utils::Ref<Tensor> broadcast(const utils::Ref<const Tensor> &t, const std::vector<size_t> &shape);
utils::Ref<Tensor> slice(const utils::Ref<const Tensor> &t, const std::vector<size_t> &begin,
                         const std::vector<size_t> &end, const std::vector<size_t> &step = {});

// Composite Ops
utils::Ref<Tensor> reshape(const utils::Ref<const Tensor> &t, const std::vector<size_t> &new_shape);
utils::Ref<Tensor> mean(const utils::Ref<const Tensor> &t, const std::vector<int> &axes = {}, bool keep_dims = false);

// Scalar Ops (Tensor, float)
utils::Ref<Tensor> add(const utils::Ref<const Tensor> &a, float val);
utils::Ref<Tensor> sub(const utils::Ref<const Tensor> &a, float val);
utils::Ref<Tensor> mul(const utils::Ref<const Tensor> &a, float val);
utils::Ref<Tensor> div(const utils::Ref<const Tensor> &a, float val);
utils::Ref<Tensor> pow(const utils::Ref<const Tensor> &a, float val);

// Scalar Ops (float, Tensor)
utils::Ref<Tensor> add(float val, const utils::Ref<const Tensor> &a);
utils::Ref<Tensor> sub(float val, const utils::Ref<const Tensor> &a);
utils::Ref<Tensor> mul(float val, const utils::Ref<const Tensor> &a);
utils::Ref<Tensor> div(float val, const utils::Ref<const Tensor> &a);
utils::Ref<Tensor> pow(float val, const utils::Ref<const Tensor> &a);

} // namespace cppgrad::ir
