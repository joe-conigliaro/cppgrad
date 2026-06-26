# CppGrad
A small C++17 autograd + neural-network library.

![CI - main branch](https://github.com/joe-conigliaro/cppgrad/actions/workflows/ci.yml/badge.svg?branch=main)

## Overview
- **IR-style graph**: Ops create new `Tensor` nodes with child links.
- **Intrusive ref counting**: Graph ownership via `utils::Ref<T>`.
- **Batch realization**: `GraphContext` / `GraphScope` batches execution.
- **Arena Allocation**: Allocate in arena when `GraphScope` is active, otherwise falls back to heap.
- **View-based layouts**: `AccessMeta` encodes `shape/strides/offset` for zero-copy movement ops.
- **Materialization when needed**: `contiguous()` (and copy paths) produce dense `offset=0` buffers.
- **Multiple backends**: CPU + Metal. The default is chosen by `void DeviceManager::init()` (Metal when available, else CPU) in src/cppgrad/backend/device_manager.cpp.
- **Executor**: Interpreter (Metal backend uses JIT Metal shader compilation).
- **Dtype**: `FLOAT32` for compute / activations; weights may be `BFLOAT16` or 8-bit (MLX affine) quantized, dequantized in-kernel (matmul / gather) on both CPU and Metal.

---

## Design invariants
- **Realized outputs are identity layout**: row-major dense with `offset = 0`.
- **Movement ops are views**: (metadata-only) until materialized.
- **Synchronization policy**: GPU work is batched; the host blocks only on explicit readback.

---

## Metal execution model
The Metal backend does not execute ops one-at-a-time. Each compute op records a self-contained
work item into a per-device `MetalExecutionContext` (a single command buffer), and that buffer is
committed (and waited on) **once** at:
- **`GraphScope` boundaries** - `GraphScope`'s destructor calls `Backend::flush_pending()`, a no-op
  for CPU and a flush of the execution context for Metal, so a scope's GPU work completes at scope
  end just like the synchronous CPU backend.
- **host readback** - the allocator's device->host / device->device / host->device copies flush pending
  compute first, so a read never races ahead of the kernels that produce its data.

---

## LLM inference (Qwen3.5 / 3.6)
Runs Qwen3.5/3.6 - including the 27B - from MLX `.safetensors` checkpoints via
`examples/llm/qwen3_inference.cpp` (`--quant` keeps weights 8-bit). Includes a faithful
GatedDeltaNet linear-attention + full-attention hybrid, an in-place (preallocated) KV /
recurrent-state cache, a byte-level BPE tokenizer, and an MLX-affine quantized matmul
(CPU + Metal, with a simdgroup GEMV for single-token decode).

Decode is memory-bandwidth-bound (reading the 8-bit weights is ~85% of traffic); set
`CPPGRAD_PROFILE=1` for a per-op memory-traffic + GPU-time breakdown, or `QWEN_TIMING=1`
for prefill/decode tokens-per-second.

### Example - Qwen3.6-27B-8bit
```sh
./build/examples/llm/qwen3_inference \
    --model /path/to/models/mlx-community/Qwen3.6-27B-8bit \
    --config 27b \
    --prompt "An elephant is" \
    --max-tokens 28 \
    --quant
```
<details>
<summary><b>View Expected Output</b></summary>

```text
CPU device registered.
METAL device registered.
[Qwen3.5/3.6] Config: 27b hidden=5120 layers=64 heads=24 kv_heads=4 head_dim=256
Default device set to: METAL
[Qwen3.5/3.6] Device: METAL
[Qwen3.5/3.6] Creating model...
[Qwen3.5/3.6] Loading weights from: /path/to/models/mlx-community/Qwen3.6-27B-8bit
[Qwen3.5/3.6] Weights loaded in 7222ms
[Qwen3] Prompt: "An elephant is" -> 3 tokens (BPE)
[Qwen3.5/3.6] Generating 28 tokens...
[Qwen3] Generated 28 tokens in 4726ms
[Qwen3] Output:  a large mammal belonging to the family Elephantidae. The only living species are the African bush elephant (Loxodonta africana),
```
</details>

---

## Quickstart
Simple linear regression with `SGD` (batched)
```cpp
#include <vector>
#include <iomanip>
#include <iostream>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/optim/sgd.h"

using namespace cppgrad;

int main() {
    backend::DeviceManager::instance().init();

    // Data: x in R^{N,1}, y = 2x + 3
    auto x = ir::from_vector<float>({0, 1, 2, 3}, {4, 1});
    auto y = ir::from_vector<float>({3, 5, 7, 9}, {4, 1});

    // Trainable parameters (canonical leaf tensors)
    auto w = ir::parameter({1, 1});
    auto b = ir::parameter({1, 1});

    optim::SGD opt({w, b}, /*lr=*/0.1f);

    for (int step = 0; step < 100; ++step) {
        // One scope per step: builds a graph, then batch-realizes at scope exit.
        ir::GraphScope scope;

        // Forward: yhat = x*w + b
        auto yhat = ir::add(ir::mul(x, w), b);

        // Loss: mean((yhat - y)^2)
        auto diff = ir::sub(yhat, y);
        auto loss = ir::mean(ir::mul(diff, diff));

        opt.zero_grad();
        loss->backward();
        opt.step();

        if (step == 0 || (step + 1) % 10 == 0) {
            // `item()` forces realization of 'loss'
            std::cout << "step " << step+1
                      << " loss=" << std::fixed << std::setprecision(6) << loss->item<float>() << "\n";
        }
    }

    return 0;
}
```

## Building
### Build Flags
- `CPPGRAD_DEBUG=true`: enables debug-only checks & logging.
- `DEBUG=true`: enables debug build (`-g -O0`).
- `SANITIZE_ADDRESS=true`: enables AddressSanitizer/ASan (`-fsanitize=address -fno-omit-frame-pointer`) .
- `SANITIZE_THREAD=true`: enables ThreadSanitizer/Tsan (`-fsanitize=thread`).
- `FFP_CONTRACT_OFF=true`: disables floating-point expression contraction (`-ffp-contract=off`).
- `FAST_MATH=false`: disables fast-math optimizations (`-fno-fast-math`).

Metal is enabled automatically on Apple platforms when `xcrun` is available - the backend is compiled
in via the `CPPGRAD_WITH_METAL` presence macro. Without it (non-Apple, or no `xcrun`) the build is CPU-only.

### Runtime flags (env)
Set at run time (not compile time); zero cost when unset.
- `CPPGRAD_PROFILE=1`: per-op memory-traffic breakdown + GPU time (decode-only for the Qwen example).
- `CPPGRAD_METAL_DISPATCH=1`: number of Metal kernels dispatched per command-buffer flush.
- `CPPGRAD_METAL_CAPTURE=N`: capture the Nth command-buffer flush to `/tmp/cppgrad_flush.gputrace` (open in Xcode; run with `METAL_CAPTURE_ENABLED=1`).
- `QWEN_TIMING=1`: prefill time and decode tokens/sec.
- `QWEN_KV_CONCAT=1`: use the concat KV-cache reference path instead of the default in-place cache (cross-check).
- `CPPGRAD_KV_DTYPE=bf16`: store the full-attention KV cache in bfloat16 (halves cache memory/bandwidth at long context and the persisted cache file; fp32-accumulate attention, so loss is bf16-rounding only). Default `f32`.

See [docs/env-flags.md](docs/env-flags.md) for all env flags.

### Building and Running

All builds use the `Makefile` (incremental, parallel, static library).

```sh
# Build + run everything
make all

# Tests
make tests                # build + run all tests
make build-tests          # build only
make run-tests            # run only

# Examples
make examples             # build + run all examples
make build-examples       # build only
make run-examples         # run only

# Cleanup
make clean
```

Every target accepts the build flags above via environment variables:

```sh
# Debug build
DEBUG=true make tests

# AddressSanitizer
SANITIZE_ADDRESS=true make build-tests

# Disable fast-math
FAST_MATH=false make examples
```

Binaries are emitted under `build/`, mirroring source paths
(e.g. `build/examples/llm/qwen3_inference`).

---

## TODO
- ~~Optimizer parameter/state updates~~ **(done)**
  - ~~Graph-based updates via `OptimizerStepOp` vs `AssignOp` vs eager `set_parameter_data`/`copy_into_parameter`.~~ Implemented via lazy `AssignOp` graph nodes (schedulable/fuseable, backend-consistent) - see `optim/{sgd,adam,adamw}.h`.
  - Future: a fused `OptimizerStepOp` (single backend kernel) for perf.
- ~~Metal streaming / async execution~~ **(done)**
  - ~~Add per-device `ExecutionContext` and batch command buffer submission.~~ Per-device `MetalExecutionContext` batches compute into one command buffer.
  - ~~Remove per-op `waitUntilCompleted`; sync only on host readback.~~ Committed at `GraphScope` boundaries (`Backend::flush_pending()`) and on host readback.
- ~~Context-aware allocator copies~~ **(done)**
  - ~~Add optional `ExecutionContext*` to allocator copy methods for async blits/uploads.~~ Allocator device↔host / device↔device copies flush pending compute first.
- Per-scope backend handle (consider)
  - Generalize the stateless `Backend::flush_pending()` hook into an opaque per-scope
    `ScopeContext` handle (null for CPU) if a backend needs genuine per-scope state - e.g.
    per-scope command buffers / memory pools, nested-scope isolation, or CPU<->GPU overlap.
    Interface sketch is in `backend.h`.
- Kernel fusion
  - Fuse elementwise chains (unary/binary) within schedules. (Profiling shows this is <7% of
    quantized-decode memory traffic, so it is a code-quality win, not a decode-speed lever.)
- CPU SIMD & BLAS / quant GEMM
  - SIMD elementwise; BLAS (or tiled GEMM) for prefill matmul. Quantized decode uses a coalesced
    simdgroup GEMV (M=1) on Metal; it currently reaches ~40% of memory bandwidth, so a
    higher-occupancy variant (larger threadgroups / multiple output columns per threadgroup) is the
    remaining decode-speed lever. CPU quant matmul is still a triple-loop reference.
- Autograd coverage (for training)
  - Backward for `GatherOp` (embedding lookup), N-D / batched `MatMul`, and a proper scatter-add
    `SLICE` backward. The library is inference-complete; these gaps block end-to-end LLM training.
- Graph lowering (consider)
  - Lower IR -> scheduled kernel regions (fusion + memory planning).

---
## License
MIT License
