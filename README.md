# cppgrad

A small C++17 autograd + neural-network library.

---

## Overview

- **IR-style graph**: Ops create new `Tensor` nodes with child links.
- **Intrusive ref counting**: Graph ownership via `utils::Ref<T>`.
- **Batch realization**: `GraphContext` / `AutoGraphScope` batches execution.
- **Arena Allocation**: Arena allocation when `AutoGraphScope` is active, otherwise falls back to heap.
- **View-based layouts**: `AccessMeta` encodes `shape/strides/offset` for zero-copy movement ops.
- **Materialization when needed**: `contiguous()` (and copy paths) produce dense `offset=0` buffers.
- **Multiple backends**: CPU + Metal.
- **Executor**: Interpreter (Metal backend uses JIT Metal shader compilation).

---

## Design invariants
- **Realized outputs are identity layout**: row-major dense with `offset = 0`.
- **Movement ops are views**: (metadata-only) until materialized.
- **Synchronization policy**: GPU work should be batched; block only on explicit host readback.
  - **Current** Status: Metal backend is still largely synchronous (per-op `waitUntilCompleted`).
  - **TODO**: Add per-device `ExecutionContext`/streaming and make allocator copies context-aware to enable async submission.

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
        ir::AutoGraphScope scope;

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

### Examples
Build via the repo script:
```sh
# Release
./build_examples.sh

# Debug
DEBUG=true ./build_examples.sh
```

### Unit Tests
Run via the repo script:
```sh
./run_tests.sh
```

---

## TODO
- Optimizer parameter/state updates
  - Graph-based updates via spcialized `OptimizerStepOp` with a dedicated backend kernel (lazy, schedulable, fuseable) vs
  - Graph-based updates via `AssignOp` (lazy, schedulable/fuseable, backend-consistent) vs
  - Eager/in-place updates via `set_parameter_data` / `copy_into_parameter` (simple, but breaks batching)
- Metal streaming / async execution
  - Add per-device `ExecutionContext` and batch command buffer submission.
  - Remove per-op `waitUntilCompleted`; sync only on host readback.
- Context-aware allocator copies
  - Add optional `ExecutionContext*` to allocator copy methods for async blits/uploads.
- Kernel fusion
  - Fuse elementwise chains (unary/binary) within schedules.
- CPU SIMD & BLAS
  - SIMD elementwise; BLAS (or tiled GEMM) for matmul.
- Graph lowering (consider)
  - Lower IR → scheduled kernel regions (fusion + memory planning).

---
## License
MIT License
