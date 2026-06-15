// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "cppgrad/backend/metal/metal_execution_context.h"
#include "cppgrad/utils/profiler.h"

namespace cppgrad {
namespace backend {
namespace metal {

MetalExecutionContext::MetalExecutionContext(void* native_device, void* native_queue)
    : _device((__bridge id<MTLDevice>)native_device),
      _queue((__bridge id<MTLCommandQueue>)native_queue) {
    _commandBuffer = [_queue commandBuffer];
}

MetalExecutionContext::~MetalExecutionContext() {
    _computeWork.clear();
    _commandBuffer = nil;
    _queue = nil;
    _device = nil;
}

void MetalExecutionContext::submit_compute(ComputeWork work) {
    _computeWork.push_back(std::move(work));
}

void MetalExecutionContext::flush() {
    if (!_computeWork.empty()) {
        if (getenv("QWEN_DISPATCH")) fprintf(stderr, "[dispatch] flush: %zu kernels\n", _computeWork.size());
        id<MTLComputeCommandEncoder> enc = [_commandBuffer computeCommandEncoder];
        for (const auto& work : _computeWork) {
            encode_work(enc, work);
        }
        [enc endEncoding];
        [_commandBuffer commit];
        [_commandBuffer waitUntilCompleted];

        // Real GPU time for this batch (profiler is a dev tool; cost only when enabled).
        if (cppgrad::utils::Profiler::enabled()) {
            const double gpu_ns = ([_commandBuffer GPUEndTime] - [_commandBuffer GPUStartTime]) * 1e9;
            cppgrad::utils::Profiler::instance().record("GPU(flush)", gpu_ns, 0, _computeWork.size());
        }

        // The committed command buffer can't accept more work.
        _computeWork.clear();
        _commandBuffer = nil;
    }
    // Always leave a fresh command buffer armed so the context is ready for the
    // next submit (readback, next scope, or pre-scope work).
    if (!_commandBuffer) _commandBuffer = [_queue commandBuffer];
}

} // namespace metal
} // namespace backend
} // namespace cppgrad
