// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdlib>
#include <algorithm>
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
    // Work is batched and flushed at scope boundaries / explicit flush_pending (see flush()).
    _computeWork.push_back(std::move(work));

    // Opt-in safety valve (CPPGRAD_METAL_MAX_KERNELS=N): bound the command buffer to N kernels by
    // flushing mid-scope. A large prefill chunk's linear-attention scan emits tens of thousands of
    // small kernels; batching them all into one command buffer pins every transient buffer resident
    // until completion and exhausts GPU memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). flush()
    // commits + waits, and intermediates stay owned by the graph (not freed), so a mid-scope flush is
    // order-preserving -- it changes only WHEN work runs, not the result. This decouples prefill chunk
    // size (throughput) from command-buffer size (memory). Off by default (0) -> unchanged behavior.
    static const size_t kMaxKernels = [] {
        const char* s = std::getenv("CPPGRAD_METAL_MAX_KERNELS");
        int v = s ? atoi(s) : 0;
        return v > 0 ? (size_t)v : (size_t)0;
    }();
    if (kMaxKernels && _computeWork.size() >= kMaxKernels) flush();
}

void MetalExecutionContext::flush() {
    if (!_computeWork.empty()) {
        if (getenv("CPPGRAD_METAL_DISPATCH")) fprintf(stderr, "[metal] flush: %zu kernels\n", _computeWork.size());

        // CPPGRAD_METAL_CAPTURE=N captures the Nth flush to /tmp/cppgrad_flush.gputrace for Xcode.
        // Run with METAL_CAPTURE_ENABLED=1. The command buffer is (re)created INSIDE the capture
        // scope so its commands are instrumented.
        static int s_fi = 0; ++s_fi;
        const char* cap = getenv("CPPGRAD_METAL_CAPTURE");
        MTLCaptureManager* capMgr = nil;
        if (cap && atoi(cap) == s_fi) {
            capMgr = [MTLCaptureManager sharedCaptureManager];
            MTLCaptureDescriptor* d = [[MTLCaptureDescriptor alloc] init];
            d.captureObject = _queue;
            d.destination = MTLCaptureDestinationGPUTraceDocument;
            d.outputURL = [NSURL fileURLWithPath:@"/tmp/cppgrad_flush.gputrace"];
            NSError* err = nil;
            if ([capMgr startCaptureWithDescriptor:d error:&err]) {
                fprintf(stderr, "[metal] capture flush %d (%zu kernels) -> /tmp/cppgrad_flush.gputrace\n", s_fi, _computeWork.size());
                _commandBuffer = [_queue commandBuffer];   // create inside capture scope
            } else { fprintf(stderr, "[metal] capture start failed: %s\n", err.localizedDescription.UTF8String); capMgr = nil; }
        }

        id<MTLComputeCommandEncoder> enc = [_commandBuffer computeCommandEncoder];
        for (const auto& work : _computeWork) {
            encode_work(enc, work);
        }
        [enc endEncoding];
        [_commandBuffer commit];
        [_commandBuffer waitUntilCompleted];

        // A GPU fault (e.g. an out-of-bounds kernel access) fails the command buffer silently:
        // no exception, and every op after the fault leaves its output buffer unwritten (reads as
        // zero). Surface it loudly - garbage results are otherwise undetectable. Reading status
        // after the wait we already do is free; the description is only built on failure.
        if (_commandBuffer.status != MTLCommandBufferStatusCompleted) {
            NSError* cbErr = _commandBuffer.error;
            fprintf(stderr, "[metal] command buffer did not complete: status=%ld error=%s (domain=%s code=%ld) [%zu kernels]\n",
                    (long)_commandBuffer.status,
                    cbErr ? cbErr.localizedDescription.UTF8String : "(none)",
                    cbErr ? cbErr.domain.UTF8String : "(none)",
                    cbErr ? (long)cbErr.code : 0L, _computeWork.size());
        }

        if (capMgr) { [capMgr stopCapture]; fprintf(stderr, "[metal] capture done\n"); }

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
