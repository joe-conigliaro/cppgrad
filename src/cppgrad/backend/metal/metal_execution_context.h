// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace cppgrad::backend::metal {

// A single small `setBytes:` binding (inline so the work item is self-contained
// and survives being copied into the pending-work queue - no dangling pointers).
struct ByteArg {
    NSUInteger index = 0;
    NSUInteger length = 0;
    unsigned char bytes[240] = {};
};

// A single pending compute dispatch. Fully self-contained / trivially copyable:
// it owns inline copies of every kernel argument and the dispatch geometry, so it
// can be recorded now and encoded later without referencing any caller stack data.
struct ComputeWork {
    id<MTLComputePipelineState> pso = nil;
    // Buffer bindings at indices 0, 1, 2, ...
    std::vector<std::pair<id<MTLBuffer>, NSUInteger>> buffers;
    // Inline `setBytes:` bindings (params structs / scalars).
    std::array<ByteArg, 5> byteArgs{};
    size_t byteArgCount = 0;

    // Dispatch geometry, filled by the submitting op to match the kernel's ABI.
    bool       useThreadgroups = false;          // dispatchThreadgroups vs dispatchThreads
    MTLSize    grid = {1, 1, 1};                  // threads (dispatchThreads) or #threadgroups
    MTLSize    threadsPerThreadgroup = {1, 1, 1};
    NSUInteger threadgroupMemoryLength = 0;       // threadgroup(0) bytes (reduce fast path)

    // Append an inline setBytes binding.
    void add_bytes(NSUInteger index, const void* src, NSUInteger length) {
        ByteArg& a = byteArgs[byteArgCount++];
        a.index = index;
        a.length = length;
        std::memcpy(a.bytes, src, length);
    }
};

// Encode a single work item into an open compute encoder.
inline void encode_work(id<MTLComputeCommandEncoder> enc, const ComputeWork& w) {
    [enc setComputePipelineState:w.pso];
    for (size_t i = 0; i < w.buffers.size(); ++i)
        [enc setBuffer:w.buffers[i].first offset:w.buffers[i].second atIndex:(NSUInteger)i];
    for (size_t i = 0; i < w.byteArgCount; ++i)
        [enc setBytes:w.byteArgs[i].bytes length:w.byteArgs[i].length atIndex:w.byteArgs[i].index];
    if (w.threadgroupMemoryLength > 0)
        [enc setThreadgroupMemoryLength:w.threadgroupMemoryLength atIndex:0];
    if (w.useThreadgroups)
        [enc dispatchThreadgroups:w.grid threadsPerThreadgroup:w.threadsPerThreadgroup];
    else
        [enc dispatchThreads:w.grid threadsPerThreadgroup:w.threadsPerThreadgroup];
}

// Batches compute work into a single command buffer and commits it once, instead
// of one command buffer (+ wait) per op. Work is flushed at GraphScope boundaries
// (Backend::flush_pending) and before any readback (the allocator's copy paths).
class MetalExecutionContext {
public:
    MetalExecutionContext(void* native_device, void* native_queue);
    ~MetalExecutionContext();

    // Record a compute work item for later encoding.
    void submit_compute(ComputeWork work);

    // Encode all pending work, commit once, wait once, then re-arm a fresh command
    // buffer so the context stays usable for the next submit. No-op when empty.
    void flush();

    id<MTLCommandQueue> command_queue() const { return _queue; }

private:
    id<MTLDevice> _device = nil;
    id<MTLCommandQueue> _queue = nil;
    id<MTLCommandBuffer> _commandBuffer = nil;
    std::vector<ComputeWork> _computeWork;
};

} // namespace cppgrad::backend::metal
