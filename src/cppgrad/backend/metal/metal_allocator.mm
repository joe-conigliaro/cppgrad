// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include "cppgrad/backend/metal/metal_allocator.h"
#include "cppgrad/backend/buffer.h"
#include "cppgrad/backend/metal/metal_execution_context.h"
#include "cppgrad/backend/metal/metal_utils.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace cppgrad::backend::metal {

// Small impl that owns device, queue, and reusable staging
class MetalAllocatorImpl {
    public:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> staging = nil;               // reusable Shared staging buffer
    MetalExecutionContext *exec_ctx = nullptr; // non-owning; set at construction

    explicit MetalAllocatorImpl(void *native_device, MetalExecutionContext *ec) {
        device = (__bridge id<MTLDevice>)native_device;
        queue = [device newCommandQueue];
        exec_ctx = ec;
    }

    ~MetalAllocatorImpl() {
        staging = nil;
        queue = nil;
        device = nil;
    }

    id<MTLBuffer> get_or_resize_staging(NSUInteger bytes) {
        if (!staging || staging.length < bytes) {
            // Shared storage for host-visible staging
            MTLResourceOptions opts = MTLResourceStorageModeShared;
            #if TARGET_OS_OSX
                // On macOS with discrete GPU, if you want to read back frequently, Shared is still fine for staging.
            #endif
            staging = [device newBufferWithLength:bytes options:opts];
        }
        return staging;
    }

    void blit_copy(id<MTLBuffer> src, id<MTLBuffer> dst, NSUInteger bytes) const {
        // Always synchronous. The async path (recording into exec_ctx) is incorrect
        // for readback: the blit would be recorded but never committed, causing the
        // subsequent memcpy to read uninitialized/garbage data from the staging
        // buffer.
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void blit_synchronize(id<MTLBuffer> buf) {
        if (exec_ctx) {
            // Async: record into execution context (sync is only needed during
            // readback, which happens after flush, so no-op here is fine).
            (void)buf;
            return;
        }
        #if TARGET_OS_OSX
            // Managed storage exists on macOS (discrete GPU)
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit synchronizeResource:buf];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        #else
            // On iOS/Apple Silicon, storage is usually Shared; no synchronize needed.
            (void)buf;
        #endif
    }
};

MetalAllocator::MetalAllocator(void *native_device, MetalExecutionContext *ec)
: _impl(std::make_unique<MetalAllocatorImpl>(native_device, ec)) {}

MetalAllocator::~MetalAllocator() = default;

// Allocate Metal buffer (Shared by default). This is host-visible and simple to
// use. If you switch to Private for performance, copy_to/from_device already
// handle staging.
// Throw (catchable) instead of returning a nil-backed Buffer when Metal can't satisfy an allocation.
// Otherwise the nil MTLBuffer is used by a later kernel/copy and faults the process with SIGBUS (no
// lag, no exception) -- e.g. a huge KV cache from a long prompt + large max_tokens. As a catchable
// std::runtime_error the server's try/catch logs it and stays up.
static void check_alloc(id<MTLBuffer> buf, size_t bytes, id<MTLDevice> dev) {
    if (buf) return;
    throw std::runtime_error("Metal buffer allocation failed: requested " + std::to_string(bytes) +
                             " bytes (device maxBufferLength=" + std::to_string((size_t)[dev maxBufferLength]) +
                             ", recommendedMaxWorkingSetSize=" + std::to_string((size_t)[dev recommendedMaxWorkingSetSize]) + ")");
}

std::shared_ptr<Buffer> MetalAllocator::allocate(size_t num_elements,
common::DType dtype) {
    const size_t bytes = num_elements * size(dtype);
    // Zero-size allocation: Metal returns nil for newBufferWithLength:0 (which check_alloc would treat
    // as failure), so hand back an empty null-backed Buffer -- matching the host-data allocate overload.
    if (bytes == 0) {
        return std::make_shared<Buffer>(nullptr, 0, dtype, DeviceType::METAL, this);
    }
    id<MTLBuffer> buf = [_impl->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    check_alloc(buf, bytes, _impl->device);
    return std::make_shared<Buffer>((__bridge_retained void *)buf, bytes, dtype, DeviceType::METAL, this);
}

std::shared_ptr<Buffer>
MetalAllocator::allocate(const void *src, size_t num_elements, common::DType dtype) {
    const size_t bytes = num_elements * size(dtype);
    if (bytes == 0) {
        return std::make_shared<Buffer>(nullptr, 0, dtype, DeviceType::METAL, this);
    }
    id<MTLBuffer> buf = [_impl->device newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
    check_alloc(buf, bytes, _impl->device);
    return std::make_shared<Buffer>((__bridge_retained void *)buf, bytes, dtype, DeviceType::METAL, this);
}

void MetalAllocator::deallocate(void *ptr) {
    if (ptr) {
        CFRelease(ptr); // release the retained MTLBuffer
    }
}

// Device -> Host
void MetalAllocator::copy_device_to_host(void *host_dst,
const Buffer &device_src) const {
    if (device_src.size_bytes() == 0 || !host_dst) return;
    if (device_src.device_type() != DeviceType::METAL) throw std::runtime_error("MetalAllocator::copy_from_device: src is not METAL");

    id<MTLBuffer> src = as_mtl_checked(device_src);
    const size_t bytes = device_src.size_bytes();

    if ((size_t)[src length] != bytes) throw std::runtime_error("MetalAllocator::copy_from_device: MTLBuffer length mismatch");

    // Readback must observe all GPU work scheduled this scope. In async mode the
    // backend batches compute into the execution context and commits at scope
    // end; flush here so the producing kernels have actually run before we read.
    if (_impl->exec_ctx) _impl->exec_ctx->flush();

    MTLResourceOptions opts = src.resourceOptions;

    if (opts & MTLResourceStorageModeShared) {
        // Shared: GPU writes go to system RAM. On Apple Silicon (unified memory)
        // this is direct.
        std::memcpy(host_dst, [src contents], bytes);
        return;
    }
    #if TARGET_OS_OSX
    if (opts & MTLResourceStorageModeManaged) {
        // Make device writes visible to CPU
        _impl->blit_synchronize(src);
        std::memcpy(host_dst, [src contents], bytes);
        return;
    }
    #endif
    // Private (or unknown): blit to staging Shared, then memcpy
    {
        id<MTLBuffer> staging = _impl->get_or_resize_staging((NSUInteger)bytes);
        _impl->blit_copy(src, staging, (NSUInteger)bytes);
        std::memcpy(host_dst, [staging contents], bytes);
    }
}

// Host -> Device
void MetalAllocator::copy_host_to_device(Buffer &device_dst,
const void *host_src) const {
    if (device_dst.size_bytes() == 0 || !host_src) return;
    if (device_dst.device_type() != DeviceType::METAL) throw std::runtime_error("MetalAllocator::copy_to_device: dst is not METAL");

    id<MTLBuffer> dst = as_mtl_checked(device_dst);
    const size_t bytes = device_dst.size_bytes();

    if ((size_t)[dst length] != bytes) throw std::runtime_error("MetalAllocator::copy_to_device: MTLBuffer length mismatch");

    // Flush batched compute first: a pending kernel may write to this buffer and,
    // running later on a separate command buffer, would clobber the data we set
    // here.
    if (_impl->exec_ctx) _impl->exec_ctx->flush();

    // Detect storage mode based on resourceOptions (Metal doesn’t expose an enum
    // on MTLBuffer directly across platforms) For simplicity, assume Shared for
    // now, but handle Private/Managed if you change allocator options.
    MTLResourceOptions opts = dst.resourceOptions;

    if (opts & MTLResourceStorageModeShared) {
        // Shared: direct memcpy; no didModifyRange (that is only valid for Managed
        // buffers - calling it on Shared trips Metal API validation).
        std::memcpy([dst contents], host_src, bytes);
        return;
    }
    #if TARGET_OS_OSX
    if (opts & MTLResourceStorageModeManaged) {
        std::memcpy([dst contents], host_src, bytes);
        [dst didModifyRange:NSMakeRange(0, bytes)];
        return;
    }
    #endif
    // Private (or unknown): stage and blit
    {
        id<MTLBuffer> staging = _impl->get_or_resize_staging((NSUInteger)bytes);
        std::memcpy([staging contents], host_src, bytes);
        _impl->blit_copy(staging, dst, (NSUInteger)bytes);
    }
}

// Device -> Device (same Metal device) GPU blit
void MetalAllocator::copy_device_to_device(Buffer &dst,
const Buffer &src) const {
    if (dst.size_bytes() == 0) return;
    if (dst.device_type() != DeviceType::METAL || src.device_type() != DeviceType::METAL) throw std::runtime_error("MetalAllocator::copy_device_to_device: both buffers must be METAL");

    id<MTLBuffer> mtl_dst = as_mtl_checked(dst);
    id<MTLBuffer> mtl_src = as_mtl_checked(src);

    const size_t bytes = dst.size_bytes();
    if ((size_t)[mtl_dst length] != bytes || (size_t)[mtl_src length] != bytes) throw std::runtime_error("MetalAllocator::copy_device_to_device: length mismatch");

    // Flush batched compute first: pending kernels may target src or dst, and
    // this blit runs on a separate command buffer with no ordering against them.
    if (_impl->exec_ctx) _impl->exec_ctx->flush();
    _impl->blit_copy(mtl_src, mtl_dst, (NSUInteger)bytes);
}

} // namespace cppgrad::backend::metal
