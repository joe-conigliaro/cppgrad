// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#import <Metal/Metal.h>
#include <stdexcept>
#include "cppgrad/backend/buffer.h"

namespace cppgrad {
namespace backend {
namespace metal {

// Safely casts the generic void* from a Buffer to the concrete id<MTLBuffer> type.
// This is the designated way for Metal backend code to access the underlying buffer object.
inline id<MTLBuffer> as_mtl(const Buffer& buf) {
    // We use a direct __bridge cast because the Buffer's void* was created
    // with __bridge_retained, and its lifetime is managed by the Buffer's destructor
    // calling deallocate, which does a CFBridgingRelease.
    // The asMTL function just needs a temporary, non-owning reference.
    return (__bridge id<MTLBuffer>)buf.data();
}

inline id<MTLBuffer> as_mtl_checked(const Buffer& buf) {
    if (buf.device_type() != DeviceType::METAL) {
        throw std::runtime_error("as_mtl_checked: not a METAL buffer");
    }
    // Buffer::data() stores a retained MTLBuffer object pointer (void*).
    // This bridge is non-owning; Buffer destructor releases it.
    return (__bridge id<MTLBuffer>)buf.data();
}

} // namespace metal
} // namespace backend
} // namespace cppgrad
