// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <utility>
#include "cppgrad/backend/buffer.h"
#include "cppgrad/common/access_meta.h"

namespace cppgrad::ir {

// Combined storage-and-view descriptor shared across tensors.
// Multiple tensors can share 'buffer' while each carries its own 'view'.
struct StorageView {
    static StorageView contiguous_from(std::shared_ptr<backend::Buffer> buf, std::vector<size_t> shape, size_t offset = 0) {
        StorageView sv;
        sv.buffer = std::move(buf);
        sv.access_meta = common::AccessMeta::contiguous_from(std::move(shape), offset);
        return sv;
    }

    static StorageView from(std::shared_ptr<backend::Buffer> buf, common::AccessMeta access_meta) {
        StorageView sv;
        sv.buffer = std::move(buf);
        sv.access_meta = std::move(access_meta);
        sv.access_meta.recompute_contiguity();
        return sv;
    }

    mutable std::shared_ptr<backend::Buffer> buffer;
    common::AccessMeta access_meta;
};

} // namespace cppgrad::ir
