// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

namespace cppgrad {
namespace utils {

class Arena {
public:
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena() = default;
    ~Arena() = default;

    // Allocate raw memory for one object with given size and alignment.
    // Does NOT construct the object.
    void* alloc_raw(std::size_t size, std::size_t align) {
        // Calculate current address as integer
        auto current_addr = reinterpret_cast<std::uintptr_t>(_current_ptr);

        // Calculate aligned address: (addr + align - 1) & ~(align - 1)
        // Note: align must be a power of 2 (standard for alignof)
        auto aligned_addr = (current_addr + (align - 1)) & ~(align - 1);

        // Calculate total bytes needed including padding
        std::size_t padding = aligned_addr - current_addr;
        std::size_t total_needed = size + padding;

        // Fast Path: Check if it fits in current chunk
        if (total_needed <= _remaining_bytes) {
            _current_ptr = reinterpret_cast<void*>(aligned_addr + size);
            _remaining_bytes -= total_needed;
            return reinterpret_cast<void*>(aligned_addr);
        }

        // Slow Path: Allocate new chunk
        // We add 'align' to the request to guarantee we can align the start of the new block
        new_chunk(size + align);

        // Recalculate alignment on the new chunk
        current_addr = reinterpret_cast<std::uintptr_t>(_current_ptr);
        aligned_addr = (current_addr + (align - 1)) & ~(align - 1);

        // We know it fits now because new_chunk allocates max(size+align, CHUNK_SIZE)
        _current_ptr = reinterpret_cast<void*>(aligned_addr + size);

        padding = aligned_addr - current_addr;
        _remaining_bytes -= (size + padding);

        return reinterpret_cast<void*>(aligned_addr);
    }

    // Allocate memory for one object of type T and construct it.
    template<typename T, typename... Args>
    T* alloc(Args&&... args) {
        static_assert(!std::is_array<T>::value, "Arena::alloc does not support arrays");

        void* mem = alloc_raw(sizeof(T), alignof(T));

        return new (mem) T(std::forward<Args>(args)...);
    }

    // Resets the arena, invalidating all previously allocated pointers.
    // Keeps the first chunk to avoid reallocation overhead, releases the rest.
    void reset() {
        if (_chunks.empty()) {
            return;
        }

        // Keep only the first chunk to reuse memory
        if (_chunks.size() > 1) {
            _chunks.resize(1);
            _chunk_sizes.resize(1);
        }

        // Reset pointers to the start of the first chunk
        _current_ptr = _chunks.front().get();
        _remaining_bytes = _chunk_sizes.front();
    }

    // Pre-allocate or resize the first chunk to at least 'bytes'.
    void reserve(std::size_t bytes) {
        const size_t target = std::max(bytes, CHUNK_SIZE_BYTES);

        if (_chunks.empty()) {
            new_chunk(target);
            return;
        }

        // If the existing first chunk is too small, we must replace it
        if (_chunk_sizes.front() < target) {
            release_all();
            new_chunk(target);
            return;
        }

        // Otherwise, just reset to the beginning
        reset();
    }

    // Release all memory back to the system and reset arena state.
    void release_all() {
        _chunks.clear();
        _chunk_sizes.clear();
        _current_ptr = nullptr;
        _remaining_bytes = 0;
    }

    // Debug helper: does this pointer come from any of our chunks?
    bool owns(const void* p) const noexcept {
#ifdef CPPGRAD_DEBUG
        if (!p) return false;
        auto addr = reinterpret_cast<std::uintptr_t>(p);
        for (size_t i = 0; i < _chunks.size(); ++i) {
            auto base = reinterpret_cast<std::uintptr_t>(_chunks[i].get());
            auto end  = base + _chunk_sizes[i];
            if (addr >= base && addr < end) return true;
        }
#endif
        return false;
    }

private:
    void new_chunk(size_t min_size) {
        size_t chunk_size = std::max(min_size, CHUNK_SIZE_BYTES);

        _chunks.push_back(std::make_unique<std::byte[]>(chunk_size));
        _chunk_sizes.push_back(chunk_size);

        _current_ptr = _chunks.back().get();
        _remaining_bytes = chunk_size;
    }

    static constexpr size_t CHUNK_SIZE_BYTES = 65536; // 64 KB

    std::vector<std::unique_ptr<std::byte[]>> _chunks;
    std::vector<size_t> _chunk_sizes;

    void*  _current_ptr = nullptr;
    size_t _remaining_bytes = 0;
};

} // namespace utils
} // namespace cppgrad
