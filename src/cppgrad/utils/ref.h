// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <atomic>
#include <cstddef>
#include <utility>
#include <stdexcept>

namespace cppgrad {
namespace utils {

// Base class for intrusive reference counting
class RefCounted {
public:
    RefCounted() = default;

    RefCounted(const RefCounted&) = delete;
    RefCounted& operator=(const RefCounted&) = delete;
    RefCounted(RefCounted&&) = delete;
    RefCounted& operator=(RefCounted&&) = delete;

    virtual ~RefCounted() = default;

    virtual void destroy() const {
        // Default behavior (Standard Heap delete).
        delete this;
    }

    void inc_ref() const noexcept {
        _ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Returns true if the object should be deleted (count dropped to 0)
    bool dec_ref() const noexcept {
        // fetch_sub returns the value before subtraction
        return _ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    long use_count() const noexcept {
        return _ref_count.load(std::memory_order_relaxed);
    }

private:
    mutable std::atomic<long> _ref_count;
};

// Intrusive Smart Pointer
template <typename T>
class Ref {
public:
    // Constructors
    Ref() : _ptr(nullptr) {}

    Ref(T* ptr) : _ptr(ptr) {
        if (_ptr) _ptr->inc_ref();
    }

    // Allow implicit conversion (e.g., Ref<Tensor> -> Ref<const Tensor>)
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    Ref(const Ref<U>& other) : _ptr(other.get()) {
        if (_ptr) _ptr->inc_ref();
    }

    // Copy
    Ref(const Ref& other) : _ptr(other._ptr) {
        if (_ptr) _ptr->inc_ref();
    }

    // Move
    Ref(Ref&& other) noexcept : _ptr(other._ptr) {
        other._ptr = nullptr;
    }

    ~Ref() {
        reset();
    }

    // Assignment
    Ref& operator=(const Ref& other) {
        if (this != &other) {
            reset();
            _ptr = other._ptr;
            if (_ptr) _ptr->inc_ref();
        }
        return *this;
    }

    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) {
            reset();
            _ptr = other._ptr;
            other._ptr = nullptr;
        }
        return *this;
    }

    // Accessors
    T* get() const noexcept { return _ptr; }
    T* operator->() const noexcept { return _ptr; }
    T& operator*() const noexcept { return *_ptr; }
    explicit operator bool() const noexcept { return _ptr != nullptr; }

    void reset() {
        if (_ptr) {
            if (_ptr->dec_ref()) {
                // Delegate deletion logic to the object
                _ptr->destroy();
            }
            _ptr = nullptr;
        }
    }

private:
    T* _ptr;
};

// Helper like std::make_shared
template<typename T, typename... Args>
Ref<T> make_ref(Args&&... args) {
    return Ref<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename U>
inline bool operator==(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() == rhs.get();
}

template <typename T, typename U>
inline bool operator!=(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() != rhs.get();
}

// Allow comparison with nullptr
template <typename T>
inline bool operator==(const Ref<T>& lhs, std::nullptr_t) noexcept {
    return lhs.get() == nullptr;
}

template <typename T>
inline bool operator==(std::nullptr_t, const Ref<T>& rhs) noexcept {
    return nullptr == rhs.get();
}

template <typename T>
inline bool operator!=(const Ref<T>& lhs, std::nullptr_t) noexcept {
    return lhs.get() != nullptr;
}

template <typename T>
inline bool operator!=(std::nullptr_t, const Ref<T>& rhs) noexcept {
    return nullptr != rhs.get();
}

template <typename T, typename U>
inline bool operator<(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() < rhs.get();
}

template <typename T, typename U>
inline bool operator>(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() > rhs.get();
}

template <typename T, typename U>
inline bool operator<=(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() <= rhs.get();
}

template <typename T, typename U>
inline bool operator>=(const Ref<T>& lhs, const Ref<U>& rhs) noexcept {
    return lhs.get() >= rhs.get();
}

} // namespace utils
} // namespace cppgrad

// std::hash Specialization
namespace std {

template <typename T>
struct hash<cppgrad::utils::Ref<T>> {
    std::size_t operator()(const cppgrad::utils::Ref<T>& ref) const noexcept {
        // Hash the underlying pointer address
        return std::hash<T*>{}(ref.get());
    }
};

} // namespace std
