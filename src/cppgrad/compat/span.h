// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cstddef>

#if __cplusplus >= 202002L
#include <span>
namespace cppgrad {
template <typename T>
using Span = std::span<T>;
}
#else
#include <utility>
#include <type_traits>

namespace cppgrad {

template <typename T>
class Span {
    static_assert(!std::is_reference<T>::value, "Span<T> requires non-reference T");
public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using size_type = std::size_t;

    constexpr Span() noexcept : _ptr(nullptr), _size(0) {}
    constexpr Span(T* ptr, size_type size) noexcept : _ptr(ptr), _size(size) {}

    // From container with data()/size() (non-const)
    template <typename Container, typename = std::enable_if_t<!std::is_const<Container>::value && std::is_convertible<decltype(std::declval<Container&>().data()), T*>::value>>
    explicit Span(Container& c) noexcept : _ptr(c.data()), _size(c.size()) {}

    // From const container (only if T is a const type)
    template <typename Container, typename = std::enable_if_t<std::is_const<T>::value && std::is_convertible<decltype(std::declval<const Container&>().data()), T*>::value>>
    explicit Span(const Container& c) noexcept : _ptr(c.data()), _size(c.size()) {}

    constexpr T* data() const noexcept { return _ptr; }
    constexpr size_type size() const noexcept { return _size; }
    constexpr bool empty() const noexcept { return _size == 0; }
    constexpr T* begin() const noexcept { return _ptr; }
    constexpr T* end() const noexcept { return _ptr + _size; }
    constexpr T& operator[](size_type i) const { return _ptr[i]; }

 private:
    T* _ptr;
    size_type _size;
};

} // namespace cppgrad
#endif
