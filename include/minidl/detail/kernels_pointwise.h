#pragma once
#include <cmath>
#include <vector>

#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl::kernels {

template <typename T, class Op>
inline void binary_contig(const Tensor& a, const Tensor& b, Tensor& out, std::size_t n) {
    detail::ElementReader ra(a.data(), a.dtype());
    detail::ElementReader rb(b.data(), b.dtype());
    auto* z = static_cast<T*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = Op::apply(ra.read_as<T>(i), rb.read_as<T>(i));
    }
}

template <typename T, class Op>
inline void binary_same_shape_strided(const Tensor& a, const Tensor& b, Tensor& out,
                                      const std::vector<std::size_t>& shape, const std::vector<std::size_t>& xs,
                                      const std::vector<std::size_t>& ys) {
    detail::ElementReader ra(a.data(), a.dtype());
    detail::ElementReader rb(b.data(), b.dtype());
    auto* z = static_cast<T*>(out.data());
    minidl::detail::NdCounter it(shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = Op::apply(ra.read_as<T>(xo), rb.read_as<T>(yo));
        it.next();
    }
}

template <typename T, class Op>
inline void binary_broadcast(const Tensor& a, const Tensor& b, Tensor& out, const std::vector<std::size_t>& out_shape,
                             const std::vector<std::size_t>& xs, const std::vector<std::size_t>& ys) {
    detail::ElementReader ra(a.data(), a.dtype());
    detail::ElementReader rb(b.data(), b.dtype());
    auto* z = static_cast<T*>(out.data());
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = Op::apply(ra.read_as<T>(xo), rb.read_as<T>(yo));
        it.next();
    }
}

// relu
template <typename T>
inline T relu_elem(const T x) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(x)) return x;
        return std::fmax(x, static_cast<T>(0));
    } else {
        return (x < static_cast<T>(0)) ? static_cast<T>(0) : x;
    }
}

template <typename T>
inline void relu_contig(T* z, const T* x, const std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; i++) {
        z[i] = relu_elem<T>(x[i]);
    }
}

template <typename T>
inline void relu_non_contig(T* z, const T* x, const std::vector<std::size_t>& out_shape,
                            const std::vector<std::size_t>& xs) noexcept {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        z[zi++] = relu_elem<T>(x[xo]);
        it.next();
    }
}

// sigmoid
template <typename T>
inline T sigmoid_elem(const T x) noexcept {
    return 1 / (1 + std::exp(-x));
}

template <typename T>
inline void sigmoid_contig(T* z, const T* x, const ::size_t n) noexcept {
    for (std::size_t i = 0; i < n; i++) {
        z[i] = sigmoid_elem<T>(x[i]);
    }
}

template <typename T>
inline void sigmoid_non_contig(T* z, const T* x, const std::vector<std::size_t>& out_shape,
                               const std::vector<std::size_t>& xs) noexcept {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        z[zi++] = sigmoid_elem<T>(x[xo]);
        it.next();
    }
}

}  // namespace minidl::kernels