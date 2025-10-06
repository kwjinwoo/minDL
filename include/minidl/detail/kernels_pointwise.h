#pragma once
#include <cmath>
#include <vector>

#include "minidl/detail/iter.h"

namespace minidl::kernels {

template <typename T, class Op>
inline void binary_contig(T* __restrict z, const T* __restrict x, const T* __restrict y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = Op::apply(x[i], y[i]);
    }
}

template <typename T, class Op>
inline void binary_same_shape_strided(T* __restrict z, const T* __restrict x, const T* __restrict y,
                                      const std::vector<std::size_t>& shape, const std::vector<std::size_t>& xs,
                                      const std::vector<std::size_t>& ys) {
    minidl::detail::NdCounter it(shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = Op::apply(x[xo], y[yo]);
        it.next();
    }
}

template <typename T, class Op>
inline void binary_broadcast(T* __restrict z, const T* __restrict x, const T* __restrict y,
                             const std::vector<std::size_t>& out_shape, const std::vector<std::size_t>& xs,
                             const std::vector<std::size_t>& ys) {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = Op::apply(x[xo], y[yo]);
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
        z[i] = relu_elem(x[i]);
    }
}

template <typename T>
inline void relu_non_contig(T* z, const T* x, const std::vector<std::size_t>& out_shape,
                            const std::vector<std::size_t>& xs) noexcept {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        z[zi++] = relu_elem(x[xo]);
        it.next();
    }
}
}  // namespace minidl::kernels