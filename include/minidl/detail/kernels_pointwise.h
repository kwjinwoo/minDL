#pragma once
#include <vector>

#include "minidl/detail/iter.h"

namespace minidl::kernels {

template <typename T, class Op>
inline void binary_contig(T* __restrict z, const T* __restrict x, const T* __restrict y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = Op::apply(x[i], y[i]);
    }
}

template <typename T, class Op>
inline void binary_same_shape_strided(T* __restrict z, const T* __restrict x, const T* __restrict y,
                                      const std::vector<std::size_t>& shape, const std::vector<std::size_t>& xs,
                                      const std::vector<std::size_t>& ys) noexcept {
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
                             const std::vector<std::size_t>& ys) noexcept {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = Op::apply(x[xo], y[yo]);
        it.next();
    }
}
}  // namespace minidl::kernels