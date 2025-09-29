#include "minidl/detail/iter.h"
#include "minidl/detail/kenerls_pointwise.h"

namespace minidl::kernels {

template <typename T>
void add_contig(T* __restrict z, const T* __restrict x, const T* __restrict y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

template void add_contig<float>(float*, const float*, const float*, std::size_t);
template void add_contig<std::int32_t>(std::int32_t*, const std::int32_t*, const std::int32_t*, std::size_t);

template <typename T>
void add_same_shape_strided(T* __restrict z, const T* __restrict x, const T* __restrict y,
                            const std::vector<std::size_t>& shape, const std::vector<std::size_t>& xs,
                            const std::vector<std::size_t>& ys) noexcept {
    minidl::detail::NdCounter it(shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = x[xo] + y[yo];
        it.next();
    }
}

template void add_same_shape_strided<float>(float*, const float*, const float*, const std::vector<std::size_t>&,
                                            const std::vector<std::size_t>&, const std::vector<std::size_t>&) noexcept;
template void add_same_shape_strided<std::int32_t>(std::int32_t*, const std::int32_t*, const std::int32_t*,
                                                   const std::vector<std::size_t>&, const std::vector<std::size_t>&,
                                                   const std::vector<std::size_t>&);

template <typename T>
void add_broadcast(T* __restrict z, const T* __restrict x, const T* __restrict y,
                   const std::vector<std::size_t>& out_shape, const std::vector<std::size_t>& xs,
                   const std::vector<std::size_t>& ys) noexcept {
    minidl::detail::NdCounter it(out_shape);
    std::size_t zi = 0;
    while (!it.done()) {
        const auto xo = minidl::detail::offset_elems(it.idx, xs);
        const auto yo = minidl::detail::offset_elems(it.idx, ys);
        z[zi++] = x[xo] + y[yo];
        it.next();
    }
}

template void add_broadcast<float>(float*, const float*, const float*, const std::vector<std::size_t>&,
                                   const std::vector<std::size_t>&, const std::vector<std::size_t>&) noexcept;
template void add_broadcast<std::int32_t>(std::int32_t*, const std::int32_t*, const std::int32_t*,
                                          const std::vector<std::size_t>&, const std::vector<std::size_t>&,
                                          const std::vector<std::size_t>&);

}  // namespace minidl::kernels
