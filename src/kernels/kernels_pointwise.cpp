#include "minidl/detail/kernels_pointwise.h"

#include "minidl/detail/binary_ops.h"

namespace minidl::kernels {

// Add instances
template void binary_contig<float, detail::AddOp<float>>(float*, const float*, const float*, std::size_t) noexcept;
template void binary_contig<std::int32_t, detail::AddOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t) noexcept;
template void binary_same_shape_strided<float, detail::AddOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&) noexcept;
template void binary_same_shape_strided<std::int32_t, detail::AddOp<std::int32_t>>(
    std::int32_t*, const std::int32_t*, const std::int32_t*, const std::vector<std::size_t>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&) noexcept;
template void binary_broadcast<float, detail::AddOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&) noexcept;
template void binary_broadcast<std::int32_t, detail::AddOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&) noexcept;

// Mul instances
template void binary_contig<float, detail::MulOp<float>>(float*, const float*, const float*, std::size_t) noexcept;
template void binary_contig<std::int32_t, detail::MulOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t) noexcept;
template void binary_same_shape_strided<float, detail::MulOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&) noexcept;
template void binary_same_shape_strided<std::int32_t, detail::MulOp<std::int32_t>>(
    std::int32_t*, const std::int32_t*, const std::int32_t*, const std::vector<std::size_t>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&) noexcept;
template void binary_broadcast<float, detail::MulOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&) noexcept;
template void binary_broadcast<std::int32_t, detail::MulOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&) noexcept;

}  // namespace minidl::kernels
