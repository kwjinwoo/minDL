#include "minidl/detail/kernels_pointwise.h"

#include "minidl/detail/binary_ops.h"

namespace minidl::kernels {

// add instances
template void binary_contig<float, detail::AddOp<float>>(float*, const float*, const float*, std::size_t);
template void binary_contig<std::int32_t, detail::AddOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t);
template void binary_same_shape_strided<float, detail::AddOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::AddOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                                   const std::int32_t*,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::AddOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::AddOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// mul instances
template void binary_contig<float, detail::MulOp<float>>(float*, const float*, const float*, std::size_t);
template void binary_contig<std::int32_t, detail::MulOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t);
template void binary_same_shape_strided<float, detail::MulOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::MulOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                                   const std::int32_t*,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::MulOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::MulOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// sub instances
template void binary_contig<float, detail::SubOp<float>>(float*, const float*, const float*, std::size_t);
template void binary_contig<std::int32_t, detail::SubOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t);
template void binary_same_shape_strided<float, detail::SubOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::SubOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                                   const std::int32_t*,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::SubOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::SubOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// div instances
template void binary_contig<float, detail::DivOp<float>>(float*, const float*, const float*, std::size_t);
template void binary_contig<std::int32_t, detail::DivOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                       const std::int32_t*, std::size_t);
template void binary_same_shape_strided<float, detail::DivOp<float>>(float*, const float*, const float*,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::DivOp<std::int32_t>>(std::int32_t*, const std::int32_t*,
                                                                                   const std::int32_t*,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::DivOp<float>>(float*, const float*, const float*,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::DivOp<float>>(std::int32_t*, const std::int32_t*,
                                                                   const std::int32_t*, const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

}  // namespace minidl::kernels
