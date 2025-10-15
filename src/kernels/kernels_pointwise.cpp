#include "minidl/detail/kernels_pointwise.h"

#include "minidl/detail/binary_ops.h"

namespace minidl::kernels {

// add instances
template void binary_contig<float, detail::AddOp<float>>(const Tensor&, const Tensor&, Tensor&, std::size_t);
template void binary_contig<std::int32_t, detail::AddOp<std::int32_t>>(const Tensor&, const Tensor&, Tensor&,
                                                                       std::size_t);
template void binary_same_shape_strided<float, detail::AddOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::AddOp<std::int32_t>>(const Tensor&, const Tensor&,
                                                                                   Tensor&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::AddOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::AddOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// mul instances
template void binary_contig<float, detail::MulOp<float>>(const Tensor&, const Tensor&, Tensor&, std::size_t);
template void binary_contig<std::int32_t, detail::MulOp<std::int32_t>>(const Tensor&, const Tensor&, Tensor&,
                                                                       std::size_t);
template void binary_same_shape_strided<float, detail::MulOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::MulOp<std::int32_t>>(const Tensor&, const Tensor&,
                                                                                   Tensor&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::MulOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::MulOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// sub instances
template void binary_contig<float, detail::SubOp<float>>(const Tensor&, const Tensor&, Tensor&, std::size_t);
template void binary_contig<std::int32_t, detail::SubOp<std::int32_t>>(const Tensor&, const Tensor&, Tensor&,
                                                                       std::size_t);
template void binary_same_shape_strided<float, detail::SubOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::SubOp<std::int32_t>>(const Tensor&, const Tensor&,
                                                                                   Tensor&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::SubOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::SubOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// div instances
template void binary_contig<float, detail::DivOp<float>>(const Tensor&, const Tensor&, Tensor&, std::size_t);
template void binary_contig<std::int32_t, detail::DivOp<std::int32_t>>(const Tensor&, const Tensor&, Tensor&,
                                                                       std::size_t);
template void binary_same_shape_strided<float, detail::DivOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&,
                                                                     const std::vector<std::size_t>&);
template void binary_same_shape_strided<std::int32_t, detail::DivOp<std::int32_t>>(const Tensor&, const Tensor&,
                                                                                   Tensor&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&,
                                                                                   const std::vector<std::size_t>&);
template void binary_broadcast<float, detail::DivOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&,
                                                            const std::vector<std::size_t>&);
template void binary_broadcast<std::int32_t, detail::DivOp<float>>(const Tensor&, const Tensor&, Tensor&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&,
                                                                   const std::vector<std::size_t>&);

// relu
template float relu_elem<float>(float) noexcept;
template int32_t relu_elem<int32_t>(int32_t) noexcept;

template void relu_contig<float>(float*, const float*, std::size_t) noexcept;
template void relu_contig<int32_t>(int32_t*, const int32_t*, std::size_t) noexcept;

template void relu_non_contig<float>(float*, const float*, const std::vector<std::size_t>&,
                                     const std::vector<std::size_t>&) noexcept;
template void relu_non_contig<int32_t>(int32_t*, const int32_t*, const std::vector<std::size_t>&,
                                       const std::vector<std::size_t>&) noexcept;

// sigmoid
template float sigmoid_elem<float>(float) noexcept;

template void sigmoid_contig<float>(float*, const float*, std::size_t) noexcept;

template void sigmoid_non_contig<float>(float*, const float*, const std::vector<std::size_t>&,
                                        const std::vector<std::size_t>&) noexcept;

}  // namespace minidl::kernels
