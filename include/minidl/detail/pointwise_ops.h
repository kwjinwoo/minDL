#pragma once
#include "minidl/detail/broadcasting.h"
#include "minidl/detail/kernels_pointwise.h"
#include "minidl/tensor.h"

namespace minidl::detail {
// imple
template <typename T>
Tensor relu_impl(const Tensor& tensor) noexcept {
    Tensor out = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.storage()->alloc_);

    const std::size_t n = out.numel();

    auto* z = static_cast<T*>(out.data());
    auto* x = static_cast<T*>(tensor.data());

    // scalar
    if (n == 0) {
        z[0] = kernels::relu_elem<T>(x[0]);
    }

    if (tensor.is_contiguous()) {
        kernels::relu_contig(z, x, out.numel());
    } else {
        kernels::relu_non_contig(z, x, out.shape().dims(), tensor.strides());
    }

    return out;
}
}  // namespace minidl::detail
