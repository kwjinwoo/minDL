#pragma once
#include "minidl/detail/broadcasting.h"
#include "minidl/detail/kernels_pointwise.h"
#include "minidl/tensor.h"

namespace minidl::detail {
// imple
template <typename T>
Tensor relu_impl(const Tensor& tensor, std::shared_ptr<GradFn> grad_fn) noexcept {
    Tensor out = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.storage()->alloc_, tensor.requires_grad());
    if (grad_fn) out.impl()->grad_fn = grad_fn;
    const std::size_t n = out.numel();

    auto* z = static_cast<T*>(out.data());
    auto* x = static_cast<T*>(tensor.data());

    // scalar
    if (n == 0) {
        z[0] = kernels::relu_elem<T>(x[0]);
    }

    if (tensor.is_contiguous()) {
        kernels::relu_contig<T>(z, x, n);
    } else {
        kernels::relu_non_contig<T>(z, x, out.shape().dims(), tensor.strides());
    }

    return out;
}

template <typename T>
Tensor sigmoid_impl(const Tensor& tensor, std::shared_ptr<GradFn> grad_fn) noexcept {
    Tensor out = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.storage()->alloc_, tensor.requires_grad());
    if (grad_fn) out.impl()->grad_fn = grad_fn;
    const std::size_t n = out.numel();

    auto* z = static_cast<T*>(out.data());
    auto* x = static_cast<T*>(tensor.data());

    if (n == 0) {
        z[0] = kernels::sigmoid_elem<T>(x[0]);
    }

    if (tensor.is_contiguous()) {
        kernels::sigmoid_contig<T>(z, x, n);
    } else {
        kernels::sigmoid_non_contig<T>(z, x, out.shape().dims(), tensor.strides());
    }
    return out;
}
}  // namespace minidl::detail
