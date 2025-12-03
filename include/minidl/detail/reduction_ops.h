#pragma once
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl::detail {
template <typename T>
Tensor sum_impl(const Tensor& tensor) {
    auto data = static_cast<const T*>(tensor.data());

    T sum_out{};
    if (tensor.is_contiguous()) {
        const auto n = tensor.numel();
        for (std::size_t i = 0; i < n; i++) {
            sum_out += data[i];
        }
    } else {
        detail::NdCounter it(tensor.shape().dims());
        const auto& strides = tensor.strides();
        while (!it.done()) {
            const auto xo = detail::offset_elems(it.idx, strides);
            sum_out += data[xo];
            it.next();
        }
    }

    return Tensor::from_scalar(sum_out, tensor.dtype(), nullptr, tensor.requires_grad());
}
}  // namespace minidl::detail
