#include <iostream>

#include "minidl/detail/binary_ops.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/pointwise_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    return detail::dispatch(
        a.dtype(), [&] { return detail::binary_impl<float, detail::AddOp<float>>(a, b); },
        [&] { return detail::binary_impl<int32_t, detail::AddOp<int32_t>>(a, b); });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return detail::dispatch(
        a.dtype(), [&] { return detail::binary_impl<float, detail::MulOp<float>>(a, b); },
        [&] { return detail::binary_impl<int32_t, detail::MulOp<int32_t>>(a, b); });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return detail::dispatch(
        a.dtype(), [&] { return detail::binary_impl<float, detail::SubOp<float>>(a, b); },
        [&] { return detail::binary_impl<int32_t, detail::SubOp<int32_t>>(a, b); });
}

Tensor div(const Tensor& a, const Tensor& b) {
    return detail::dispatch(
        a.dtype(), [&] { return detail::binary_impl<float, detail::DivOp<float>>(a, b); },
        [&] { return detail::binary_impl<int32_t, detail::DivOp<int32_t>>(a, b); });
}

// relu
Tensor relu(const Tensor& tensor) {
    return detail::dispatch(
        tensor.dtype(), [&] { return detail::relu_impl<float>(tensor); },
        [&] { return detail::relu_impl<int32_t>(tensor); });
}

}  // namespace minidl::ops
