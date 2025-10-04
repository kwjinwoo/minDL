#include <iostream>

#include "minidl/detail/binary_ops.h"
#include "minidl/detail/dispatch.h"
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

}  // namespace minidl::ops
