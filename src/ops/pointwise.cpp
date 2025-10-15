#include <iostream>

#include "minidl/detail/binary_ops.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/dtype_promotion.h"
#include "minidl/detail/pointwise_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::AddOp<float>>(a, b, promoted_dtype); },
        [&] { return detail::binary_impl<int32_t, detail::AddOp<int32_t>>(a, b, promoted_dtype); });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::MulOp<float>>(a, b, promoted_dtype); },
        [&] { return detail::binary_impl<int32_t, detail::MulOp<int32_t>>(a, b, promoted_dtype); });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::SubOp<float>>(a, b, promoted_dtype); },
        [&] { return detail::binary_impl<int32_t, detail::SubOp<int32_t>>(a, b, promoted_dtype); });
}

Tensor div(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::DivOp<float>>(a, b, promoted_dtype); },
        [&] { return detail::binary_impl<int32_t, detail::DivOp<int32_t>>(a, b, promoted_dtype); });
}

// relu
Tensor relu(const Tensor& tensor) {
    return detail::dispatch(
        tensor.dtype(), [&] { return detail::relu_impl<float>(tensor); },
        [&] { return detail::relu_impl<int32_t>(tensor); });
}

// sigmoid
Tensor sigmoid(const Tensor& tensor) {
    return detail::dispatch(
        tensor.dtype(), [&] { return detail::sigmoid_impl<float>(tensor); },
        [&] {
            throw std::runtime_error("sigmoid: (TODO) Not yet implemented type propagation.");
            return tensor;
        });
}

}  // namespace minidl::ops
