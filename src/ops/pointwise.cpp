#include <iostream>

#include "minidl/autograd/grad_fn.h"
#include "minidl/detail/binary_ops.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/dtype_promotion.h"
#include "minidl/detail/pointwise_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    std::shared_ptr<GradFn> grad_fn;
    if (a.requires_grad() || b.requires_grad()) grad_fn = std::make_shared<AddBackward>(a, b);
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::AddOp<float>>(a, b, promoted_dtype, grad_fn); },
        [&] { return detail::binary_impl<int32_t, detail::AddOp<int32_t>>(a, b, promoted_dtype, grad_fn); });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    std::shared_ptr<GradFn> grad_fn;
    if (a.requires_grad() || b.requires_grad()) grad_fn = std::make_shared<MulBackward>(a, b);
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::MulOp<float>>(a, b, promoted_dtype, grad_fn); },
        [&] { return detail::binary_impl<int32_t, detail::MulOp<int32_t>>(a, b, promoted_dtype, grad_fn); });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    std::shared_ptr<GradFn> grad_fn;
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::SubOp<float>>(a, b, promoted_dtype, grad_fn); },
        [&] { return detail::binary_impl<int32_t, detail::SubOp<int32_t>>(a, b, promoted_dtype, grad_fn); });
}

Tensor div(const Tensor& a, const Tensor& b) {
    auto promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    std::shared_ptr<GradFn> grad_fn;
    return detail::dispatch(
        promoted_dtype, [&] { return detail::binary_impl<float, detail::DivOp<float>>(a, b, promoted_dtype, grad_fn); },
        [&] { return detail::binary_impl<int32_t, detail::DivOp<int32_t>>(a, b, promoted_dtype, grad_fn); });
}

// relu
Tensor relu(const Tensor& tensor) {
    std::shared_ptr<GradFn> grad_fn;
    if (tensor.requires_grad()) grad_fn = std::make_shared<ReluBackward>(tensor);
    return detail::dispatch(
        tensor.dtype(), [&] { return detail::relu_impl<float>(tensor, grad_fn); },
        [&] { return detail::relu_impl<int32_t>(tensor, grad_fn); });
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
