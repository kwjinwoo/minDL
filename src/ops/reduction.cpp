#include "minidl/autograd/grad_fn.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/reduction_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor sum(const Tensor& tensor, const std::vector<std::size_t>& axes, bool keepdims) {
    Tensor y = detail::dispatch(
        tensor.dtype(), [&] { return detail::sum_impl<float>(tensor, axes, keepdims); },
        [&] { return detail::sum_impl<int32_t>(tensor, axes, keepdims); });

    if (tensor.requires_grad()) {
        y.impl()->grad_fn = std::make_shared<SumBackward>(tensor, tensor.shape(), axes, keepdims);
    }
    return y;
}

Tensor sum(const Tensor& tensor, bool keepdims) {
    std::vector<std::size_t> axes(tensor.rank());
    for (std::size_t i = 0; i < axes.size(); i++) {
        axes[i] = i;
    }
    Tensor y = detail::dispatch(
        tensor.dtype(), [&] { return detail::sum_impl<float>(tensor, axes, keepdims); },
        [&] { return detail::sum_impl<int32_t>(tensor, axes, keepdims); });

    if (tensor.requires_grad()) {
        y.impl()->grad_fn = std::make_shared<SumBackward>(tensor, tensor.shape(), axes, keepdims);
    }
    return y;
}

}  // namespace minidl::ops
