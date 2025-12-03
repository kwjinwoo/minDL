#include "minidl/autograd/grad_fn.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/reduction_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor sum(const Tensor& tensor) {
    Tensor y = detail::dispatch(
        tensor.dtype(), [&] { return detail::sum_impl<float>(tensor); },
        [&] { return detail::sum_impl<int32_t>(tensor); });

    if (tensor.requires_grad()) {
        y.impl()->grad_fn = std::make_shared<SumBackward>(tensor);
    }
    return y;
}

}  // namespace minidl::ops
