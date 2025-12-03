#include "minidl/detail/dispatch.h"
#include "minidl/detail/reduction_ops.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor sum(const Tensor& tensor) {
    return detail::dispatch(
        tensor.dtype(), [&] { return detail::sum_impl<float>(tensor); },
        [&] { return detail::sum_impl<int32_t>(tensor); });
}

}  // namespace minidl::ops
