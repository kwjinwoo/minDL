#include "minidl/detail/dispatch.h"
#include "minidl/detail/dtype_promotion.h"
#include "minidl/detail/kernels_matmul.h"
#include "minidl/ops.h"

namespace minidl::ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    DType promoted_dtype = detail::promote_dtype(a.dtype(), b.dtype());
    return detail::dispatch(
        promoted_dtype, [&] { return detail::naive_2d_matmul<float>(a, b, promoted_dtype); },
        [&] { return detail::naive_2d_matmul<int32_t>(a, b, promoted_dtype); });
}

}  // namespace minidl::ops
