#include <iostream>

#include "minidl/detail/broadcasting.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/kenerls_pointwise.h"
#include "minidl/ops.h"
#include "minidl/shape.h"

namespace minidl::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype()) throw std::runtime_error("add: dtype mismatch.");

    const auto out_shape = detail::compute_broadcast_shape(a.shape().dims(), b.shape().dims());

    Tensor out = Tensor::zeros(Shape(out_shape), a.dtype(), a.storage()->alloc_);
    const std::size_t n = out.numel();
    if (n == 0) return out;

    auto xs = detail::expand_strides_for_broadcast(a.shape().dims(), a.strides(), out_shape);
    auto ys = detail::expand_strides_for_broadcast(b.shape().dims(), b.strides(), out_shape);

    const bool same_shape = (a.shape().dims() == b.shape().dims());
    const bool same_strides = same_shape && (a.strides() == b.strides());
    const bool cont_all = a.is_contiguous() && b.is_contiguous() && out.is_contiguous();
    const bool no_bcast = same_shape && same_strides;

    detail::dispatch(
        a.dtype(),
        [&] {
            auto* z = static_cast<float*>(out.data());
            auto* x = static_cast<const float*>(a.data());
            auto* y = static_cast<const float*>(b.data());

            if (cont_all && no_bcast) {
                kernels::add_contig<float>(z, x, y, out.numel());
            } else if (no_bcast) {
                kernels::add_same_shape_strided<float>(z, x, y, out_shape, a.strides(), b.strides());
            } else {
                kernels::add_broadcast<float>(z, x, y, out_shape, xs, ys);
            }
        },
        [&] {
            auto* z = static_cast<int32_t*>(out.data());
            auto* x = static_cast<const int32_t*>(a.data());
            auto* y = static_cast<const int32_t*>(b.data());

            if (cont_all && no_bcast) {
                kernels::add_contig<int32_t>(z, x, y, out.numel());
            } else if (no_bcast) {
                kernels::add_same_shape_strided<int32_t>(z, x, y, out_shape, a.strides(), b.strides());
            } else {
                kernels::add_broadcast<int32_t>(z, x, y, out_shape, xs, ys);
            }
        });
    return out;
}

}  // namespace minidl::ops
