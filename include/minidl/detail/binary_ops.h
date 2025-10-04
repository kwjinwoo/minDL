#pragma once
#include "minidl/detail/broadcasting.h"
#include "minidl/detail/kernels_pointwise.h"
#include "minidl/tensor.h"

namespace minidl::detail {

// Functors
template <typename T>
struct AddOp {
    static inline T apply(T a, T b) noexcept { return a + b; }
};

template <typename T>
struct MulOp {
    static inline T apply(T a, T b) noexcept { return a * b; }
};

// impl
template <typename T, class Op>
Tensor binary_impl(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype()) throw std::runtime_error("binary_impl: dtype mismatch.");

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

    auto* z = static_cast<T*>(out.data());
    auto* x = static_cast<const T*>(a.data());
    auto* y = static_cast<const T*>(b.data());

    if (cont_all && no_bcast) {
        kernels::binary_contig<T, Op>(z, x, y, out.numel());
    } else if (no_bcast) {
        kernels::binary_same_shape_strided<T, Op>(z, x, y, out_shape, a.strides(), b.strides());
    } else {
        kernels::binary_broadcast<T, Op>(z, x, y, out_shape, xs, ys);
    }
    return out;
}

}  // namespace minidl::detail
