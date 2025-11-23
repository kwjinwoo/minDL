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

template <typename T>
struct SubOp {
    static inline T apply(T a, T b) noexcept { return a - b; }
};

template <typename T>
struct DivOp {
    static inline T apply(T a, T b) {
        if constexpr (std::is_floating_point_v<T>) {
            if (b == static_cast<T>(0)) return std::numeric_limits<T>::quiet_NaN();
            return a / b;
        } else if constexpr (std::is_integral_v<T>) {
            if (b == 0) throw std::runtime_error("Div: Integer division by zero.");
            return a / b;
        }
    }
};

// impl
template <typename T, class Op>
Tensor binary_impl(const Tensor& a, const Tensor& b, DType promoted_dtype, std::shared_ptr<GradFn>& grad_fn) {
    const auto out_shape = detail::compute_broadcast_shape(a.shape().dims(), b.shape().dims());
    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor out = Tensor::zeros(Shape(out_shape), promoted_dtype, a.storage()->alloc_, requires_grad);
    if (grad_fn) out.impl()->grad_fn = grad_fn;
    const std::size_t n = out.numel();
    if (n == 0) return out;

    auto xs = detail::expand_strides_for_broadcast(a.shape().dims(), a.strides(), out_shape);
    auto ys = detail::expand_strides_for_broadcast(b.shape().dims(), b.strides(), out_shape);

    const bool same_shape = (a.shape().dims() == b.shape().dims());
    const bool same_strides = same_shape && (a.strides() == b.strides());
    const bool cont_all = a.is_contiguous() && b.is_contiguous() && out.is_contiguous();
    const bool no_bcast = same_shape && same_strides;

    if (cont_all && no_bcast) {
        kernels::binary_contig<T, Op>(a, b, out, out.numel());
    } else if (no_bcast) {
        kernels::binary_same_shape_strided<T, Op>(a, b, out, out_shape, a.strides(), b.strides());
    } else {
        kernels::binary_broadcast<T, Op>(a, b, out, out_shape, xs, ys);
    }
    return out;
}

}  // namespace minidl::detail
