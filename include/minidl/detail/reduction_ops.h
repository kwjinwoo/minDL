#pragma once
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl::detail {
template <typename T>
Tensor sum_impl(const Tensor& tensor, const std::vector<std::size_t>& axes, bool keepdims) {
    if (tensor.rank() == 0) return tensor;

    const auto r = tensor.rank();
    std::vector<bool> seen(r, axes.empty());
    for (auto axis : axes) {
        if (axis >= r) throw std::runtime_error("sum: axis out of range.");
        seen[axis] = true;
    }

    const auto& dims = tensor.shape().dims();
    std::vector<std::size_t> out_shape;
    out_shape.reserve(r);
    if (keepdims) {
        out_shape.resize(r);

        for (std::size_t i = 0; i < r; i++) {
            out_shape[i] = seen[i] ? 1 : dims[i];
        }
    } else {
        for (std::size_t i = 0; i < r; i++) {
            if (!seen[i]) out_shape.push_back(dims[i]);
        }
    }

    std::vector<std::size_t> kept_axes;
    if (!keepdims) {
        for (std::size_t i = 0; i < r; ++i) {
            if (!seen[i]) kept_axes.push_back(i);
        }
    }

    Tensor out = Tensor::zeros(Shape(out_shape), tensor.dtype(), tensor.storage()->alloc_, tensor.requires_grad());

    auto* out_data = static_cast<T*>(out.data());
    const auto* in_data = static_cast<const T*>(tensor.data());

    const auto& in_strides = tensor.strides();
    const auto& out_strides = out.strides();
    detail::NdCounter counter(dims);
    std::vector<std::size_t> out_idx(out.rank());
    const bool all_reduced = (!keepdims && kept_axes.empty());
    for (; !counter.done(); counter.next()) {
        auto idx = counter.idx;

        std::size_t in_off = detail::offset_elems(idx, in_strides);
        std::size_t out_off = 0;
        if (keepdims) {
            for (std::size_t i = 0; i < r; i++) {
                out_idx[i] = seen[i] ? 0 : idx[i];
            }
            out_off = detail::offset_elems(out_idx, out_strides);
        } else {
            if (all_reduced) {
                out_off = 0;
            } else {
                for (std::size_t k = 0; k < kept_axes.size(); ++k) {
                    out_idx[k] = idx[kept_axes[k]];
                }
                out_off = detail::offset_elems(out_idx, out_strides);
            }
        }

        out_data[out_off] += in_data[in_off];
    }
    return out;
}
}  // namespace minidl::detail
