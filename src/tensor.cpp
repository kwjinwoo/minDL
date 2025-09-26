#include "minidl/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minidl/allocators/default.h"
#include "minidl/detail/broadcasting.h"
#include "minidl/detail/iter.h"

namespace minidl {

Tensor Tensor::operator+(const Tensor& rhs) const {
    if (dtype_ != rhs.dtype_) {
        throw std::runtime_error("DType Must be same.");
    }
    const auto out_shape = detail::compute_broadcast_shape(shape_.dims(), rhs.shape_.dims());

    Tensor out = zeros(Shape(out_shape), dtype_, storage_->alloc_);
    const std::size_t n = out.numel();
    if (n == 0) return out;

    auto add_kernel = [&](auto* z, auto* x, auto* y) {
        // Fast path
        if (shape_.dims() == rhs.shape_.dims() && is_contiguous() && rhs.is_contiguous() &&
            strides() == rhs.strides()) {
            for (std::size_t i = 0; i < n; i++) {
                z[i] = x[i] + y[i];
            }
            return;
        }

        detail::NdCounter counter(out_shape);
        const auto& xs = detail::expand_strides_for_broadcast(shape_.dims(), strides_, out_shape);
        const auto& ys = detail::expand_strides_for_broadcast(rhs.shape_.dims(), rhs.strides_, out_shape);
        std::size_t z_offset = 0;

        while (!counter.done()) {
            auto x_offset = detail::offset_elems(counter.idx, xs);
            auto y_offset = detail::offset_elems(counter.idx, ys);
            z[z_offset] = x[x_offset] + y[y_offset];
            z_offset += 1;
            counter.next();
        }
    };

    switch (dtype_) {
        case DType::f32: {
            auto* z = static_cast<float*>(out.data());
            auto* x = static_cast<const float*>(data());
            auto* y = static_cast<const float*>(rhs.data());
            add_kernel(z, x, y);
            break;
        }
        case DType::i32: {
            auto* z = static_cast<std::int32_t*>(out.data());
            auto* x = static_cast<const std::int32_t*>(data());
            auto* y = static_cast<const std::int32_t*>(rhs.data());
            add_kernel(z, x, y);
            break;
        }
        default:
            throw std::runtime_error("add: unsupported dtype");
    }

    return out;
}

}  // namespace minidl
