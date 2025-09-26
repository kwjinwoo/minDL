#include "minidl/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minidl/allocators/default.h"
#include "minidl/detail/broadcasting.h"
#include "minidl/detail/iter.h"

namespace minidl {

Tensor Tensor::view(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::runtime_error("view: new_shape.numel() must equal the current numel().");
    }
    if (numel() != 0 && !is_contiguous()) {
        throw std::runtime_error("view: tensor must be contiguous (use reshape for non-contiguous).");
    }

    Tensor out = *this;
    out.shape_ = new_shape;
    out.strides_ = default_strides(new_shape);
    return out;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::runtime_error("reshape: new_shape.numel() must equal the current numel().");
    }
    if (numel() == 0 || is_contiguous()) {
        Tensor new_tensor = *this;
        new_tensor.shape_ = new_shape;
        new_tensor.strides_ = default_strides(new_shape);
        return new_tensor;
    }

    Tensor new_tensor = this->contiguous();
    new_tensor.shape_ = new_shape;
    new_tensor.strides_ = default_strides(new_shape);
    return new_tensor;
}

Tensor Tensor::transpose(const std::initializer_list<std::size_t> axes_ilist) const {
    const std::size_t n = rank();
    if (axes_ilist.size() != n) throw std::runtime_error("axis Size Must be same with rank.");

    std::vector<std::size_t> axes(axes_ilist.begin(), axes_ilist.end());

    std::vector<bool> seen(n, false);
    for (auto a : axes) {
        if (a >= n) throw std::runtime_error("axis index out of range");
        if (seen[a]) throw std::runtime_error("duplicate axis");
        seen[a] = true;
    }

    bool identity = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (axes[i] != i) {
            identity = false;
            break;
        }
    }
    if (identity) return *this;

    Tensor new_tensor = *this;
    std::vector<std::size_t> new_shape(n);
    std::vector<std::size_t> new_strides(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = axes[i];
        new_shape[i] = shape_[src];
        new_strides[i] = strides_[src];
    }

    new_tensor.shape_ = Shape(new_shape);
    new_tensor.strides_ = std::move(new_strides);

    return new_tensor;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    if (numel() == 0) {
        Tensor t(shape_, dtype_, std::make_shared<Storage>(storage_->alloc_));
        t.storage_->nbytes = 0;
        t.storage_->data = nullptr;
        return t;
    }

    auto item = itemsize();
    auto alloc = storage_->alloc_;
    auto new_storage = std::make_shared<Storage>(alloc);
    new_storage->nbytes = nbytes();
    new_storage->data = alloc->allocate(new_storage->nbytes);

    Tensor new_tensor(shape_, dtype_, new_storage);

    // data iter
    const auto* src = static_cast<const std::byte*>(data());
    auto* dst = static_cast<std::byte*>(new_tensor.data());

    const auto& dims = shape_.dims();
    const auto& st = strides_;

    detail::NdCounter counter(dims);
    std::size_t dst_offset = 0;
    while (counter.done()) {
        auto src_offset = detail::offset_elems(counter.idx, st) * item;
        std::memcpy(dst + dst_offset, src + src_offset, item);
        dst_offset += item;
        counter.next();
    }
    return new_tensor;
}

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
