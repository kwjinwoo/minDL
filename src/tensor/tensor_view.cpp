#include <cstddef>

#include "minidl/allocators/default.h"
#include "minidl/autograd/grad_fn.h"
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl {

Tensor Tensor::view(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::runtime_error("view: new_shape.numel() must equal the current numel().");
    }
    if (numel() != 0 && !is_contiguous()) {
        throw std::runtime_error("view: tensor must be contiguous (use reshape for non-contiguous).");
    }

    auto new_impl_ = std::make_shared<TensorImpl>(*impl_);

    new_impl_->shape = new_shape;
    new_impl_->strides = default_strides(new_shape);

    new_impl_->grad.reset();
    new_impl_->grad_fn.reset();

    Tensor out;
    out.impl_ = std::move(new_impl_);

    if (requires_grad()) {
        out.impl_->grad_fn = std::make_shared<ViewBackward>(*this, shape());
        out.impl_->requires_grad = true;
    }
    return out;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::runtime_error("reshape: new_shape.numel() must equal the current numel().");
    }
    if (numel() == 0 || is_contiguous()) {
        auto new_impl_ = std::make_shared<TensorImpl>(*impl_);
        new_impl_->shape = new_shape;
        new_impl_->strides = default_strides(new_shape);

        new_impl_->grad.reset();
        new_impl_->grad_fn.reset();

        Tensor out;
        out.impl_ = std::move(new_impl_);

        if (requires_grad()) {
            out.impl_->grad_fn = std::make_shared<ReshapeBackward>(*this, shape());
            out.impl_->requires_grad = true;
        }
        return out;
    }

    Tensor new_tensor = this->contiguous();
    new_tensor.impl_->shape = new_shape;
    new_tensor.impl_->strides = default_strides(new_shape);

    if (requires_grad()) {
        new_tensor.impl_->grad_fn = std::make_shared<ReshapeBackward>(*this, shape());
        new_tensor.impl_->requires_grad = true;
    }

    return new_tensor;
}

Tensor Tensor::transpose(const std::vector<std::size_t> axes) const {
    const std::size_t n = rank();
    if (axes.size() != n) throw std::runtime_error("axis Size Must be same with rank.");

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

    std::vector<std::size_t> new_shape(n);
    std::vector<std::size_t> new_strides(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = axes[i];
        new_shape[i] = impl_->shape[src];
        new_strides[i] = impl_->strides[src];
    }

    // create new impl
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->shape = Shape(new_shape);
    new_impl->strides = std::move(new_strides);

    new_impl->grad.reset();
    new_impl->grad_fn.reset();

    Tensor out;
    out.impl_ = std::move(new_impl);

    if (impl_->requires_grad) {
        auto grad_fn = std::make_shared<TransposeBackward>(*this, std::vector<std::size_t>(axes.begin(), axes.end()));
        out.impl()->grad_fn = grad_fn;
        out.impl()->requires_grad = true;
    }
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    if (numel() == 0) {
        Tensor t(shape(), dtype(), std::make_shared<Storage>(impl_->storage->alloc_));
        t.storage()->nbytes = 0;
        t.storage()->data = nullptr;
        return t;
    }

    auto item = itemsize();
    auto alloc = storage()->alloc_;
    auto new_storage = std::make_shared<Storage>(alloc);
    new_storage->nbytes = nbytes();
    new_storage->data = alloc->allocate(new_storage->nbytes);

    Tensor new_tensor(shape(), dtype(), new_storage);

    // data iter
    const auto* src = static_cast<const std::byte*>(data());
    auto* dst = static_cast<std::byte*>(new_tensor.data());

    const auto& dims = shape().dims();
    const auto& st = strides();

    detail::NdCounter counter(dims);
    std::size_t dst_offset = 0;
    while (!counter.done()) {
        auto src_offset = detail::offset_elems(counter.idx, st) * item;
        std::memcpy(dst + dst_offset, src + src_offset, item);
        dst_offset += item;
        counter.next();
    }

    return new_tensor;
}

}  // namespace minidl
