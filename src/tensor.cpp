#include "minidl/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minidl/allocators/default.h"
#include "minidl/indexing.h"

namespace minidl {

std::vector<std::size_t> Tensor::default_strides(const Shape& shape) {
    // stride in element
    const auto dims = shape.dims();
    const int64_t n = static_cast<int64_t>(shape.rank());

    std::vector<std::size_t> strides;

    if (n == 0) {
        // empty strides
        return strides;
    }

    strides.resize(n);
    strides[n - 1] = 1;

    for (int64_t i = n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

void Tensor::fill_ones_(void* data, size_t numel, DType dtype) {
    if (!data) return;
    switch (dtype) {
        case DType::f32: {
            auto* x = static_cast<float*>(data);
            std::fill_n(x, numel, 1.0f);
            break;
        }
        case DType::i32: {
            auto* x = static_cast<std::int32_t*>(data);
            std::fill_n(x, numel, 1);
            break;
        }
        default:
            throw std::runtime_error("Unsupported DType in fill_ones");
    }
}

bool Tensor::is_contiguous() const noexcept {
    // 0 element is conventionally contiguous
    if (numel() == 0) return true;

    std::int64_t expected = 1;
    for (std::int64_t d = static_cast<std::int64_t>(rank()) - 1; d >= 0; d--) {
        const int64_t dim = static_cast<std::int64_t>(shape_[d]);
        const int64_t s = static_cast<std::int64_t>(strides_[d]);

        if (dim == 1) {
            continue;
        }

        if (s != expected) {
            return false;
        }
        expected *= dim;
    }
    return true;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, std::shared_ptr<Allocator> alloc) {
    if (alloc == nullptr) alloc = get_default_allocator();
    auto storage = std::make_shared<Storage>(alloc);

    Tensor t(shape, dtype, storage);
    t.strides_ = t.default_strides(shape);

    t.storage_->nbytes = t.numel() * t.itemsize();

    if (t.nbytes() == 0) {
        t.storage_->data = nullptr;
        return t;
    }
    t.storage_->data = t.storage_->alloc_->allocate(t.nbytes());

    if (!t.data()) throw std::bad_alloc{};
    std::memset(t.data(), 0, t.nbytes());
    return t;
}

Tensor Tensor::ones(const Shape& shape, DType dtype, std::shared_ptr<Allocator> alloc) {
    if (alloc == nullptr) alloc = get_default_allocator();
    auto storage = std::make_shared<Storage>(alloc);

    Tensor t(shape, dtype, storage);
    t.strides_ = t.default_strides(shape);

    t.storage_->nbytes = t.numel() * t.itemsize();

    if (t.nbytes() == 0) {
        t.storage_->data = nullptr;
        return t;
    }
    t.storage_->data = t.storage_->alloc_->allocate(t.nbytes());
    if (!t.data()) throw std::bad_alloc{};

    t.fill_ones_(t.data(), t.numel(), dtype);
    return t;
}

Tensor Tensor::arange(std::size_t size, DType dtype, std::shared_ptr<Allocator> alloc) {
    if (alloc == nullptr) alloc = get_default_allocator();
    auto storage = std::make_shared<Storage>(alloc);

    Shape s({size});
    Tensor t(s, dtype, storage);
    t.strides_ = t.default_strides(s);

    t.storage_->nbytes = t.numel() * t.itemsize();

    if (t.nbytes() == 0) {
        t.storage_->data = nullptr;
        return t;
    }

    t.storage_->data = t.storage_->alloc_->allocate(t.nbytes());
    if (!t.storage_->data) throw std::bad_alloc();

    const std::size_t n = t.numel();
    if (dtype == DType::f32) {
        auto* x = static_cast<float*>(t.data());
        float v = 0.0f;
        for (std::size_t i = 0; i < n; i++) {
            x[i] = v;
            v += 1.0f;
        }
    } else if (dtype == DType::i32) {
        auto* x = static_cast<std::int32_t*>(t.data());
        std::int32_t v = 0;
        for (std::size_t i = 0; i < n; i++) {
            x[i] = v;
            v += 1;
        }
    } else {
        throw std::runtime_error("Unsupported DType in arange");
    }
    return t;
}

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

    NdCounter counter(dims);
    std::size_t dst_offset = 0;
    while (counter.done()) {
        auto src_offset = offset_elems(counter.idx, st) * item;
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
    // not yet implement broadcasting. for now, only support element wise add.
    if (shape_.dims() != rhs.shape_.dims()) {
        throw std::runtime_error("Shape Must be same.");
    }

    Tensor out = zeros(shape_, dtype_, storage_->alloc_);
    const std::size_t n = numel();

    if (n == 0) return out;

    // x + y = z
    auto add_kernel = [&](auto* z, auto* x, auto* y) {
        // Fast path
        if (is_contiguous() && rhs.is_contiguous() && strides() == rhs.strides()) {
            for (std::size_t i = 0; i < n; i++) {
                z[i] = x[i] + y[i];
            }
            return;
        }

        NdCounter counter(shape_.dims());
        const auto& xs = strides();
        const auto& ys = rhs.strides();
        std::size_t z_offset = 0;
        for (; counter.done(); counter.next()) {
            auto x_offset = offset_elems(counter.idx, xs);
            auto y_offset = offset_elems(counter.idx, ys);

            z[z_offset] = x[x_offset] + y[y_offset];
            z_offset += 1;
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
