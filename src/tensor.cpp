#include "minidl/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minidl/allocators/default.h"

namespace minidl {

std::vector<int64_t> Tensor::default_strides(const Shape& shape) {
    // stride in element
    const auto dims = shape.dims();
    const int64_t n = static_cast<int64_t>(shape.rank());

    std::vector<int64_t> strides;

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

void Tensor::fill_ones_(void* data, int64_t numel, DType dtype) {
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

    int64_t expected = 1;
    for (int64_t d = static_cast<int64_t>(rank()) - 1; d >= 0; d--) {
        const int64_t dim = static_cast<int64_t>(shape_[d]);
        const int64_t s = strides_[d];

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
    const int64_t r = rank();

    for (int64_t linear = 0; linear < numel(); linear++) {
        int64_t rem = linear;
        int64_t scr_elem_offset = 0;

        for (int64_t d = r - 1; d >= 0; d--) {
            const std::size_t dim = dims[d];
            const std::size_t idx_d = rem % dim;

            rem /= dim;
            scr_elem_offset += idx_d * st[d];
        }
        std::memcpy(dst + linear * item, src + scr_elem_offset * item, item);
    }
    return new_tensor;
}
}  // namespace minidl
