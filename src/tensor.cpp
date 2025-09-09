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
            auto* x = static_cast<int*>(data);
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
}  // namespace minidl
