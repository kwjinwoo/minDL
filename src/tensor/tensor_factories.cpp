#include "minidl/allocators/default.h"
#include "minidl/tensor.h"

namespace minidl {

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

}  // namespace minidl
