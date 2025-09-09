#include "minidl/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minidl/allocators/default.h"

namespace minidl {
std::vector<int64_t> Tensor::default_strides(const Shape& shape, const DType& dtype) {
    const auto dims = shape.dims();
    const int64_t n = static_cast<int64_t>(shape.rank());

    std::vector<int64_t> strides;

    if (n == 0) {
        // empty strides
        return strides;
    }

    strides.resize(n);

    int64_t item_size = static_cast<int64_t>(size_of(dtype));

    strides[n - 1] = item_size;

    for (int64_t i = n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, std::shared_ptr<Allocator> alloc) {
    if (alloc == nullptr) alloc = get_default_allocator();
    auto storage = std::make_shared<Storage>(alloc);

    Tensor t(shape, dtype, storage);
    t.strides_ = t.default_strides(shape, dtype);

    t.storage_->nbytes = shape.numel() * size_of(dtype);
    t.storage_->data = (t.storage_->nbytes == 0) ? nullptr : t.storage_->alloc_->allocate(t.storage_->nbytes);

    if (t.storage_->data) {
        std::memset(t.storage_->data, 0, t.storage_->nbytes);
    }
    return t;
}
}  // namespace minidl
