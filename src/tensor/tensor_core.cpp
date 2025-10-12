#include "minidl/tensor.h"

namespace minidl {

// constructor and deleter
Tensor::Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage)
    : shape_(shape), dtype_(dtype), storage_(std::move(storage)) {}
Tensor::~Tensor() = default;

// to
Tensor Tensor::to(DType d) noexcept {
    Tensor out = *this;
    if (d == out.dtype_) return out;
    out.dtype_ = d;
    return out;
}

}  // namespace minidl
