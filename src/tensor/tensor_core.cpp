#include "minidl/tensor.h"

namespace minidl {

// constructor and deleter
Tensor::Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage)
    : shape_(shape), dtype_(dtype), storage_(std::move(storage)) {}
Tensor::~Tensor() = default;

// to
void Tensor::to(DType d) noexcept {
    if (d == dtype_) return;
    dtype_ = d;
}

}  // namespace minidl
