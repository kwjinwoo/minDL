#pragma once
#include "minidl/tensor.h"

namespace minidl::ops {

// binary ops
Tensor add(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor mul(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor sub(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor div(const Tensor& /*lhs*/, const Tensor& /*rhs*/);

// relu
Tensor relu(const Tensor& /*tensor*/);

// sigmoid
Tensor sigmoid(const Tensor& /*tensor*/);

// matmul
Tensor matmul(const Tensor& /*lhs*/, const Tensor& /*rhs*/);

// sum
Tensor sum(const Tensor& tensor, const std::vector<std::size_t>& axes, bool keepdims = true);
Tensor sum(const Tensor& tensor, bool keepdims = true);
}  // namespace minidl::ops
