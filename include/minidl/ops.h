#pragma once
#include "minidl/tensor.h"

namespace minidl::ops {

Tensor add(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor mul(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor sub(const Tensor& /*lhs*/, const Tensor& /*rhs*/);
Tensor div(const Tensor& /*lhs*/, const Tensor& /*rhs*/);

}  // namespace minidl::ops
