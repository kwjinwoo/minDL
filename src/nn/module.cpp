#include "minidl/nn.h"
#include "minidl/ops.h"
#include "minidl/tensor.h"

namespace minidl::nn {

Linear::Linear(std::size_t in_features, std::size_t out_features, bool use_bias)
    : Module("Linear"), use_bias_(use_bias) {
    float bound = std::sqrt(6.0f / in_features);
    weight_ = Tensor::rand_uniform({out_features, in_features}, -bound, bound, DType::f32, nullptr, true);
    register_parameter("weight", &weight_);
    if (use_bias_) {
        bias_ = Tensor::zeros({out_features}, DType::f32, nullptr, true);
        register_parameter("bias", &bias_);
    }
}

Tensor Linear::forward(const Tensor& x) const {
    Tensor matmul_out = ops::matmul(x, weight_.transpose({1, 0}));
    if (use_bias_) {
        return ops::add(matmul_out, bias_);
    } else {
        return matmul_out;
    }
}

}  // namespace minidl::nn
