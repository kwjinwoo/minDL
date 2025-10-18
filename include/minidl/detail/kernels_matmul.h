#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl::detail {

template <typename T>
Tensor naive_2d_matmul(const Tensor& a, const Tensor& b, DType promoted_dtype) {
    // rank check
    if (a.rank() != 2 || b.rank() != 2) throw std::runtime_error("naive_2d_matmul: both inputs must be rank-2");

    // check inner dims
    if (a.shape().dims()[1] != b.shape().dims()[0])
        throw std::runtime_error("naive_2d_matmul: inner dims must match (a.shape[1] == b.shape[0])");

    const std::size_t M = a.shape().dims()[0];
    const std::size_t K = a.shape().dims()[1];
    const std::size_t N = b.shape().dims()[1];

    // out tensor
    auto c = Tensor::zeros(Shape({M, N}), promoted_dtype);
    // data reader
    detail::ElementReader ra(a.data(), a.dtype());
    detail::ElementReader rb(b.data(), b.dtype());

    // strides
    const auto& astr = a.strides();
    const auto& bstr = b.strides();
    const auto& cstr = c.strides();

    T* z = static_cast<T*>(c.data());
    for (std::size_t i = 0; i < M; i++) {
        const auto a_base = astr[0] * i;  // (i, 0)
        for (std::size_t j = 0; j < N; j++) {
            T acc = static_cast<T>(0);

            const auto b_base = bstr[1] * j;  // (0, j)
            for (std::size_t k = 0; k < K; k++) {
                const auto a_offset = astr[1] * k + a_base;
                const auto b_offset = bstr[0] * k + b_base;
                acc += ra.read_as<T>(a_offset) * rb.read_as<T>(b_offset);
            }
            const auto z_offset = cstr[0] * i + cstr[1] * j;
            z[z_offset] = acc;
        }
    }
    return c;
}

}  // namespace minidl::detail
