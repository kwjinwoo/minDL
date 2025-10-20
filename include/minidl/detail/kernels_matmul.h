#include "minidl/detail/broadcasting.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl::detail {

using Vec = std::vector<std::size_t>;

Vec get_batch_shape(const Vec& dims) {
    const auto size = dims.size();
    if (size < 2) throw std::runtime_error("get_batch_shape: Tensor rank must be >= 2");

    Vec batch_shape;
    if (size > 2) batch_shape.insert(batch_shape.end(), dims.begin(), dims.end() - 2);
    return batch_shape;
}

inline Vec get_output_shape(const Vec& broadcast_batch_shape, const std::size_t M, const std::size_t N) {
    Vec out = broadcast_batch_shape;
    out.push_back(M);
    out.push_back(N);
    return out;
}

template <typename T>
void gemm2d_native(void* __restrict x, void* __restrict y, T* z, const std::size_t M, const std::size_t N,
                   const std::size_t K, const Vec& astr, const Vec& bstr, const Vec& cstr, DType x_dtype, DType y_dtype,
                   const std::size_t x_batch_offset, const std::size_t y_batch_offset,
                   const std::size_t z_batch_offset) {
    // data reader
    detail::ElementReader rx(x, x_dtype);
    detail::ElementReader ry(y, y_dtype);

    for (std::size_t i = 0; i < M; i++) {
        const auto a_base = astr[0] * i;  // (i, 0)
        for (std::size_t j = 0; j < N; j++) {
            T acc = static_cast<T>(0);

            const auto b_base = bstr[1] * j;  // (0, j)
            for (std::size_t k = 0; k < K; k++) {
                const auto a_offset = astr[1] * k + a_base + x_batch_offset;
                const auto b_offset = bstr[0] * k + b_base + y_batch_offset;
                acc += rx.read_as<T>(a_offset) * ry.read_as<T>(b_offset);
            }
            const auto z_offset = cstr[0] * i + cstr[1] * j + z_batch_offset;
            z[z_offset] = acc;
        }
    }
}

template <typename T>
Tensor batched_gemm2d_native(const Tensor& a, const Tensor& b, DType promote_dtype) {
    // get batch shapes and strides
    const Vec a_batch_shape = get_batch_shape(a.shape().dims());
    const Vec b_batch_shape = get_batch_shape(b.shape().dims());
    const Vec a_batch_strides = get_batch_shape(a.strides());
    const Vec b_batch_strides = get_batch_shape(b.strides());

    // get 2d shapes and strides
    const Vec a_gemm2d_shape(a.shape().dims().end() - 2, a.shape().dims().end());
    const Vec b_gemm2d_shape(b.shape().dims().end() - 2, b.shape().dims().end());
    const Vec a_gemm2d_strides(a.strides().end() - 2, a.strides().end());
    const Vec b_gemm2d_strides(b.strides().end() - 2, b.strides().end());

    // inner dim check
    if (a_gemm2d_shape[1] != b_gemm2d_shape[0])
        throw std::runtime_error("batched_gemm2d_native: inner dims must match (a.shape[1] == b.shape[0])");
    const std::size_t M = a_gemm2d_shape[0];
    const std::size_t K = a_gemm2d_shape[1];
    const std::size_t N = b_gemm2d_shape[1];

    // get broadcast shape
    Vec broadcast_batch_shape = detail::compute_broadcast_shape(a_batch_shape, b_batch_shape);
    Vec a_expanded_strides =
        detail::expand_strides_for_broadcast(a_batch_shape, a_batch_strides, broadcast_batch_shape);
    Vec b_expanded_strides =
        detail::expand_strides_for_broadcast(b_batch_shape, b_batch_strides, broadcast_batch_shape);

    // zero c
    const auto out_shape = get_output_shape(broadcast_batch_shape, M, N);
    Tensor c = Tensor::zeros(Shape(out_shape), promote_dtype);
    const Vec c_batch_strides(get_batch_shape(c.strides()));
    const Vec c_gemm_strides(c.strides().end() - 2, c.strides().end());
    T* c_data = static_cast<T*>(c.data());

    // iter batch
    const std::size_t num_batches =
        std::accumulate(broadcast_batch_shape.begin(), broadcast_batch_shape.end(), 1, std::multiplies<std::size_t>());
    const Vec radix = compute_radix(broadcast_batch_shape);

#pragma omp parallel for schedule(static)
    for (std::int64_t blin = 0; blin < (std::int64_t)num_batches; ++blin) {
        const auto a_batch_offset =
            detail::linear_to_offset((std::size_t)blin, broadcast_batch_shape, radix, a_expanded_strides);
        const auto b_batch_offset =
            detail::linear_to_offset((std::size_t)blin, broadcast_batch_shape, radix, b_expanded_strides);
        const auto c_batch_offset =
            detail::linear_to_offset((std::size_t)blin, broadcast_batch_shape, radix, c_batch_strides);

        gemm2d_native<T>(a.data(), b.data(), c_data, M, N, K, a_gemm2d_strides, b_gemm2d_strides, c_gemm_strides,
                         a.dtype(), b.dtype(), a_batch_offset, b_batch_offset, c_batch_offset);
    }
    return c;
}

}  // namespace minidl::detail
