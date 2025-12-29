#include <algorithm>
#include <cmath>

#include "minidl/detail/iter.h"
#include "minidl/ops.h"

namespace minidl::ops {

inline Tensor reduce_max(const Tensor& tensor) {
    if (tensor.rank() != 2) throw std::runtime_error("reduce_max: tensor rank Must be 2.");
    const std::vector<std::size_t> dims = tensor.shape().dims();
    const std::vector<std::size_t>& strides = tensor.strides();
    const float* data = static_cast<const float*>(tensor.data());
    std::vector<std::size_t> out_shape = {dims[0], 1};

    Tensor out = Tensor::zeros(Shape(out_shape));
    const std::vector<std::size_t> out_strides = out.strides();
    float* out_data = static_cast<float*>(out.data());
    std::vector<std::size_t> out_idx(2, 0);  // only last dim is reudced.
    detail::NdCounter counter(dims);
    for (; !counter.done(); counter.next()) {
        auto idx = counter.idx;
        out_idx[0] = idx[0];  // batch dim

        auto in_off = detail::offset_elems(idx, strides);
        auto out_off = detail::offset_elems(out_idx, out_strides);

        // assign first row value.
        if (idx[1] == 0) {
            out_data[out_off] = data[in_off];
            continue;
        }

        if (out_data[out_off] < data[in_off]) out_data[out_off] = data[in_off];
    }
    return out;
}

inline Tensor exp(const Tensor& tensor) {
    const float* data = static_cast<const float*>(tensor.data());
    std::vector<std::size_t> strides = tensor.strides();

    Tensor out = Tensor::zeros_like(tensor);
    float* out_data = static_cast<float*>(out.data());
    std::vector<std::size_t> out_strides = out.strides();

    detail::NdCounter counter(tensor.shape().dims());
    for (; !counter.done(); counter.next()) {
        auto idx = counter.idx;

        auto in_off = detail::offset_elems(idx, strides);
        auto out_off = detail::offset_elems(idx, out_strides);  // tensor and out has same shape.

        out_data[out_off] = std::exp(data[in_off]);
    }
    return out;
}

inline Tensor neg_log_softmax(const Tensor& numerator, const Tensor& denominator) {
    constexpr float eps = 1e-6f;

    if (numerator.rank() != 2 || denominator.rank() != 2)
        throw std::runtime_error("neg_log_softmax: numerator or denominator Must have 2 rank.");
    if (denominator.shape().dims()[0] != numerator.shape().dims()[0] || denominator.shape().dims()[1] != 1)
        throw std::runtime_error(
            "neg_log_softmax: numerator and denominator Must have same batch size and denominator is already reduced.");
    // numerator shape : [N, C]
    // denominator shape : [N, 1]
    const std::vector<std::size_t>& numerator_dims = numerator.shape().dims();
    const std::vector<std::size_t>& numerator_strides = numerator.strides();
    const float* numerator_data = static_cast<const float*>(numerator.data());

    const std::vector<std::size_t>& denominator_strides = denominator.strides();
    const float* denominator_data = static_cast<const float*>(denominator.data());

    Tensor out = Tensor::zeros_like(numerator);
    float* out_data = static_cast<float*>(out.data());
    const std::vector<std::size_t>& out_strides = out.strides();

    detail::NdCounter counter(numerator_dims);
    std::vector<std::size_t> denominator_idx(2, 0);
    for (; !counter.done(); counter.next()) {
        auto idx = counter.idx;
        denominator_idx[0] = idx[0];

        auto numerator_off = detail::offset_elems(idx, numerator_strides);
        auto out_off = detail::offset_elems(idx, out_strides);
        auto denominator_off = detail::offset_elems(denominator_idx, denominator_strides);

        out_data[out_off] =
            -(numerator_data[numerator_off] - std::log(std::max(denominator_data[denominator_off], eps)));
    }
    return out;
}

inline Tensor gather_sum(const Tensor& input, const Tensor& target) {
    // input shape : [N, C]
    // target shape : [N, 1], dtype Must be int32
    const std::vector<std::size_t>& dims = input.shape().dims();
    const std::size_t batch_size = dims[0];
    const std::vector<std::size_t>& inp_strides = input.strides();
    const std::vector<std::size_t>& target_strides = target.strides();

    const float* inp_data = static_cast<const float*>(input.data());
    const int32_t* target_data = static_cast<const int32_t*>(target.data());

    Tensor out = Tensor::zeros(Shape({batch_size, 1}));
    const std::vector<std::size_t>& out_strides = out.strides();
    float* out_data = static_cast<float*>(out.data());

    std::vector<std::size_t> target_idx(2, 0);
    std::vector<std::size_t> out_idx(2, 0);
    std::vector<std::size_t> inp_idx(2, 0);
    for (std::size_t i = 0; i < batch_size; i++) {
        target_idx[0] = i;
        out_idx[0] = i;
        inp_idx[0] = i;

        auto target_off = detail::offset_elems(target_idx, target_strides);
        const std::size_t class_ = static_cast<const std::size_t>(target_data[target_off]);

        inp_idx[1] = class_;
        auto inp_off = detail::offset_elems(inp_idx, inp_strides);

        auto out_off = detail::offset_elems(out_idx, out_strides);
        out_data[out_off] = inp_data[inp_off];
    }

    Tensor reduced = ops::sum(out, false);
    return reduced;
}

Tensor cross_entropy(const Tensor& input, const Tensor& target) {
    // support only 2D (N, C) case
    if (input.rank() != 2) throw std::runtime_error("cross_entropy: Input Tensor rank Must be 2.");

    // support only fp32
    if (input.dtype() != DType::f32) throw std::runtime_error("cross_entroph: Input Tensor's DType Must be f32.");

    const float batch_size = static_cast<const float>(input.shape().dims()[0]);
    // softmax about axis 1
    auto max_val = reduce_max(input);
    Tensor sub_val = ops::sub(input, max_val);  // x - x_max, broadcast is guranteed.

    Tensor numerator = sub_val;
    Tensor denominator = ops::sum(exp(numerator), {1}, true);

    // log
    Tensor log_val = neg_log_softmax(numerator, denominator);

    // gather sum
    Tensor out = gather_sum(log_val, target);
    out = ops::div(out, Tensor::from_scalar(batch_size));

    // later, should implement backward fn
    // out.impl()->grad_fn = std::make_shared<CrossEntropyBackward>(input, ...);
    return out;
}
}  // namespace minidl::ops
