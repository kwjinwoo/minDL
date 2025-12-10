#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(SumOp, SimpleFloatValues) {
    // Tensor: [1.0, 2.0, 3.0, 4.0]
    Tensor x = Tensor::zeros({4}, DType::f32, nullptr, false);
    float* x_data = static_cast<float*>(x.data());
    x_data[0] = 1.0f;
    x_data[1] = 2.0f;
    x_data[2] = 3.0f;
    x_data[3] = 4.0f;

    Tensor y = ops::sum(x);

    ASSERT_EQ(y.numel(), 1);           // scalar
    EXPECT_EQ(y.dtype(), DType::f32);  // same dtype

    float* y_data = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(*y_data, 10.0f);
}

TEST(SumOp, NegativeAndZeroValues) {
    // Tensor: [ -2, 0, 5 ]
    Tensor x = Tensor::zeros({3}, DType::f32, nullptr, false);
    float* d = static_cast<float*>(x.data());
    d[0] = -2.0f;
    d[1] = 0.0f;
    d[2] = 5.0f;

    Tensor y = ops::sum(x);
    float* yd = static_cast<float*>(y.data());

    EXPECT_FLOAT_EQ(*yd, 3.0f);
}

TEST(SumOp, Contiguous2DTensor) {
    // 2x3 matrix:
    // [1, 2, 3]
    // [4, 5, 6]  → sum = 21
    Tensor x = Tensor::zeros({2, 3}, DType::f32, nullptr, false);
    float* d = static_cast<float*>(x.data());
    for (int i = 0; i < 6; ++i) d[i] = float(i + 1);

    Tensor y = ops::sum(x);
    float* yd = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(*yd, 21.0f);
}

TEST(SumOp, NonContiguousTensor_Transpose) {
    // ([[1, 2, 3],
    //   [4, 5, 6]]).transpose → non-contiguous view
    Tensor x = Tensor::zeros({2, 3}, DType::f32, nullptr, false);

    float* d = static_cast<float*>(x.data());
    for (int i = 0; i < 6; ++i) d[i] = float(i + 1);

    Tensor x_t = x.transpose({1, 0});  // shape = [3,2], non-contiguous
    ASSERT_FALSE(x_t.is_contiguous());

    Tensor y = ops::sum(x_t);
    float* yd = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(*yd, 21.0f);  // same values, same sum
}

TEST(SumOp, EmptyTensor) {
    Tensor x = Tensor::zeros({0}, DType::f32, nullptr, false);
    Tensor y = ops::sum(x);

    float* yd = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(*yd, 0.0f);  // convention: empty sum = 0
}

static Tensor make_2x3_tensor_1_to_6(bool requires_grad = false) {
    Tensor x = Tensor::zeros({2, 3}, DType::f32, nullptr, requires_grad);
    auto* data = static_cast<float*>(x.data());
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    data[4] = 5.0f;
    data[5] = 6.0f;
    return x;
}

TEST(SumOp, SumAllNoKeepdims) {
    auto x = make_2x3_tensor_1_to_6(false);
    std::vector<std::size_t> axes;  // empty
    Tensor y = ops::sum(x, axes, /*keepdims=*/false);

    // rank 0 scalar라고 가정: shape().dims().size() == 0
    const auto& out_dims = y.shape().dims();
    EXPECT_EQ(out_dims.size(), 0u);

    auto* y_data = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(y_data[0], 21.0f);
}

TEST(SumOp, SumAllKeepdims) {
    auto x = make_2x3_tensor_1_to_6(false);

    std::vector<std::size_t> axes;  // empty
    Tensor y = ops::sum(x, axes, /*keepdims=*/true);

    const auto& out_dims = y.shape().dims();
    ASSERT_EQ(out_dims.size(), 2u);
    EXPECT_EQ(out_dims[0], 1u);
    EXPECT_EQ(out_dims[1], 1u);

    auto* y_data = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(y_data[0], 21.0f);
}